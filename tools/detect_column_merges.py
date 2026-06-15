#!/usr/bin/env python3
"""Find pages with *genuine* column-merge alignment failures.

Background
----------
``align_ocr.py`` flags ``possible_column_merge`` from the Surya:Gemini
line-count ratio, but that signal is noisy: it also fires when Surya simply
over-segments a low-resolution scan (see the empirical test in
docs/comparison-htr-alto-pipeline.md). A *genuine* column merge is different —
it shows up as a single aligned line that contains **two entries**, because
Gemini read horizontally across two columns and concatenated left-column-row-N
with right-column-row-N.

Detection modes
---------------
``--mode address`` (default) — precise, for directory volumes. Fingerprints a
  merge as **two street addresses on one line** (each entry carries its own
  address). High precision, but only works where entries are addresses.

``--mode generic`` — content-agnostic, for *any* volume (copyright ledgers,
  passenger lists, name/date rosters, flower dictionaries, …). On a page that
  Surya geometry shows is **multi-column**, a line whose character *and* word
  counts are a strong outlier above the page median (≈ two entries concatenated)
  is flagged. No knowledge of the entry content is required. The multi-column
  gate keeps single-column prose — where long lines are normal — from tripping it.

``--mode both`` — union of the two (address precision plus generic coverage).

Resolution
----------
Each row reports the page's *download* width (Surya image width) and estimated
column count; the line-crop fallback only produces clean crops at ≳1500px width,
so low-res pages are de-prioritized.

Usage
-----
    python tools/detect_column_merges.py output/                         # directory volumes
    python tools/detect_column_merges.py output/copyright_ledgers_... --mode generic
    python tools/detect_column_merges.py output/ --mode both --show 3
    python tools/detect_column_merges.py output/ --mode generic --outlier-factor 1.7

Output is a ranked table; ``--show N`` prints N suspected merged lines per page
so the heuristic can be eyeballed.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

# Street-type suffixes seen across these directories. Matched as whole tokens,
# optionally abbreviated with a trailing period.
_STREET_SUFFIX = re.compile(
    r"\b(?:av|ave|st|street|blvd|bd|rd|road|ter|terr|pl|place|dr|drive|ct|"
    r"court|ln|lane|hwy|pkwy|sq|al|alley|cir)\b\.?",
    re.IGNORECASE,
)
# A street number: 1–5 digits, optionally with a 1/2 fraction, as a whole token.
_STREET_NUM = re.compile(r"\b\d{1,5}(?:\s*1/2)?\b")
# Entry separator used in these directories (em-dash between name and address).
_EMDASH = "—"


# ---------------------------------------------------------------------------
# Address signal (precise, directory volumes)
# ---------------------------------------------------------------------------

def _address_count(text: str) -> int:
    """Heuristic count of distinct street addresses in a line.

    An address ≈ a street number followed (within a short window) by a street
    suffix. Counting matched (number, suffix) pairs left-to-right avoids
    double-counting a lone year or membership figure that has no suffix.
    """
    nums = [m.start() for m in _STREET_NUM.finditer(text)]
    sufs = [m.start() for m in _STREET_SUFFIX.finditer(text)]
    if not nums or not sufs:
        return 0
    count = 0
    si = 0
    for npos in nums:
        # find the first suffix that starts after this number, within 40 chars
        while si < len(sufs) and sufs[si] < npos:
            si += 1
        if si < len(sufs) and sufs[si] - npos <= 40:
            count += 1
            si += 1  # consume this suffix so it can't pair with a later number
    return count


def _looks_merged(text: str, median_len: int) -> bool:
    """True if a line looks like two concatenated directory entries (address signal)."""
    if _address_count(text) >= 2:
        return True
    # Fallback: very long line carrying two name→address separators.
    if len(text) > max(80, int(median_len * 2.0)) and text.count(_EMDASH) >= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Generic signal (content-agnostic): length outlier on a multi-column page
# ---------------------------------------------------------------------------

def _estimate_columns(boxes: list, page_width: int) -> int:
    """Estimate the number of text columns from line boxes via left-edge clustering.

    Bins each box's left edge; a bin holding ≥12% of boxes is a column anchor,
    and anchors separated by a >10%-page-width gap are distinct columns. Returns
    1 when there are too few boxes to judge (so single-column prose is exempt
    from the generic merge signal).
    """
    if len(boxes) < 6 or page_width <= 0:
        return 1
    bin_w = max(page_width // 20, 1)
    hist: dict[int, int] = {}
    for b in boxes:
        key = int(b[0]) // bin_w
        hist[key] = hist.get(key, 0) + 1
    thresh = len(boxes) * 0.12
    peaks = sorted(k for k, c in hist.items() if c >= thresh)
    if not peaks:
        return 1
    gap_bins = max(1, int(page_width * 0.10) // bin_w)
    cols = 1
    for prev, cur in zip(peaks, peaks[1:]):
        if cur - prev > gap_bins:
            cols += 1
    return cols


def _length_outlier_lines(texts: list, factor: float) -> list:
    """Content-agnostic merge candidates: strong char- *and* word-count outliers.

    A line is flagged when both its character length and its word count are at
    least ``factor`` × the page median — the signature of two entries fused into
    one line. Requiring both guards against a single legitimately long field.
    The caller gates this on a multi-column page; on single-column pages long
    lines are normal and this should not run.
    """
    real = [t for t in texts if t.strip()]
    if len(real) < 8:
        return []
    med_len = statistics.median(len(t) for t in real)
    med_wc = statistics.median(len(t.split()) for t in real)
    if med_len <= 0 or med_wc <= 0:
        return []
    return [
        t for t in real
        if len(t) >= factor * med_len and len(t.split()) >= factor * med_wc
    ]


def _merged_lines(
    texts: list, mode: str, factor: float, column_count: int, median_len: int
) -> list:
    """Return the suspected-merged lines for a page under the chosen mode."""
    addr = (
        [t for t in texts if _looks_merged(t, median_len)]
        if mode in ("address", "both") else []
    )
    generic = (
        _length_outlier_lines(texts, factor)
        if mode in ("generic", "both") and column_count >= 2 else []
    )
    if mode == "address":
        return addr
    if mode == "generic":
        return generic
    addr_set = set(addr)  # union, address first, no double-count
    return addr + [t for t in generic if t not in addr_set]


# ---------------------------------------------------------------------------
# Surya geometry (download width + column count)
# ---------------------------------------------------------------------------

def _find_surya(aligned_path: Path) -> "Path | None":
    """Locate the Surya JSON for an aligned page, across model-slug variations."""
    base = aligned_path.name.split("_gemini")[0]
    alt = re.sub(r"_[^_]+(?:-[^_]+)*_aligned\.json$", "", aligned_path.name)
    for stem in (base, alt):
        cands = list(aligned_path.parent.glob(stem + "*_surya.json"))
        if cands:
            return cands[0]
    return None


def _surya_geometry(path: "Path | None") -> "tuple[int, int]":
    """Return (image_width, column_count) from a Surya JSON, or (0, 1)."""
    if path is None:
        return 0, 1
    try:
        d = json.loads(path.read_text())
    except Exception:
        return 0, 1
    width = int(d.get("image_width") or 0)
    boxes = [ln["bbox"] for ln in d.get("lines", []) if ln.get("bbox")]
    return width, _estimate_columns(boxes, width)


def scan(root: Path, min_width: int, show: int, top: int, mode: str, factor: float) -> None:
    rows = []
    for p in sorted(root.rglob("*_aligned.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        lines = [l.get("gemini_text", "") for l in (d.get("lines") or [])]
        if len(lines) < 8:
            continue
        width, columns = _surya_geometry(_find_surya(p))
        # Geometry fallback: if no Surya JSON, estimate columns from aligned bboxes.
        if columns < 2 and width == 0:
            albx = [l["bbox"] for l in (d.get("lines") or []) if l.get("bbox")]
            pw = int(d.get("canvas_width") or (max((b[2] for b in albx), default=0)))
            columns = _estimate_columns(albx, pw)
        median_len = int(statistics.median([len(t) for t in lines])) or 1
        merged = _merged_lines(lines, mode, factor, columns, median_len)
        if not merged:
            continue
        try:
            vol = p.relative_to(root).parts[0]
        except Exception:
            vol = p.parent.name
        rows.append({
            "vol": vol,
            "page": p.name.split("_gemini")[0],
            "width": width,
            "cols": columns,
            "n_lines": len(lines),
            "n_merged": len(merged),
            "rate": len(merged) / len(lines),
            "examples": merged,
        })

    # Rank: prefer adequate resolution, then absolute merged-line count.
    rows.sort(key=lambda r: (r["width"] >= min_width, r["n_merged"], r["rate"]), reverse=True)

    print(f"{'merged':>6} {'/lines':>6} {'rate':>5} {'cols':>4} {'width':>5}  volume / page")
    print("-" * 104)
    for r in rows[:top]:
        ok = "✓" if r["width"] >= min_width else " "
        print(f"{r['n_merged']:>6} {r['n_lines']:>6} {r['rate']*100:>4.0f}% "
              f"{r['cols']:>4} {r['width']:>5}{ok} {r['vol'][:40]}  {r['page'][-26:]}")
        for ex in r["examples"][:show]:
            print(f"          ↳ {ex[:96]}")
    adequate = [r for r in rows if r["width"] >= min_width]
    print(f"\n{len(rows)} page(s) with ≥1 suspected merged line "
          f"(mode={mode}; {len(adequate)} at ≥{min_width}px, marked ✓).")
    print(
        "Worklist: re-OCR the pages above with a stronger model first "
        "(e.g. --ocr-model gemini-3-flash-preview, then --align-ocr --force) — "
        "that clears most merges at 1x cost. Reserve per-line crop OCR for ✓ "
        "pages that still merge afterward. See docs/comparison-htr-alto-pipeline.md."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Directory to scan (e.g. output/ or one volume)")
    ap.add_argument("--mode", choices=["address", "generic", "both"], default="address",
                    help="Merge signal: 'address' (precise, directory volumes; default), "
                         "'generic' (content-agnostic length outlier on multi-column "
                         "pages — any volume type), or 'both' (union).")
    ap.add_argument("--outlier-factor", type=float, default=1.8, metavar="F",
                    help="generic/both: flag a line when its char and word counts are "
                         "both ≥ F × the page median (default 1.8).")
    ap.add_argument("--min-width", type=int, default=1500,
                    help="Download width considered adequate for line-crop (default 1500)")
    ap.add_argument("--show", type=int, default=2,
                    help="Suspected merged lines to print per page (default 2)")
    ap.add_argument("--top", type=int, default=30, help="Pages to list (default 30)")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"not found: {root}")
    scan(root, args.min_width, args.show, args.top, args.mode, args.outlier_factor)


if __name__ == "__main__":
    main()
