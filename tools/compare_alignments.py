#!/usr/bin/env python3
"""Compare two sets of aligned OCR JSONs and score which alignment is better.

Alignment quality is measured without ground truth by exploiting a fact the
pipeline already gives us: every aligned line carries a Surya *bbox*, and the
Surya JSON records what text Surya itself read inside that bbox.  If alignment
assigned the right box to a line, the Gemini text and the Surya text for that
box describe the same words.  When Needleman-Wunsch mis-commits an anchor and
slides a run of lines onto the wrong boxes, that agreement collapses.

Metrics per page.  The starred ones are GEOMETRIC -- they do not depend on the
text-similarity objective Needleman-Wunsch itself maximises, and are therefore
the trustworthy signals.  A text-agreement score is partly circular: NW binds
"Funeral Home" to a *different* ad's "FUNERAL" box precisely because they look
alike, and a similarity metric scores that mistake highly.

  inversions*  normalised Kendall-tau distance between the aligned list order
               and true geometric reading order (column-major, top-to-bottom).
               0.0 is perfect.  The headline number.  (lower is better)
  order_viol*  count of backwards y-jumps within a single column -- a coarser,
               more localisable version of the same signal.  (lower is better)
  dup_bbox*    Gemini lines sharing one Surya box; always an error.  (lower)
  unmatched    Gemini lines left with no bbox at all.  (lower is better)
  coverage     fraction of aligned lines that found a matching Surya bbox.
  agreement    mean text similarity between each line's gemini_text and the
               Surya text at its assigned box.  Catches *gross* misplacement
               (a line thrown across the page) but is blind to swaps between
               similar-looking lines -- read it as a floor, not a verdict.

Usage:

    python tools/compare_alignments.py \
        --old  output/_ocr_backups/morticians_2026-08-24_aligned_OLD \
        --new  output/the_national.../nationaldirectornati \
        --surya output/the_national.../nationaldirectornati

    # write the per-page table for spreadsheet triage
    python tools/compare_alignments.py --old A --new B --surya S --csv report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The aligner itself now uses these to choose between candidate alignments,
# so they live in utils/ — a pipeline stage must not depend on a tool.
from utils.column_utils import (  # noqa: E402
    assign_columns as _assign_columns,
    bbox_centre as _centre,
    inversion_rate as _inversion_rate,
)

_ALIGNED_RE = re.compile(r"^(?P<stem>.+?)_[^_]+(?:-[^_]+)*_aligned\.json$")
_WS = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    """Lowercase and strip everything but alphanumerics for fuzzy comparison."""
    return _WS.sub("", (s or "").lower())


def _sim(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _stem_of_aligned(path: Path) -> str | None:
    m = _ALIGNED_RE.match(path.name)
    return m.group("stem") if m else None


def _index_aligned(d: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(d.glob("*_aligned.json")):
        stem = _stem_of_aligned(p)
        if stem:
            out[stem] = p
    return out


def _index_surya(d: Path) -> dict[str, Path]:
    return {p.name[: -len("_surya.json")]: p for p in sorted(d.glob("*_surya.json"))}


def _load_surya(path: Path) -> tuple[int, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("image_width") or 0), data.get("lines", [])


def score_page(aligned_path: Path, surya_path: Path, bad_threshold: float) -> dict | None:
    """Score one page.  Returns None when the page can't be scored."""
    aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    lines = aligned.get("lines", [])
    if not lines:
        return None
    img_w, surya_lines = _load_surya(surya_path)
    if not surya_lines or not img_w:
        return None

    # NOTE: aligned "bbox" is in image pixels, the same space as Surya.  Only
    # "canvas_fragment" (#xywh=) is rescaled into IIIF canvas space, so no
    # conversion is needed here -- rescaling bbox would break the join.

    # Index Surya lines by bbox centre for nearest-neighbour lookup.
    centres = [(_centre(sl["bbox"]), sl) for sl in surya_lines]
    # Tolerance: a box is "the same box" if its centre is within half a median
    # line height.  Generous enough for rounding, tight enough to not alias.
    heights = [sl["bbox"][3] - sl["bbox"][1] for sl in surya_lines]
    tol = max(8.0, statistics.median(heights) / 2.0)

    sims: list[float] = []
    matched = 0
    placed: list[tuple[float, float]] = []  # (x, y) in image coords, in list order

    for ln in lines:
        bbox = ln.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        cx, cy = _centre(bbox)
        best, best_d = None, float("inf")
        for (sx, sy), sl in centres:
            d = abs(sx - cx) + abs(sy - cy)
            if d < best_d:
                best, best_d = sl, d
        if best is None or best_d > tol * 4:
            continue
        matched += 1
        sims.append(_sim(ln.get("gemini_text", ""), best.get("text", "")))
        placed.append((cx, cy))

    if not sims:
        return None

    # Reading-order violations, computed per column.  Columns are inferred from
    # the placed x-centres by splitting on the widest gaps, which avoids
    # depending on align_ocr's own column planner (the thing under test).
    order_viol = _order_violations(placed)
    inversions = _inversion_rate(placed)

    # Two Gemini lines bound to the same Surya box is always an error.
    seen: dict[tuple, int] = {}
    for ln in lines:
        b = ln.get("bbox")
        if b and len(b) == 4:
            k = tuple(b)
            seen[k] = seen.get(k, 0) + 1
    dup_bbox = sum(n - 1 for n in seen.values() if n > 1)

    agreement = sum(sims) / len(sims)
    bad = sum(1 for s in sims if s < bad_threshold)
    return {
        "lines": len(lines),
        "scored": len(sims),
        "inversions": inversions,
        "order_viol": order_viol,
        "dup_bbox": dup_bbox,
        "agreement": agreement,
        "bad": bad,
        "bad_frac": bad / len(sims),
        "coverage": matched / len(lines),
        "unmatched": len(aligned.get("unmatched_gemini", [])),
    }


def _order_violations(placed: list[tuple[float, float]], n_cols_max: int = 4) -> int:
    """Count y-decreases between consecutive lines that sit in the same column.

    A coarser companion to :func:`_inversion_rate`: it localises *where* the
    alignment jumps backwards rather than scoring the whole permutation.
    Column-to-column resets are not counted.
    """
    if len(placed) < 3:
        return 0
    cols = _assign_columns(placed, n_cols_max)
    viol = 0
    last_y: dict[int, float] = {}
    for (x, y), c in zip(placed, cols):
        if c in last_y and y < last_y[c] - 1e-9:
            viol += 1
        last_y[c] = y
    return viol


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, type=Path, help="Directory of baseline *_aligned.json")
    ap.add_argument("--new", required=True, type=Path, help="Directory of candidate *_aligned.json")
    ap.add_argument("--surya", required=True, type=Path, help="Directory of *_surya.json")
    ap.add_argument("--bad-threshold", type=float, default=0.40,
                    help="Agreement below this counts a line as mis-assigned (default: 0.40)")
    ap.add_argument("--top", type=int, default=10, help="How many biggest movers to list (default: 10)")
    ap.add_argument("--csv", type=Path, default=None, help="Write the per-page table here")
    args = ap.parse_args()

    old_idx, new_idx, sur_idx = _index_aligned(args.old), _index_aligned(args.new), _index_surya(args.surya)
    stems = sorted(set(old_idx) & set(new_idx) & set(sur_idx))
    if not stems:
        raise SystemExit(
            f"No overlapping pages.\n  old={len(old_idx)} new={len(new_idx)} surya={len(sur_idx)}"
        )
    print(f"Scoring {len(stems)} page(s) present in old, new, and surya sets…\n")

    rows = []
    for stem in stems:
        o = score_page(old_idx[stem], sur_idx[stem], args.bad_threshold)
        n = score_page(new_idx[stem], sur_idx[stem], args.bad_threshold)
        if not o or not n:
            continue
        rows.append({"page": stem[:4], "old": o, "new": n})

    if not rows:
        raise SystemExit("No pages could be scored.")

    def agg(side: str, key: str) -> float:
        return sum(r[side][key] for r in rows) / len(rows)

    def tot(side: str, key: str) -> int:
        return sum(r[side][key] for r in rows)

    print(f"{'metric':<22}{'OLD':>12}{'NEW':>12}{'change':>14}")
    print("-" * 60)
    for label, key, kind, better in (
        ("inversion rate*", "inversions", "mean", "down"),
        ("order violations*", "order_viol", "sum", "down"),
        ("duplicate bboxes*", "dup_bbox", "sum", "down"),
        ("unmatched lines", "unmatched", "sum", "down"),
        ("coverage (mean)", "coverage", "mean", "up"),
        ("text agreement", "agreement", "mean", "up"),
        ("low-agreement lines", "bad", "sum", "down"),
    ):
        ov = agg("old", key) if kind == "mean" else tot("old", key)
        nv = agg("new", key) if kind == "mean" else tot("new", key)
        delta = nv - ov
        good = (delta > 0) if better == "up" else (delta < 0)
        mark = "  ✓" if (good and abs(delta) > 1e-9) else ("  ✗" if abs(delta) > 1e-9 else "   ")
        fmt = "{:>12.4f}" if kind == "mean" else "{:>12.0f}"
        print(f"{label:<22}" + fmt.format(ov) + fmt.format(nv)
              + ("{:>+12.4f}".format(delta) if kind == "mean" else "{:>+12.0f}".format(delta)) + mark)

    improved = sum(1 for r in rows if r["new"]["inversions"] < r["old"]["inversions"] - 1e-9)
    regressed = sum(1 for r in rows if r["new"]["inversions"] > r["old"]["inversions"] + 1e-9)
    print("\n* geometric metrics -- independent of the text-similarity objective NW optimises.")
    print(f"\npages improved: {improved}   regressed: {regressed}   unchanged: {len(rows)-improved-regressed}")

    movers = sorted(rows, key=lambda r: r["old"]["inversions"] - r["new"]["inversions"])
    print(f"\nWorst {args.top} regressions (inversion rate old → new):")
    for r in movers[: args.top]:
        d = r["old"]["inversions"] - r["new"]["inversions"]
        if d >= 0:
            print("  (none)")
            break
        print(f"  page {r['page']}  {r['old']['inversions']:.4f} → {r['new']['inversions']:.4f}"
              f"   order_viol {r['old']['order_viol']} → {r['new']['order_viol']}")
    print(f"\nBest {args.top} improvements (inversion rate old → new):")
    for r in reversed(movers[-args.top:]):
        d = r["old"]["inversions"] - r["new"]["inversions"]
        if d <= 0:
            print("  (none)")
            break
        print(f"  page {r['page']}  {r['old']['inversions']:.4f} → {r['new']['inversions']:.4f}"
              f"   order_viol {r['old']['order_viol']} → {r['new']['order_viol']}")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            keys = ["inversions", "order_viol", "dup_bbox", "unmatched", "coverage", "agreement", "bad", "bad_frac", "lines", "scored"]
            w = csv.writer(fh)
            w.writerow(["page"] + [f"old_{k}" for k in keys] + [f"new_{k}" for k in keys])
            for r in rows:
                w.writerow([r["page"]] + [r["old"][k] for k in keys] + [r["new"][k] for k in keys])
        print(f"\nPer-page table → {args.csv}")


if __name__ == "__main__":
    main()
