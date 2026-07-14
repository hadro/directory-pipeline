#!/usr/bin/env python3
"""Score and compare OCR-engine alignment quality from aligned JSON.

This is the Phase-1 validation instrument for the local-OCR backend
(`docs/plans/local-ocr-backend.md`). It does **not** run any model — it reads
the `*_{slug}_aligned.json` files that `align_ocr.py` already produced and
scores how cleanly each OCR engine's text aligned to Surya's per-line
geometry.

Because every engine aligns against the *same* Surya JSON, the
engine-discriminating signals live entirely in the aligned JSON:

  - matched lines        len(["lines"])              — Surya lines that got text
  - unmatched OCR lines  len(["unmatched_gemini"])   — OCR lines with no home
  - possible_column_merge / needs_review flags

Derived per-page metrics:
  - match_rate = matched / (matched + unmatched_ocr)
        fraction of an engine's OCR lines that aligned to geometry.
        This is the "alignment %" the plan gates on (≥ ~90%).
  - coverage  = matched / total_surya_lines
        fraction of Surya geometry that received text (needs the sibling
        *_surya.json; skipped if absent).

Geometry alone cannot rank OCR *accuracy*: it is blind to character errors and
in-line column-bleed, and it penalises an engine that merges wrapped entries
(as Gemini does) differently than Surya segments them. So head-to-head mode
(--baseline/--candidate) also reports text-quality metrics from the raw OCR
.txt files:

  - word_recall / word_precision: order-independent bag-of-words overlap with
        the baseline — merging vs splitting lines leaves the multiset unchanged,
        so a low score means genuinely misread words, not segmentation style.
  - line_agreement: text similarity on Surya boxes both engines matched —
        catches per-line OCR error and column-bleed. Pass --no-text to skip.

Usage
-----
    # Summary table across every engine that has aligned JSON in a dir:
    python analysis/compare_alignment.py output/<slug>/

    # Head-to-head: list pages where CANDIDATE is worse than BASELINE
    python analysis/compare_alignment.py output/<slug>/ \
        --baseline gemini-3.1-flash-lite-preview --candidate chandra-ocr-2

    # Tunables / output
    python analysis/compare_alignment.py output/<slug>/ --gate 0.90 --delta 0.05
    # In head-to-head mode --csv also writes report_text.csv (the text metrics):
    python analysis/compare_alignment.py output/<slug>/ \
        --baseline gemini-3.1-flash-lite-preview --candidate chandra-ocr-2 --csv report.csv
"""

import argparse
import csv
import difflib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

_ALIGNED_RX = re.compile(r"_([^_]+)_aligned\.json$")
_WORD_RX = re.compile(r"\w+", re.UNICODE)
# Matched lines whose text agreement with the baseline falls below this are
# counted as disagreements (OCR error or in-line column-bleed on that box).
_LINE_AGREE_BAD = 0.6


def _page_key(name: str, slug: str) -> str:
    """Strip the ``_{slug}_aligned.json`` suffix to a per-page identity that is
    shared across engines (so baseline/candidate pages can be paired)."""
    return name[: -len(f"_{slug}_aligned.json")]


def _surya_total(aligned_path: Path, slug: str) -> int | None:
    """Total Surya line count for this page, from the sibling *_surya.json."""
    surya_path = Path(str(aligned_path)[: -len(f"_{slug}_aligned.json")] + "_surya.json")
    if not surya_path.exists():
        return None
    try:
        data = json.loads(surya_path.read_text(encoding="utf-8"))
        return len(data.get("lines", []))
    except (json.JSONDecodeError, OSError):
        return None


def _score_page(path: Path, slug: str) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    matched = len(d.get("lines", []))
    unmatched = len(d.get("unmatched_gemini", []))
    denom = matched + unmatched
    surya_total = _surya_total(path, slug)
    return {
        "page": _page_key(path.name, slug),
        "path": str(path),
        "matched": matched,
        "unmatched": unmatched,
        "match_rate": (matched / denom) if denom else None,
        "coverage": (matched / surya_total) if surya_total else None,
        "surya_total": surya_total,
        "column_merge": bool(d.get("possible_column_merge")),
        "needs_review": bool(d.get("needs_review")),
        "surya_conf": d.get("surya_median_confidence"),
    }


# --------------------------------------------------------------------------- #
# Text-quality comparison (accuracy, not just geometry)
# --------------------------------------------------------------------------- #
# Geometry metrics (coverage / match_rate) cannot see character accuracy or
# in-line column-bleed, and they penalise an engine for merging wrapped entries
# differently than Surya segments them. These text metrics fix both:
#   - word_recall / word_precision: order-independent bag-of-words overlap with
#     the baseline. Merging vs splitting lines leaves the word multiset
#     unchanged, so this isolates transcription accuracy (a misread word simply
#     fails to match). Recall = how much baseline text the candidate recovered;
#     precision = how much candidate text is also in the baseline (low precision
#     flags noise/duplication/hallucination).
#   - line_agreement: for Surya boxes *both* engines matched, similarity of the
#     two texts. Catches per-line OCR error and column-bleed (a box whose text
#     came from the wrong column won't match the baseline's text for that box).


def _norm_words(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD_RX.findall(text))


def _sibling_text(path: Path, slug: str) -> str | None:
    """Read the raw OCR text ({stem}_{slug}.txt) beside an aligned JSON."""
    txt = Path(str(path)[: -len(f"_{slug}_aligned.json")] + f"_{slug}.txt")
    try:
        return txt.read_text(encoding="utf-8") if txt.exists() else None
    except OSError:
        return None


def _word_recall(ref_text: str, cand_text: str) -> tuple[float | None, float | None]:
    ref, cand = _norm_words(ref_text), _norm_words(cand_text)
    ref_n = sum(ref.values())
    if not ref_n:
        return None, None
    overlap = sum((ref & cand).values())  # Counter & == per-token min
    cand_n = sum(cand.values())
    return overlap / ref_n, (overlap / cand_n if cand_n else 0.0)


def _line_text_by_box(path: Path) -> dict[tuple, str]:
    """Map each matched Surya bbox -> its text. Both engines align against the
    same Surya geometry, so the bbox is a stable cross-engine join key."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[tuple, str] = {}
    for ln in d.get("lines", []):
        bbox = ln.get("bbox")
        if bbox:
            out[tuple(round(c) for c in bbox)] = ln.get("gemini_text", "")
    return out


def _line_agreement(base_path: Path, cand_path: Path) -> tuple[float | None, int, int]:
    base, cand = _line_text_by_box(base_path), _line_text_by_box(cand_path)
    shared = base.keys() & cand.keys()
    if not shared:
        return None, 0, 0
    ratios = [
        difflib.SequenceMatcher(
            None, " ".join(base[k].split()).lower(), " ".join(cand[k].split()).lower()
        ).ratio()
        for k in shared
    ]
    bad = sum(1 for r in ratios if r < _LINE_AGREE_BAD)
    return statistics.mean(ratios), bad, len(shared)


def text_page(base: dict, cand: dict, base_slug: str, cand_slug: str) -> dict | None:
    """Text-quality metrics for one page (candidate vs baseline)."""
    b_txt = _sibling_text(Path(base["path"]), base_slug)
    c_txt = _sibling_text(Path(cand["path"]), cand_slug)
    if b_txt is None or c_txt is None:
        return None
    recall, precision = _word_recall(b_txt, c_txt)
    agree, bad, shared = _line_agreement(Path(base["path"]), Path(cand["path"]))
    return {
        "page": cand["page"],
        "word_recall": recall,
        "word_precision": precision,
        "line_agreement": agree,
        "bad_lines": bad,
        "shared_lines": shared,
    }


def collect(root: Path) -> dict[str, dict[str, dict]]:
    """Return {slug: {page_key: metrics}} for every aligned JSON under *root*."""
    by_slug: dict[str, dict[str, dict]] = {}
    for path in sorted(root.rglob("*_aligned.json")):
        m = _ALIGNED_RX.search(path.name)
        if not m:
            continue
        slug = m.group(1)
        page = _score_page(path, slug)
        if page is None:
            continue
        by_slug.setdefault(slug, {})[page["page"]] = page
    return by_slug


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def summarize(slug: str, pages: dict[str, dict]) -> dict:
    rows = list(pages.values())
    rates = [r["match_rate"] for r in rows if r["match_rate"] is not None]
    covs = [r["coverage"] for r in rows if r["coverage"] is not None]
    return {
        "slug": slug,
        "pages": len(rows),
        "mean_match_rate": _mean(rates),
        "median_match_rate": statistics.median(rates) if rates else None,
        "mean_coverage": _mean(covs),
        "total_matched": sum(r["matched"] for r in rows),
        "total_unmatched": sum(r["unmatched"] for r in rows),
        "column_merge_pages": sum(1 for r in rows if r["column_merge"]),
        "needs_review_pages": sum(1 for r in rows if r["needs_review"]),
    }


def _pct(x: float | None) -> str:
    return f"{x * 100:5.1f}%" if x is not None else "   n/a"


def print_summary(summaries: list[dict], gate: float) -> None:
    hdr = (
        f"{'engine':<34} {'pages':>5} {'match':>7} {'median':>7} "
        f"{'cover':>7} {'unmatch':>8} {'colMrg':>7} {'review':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        flag = "" if (s["mean_match_rate"] or 0) >= gate else "  ⚠ below gate"
        print(
            f"{s['slug']:<34} {s['pages']:>5} "
            f"{_pct(s['mean_match_rate']):>7} {_pct(s['median_match_rate']):>7} "
            f"{_pct(s['mean_coverage']):>7} {s['total_unmatched']:>8} "
            f"{s['column_merge_pages']:>7} {s['needs_review_pages']:>7}{flag}"
        )
    print(f"\ngate = mean match_rate ≥ {gate * 100:.0f}%")


def compare(
    baseline: dict[str, dict], candidate: dict[str, dict], delta: float
) -> list[dict]:
    """Pages where candidate match_rate drops > delta vs baseline, or where the
    candidate newly trips possible_column_merge. Worst regression first."""
    out = []
    for page, c in candidate.items():
        b = baseline.get(page)
        if b is None:
            continue
        br, cr = b["match_rate"], c["match_rate"]
        regressed = br is not None and cr is not None and (br - cr) > delta
        new_merge = c["column_merge"] and not b["column_merge"]
        if regressed or new_merge:
            out.append(
                {
                    "page": page,
                    "base_rate": br,
                    "cand_rate": cr,
                    "drop": (br - cr) if (br is not None and cr is not None) else None,
                    "new_column_merge": new_merge,
                    "cand_unmatched": c["unmatched"],
                }
            )
    out.sort(key=lambda r: (r["drop"] is None, -(r["drop"] or 0)))
    return out


def print_comparison(rows: list[dict], base_slug: str, cand_slug: str) -> None:
    print(f"\nRegressions: {cand_slug} vs baseline {base_slug}")
    if not rows:
        print("  none — candidate matches or beats baseline on every shared page.")
        return
    print(f"{'page':<48} {'base':>6} {'cand':>6} {'drop':>6} {'unmatch':>8}  flags")
    print("-" * 84)
    for r in rows:
        flags = "NEW column_merge" if r["new_column_merge"] else ""
        print(
            f"{r['page']:<48} {_pct(r['base_rate']):>6} {_pct(r['cand_rate']):>6} "
            f"{_pct(r['drop']):>6} {r['cand_unmatched']:>8}  {flags}"
        )
    print(f"\n{len(rows)} page(s) to eyeball (open the *_aligned.json or the page image).")


def compare_text(
    baseline: dict[str, dict], candidate: dict[str, dict], base_slug: str, cand_slug: str
) -> list[dict]:
    """Per-page text-quality rows for pages shared by both engines (those whose
    raw OCR .txt files are both present). Worst word_recall first."""
    rows = []
    for page, c in candidate.items():
        b = baseline.get(page)
        if b is None:
            continue
        tp = text_page(b, c, base_slug, cand_slug)
        if tp is not None:
            rows.append(tp)
    rows.sort(key=lambda r: (r["word_recall"] is None, r["word_recall"] or 0.0))
    return rows


def print_text_comparison(
    rows: list[dict], base_slug: str, cand_slug: str, worst: int = 15
) -> None:
    print(f"\nText quality: {cand_slug} vs baseline {base_slug}")
    print("  (word_recall/precision are order-independent — merging vs splitting "
          "lines\n   does not affect them; line_agree is per-Surya-box text similarity)")
    if not rows:
        print("  no pages with both engines' raw .txt present — nothing to compare.")
        return

    def _agg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else None

    print(
        f"\n  aggregate over {len(rows)} pages: "
        f"recall={_pct(_agg('word_recall'))}  precision={_pct(_agg('word_precision'))}  "
        f"line_agree={_pct(_agg('line_agreement'))}  "
        f"bad_lines={sum(r['bad_lines'] for r in rows)}"
    )
    print(f"\n  worst {min(worst, len(rows))} pages by word_recall:")
    print(f"  {'page':<46} {'recall':>7} {'prec':>7} {'lineAgr':>8} {'bad/shared':>11}")
    print("  " + "-" * 82)
    for r in rows[:worst]:
        bad_shared = f"{r['bad_lines']}/{r['shared_lines']}"
        print(
            f"  {r['page']:<46} {_pct(r['word_recall']):>7} {_pct(r['word_precision']):>7} "
            f"{_pct(r['line_agreement']):>8} {bad_shared:>11}"
        )


def write_csv(path: Path, by_slug: dict[str, dict[str, dict]]) -> None:
    cols = [
        "slug", "page", "matched", "unmatched", "match_rate", "coverage",
        "surya_total", "column_merge", "needs_review", "surya_conf",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for slug, pages in by_slug.items():
            for page in pages.values():
                w.writerow({"slug": slug, **{k: page.get(k) for k in cols if k != "slug"}})
    print(f"\nwrote per-page geometry metrics → {path}")


def write_text_csv(path: Path, rows: list[dict]) -> None:
    """Per-page text-quality metrics (head-to-head only). Kept in its own file
    because its grain is (baseline, candidate, page) — distinct from the
    geometry CSV's (engine, page) — so the two never share rows cleanly."""
    cols = ["page", "word_recall", "word_precision", "line_agreement", "bad_lines", "shared_lines"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"wrote per-page text metrics → {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path, help="output dir to scan (recursive)")
    ap.add_argument("--baseline", help="slug to treat as ground-truth in head-to-head mode")
    ap.add_argument("--candidate", help="slug to compare against the baseline")
    ap.add_argument("--gate", type=float, default=0.90, help="mean match_rate pass threshold (default 0.90)")
    ap.add_argument("--delta", type=float, default=0.05, help="per-page drop that counts as a regression (default 0.05)")
    ap.add_argument("--worst", type=int, default=15, help="rows to show in the worst-by-recall text table (default 15)")
    ap.add_argument("--no-text", action="store_true", help="skip the text-quality comparison (geometry only)")
    ap.add_argument("--csv", type=Path, help="write per-page geometry CSV; in head-to-head mode also writes a sibling <name>_text.csv with the text metrics")
    args = ap.parse_args(argv)

    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 2

    by_slug = collect(args.directory)
    if not by_slug:
        print(f"no *_aligned.json found under {args.directory}", file=sys.stderr)
        return 1

    summaries = [summarize(slug, pages) for slug, pages in sorted(by_slug.items())]
    print_summary(summaries, args.gate)

    if args.baseline or args.candidate:
        missing = [s for s in (args.baseline, args.candidate) if s and s not in by_slug]
        if missing:
            print(f"\nslug(s) not present: {', '.join(missing)}", file=sys.stderr)
            print(f"available: {', '.join(sorted(by_slug))}", file=sys.stderr)
            return 1
        if not (args.baseline and args.candidate):
            print("\n--baseline and --candidate must be given together", file=sys.stderr)
            return 2
        rows = compare(by_slug[args.baseline], by_slug[args.candidate], args.delta)
        print_comparison(rows, args.baseline, args.candidate)
        if not args.no_text:
            trows = compare_text(
                by_slug[args.baseline], by_slug[args.candidate], args.baseline, args.candidate
            )
            print_text_comparison(trows, args.baseline, args.candidate, args.worst)
            if args.csv:
                write_text_csv(
                    args.csv.with_name(args.csv.stem + "_text" + args.csv.suffix), trows
                )

    if args.csv:
        write_csv(args.csv, by_slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
