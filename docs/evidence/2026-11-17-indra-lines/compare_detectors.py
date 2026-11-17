#!/usr/bin/env python3
"""Surya's line boxes against Indra's, on the same pages, without gold.

For every page that has Surya lines in the volume, Indra lines in the mirror
(tools/indra_lines.py) and Gemini OCR text, Gemini's text is aligned onto each
detector's lines with the live aligner, in memory (nothing is written), and
each side is reported:

  lines/gem  detector lines per Gemini line, after the aligner's own filters.
             Under ~0.9 means boxes hold several printed lines.
  tall       boxes taller than 2x the page's median line height: the
             under-segmentation signature of the panel's dense failures.
  empty      share of lines with no text. Indra blanks the text of any line
             under its confidence floor (0.5) and keeps the box.
  unboxed    share of Gemini lines the aligner left without a box.
  disagree   share of bound lines whose Gemini text (first 60 chars) is under
             0.5 similar to the detector's own reading of that box. On the
             2026-10-17 panel this ranked pages as the gold did.
  col        the per-column path completed.

disagree is judged against each detector's OWN text, so a detector that reads
worse scores worse even when its boxes are right. Use it to find pages that
differ, then look at them. gold_panel.py is the judge.

    ./.venv/bin/python docs/evidence/2026-11-17-indra-lines/compare_detectors.py output/tulsa_1921/
    ... compare_detectors.py output/tulsa_1921/ --limit 150 --seed 0 --csv /tmp/tulsa.csv
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, align, bucket, page_stats, pick_gemini, read_gemini  # noqa: E402

sys.path.insert(0, str(REPO))
from tools.indra_lines import DEFAULT_MIRROR, mirror_root_for, select_images  # noqa: E402

METRICS = ("per_gem", "tall", "empty", "unboxed", "disagree")


def eligible_pages(volume: Path, mirror: Path, model: str | None):
    """(page key, Surya json, Indra json, Gemini text) for pages with all three."""
    for img in select_images(volume, exclude=DEFAULT_MIRROR):
        rel = img.relative_to(volume)
        surya = img.parent / f"{img.stem}_surya.json"
        indra = mirror / rel.parent / f"{img.stem}_surya.json"
        gem = pick_gemini(img, model)
        if not (surya.exists() and indra.exists() and gem):
            continue
        if gem.read_text(encoding="utf-8", errors="replace").lstrip().startswith("[OCR FAILED"):
            continue
        yield f"{volume.name}/{rel}", surya, indra, gem


def indra_ms(indra_json: Path) -> float | None:
    try:
        return json.loads(indra_json.read_text(encoding="utf-8"))["detector"]["timings_ms"].get("total_ms")
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def fmt(x, pct=False):
    if x is None:
        return "    -"
    return f"{x * 100:4.0f}%" if pct else f"{x:5.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volumes", nargs="+", type=Path)
    ap.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR)
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--ocr-model", help="Gemini OCR slug to align (default: as the panel picks)")
    ap.add_argument("--pages", metavar="GLOB", help="Only image filenames matching this glob")
    ap.add_argument("--limit", type=int, default=0, help="Random sample of N pages (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--indra-min-conf", type=float, default=0.35,
                    help="parse_surya's confidence floor for Indra lines (default 0.35, the "
                         "pipeline's). 0.5 drops the lines Indra returned without text.")
    ap.add_argument("--csv", type=Path, help="Write one row per page here")
    args = ap.parse_args()

    pages = []
    for vol in args.volumes:
        mirror = mirror_root_for(vol, args.mirror_root, args.output_root)
        found = list(eligible_pages(vol, mirror, args.ocr_model))
        if args.pages:
            found = [p for p in found if fnmatch.fnmatch(Path(p[0]).name, args.pages)]
        if not found:
            print(f"{vol}: no page has Surya lines, Indra lines in {mirror} and Gemini text",
                  file=sys.stderr)
        pages += found
    if args.limit and len(pages) > args.limit:
        pages = sorted(random.Random(args.seed).sample(pages, args.limit))
    if not pages:
        sys.exit("Nothing to compare. Run tools/indra_lines.py on the volume first.")

    rows = []
    hdr = "  ".join(f"{m:>8s}" for m in ("lines/gem", "tall", "empty", "unboxed", "disagree", "col"))
    print(f"{'page':38s} {'gem':>4s} | surya {hdr} | indra {hdr}")
    for key, surya, indra, gem in pages:
        gl, ads = read_gemini(gem)
        if not gl:
            continue
        row = {"page": key, "gemini_lines": len(gl), "indra_ms": indra_ms(indra)}
        cells = []
        for name, path, floor in (("surya", surya, 0.35), ("indra", indra, args.indra_min_conf)):
            lines, bound, unmatched, ran = align(path, gl, ads, floor)
            st = page_stats(lines, bound, unmatched, len(gl))
            st["col"] = ran
            row.update({f"{name}_{k}": v for k, v in st.items()})
            cells.append("  ".join([fmt(st["per_gem"]).rjust(8), f"{st['tall']:8d}",
                                    fmt(st["empty"], True).rjust(8), fmt(st["unboxed"], True).rjust(8),
                                    fmt(st["disagree"], True).rjust(8), ("yes" if ran else "no").rjust(8)]))
        rows.append(row)
        print(f"{key[-38:]:38s} {len(gl):4d} | surya {cells[0]} | indra {cells[1]}")

    print(f"\n{len(rows)} page(s). Medians:")
    for m in METRICS:
        vals = {d: [r[f"{d}_{m}"] for r in rows if r[f"{d}_{m}"] is not None] for d in ("surya", "indra")}
        pct = m != "per_gem" and m != "tall"
        med = {d: (statistics.median(v) if v else None) for d, v in vals.items()}
        print(f"  {m:9s} surya {fmt(med['surya'], pct)}   indra {fmt(med['indra'], pct)}")

    trans = Counter((bucket(r["surya_disagree"]), bucket(r["indra_disagree"])) for r in rows)
    order = ("clean", "borderline", "failing", "none")
    print("\nPages by disagreement bucket (rows: Surya, columns: Indra):")
    print(f"  {'':11s}" + "".join(f"{c:>11s}" for c in order))
    for s in order:
        print(f"  {s:11s}" + "".join(f"{trans[(s, i)]:11d}" for i in order))

    better = [r["page"] for r in rows if bucket(r["surya_disagree"]) == "failing"
              and bucket(r["indra_disagree"]) == "clean"]
    worse = [r["page"] for r in rows if bucket(r["surya_disagree"]) == "clean"
             and bucket(r["indra_disagree"]) == "failing"]
    print(f"\nSurya failing, Indra clean ({len(better)}): {', '.join(better[:20]) or '-'}")
    print(f"Surya clean, Indra failing ({len(worse)}): {', '.join(worse[:20]) or '-'}")
    print(f"Column path completed: surya {sum(r['surya_col'] for r in rows)}, "
          f"indra {sum(r['indra_col'] for r in rows)} of {len(rows)} pages")

    ms = sorted(r["indra_ms"] for r in rows if r["indra_ms"])
    if ms:
        print(f"Indra time per page (its own timer): median {ms[len(ms) // 2] / 1000:.2f}s, "
              f"p90 {ms[int(len(ms) * 0.9)] / 1000:.2f}s. Wall time incl. image load is in "
              "the mirror's indra_run.json.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Per-page rows: {args.csv}")


if __name__ == "__main__":
    main()
