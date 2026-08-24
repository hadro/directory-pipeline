#!/usr/bin/env python3
"""Detect column layout using Surya's neural text-line detector.

Drop-in replacement for detect_columns.py: produces the same
columns_report.csv schema so subsequent pipeline stages work without
modification.

Why Surya over pixel projection?
  detect_columns.py uses a dark-pixel projection profile to find gutters.
  It can misfire on pages with full-width decorative elements, ads, or
  illustrations that span the column divide.  Surya's DetectionPredictor
  (CNN-based) finds actual text-line bounding boxes and infers column
  structure from where text physically lives — more robust for mixed layouts.

Column heuristic
  Lines are split left/right at the image midpoint.  If the two groups
  overlap vertically by ≥ 30% of their combined span, the page is 2-column.
  Gutter x-position is the midpoint between the rightmost left-column edge
  and the leftmost right-column edge.

Confidence mapping
  2-col, overlap ≥ 0.60  → "high"
  2-col, overlap 0.30–0.59 → "medium"
  1-col (no gutter)        → "high"

Output
  output_dir/columns_report.csv  (same fields as detect_columns.py)

Requirements
  uv sync --extra gpu

Usage
-----
    python surya_detect.py output/greenbooks
    python surya_detect.py output/greenbooks --force
    python surya_detect.py output/greenbooks --batch-size 8
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from pipeline.align_ocr import _column_breaks

# Suppress Surya's per-batch tqdm bars
os.environ.setdefault("SURYA_DISABLE_TQDM", "true")
os.environ.setdefault("DISABLE_TQDM", "true")

# ---------------------------------------------------------------------------
# Constants — must match detect_columns.py so run_ocr.py reads the same file
# ---------------------------------------------------------------------------
REPORT_FILENAME = "columns_report.csv"
FIELDNAMES = [
    "image", "num_columns", "confidence", "gutter_x_positions",
]
MIN_OVERLAP_RATIO = 0.3
# Minimum gap between left- and right-column x1 clusters (as a fraction of
# page width) needed to declare a 2-column layout.  8 % ≈ 150 px at 1920 px.
MIN_GUTTER_GAP    = 0.08
# A reported column must hold at least this fraction of the page's lines, and
# the reported columns together at least this much of the page.  Mirrors the
# thresholds in align_ocr.plan_columns so both agree on the column count.
MIN_COLUMN_FRAC     = 0.10
MIN_COLUMN_COVERAGE = 0.80


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _find_gutter(bboxes: list, image_width: int) -> float | None:
    """
    Find the column gutter x-position by locating the largest gap in the
    distribution of bbox left-edge (x1) positions.

    Using x1 rather than x-centre means cross-column merges (bboxes whose
    leader dots bridge the gutter) are invisible to the detector: they always
    start at the left column's x1 and stay inside the left x1 cluster.

    Returns the gap midpoint, or None when the gap is below MIN_GUTTER_GAP ×
    image_width (single-column or unresolvable layout).
    """
    if len(bboxes) < 4:
        return None
    x1s = sorted(b[0] for b in bboxes)
    max_gap, gutter_x = 0.0, None
    for i in range(len(x1s) - 1):
        gap = x1s[i + 1] - x1s[i]
        if gap > max_gap:
            max_gap, gutter_x = gap, (x1s[i] + x1s[i + 1]) / 2
    return gutter_x if max_gap >= MIN_GUTTER_GAP * image_width else None


def _analyze_bboxes(bboxes: list, image_width: int) -> dict:
    """
    Derive column layout from a list of line bboxes (each is [x1, y1, x2, y2]).

    Handles any number of columns.  Gutters come from align_ocr's coverage
    valley detector — the same one the aligner uses — so the report and the
    reading order agree about where the columns are.  A layout is accepted
    when every reported column holds at least MIN_COLUMN_FRAC of the lines,
    the columns together account for MIN_COLUMN_COVERAGE of them, and adjacent
    columns overlap vertically by at least MIN_OVERLAP_RATIO (which rejects
    stacked blocks that merely happen to sit at different x positions).

    Returns a dict with the columns_report.csv fields (excluding 'image').
    """
    single = {
        "num_columns":        1,
        "confidence":         "high",
        "gutter_x_positions": "",
    }
    if not bboxes:
        return {**single, "confidence": "low"}
    if len(bboxes) < 4:
        return single

    wrapped = [{"bbox": list(b)} for b in bboxes]
    breaks  = _column_breaks(wrapped, image_width)
    if not breaks:
        return single

    def col_index(x1: float) -> int:
        for i, b in enumerate(breaks):
            if x1 < b:
                return i
        return len(breaks)

    # Spanning lines (crossing a gutter) are excluded from the column tally.
    body = [b for b in bboxes if not any(b[0] < g < b[2] for g in breaks)]
    if not body:
        return single

    counts: dict[int, int] = {}
    for b in body:
        ci = col_index(b[0])
        counts[ci] = counts.get(ci, 0) + 1
    substantial = sorted(c for c, n in counts.items()
                         if n / len(body) >= MIN_COLUMN_FRAC)
    if len(substantial) < 2:
        return single
    if sum(counts[c] for c in substantial) / len(body) < MIN_COLUMN_COVERAGE:
        return single

    groups = [[b for b in body if col_index(b[0]) == c] for c in substantial]
    spans  = [(min(b[1] for b in g), max(b[3] for b in g)) for g in groups]
    overlaps = []
    for (ay1, ay2), (by1, by2) in zip(spans, spans[1:]):
        overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
        total   = max(ay2, by2) - min(ay1, by1)
        overlaps.append(overlap / total if total > 0 else 0.0)
    if not overlaps or min(overlaps) < MIN_OVERLAP_RATIO:
        return single

    # Report the gutters that actually separate the accepted columns, placed at
    # the midpoint between neighbouring columns' content edges.
    gutters = [
        int((max(b[2] for b in left) + min(b[0] for b in right)) / 2)
        for left, right in zip(groups, groups[1:])
    ]
    confidence = "high" if min(overlaps) >= 0.6 else "medium"
    return {
        "num_columns":        len(substantial),
        "confidence":         confidence,
        "gutter_x_positions": ";".join(str(g) for g in gutters),
    }

# ---------------------------------------------------------------------------
# Image selection (mirrors detect_columns.py: prefer split halves)
# ---------------------------------------------------------------------------

def _select_images(output_root: Path) -> list[Path]:
    """
    Return the images to process.

    If a spread has already been split (_left/_right files exist), use those
    and skip the original — same logic as detect_columns.py.

    Pipeline-generated derivative images are always excluded, identified by
    known stem suffixes (_viz, _surya_det).
    """
    # Stems ending with any of these are pipeline outputs, not source scans.
    _SKIP_SUFFIXES = ("_viz", "_surya_det")

    all_jpgs = sorted(output_root.rglob("*.jpg"))
    images: list[Path] = []
    for p in all_jpgs:
        if any(p.stem.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        left  = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue   # original will be processed as split halves
        images.append(p)
    return images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect column layout using Surya text-line detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Root images directory (e.g. output/greenbooks)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-analyze all images even if columns_report.csv already exists",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        dest="batch_size",
        metavar="N",
        help="Images per Surya inference batch (default: 16; reduce if OOM)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-image progress output",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    if not output_root.exists():
        print(f"Error: directory not found: {output_root}", file=sys.stderr)
        sys.exit(1)

    report_path = output_root / REPORT_FILENAME
    if report_path.exists() and not args.force:
        print(
            f"columns_report.csv already exists ({report_path}). "
            "Use --force to re-analyze.",
            file=sys.stderr,
        )
        sys.exit(0)

    images = _select_images(output_root)
    if not images:
        print(f"No .jpg files found under {output_root}", file=sys.stderr)
        sys.exit(0)

    total = len(images)
    if not args.quiet:
        print(
            f"Surya column detection: {total} image(s) in {output_root}",
            file=sys.stderr,
        )

    # ---- Load model --------------------------------------------------------
    if not args.quiet:
        print("Loading Surya DetectionPredictor...", file=sys.stderr)
    try:
        from surya.detection import DetectionPredictor
    except ImportError as exc:
        print(
            f"Error: {exc}\nInstall with: uv sync --extra gpu",
            file=sys.stderr,
        )
        sys.exit(1)

    from PIL import Image as PILImage

    det_predictor = DetectionPredictor()
    if not args.quiet:
        print("Model loaded.\n", file=sys.stderr)

    # ---- Batch inference ---------------------------------------------------
    rows: list[dict] = []
    batch_size = max(1, args.batch_size)

    for batch_start in range(0, total, batch_size):
        batch_paths = images[batch_start : batch_start + batch_size]
        batch_imgs  = [PILImage.open(p).convert("RGB") for p in batch_paths]

        det_results = det_predictor(batch_imgs)

        for img_path, pil_img, det_result in zip(batch_paths, batch_imgs, det_results):
            w, _ = pil_img.size
            bboxes = [b.bbox for b in det_result.bboxes]
            row = _analyze_bboxes(bboxes, w)
            row["image"] = img_path.name
            rows.append(row)

            if not args.quiet:
                n_lines = len(bboxes)
                cols    = row["num_columns"]
                conf    = row["confidence"]
                gutter  = row["gutter_x_positions"] or "—"
                print(
                    f"  [{len(rows):04d}/{total}] "
                    f"{cols}-col ({conf:6s}, "
                    f"gutter={gutter}, lines={n_lines}): {img_path.name}",
                    file=sys.stderr,
                )

    # ---- Write report ------------------------------------------------------
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if not args.quiet:
        col_counts: dict[int, int] = {}
        for r in rows:
            n = int(r.get("num_columns", 1))
            col_counts[n] = col_counts.get(n, 0) + 1
        summary = ", ".join(
            f"{count} × {cols}-col" for cols, count in sorted(col_counts.items())
        )
        print(f"\nDone. {total} image(s): {summary}.", file=sys.stderr)
        print(f"Report → {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
