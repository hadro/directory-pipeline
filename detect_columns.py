#!/usr/bin/env python3
"""Detect column layout per image to guide Tesseract PSM selection.

Analyzes each image using a vertical pixel projection profile: dark-pixel
density is averaged per column (via a fast PIL single-row resize), smoothed,
and then low-density gutters between the page margins are counted to determine
the number of columns.

Produces columns_report.csv in the images directory:

    image, num_columns, confidence, recommended_psm, gutter_x_positions

run_ocr.py reads this report and applies the recommended PSM per image instead
of a single global setting. The global --psm flag always takes precedence.

PSM recommendations
-------------------
  1 column   → PSM 4  (single column of uniform text)
  2+ columns → PSM 1  (automatic page segmentation with OSD)

Confidence levels
-----------------
  high   — gutter prominence ≥ 45% (clear dip relative to neighboring columns)
  medium — gutter prominence 25–44%
  low    — ambiguous; column count may be unreliable

Algorithm
---------
Following Bell et al. (2020) / directoreadr: sum (average) black pixels in
each vertical strip of the image and find dips in the resulting profile.
Only the inner 80% of the page width is examined (outermost 10% each side
is treated as margin). Smoothing radius is ~1% of page width.

Usage
-----
    python detect_columns.py images/greenbooks
    python detect_columns.py images/greenbooks --threshold 0.08 --workers 8
    python detect_columns.py images/greenbooks --max-columns 4 --force
    python detect_columns.py images/greenbooks --force
"""

import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print(
        "Error: Pillow is required. Install with: uv add pillow",
        file=sys.stderr,
    )
    sys.exit(1)

_print_lock = threading.Lock()

REPORT_FILENAME = "columns_report.csv"
FIELDNAMES = [
    "image", "num_columns", "confidence", "recommended_psm", "gutter_x_positions",
]

# Tesseract PSM values assigned by detected column count
PSM_SINGLE_COLUMN = 4   # assume a single column of uniform text
PSM_MULTI_COLUMN  = 1   # automatic page segmentation with orientation detection


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def _moving_average(values: list[float], radius: int) -> list[float]:
    """Box-filter moving average with radius `radius`."""
    n = len(values)
    result = []
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def _find_gutters_recursive(
    smoothed: list[float],
    left: int,
    right: int,
    min_prominence: float,
    min_width: int,
    max_depth: int,
) -> list[tuple[int, float]]:
    """
    Recursively find column gutter positions in smoothed[left:right].

    Finds the deepest valley, checks its prominence relative to the peaks on
    each side, then recurses into each half-region.  The key property: the
    "right peak" for a gradual density slope (NOT a gutter) is the gutter
    floor itself, so slope regions compute near-zero prominence and are
    automatically rejected.

    Returns a sorted list of (inner_index, prominence) pairs.
    """
    if right - left < min_width * 2 or max_depth <= 0:
        return []

    # Search only the inner 70% of the current region (exclude 15% from each
    # side).  This prevents scan-edge artifacts — sparse white margins before
    # the first text column, spine shadow — from being picked up as a gutter.
    # The 15/85 constraint also prevents the "gutter approach" slope at the
    # edge of each sub-region from registering as a false inner gutter when
    # recursing.  A genuine column gutter for any split up to ≈80/20 will
    # still fall within [15%, 85%] of the region.
    region = right - left
    boundary = max(min_width, region * 15 // 100)
    search_left  = left  + boundary
    search_right = right - boundary
    if search_right - search_left < 2:
        return []

    sub = smoothed[search_left:search_right]
    if not sub:
        return []

    local_min_val = min(sub)
    local_min_idx = sub.index(local_min_val) + search_left

    # Peaks are still measured over the full [left, right] range so that the
    # text columns outside the exclusion zone count toward the reference height.
    left_peak  = max(smoothed[left:local_min_idx])         if local_min_idx > left      else 0.0
    right_peak = max(smoothed[local_min_idx + 1:right])    if local_min_idx + 1 < right else 0.0
    ref = min(left_peak, right_peak)

    if ref < 0.01:
        return []

    prominence = (ref - local_min_val) / ref
    if prominence < min_prominence:
        return []

    result = [(local_min_idx, prominence)]
    result += _find_gutters_recursive(
        smoothed, left, local_min_idx, min_prominence, min_width, max_depth - 1,
    )
    result += _find_gutters_recursive(
        smoothed, local_min_idx + 1, right, min_prominence, min_width, max_depth - 1,
    )
    return sorted(result, key=lambda x: x[0])


def column_projection(
    image_path: Path,
    row_margin_frac: float = 0.15,
) -> tuple[list[float], int, int]:
    """
    Compute dark-pixel density per column using a fast PIL single-row resize.

    Returns (projection, width, height) where projection[x] is the average
    inverted brightness of column x normalized to 0.0–1.0 (higher = more text).

    row_margin_frac rows are stripped from the top and bottom before collapsing.
    This excludes full-width headers, footers, and advertisement blocks that
    would otherwise deposit density across the column gutter, masking it.
    """
    img = Image.open(image_path).convert("L")
    w, h = img.size
    if row_margin_frac > 0:
        top = int(h * row_margin_frac)
        bot = int(h * (1.0 - row_margin_frac))
        if bot > top:
            img = img.crop((0, top, w, bot))
    # Invert: text pixels (dark) → high value; white background → 0
    inv = img.point(lambda p: 255 - p)
    # Collapse to one row — each pixel is the mean column brightness
    col_row = inv.resize((w, 1), Image.BOX)
    proj = [v / 255.0 for v in col_row.getdata()]
    return proj, w, h  # h is the original (uncropped) page height



def find_gutters(
    proj: list[float],
    page_width: int,
    margin_frac: float = 0.10,
    gutter_threshold: float = 0.25,
    min_gutter_width_frac: float = 0.01,
    max_columns: int = 2,
) -> tuple[list[int], str]:
    """
    Find column gutter x-positions within the inner portion of a projection.

    Uses recursive valley prominence: finds the deepest valley in the inner
    region, measures how prominent it is relative to the peaks on each side,
    then recurses into each half to find additional gutters.

        prominence = (min(left_peak, right_peak) - valley) / min(left_peak, right_peak)

    Supports up to max_columns columns (max_columns-1 gutters).  Gradual density
    slopes that are not genuine gutters naturally yield near-zero prominence and
    are filtered out.

    Parameters
    ----------
    proj : per-column dark-pixel density (0.0–1.0)
    page_width : full image width in pixels
    margin_frac : fraction of width to ignore at each side as page margin
    gutter_threshold : minimum valley prominence to qualify as a gutter
        (default: 0.25 — gutter must be ≥ 25% lower than neighboring peaks)
    min_gutter_width_frac : minimum gutter width as a fraction of page width
    max_columns : maximum number of columns to detect (default: 2).
        Limits recursion depth: max_depth = max_columns - 1.  Keeping this
        at 2 avoids false sub-column detections from typography variations
        (short category headers, advertisement boxes, index leader dots)
        that can resemble additional gutters within a single column.

    Returns
    -------
    (gutter_centers, confidence)
        gutter_centers : list of x pixel positions of gutter midpoints
        confidence : 'high', 'medium', or 'low'
    """
    left  = int(page_width * margin_frac)
    right = int(page_width * (1.0 - margin_frac))
    inner = proj[left:right]

    if not inner:
        return [], "low"

    # Light smoothing — projection is already naturally smoothed by column
    # averaging over the full page height; only reduce fine character noise.
    smooth_radius = max(1, page_width // 400)  # ~0.25% of page width
    smoothed = _moving_average(inner, smooth_radius)

    if max(smoothed) < 0.01:
        return [], "low"  # blank or near-blank page

    min_gutter_px = max(2, int(page_width * min_gutter_width_frac))
    max_depth = max(1, max_columns - 1)

    gutter_data = _find_gutters_recursive(
        smoothed, 0, len(smoothed), gutter_threshold, min_gutter_px, max_depth=max_depth,
    )

    # Convert inner indices to image x-coordinates
    gutters = [left + idx for idx, _ in gutter_data]

    # Confidence based on the most prominent detected gutter
    if not gutters:
        confidence = "high"  # unambiguously single-column
    else:
        max_prominence = max(prom for _, prom in gutter_data)
        if max_prominence >= 0.45:
            confidence = "high"
        elif max_prominence >= 0.25:
            confidence = "medium"
        else:
            confidence = "low"

    return gutters, confidence


def analyze_image(
    image_path: Path,
    margin_frac: float,
    gutter_threshold: float,
    row_margin_frac: float = 0.15,
    max_columns: int = 2,
) -> dict:
    """
    Analyze one image for column layout.
    Returns a result dict with FIELDNAMES keys plus '_error' for internal use.
    """
    try:
        proj, w, _ = column_projection(image_path, row_margin_frac=row_margin_frac)
        gutters, confidence = find_gutters(
            proj, w,
            margin_frac=margin_frac,
            gutter_threshold=gutter_threshold,
            max_columns=max_columns,
        )
        num_columns = len(gutters) + 1
        recommended_psm = PSM_MULTI_COLUMN if num_columns > 1 else PSM_SINGLE_COLUMN
        return {
            "image": image_path.name,
            "num_columns": num_columns,
            "confidence": confidence,
            "recommended_psm": recommended_psm,
            "gutter_x_positions": ";".join(str(x) for x in gutters),
            "_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "image": image_path.name,
            "num_columns": 1,
            "confidence": "low",
            "recommended_psm": PSM_SINGLE_COLUMN,
            "gutter_x_positions": "",
            "_error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect column layout per image to guide Tesseract PSM selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images_dir",
        help="Root images directory (e.g. images/greenbooks)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.25,
        metavar="FRAC",
        help=(
            "Minimum relative prominence for a dip to count as a column gutter "
            "(default: 0.25). A position is a gutter if it is at least this much "
            "lower than the local text-column density on either side: "
            "(local_max - value) / local_max >= threshold. "
            "Lower = only very deep gutters detected; "
            "higher = more sensitive but may produce false positives."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.10,
        metavar="FRAC",
        help=(
            "Fraction of page width to ignore at each edge as margin "
            "(default: 0.10 = ignore outermost 10%% on each side)."
        ),
    )
    parser.add_argument(
        "--row-margin",
        type=float,
        default=0.15,
        dest="row_margin",
        metavar="FRAC",
        help=(
            "Fraction of page height to strip from top and bottom before "
            "computing the column projection (default: 0.15). Excludes "
            "full-width headers, footers, and advertisement blocks that "
            "would otherwise inflate density in the column gutter. "
            "Increase if full-width elements extend deeper into the page."
        ),
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=2,
        dest="max_columns",
        metavar="N",
        help=(
            "Maximum number of columns to detect (default: 2). "
            "Limits how deep the recursive gutter search goes "
            "(max_depth = N - 1). The default of 2 avoids false "
            "sub-column detections from advertisement boxes, index "
            "leader dots, and short category headers within a single "
            "column. Increase only for documents with genuine 3+ "
            "column layouts."
        ),
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=min(4, os.cpu_count() or 1),
        metavar="N",
        help="Parallel workers (default: 4)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-analyze all images even if columns_report.csv already exists",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-image progress output",
    )
    args = parser.parse_args()

    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    report_path = images_root / REPORT_FILENAME
    if report_path.exists() and not args.force:
        print(
            f"columns_report.csv already exists ({report_path}). "
            "Use --force to re-analyze.",
            file=sys.stderr,
        )
        sys.exit(0)

    # Split-aware image selection — same logic used across the pipeline
    all_jpgs = sorted(images_root.rglob("*.jpg"))
    images = []
    for p in all_jpgs:
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        left = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        images.append(p)

    if not images:
        print(f"No .jpg files found under {images_root}", file=sys.stderr)
        sys.exit(0)

    total = len(images)
    if not args.quiet:
        print(
            f"Analyzing {total} image(s) for column layout, {args.workers} worker(s)…",
            file=sys.stderr,
        )

    # Pre-size list so we can store results by original index (preserves order)
    results: list[dict] = [{}] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_image, img, args.margin, args.threshold, args.row_margin,
                args.max_columns,
            ): (i, img)
            for i, img in enumerate(images)
        }
        for future in as_completed(futures):
            idx, image_path = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "image": image_path.name,
                    "num_columns": 1,
                    "confidence": "low",
                    "recommended_psm": PSM_SINGLE_COLUMN,
                    "gutter_x_positions": "",
                    "_error": str(exc),
                }
            results[idx] = result

            if not args.quiet:
                err = result.get("_error")
                if err:
                    _log(f"[{completed:04d}/{total}] FAILED:  {image_path.name}: {err}")
                else:
                    cols = result["num_columns"]
                    conf = result["confidence"]
                    psm  = result["recommended_psm"]
                    _log(
                        f"[{completed:04d}/{total}] "
                        f"{cols}-col ({conf:6s}, PSM {psm}): {image_path.name}"
                    )

    # Write report in original image order
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})

    if not args.quiet:
        col_counts: dict[int, int] = {}
        for r in results:
            n = int(r.get("num_columns", 1))
            col_counts[n] = col_counts.get(n, 0) + 1
        summary = ", ".join(
            f"{count} × {cols}-col" for cols, count in sorted(col_counts.items())
        )
        print(f"\nDone. {total} image(s): {summary}.", file=sys.stderr)
        print(f"Report → {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
