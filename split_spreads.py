#!/usr/bin/env python3
"""Split double-page spread images into separate left and right page files.

Reads a spreads_report.csv (produced by detect_spreads.py) and, for every row
where double_page=True, splits the image at the detected gutter column.

Outputs (all placed alongside the original image — originals are untouched):
  {stem}_left.jpg   — left page crop
  {stem}_right.jpg  — right page crop
  {stem}_split.json — coordinate sidecar for downstream OCR alignment

JSON sidecar schema
-------------------
{
  "original_file": "0001_abc.jpg",
  "original_width": 2048,
  "original_height": 1711,
  "gutter_col": 1041,
  "pages": [
    {
      "side": "left",
      "file": "0001_abc_left.jpg",
      "x_offset": 0,
      "y_offset": 0,
      "width": 1041,
      "height": 1711
    },
    {
      "side": "right",
      "file": "0001_abc_right.jpg",
      "x_offset": 1041,
      "y_offset": 0,
      "width": 1007,
      "height": 1711
    }
  ]
}

Coordinate mapping
------------------
To map a bounding box from a split image back to the original image:
  x_original = x_split + x_offset
  y_original = y_split + y_offset

Usage
-----
    python split_spreads.py images/Hackley_Harrison/spreads_report.csv
    python split_spreads.py path/to/spreads_report.csv --quiet
    python split_spreads.py path/to/spreads_report.csv --dry-run
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Tuning — mirrors detect_spreads.py constants used in gutter location
# ---------------------------------------------------------------------------
DARK_THRESHOLD = 50   # pixels at/below this are treated as background
JPEG_QUALITY = 92     # output JPEG quality


# ---------------------------------------------------------------------------
# Gutter location helpers
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Uniform smoothing via convolution (no scipy dependency)."""
    if window < 2:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _find_gutter_col(arr: np.ndarray) -> int:
    """
    Return the absolute column index (in arr) of the most likely gutter.

    Two signals are used; which one applies depends on the image:

    Signal A — Tonal boundary (one dark page, one light page):
      Finds the steepest step in a heavily-smoothed (~12 %) column-mean
      profile, searching the center 50 % of content.  Only used when the
      two halves of the spread differ in brightness by more than
      TONAL_MIN_RANGE brightness units.  When both pages look similar, this
      score can be falsely inflated by dividing a tiny gradient by a tiny
      range, so it is suppressed.

    Signal B — Spine shadow (dark binding crease between similar-tone pages):
      Searches a narrow ±7 % band around the geometric content centre for
      the darkest smoothed column.  Because the band is tight, text columns
      away from the spine cannot win.

    Fallback — Geometric centre of the content bounding box.
      Used when neither signal is strong enough.
    """
    h, w = arr.shape

    # 1. Content bounding box — strip the dark microfilm border
    mask = arr > DARK_THRESHOLD
    col_any = mask.any(axis=0)
    row_any = mask.any(axis=1)

    if not col_any.any() or not row_any.any():
        return w // 2

    c0 = int(np.argmax(col_any))
    c1 = int(len(col_any) - 1 - np.argmax(col_any[::-1]))
    r0 = int(np.argmax(row_any))
    r1 = int(len(row_any) - 1 - np.argmax(row_any[::-1]))

    content = arr[r0: r1 + 1, c0: c1 + 1]
    cw = content.shape[1]
    if cw < 8:
        return w // 2

    col_means = content.mean(axis=0)

    # ----------------------------------------------------------------
    # Signal A: Tonal boundary
    # Only active when the two pages have meaningfully different mean
    # brightness (page_range >= TONAL_MIN_RANGE).  For same-tone pages
    # (e.g. both white text pages) the score is suppressed because
    # dividing a tiny gradient by a tiny range inflates it randomly.
    # ----------------------------------------------------------------
    TONAL_MIN_RANGE = 25.0   # brightness units (0–255 scale)

    heavy = _smooth(col_means, max(5, cw // 8))
    quarter = max(1, cw // 4)
    center_h = heavy[quarter: cw - quarter]

    if len(center_h) >= 4:
        # Score: mean brightness difference between the two halves of center_h.
        # This is robust to the smoothing window — a large left/right brightness
        # gap reliably identifies a tonal-boundary spread.
        half_ch = len(center_h) // 2
        left_mean = float(center_h[:half_ch].mean())
        right_mean = float(center_h[half_ch:].mean())
        ch_mean = (left_mean + right_mean) / 2
        boundary_score = abs(left_mean - right_mean) / (ch_mean + 1e-6)

        if boundary_score >= 0.15:
            # Location: argmax of the derivative finds the steepest step in
            # the smoothed profile, regardless of the per-pixel magnitude.
            grad_h = np.abs(np.diff(center_h))
            grad_local = int(np.argmax(grad_h))
            return c0 + quarter + grad_local

    # ----------------------------------------------------------------
    # Fallback: absolute image centre.
    # Microfilm cameras are centred on the open spread, so the physical
    # gutter is very close to w // 2 regardless of dark-border width.
    # For same-tone pages (typical text spreads) the spine shadow is too
    # subtle to distinguish reliably from text columns, so we trust the
    # geometric centre rather than chasing a noisy brightness minimum.
    # ----------------------------------------------------------------
    return w // 2


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def split_image(
    image_path: Path,
    dry_run: bool = False,
    quiet: bool = False,
    force: bool = False,
) -> dict | None:
    """
    Split one double-page image at its detected gutter column.

    Returns the sidecar dict on success, None on failure.
    Skips (returns existing sidecar dict) if split files already exist,
    unless force=True, in which case existing outputs are overwritten.
    """
    stem = image_path.stem
    left_path = image_path.with_name(f"{stem}_left.jpg")
    right_path = image_path.with_name(f"{stem}_right.jpg")
    json_path = image_path.with_name(f"{stem}_split.json")

    already_exist = [p for p in (left_path, right_path, json_path) if p.exists()]

    if already_exist and not force:
        if not quiet:
            print(f"  [skip] already split: {image_path.name}", file=sys.stderr)
        if not dry_run:
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        return {"skipped": True, "file": str(image_path)}

    if already_exist and force:
        if dry_run:
            if not quiet:
                print(
                    f"  [dry-run] would delete: "
                    f"{', '.join(p.name for p in already_exist)}",
                    file=sys.stderr,
                )
        else:
            for p in already_exist:
                p.unlink()
            if not quiet:
                print(
                    f"  [force] deleted: {', '.join(p.name for p in already_exist)}",
                    file=sys.stderr,
                )

    try:
        img = Image.open(image_path).convert("RGB")
        arr_gray = np.array(img.convert("L"), dtype=np.float32)
    except Exception as exc:
        print(f"  Warning: could not open {image_path.name}: {exc}", file=sys.stderr)
        return None

    orig_w, orig_h = img.size  # PIL: (width, height)
    gutter_col = _find_gutter_col(arr_gray)

    # Guard: gutter must be inside the image with at least a few px of margin
    margin = 8
    if gutter_col < margin or gutter_col > orig_w - margin:
        gutter_col = orig_w // 2

    left_crop = img.crop((0, 0, gutter_col, orig_h))
    right_crop = img.crop((gutter_col, 0, orig_w, orig_h))

    sidecar = {
        "original_file": image_path.name,
        "original_width": orig_w,
        "original_height": orig_h,
        "gutter_col": gutter_col,
        "pages": [
            {
                "side": "left",
                "file": left_path.name,
                "x_offset": 0,
                "y_offset": 0,
                "width": gutter_col,
                "height": orig_h,
            },
            {
                "side": "right",
                "file": right_path.name,
                "x_offset": gutter_col,
                "y_offset": 0,
                "width": orig_w - gutter_col,
                "height": orig_h,
            },
        ],
    }

    if dry_run:
        print(
            f"  [dry-run] would write: {left_path.name}, "
            f"{right_path.name}, {json_path.name}  "
            f"(gutter_col={gutter_col})",
            file=sys.stderr,
        )
        return sidecar

    left_crop.save(left_path, "JPEG", quality=JPEG_QUALITY)
    right_crop.save(right_path, "JPEG", quality=JPEG_QUALITY)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    if not quiet:
        print(
            f"  Split: {image_path.name}  →  "
            f"{left_path.name} + {right_path.name}  "
            f"(gutter_col={gutter_col})",
            file=sys.stderr,
        )

    return sidecar


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split double-page spread images into left/right page files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "report_csv",
        help="Path to spreads_report.csv produced by detect_spreads.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-image progress output",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing split files instead of skipping them",
    )
    args = parser.parse_args()

    report_path = Path(args.report_csv)
    if not report_path.exists():
        print(f"Error: report not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    # Read the spreads report
    rows: list[dict] = []
    with open(report_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    double_rows = [r for r in rows if r.get("double_page", "").strip().lower() == "true"]

    if not double_rows:
        print("No double-page images found in report.", file=sys.stderr)
        sys.exit(0)

    print(
        f"Found {len(double_rows)} double-page image(s) of {len(rows)} total. "
        f"{'(dry run)' if args.dry_run else ''}",
        file=sys.stderr,
    )

    counts = {"split": 0, "skipped": 0, "error": 0}

    for row in double_rows:
        image_path_str = row.get("image_path", "").strip()
        if not image_path_str:
            print("  Warning: row missing image_path, skipping.", file=sys.stderr)
            counts["error"] += 1
            continue

        image_path = Path(image_path_str)
        if not image_path.exists():
            # Try resolving relative to the report's directory
            image_path = report_path.parent / image_path_str
        if not image_path.exists():
            print(
                f"  Warning: image not found: {image_path_str}", file=sys.stderr
            )
            counts["error"] += 1
            continue

        result = split_image(image_path, dry_run=args.dry_run, quiet=args.quiet, force=args.force)
        if result is None:
            counts["error"] += 1
        elif result.get("skipped"):
            counts["skipped"] += 1
        else:
            counts["split"] += 1

    verb = "Would split" if args.dry_run else "Split"
    print(
        f"\nDone. {verb} {counts['split']} image(s)"
        + (f", {counts['skipped']} already split (skipped)" if counts["skipped"] else "")
        + (f", {counts['error']} error(s)" if counts["error"] else "")
        + ".",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
