#!/usr/bin/env python3
"""Detect double-page spreads in digitized document images.

Analyzes images to determine whether they contain two facing pages (a book
spread captured in a single scan) versus a single page, which may itself have
multiple text columns.

This situation commonly occurs with microfilm digitization where the camera
photographed both the left and right pages of an open book in one frame.

Detection uses local image analysis — no API calls required:
  1. Locate the content area within the dark microfilm border
  2. Check the aspect ratio of that content area
  3. Scan the central vertical band for a persistent gutter or seam

The gutter check handles three physical forms:
  - A dark shadow (spine pressing against the glass)
  - A white gap (pages splaying open)
  - A tonal boundary (e.g. a dark cover page beside a white content page)

Output is a CSV report. No image files are modified.

Usage
-----
    python detect_spreads.py images/Hackley_Harrison
    python detect_spreads.py images/Hackley_Harrison/4f7822b0-...
    python detect_spreads.py images/ --output spreads_report.csv
    python detect_spreads.py images/Hackley_Harrison --threshold 0.08
    python detect_spreads.py images/Hackley_Harrison --csv collection_csv/Hackley_Harrison.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Tuning parameters — adjust via --threshold / --aspect for edge cases
# ---------------------------------------------------------------------------
DARK_THRESHOLD = 50        # pixels at/below this are microfilm background
MIN_ASPECT_RATIO = 1.15    # content narrower than this is portrait → single page
CENTER_BAND_FRAC = 0.40    # fraction of content width searched for the gutter
DEFAULT_GUTTER_THRESHOLD = 0.12   # min fractional brightness difference to flag a gutter
MIN_CONSISTENCY = 0.50     # fraction of vertical strips that must confirm the gutter
MICROFORM_THRESHOLD_FACTOR = 0.6  # multiply threshold by this for known microform items


# ---------------------------------------------------------------------------
# Collection CSV helpers
# ---------------------------------------------------------------------------

def load_microform_lookup(csv_path: Path) -> dict[str, bool]:
    """
    Read a collection CSV (produced by nypl_collection_csv.py or loc_collection_csv.py)
    and return a dict mapping item_id → microform (bool).

    Rows without an 'item_id' or 'microform' column are skipped silently.
    """
    lookup: dict[str, bool] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item_id = row.get("item_id", "").strip()
            raw = row.get("microform", "").strip().lower()
            if item_id:
                lookup[item_id] = raw in ("true", "1", "yes")
    return lookup


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Uniform smoothing via convolution (no scipy dependency)."""
    if window < 2:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _gutter_score(band: np.ndarray) -> float:
    """
    Given a 2-D image strip (H × W), return a score for how strongly
    a vertical gutter or seam appears at the center.

    Combines two signals:
      1. Narrow-gutter score: the minimum column brightness in the center
         quarter vs the outer quarters (catches spine shadows and white gaps).
      2. Tonal-boundary score: the absolute brightness difference between
         the left and right halves (catches a dark cover page beside a white
         content page, which produces no narrow gutter but a strong left/right
         imbalance).

    Returns a value in [0, 1]; higher means a stronger spread signal.
    """
    col_means = band.mean(axis=0)
    bw = len(col_means)
    if bw < 8:
        return 0.0

    smoothed = _smooth(col_means, max(3, bw // 20))

    quarter = bw // 4

    # --- Signal 1: narrow gutter (minimum in center half vs outer quarters) ---
    outer_val = (smoothed[:quarter].mean() + smoothed[-quarter:].mean()) / 2
    center_min = smoothed[quarter: bw - quarter].min()
    if outer_val > 0:
        # Dark spine: center_min << outer_val → positive score
        # White gap: center_min >> outer_val → negative abs difference → still positive
        narrow_score = float(abs(center_min - outer_val) / outer_val)
    else:
        narrow_score = 0.0

    # --- Signal 2: left/right tonal boundary (covers spread across whole image) ---
    half = bw // 2
    left_val = smoothed[:half].mean()
    right_val = smoothed[half:].mean()
    page_mean = (left_val + right_val) / 2
    if page_mean > 0:
        boundary_score = float(abs(left_val - right_val) / page_mean)
    else:
        boundary_score = 0.0

    return max(narrow_score, boundary_score)


def analyze_image(
    image_path: Path,
    gutter_threshold: float = DEFAULT_GUTTER_THRESHOLD,
    min_aspect: float = MIN_ASPECT_RATIO,
    microform_prior: bool = False,
) -> dict:
    """
    Analyze one image for double-page spread characteristics.

    If microform_prior is True (item is known to come from microfilm/microform),
    the gutter threshold is reduced by MICROFORM_THRESHOLD_FACTOR so that faint
    gutters still trigger a DOUBLE detection.

    Returns a dict with keys:
        double_page      bool
        confidence       'high' | 'medium' | 'low'
        reason           short string
        aspect_ratio     float | None
        gutter_score     float   (0 = no gutter, higher = stronger signal)
        consistency      float   (fraction of vertical strips confirming gutter)
        microform_prior  bool
    """
    effective_threshold = (
        gutter_threshold * MICROFORM_THRESHOLD_FACTOR
        if microform_prior
        else gutter_threshold
    )
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # ------------------------------------------------------------------
    # 1. Find content bounding box — strip the dark microfilm border
    # ------------------------------------------------------------------
    mask = arr > DARK_THRESHOLD
    row_any = mask.any(axis=1)
    col_any = mask.any(axis=0)

    blank_result = {
        "double_page": False, "confidence": "high", "reason": "blank",
        "aspect_ratio": None, "gutter_score": 0.0, "consistency": 0.0,
        "microform_prior": microform_prior,
    }

    if not row_any.any() or not col_any.any():
        return blank_result

    r0 = int(np.argmax(row_any))
    r1 = int(len(row_any) - 1 - np.argmax(row_any[::-1]))
    c0 = int(np.argmax(col_any))
    c1 = int(len(col_any) - 1 - np.argmax(col_any[::-1]))

    content = arr[r0: r1 + 1, c0: c1 + 1]
    ch, cw = content.shape
    if ch == 0 or cw == 0:
        return blank_result

    aspect = cw / ch

    # ------------------------------------------------------------------
    # 2. Quick reject: portrait content is almost never a spread
    # ------------------------------------------------------------------
    if aspect < min_aspect:
        return {
            "double_page": False, "confidence": "high", "reason": "portrait",
            "aspect_ratio": round(aspect, 3),
            "gutter_score": 0.0, "consistency": 0.0,
            "microform_prior": microform_prior,
        }

    # ------------------------------------------------------------------
    # 3. Examine the central band for a vertical gutter / seam
    # ------------------------------------------------------------------
    cx = cw // 2
    half_band = int(cw * CENTER_BAND_FRAC / 2)
    bl = max(0, cx - half_band)
    br = min(cw, cx + half_band)
    band = content[:, bl:br]

    overall_score = _gutter_score(band)

    # ------------------------------------------------------------------
    # 4. Vertical consistency — the gutter must run top-to-bottom.
    #    A column separator on a single page would not.
    # ------------------------------------------------------------------
    N_STRIPS = 6
    strip_h = ch // N_STRIPS
    strip_scores = []
    for i in range(N_STRIPS):
        strip = content[i * strip_h: (i + 1) * strip_h, bl:br]
        if strip.shape[0] > 0:
            strip_scores.append(_gutter_score(strip))

    half_threshold = effective_threshold * 0.5
    n_consistent = sum(1 for s in strip_scores if s >= half_threshold)
    consistency = n_consistent / len(strip_scores) if strip_scores else 0.0

    # ------------------------------------------------------------------
    # 5. Decision
    # ------------------------------------------------------------------
    is_double = (
        overall_score >= effective_threshold
        and consistency >= MIN_CONSISTENCY
    )

    if is_double:
        confidence = (
            "high"
            if overall_score >= effective_threshold * 1.5 and consistency >= 0.70
            else "medium"
        )
        reason = "double_page_spread"
    else:
        if overall_score < effective_threshold * 0.4:
            confidence = "high"
        elif consistency < MIN_CONSISTENCY:
            confidence = "medium"
        else:
            confidence = "low"
        reason = "single_page"

    return {
        "double_page": is_double,
        "confidence": confidence,
        "reason": reason,
        "aspect_ratio": round(aspect, 3),
        "gutter_score": round(overall_score, 3),
        "consistency": round(consistency, 3),
        "microform_prior": microform_prior,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect double-page spreads in digitized document images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images_dir",
        help="Directory to scan (e.g. images/Hackley_Harrison or a single item UUID dir)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: <images_dir>/spreads_report.csv)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_GUTTER_THRESHOLD,
        metavar="FRAC",
        help=(
            f"Gutter brightness threshold 0–1 (default: {DEFAULT_GUTTER_THRESHOLD}). "
            "Lower = more sensitive to faint gutters."
        ),
    )
    parser.add_argument(
        "--aspect",
        type=float,
        default=MIN_ASPECT_RATIO,
        metavar="RATIO",
        help=(
            f"Minimum content aspect ratio to be a spread candidate "
            f"(default: {MIN_ASPECT_RATIO}). Width ÷ height of the content area."
        ),
    )
    parser.add_argument(
        "--csv", "-c",
        default=None,
        metavar="CSV",
        help=(
            "Optional collection CSV (from nypl_collection_csv.py). "
            "When provided, items flagged microform=True use a lower gutter "
            f"threshold ({MICROFORM_THRESHOLD_FACTOR}× the normal value) so "
            "faint gutters still register as double-page spreads."
        ),
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-analyze all images even if spreads_report.csv already exists",
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

    out_path = Path(args.output) if args.output else images_root / "spreads_report.csv"

    if out_path.exists() and not args.force:
        print(
            f"spreads_report.csv already exists ({out_path}). "
            "Use --force to re-analyze.",
            file=sys.stderr,
        )
        sys.exit(0)

    images = sorted(
        p for p in images_root.rglob("*.jpg")
        if not (p.stem.endswith("_left") or p.stem.endswith("_right"))
    )
    if not images:
        print(f"No .jpg files found under {images_root}", file=sys.stderr)
        sys.exit(0)

    # Load microform lookup from collection CSV if supplied
    microform_lookup: dict[str, bool] = {}
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        microform_lookup = load_microform_lookup(csv_path)
        n_microform = sum(microform_lookup.values())
        print(
            f"Loaded {len(microform_lookup)} item(s) from CSV "
            f"({n_microform} flagged microform).",
            file=sys.stderr,
        )

    print(f"Scanning {len(images)} image(s)…", file=sys.stderr)

    fieldnames = [
        "image_path", "item_id", "filename",
        "double_page", "confidence", "reason",
        "aspect_ratio", "gutter_score", "consistency",
        "microform_prior",
    ]

    counts = {"double": 0, "single": 0, "low_conf": 0, "error": 0}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, img_path in enumerate(images, 1):
            # Extract item_id: images/<collection>/<item_id>/<filename>
            # or images/<item_id>/<filename>
            parts = img_path.parts
            item_id = parts[-2] if len(parts) >= 2 else ""

            microform_prior = microform_lookup.get(item_id, False)

            try:
                result = analyze_image(
                    img_path, args.threshold, args.aspect, microform_prior
                )
            except Exception as exc:  # noqa: BLE001
                counts["error"] += 1
                print(
                    f"  Warning: could not analyze {img_path.name}: {exc}",
                    file=sys.stderr,
                )
                continue

            writer.writerow({
                "image_path": str(img_path),
                "item_id": item_id,
                "filename": img_path.name,
                **result,
            })

            if result["double_page"]:
                counts["double"] += 1
            else:
                counts["single"] += 1
            if result["confidence"] == "low":
                counts["low_conf"] += 1

            if not args.quiet:
                flag = "DOUBLE" if result["double_page"] else "single"
                print(
                    f"  [{i:04d}/{len(images)}] {flag:6s} "
                    f"({result['confidence']:6s})  "
                    f"aspect={result['aspect_ratio']}  "
                    f"gutter={result['gutter_score']}  "
                    f"consist={result['consistency']}  "
                    f"{img_path.name}",
                    file=sys.stderr,
                )

    total = counts["double"] + counts["single"]
    print(
        f"\nDone. {total} image(s) analyzed: "
        f"{counts['double']} double-page, {counts['single']} single-page"
        + (f", {counts['low_conf']} low-confidence" if counts["low_conf"] else "")
        + (f", {counts['error']} error(s)" if counts["error"] else "")
        + ".",
        file=sys.stderr,
    )
    print(f"Report → {out_path}", file=sys.stderr)

    if counts["low_conf"]:
        print(
            "Tip: low-confidence results may benefit from --threshold adjustment. "
            "Lower the threshold to catch fainter gutters, raise it to reduce false positives.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
