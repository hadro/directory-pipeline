#!/usr/bin/env python3
"""Retry Tesseract OCR with alternative PSM settings for poorly-aligned pages.

For each page whose aligned JSON has more than UNMATCHED_FRAC_THRESHOLD of
Gemini lines unmatched (and at least UNMATCHED_COUNT_MIN unmatched lines),
re-runs Tesseract with alternative page segmentation modes and keeps
whichever setting yields the fewest unmatched lines.  The winning hOCR and
aligned JSON are written back to disk so the improvement is permanent.

Retry sequence (tried in order):
  1. --psm 11              (sparse text — best for isolated headings)
  2. --psm 6  --oem 2      (uniform block + legacy+LSTM combined)
  3. --psm 3  --oem 0      (auto layout + legacy engine only)

Usage
-----
    python improve_alignment.py images/greenbooks/feb978b0 --model gemini-2.0-flash
    python improve_alignment.py images/greenbooks/ --model gemini-2.0-flash --workers 4
    python improve_alignment.py images/greenbooks/ --model gemini-2.0-flash --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import align_ocr

DEFAULT_FRAC_THRESHOLD  = 0.05
DEFAULT_COUNT_MIN       = 4

# Tried in order; first that beats the current best wins that round.
# All rounds are evaluated and the global best is kept.
RETRY_SETTINGS: list[dict] = [
    {"psm": 11, "oem": None},  # sparse text — best for isolated headings
    {"psm":  6, "oem": 2   },  # uniform block + legacy+LSTM combined
    {"psm":  3, "oem": 0   },  # auto layout + legacy engine only
]

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

def _unmatched_stats(aligned: dict) -> tuple[int, float]:
    """Return (n_unmatched, unmatched_fraction) from an aligned JSON dict."""
    n_matched   = len(aligned.get("lines", []))
    n_unmatched = len(aligned.get("unmatched_gemini", []))
    total = n_matched + n_unmatched
    return n_unmatched, (n_unmatched / total) if total > 0 else 0.0


def _needs_retry(aligned: dict, frac_threshold: float, count_min: int) -> bool:
    n_unmatched, frac = _unmatched_stats(aligned)
    return n_unmatched >= count_min and frac > frac_threshold


# ---------------------------------------------------------------------------
# Tesseract retry
# ---------------------------------------------------------------------------

def _run_tesseract(
    image_path: Path,
    hocr_out_path: Path,
    psm: int | None,
    oem: int | None,
) -> bool:
    """
    Run Tesseract on image_path, writing hOCR output to hocr_out_path.
    hocr_out_path should include the .hocr suffix; Tesseract is given the
    stem (it appends .hocr itself).  Returns True on success.
    """
    # Tesseract needs the output *base* path (without extension)
    output_base = hocr_out_path.with_suffix("")
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"  # mirrors run_ocr.py

    cmd = ["tesseract", str(image_path), str(output_base), "-l", "eng"]
    if psm is not None:
        cmd += ["--psm", str(psm)]
    if oem is not None:
        cmd += ["--oem", str(oem)]
    # Suppress dictionary correction to preserve proper nouns / street names
    # (matches run_ocr.py defaults)
    cmd += ["-c", "load_system_dawg=0", "-c", "load_freq_dawg=0"]
    cmd += ["hocr"]

    result = subprocess.run(cmd, capture_output=True, env=env)
    return result.returncode == 0


def _align_from_hocr(
    hocr_path: Path,
    gemini_lines: list[str],
    canvas_uri: str,
    canvas_w: int,
    canvas_h: int,
) -> tuple[list[dict], list[str]]:
    """
    Parse hocr_path and run NW alignment against gemini_lines.
    Returns (result_lines, unmatched_gemini).
    """
    page_bbox, lines = align_ocr.parse_hocr(hocr_path)
    img_w = page_bbox[2]
    img_h = page_bbox[3]
    page_w = img_w - page_bbox[0]

    lines = align_ocr.sort_by_reading_order(lines, page_w)
    lines = align_ocr.filter_short_lines(lines)

    def fragment(bbox: list[int]) -> str:
        if not canvas_uri:
            return ""
        return align_ocr._canvas_fragment(
            canvas_uri, bbox, img_w, img_h, canvas_w, canvas_h
        )

    return align_ocr._build_word_aligned_lines(lines, gemini_lines, fragment)


# ---------------------------------------------------------------------------
# Per-image improvement
# ---------------------------------------------------------------------------

def try_improve(
    image_path: Path,
    model: str,
    frac_threshold: float,
    count_min: int,
    dry_run: bool,
    quiet: bool,
) -> str:
    """
    Check alignment quality for one image and retry Tesseract if poor.

    Returns one of:
      'improved'      — found a better setting; files updated on disk
      'already_good'  — unmatched fraction is within threshold
      'no_improvement'— retried but nothing beat the original
      'missing'       — aligned JSON or source files not found
      'failed'        — unexpected error
    """
    slug = align_ocr.model_slug(model)
    stem = image_path.stem
    aligned_path = image_path.parent / f"{stem}_{slug}_aligned.json"
    hocr_path    = image_path.parent / f"{stem}_tesseract.hocr"
    gemini_txt   = image_path.parent / f"{stem}_{slug}.txt"

    if not aligned_path.exists() or not hocr_path.exists() or not gemini_txt.exists():
        return "missing"

    try:
        aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    except Exception:
        return "failed"

    if not _needs_retry(aligned, frac_threshold, count_min):
        return "already_good"

    n_orig, frac_orig = _unmatched_stats(aligned)

    # Load Gemini lines and IIIF canvas info once
    gemini_text  = gemini_txt.read_text(encoding="utf-8")
    gemini_lines = [ln for ln in (ln.strip() for ln in gemini_text.splitlines()) if ln]

    image_id     = align_ocr._extract_image_id(image_path.name)
    manifest     = image_path.parent / "manifest.json"
    canvas_uri, canvas_w, canvas_h = align_ocr.load_canvas_info(manifest, image_id)

    original_unmatched_set = set(aligned.get("unmatched_gemini", []))

    # Score: (rescued_from_original, -total_unmatched) — higher is better.
    # Primary key: how many of the original unmatched lines are now matched.
    # Tiebreaker: fewer total unmatched.  This lets a retry win even when PSM 11
    # finds the missing headings but also scrambles some body-text matches
    # (total count stays equal), which the old strict "n < best_n" would reject.
    best_score         = (0, -n_orig)
    best_result_lines  = None
    best_unmatched     = None
    best_setting       = None
    best_hocr_bytes    = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for setting in RETRY_SETTINGS:
            psm = setting["psm"]
            oem = setting["oem"]
            tmp_hocr = tmp / f"retry_psm{psm}_oem{oem}.hocr"

            if not dry_run:
                ok = _run_tesseract(image_path, tmp_hocr, psm, oem)
                if not ok or not tmp_hocr.exists():
                    continue

                try:
                    result_lines, unmatched = _align_from_hocr(
                        tmp_hocr, gemini_lines, canvas_uri, canvas_w, canvas_h,
                    )
                except Exception:
                    continue

                rescued = len(original_unmatched_set - set(unmatched))
                score = (rescued, -len(unmatched))
                if score > best_score:
                    best_score        = score
                    best_result_lines = result_lines
                    best_unmatched    = unmatched
                    best_setting      = setting
                    best_hocr_bytes   = tmp_hocr.read_bytes()

    if dry_run:
        # In dry-run mode just report which pages would be retried
        label = f"psm={RETRY_SETTINGS[0]['psm']}"
        _log(
            f"  [dry run] Would retry: {image_path.name}  "
            f"unmatched={n_orig} ({frac_orig:.0%})  "
            f"first attempt: {label}"
        )
        return "improved"  # report as if improved for counting purposes

    if best_result_lines is None or best_hocr_bytes is None:
        return "no_improvement"

    # Write the improved aligned JSON (preserving all other fields)
    improved = dict(aligned)
    improved["lines"]            = best_result_lines
    improved["unmatched_gemini"] = best_unmatched
    improved["retry_psm"]        = best_setting["psm"]
    improved["retry_oem"]        = best_setting["oem"]
    aligned_path.write_text(
        json.dumps(improved, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Replace the hOCR with the winning version so future re-runs use it
    hocr_path.write_bytes(best_hocr_bytes)

    if not quiet:
        rescued_count = best_score[0]
        _log(
            f"  Improved: {aligned_path.name}  "
            f"rescued {rescued_count}/{n_orig} unmatched  "
            f"total {n_orig}→{len(best_unmatched)}  "
            f"psm={best_setting['psm']} oem={best_setting['oem']}"
        )
    return "improved"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retry Tesseract OCR with alternative PSM settings for poorly-aligned pages."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images_dir",
        help=(
            "Item images directory containing *_aligned.json files "
            "(e.g. images/greenbooks/feb978b0), or a parent directory to "
            "process all item subdirectories."
        ),
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        metavar="MODEL",
        help="Gemini model used for alignment (e.g. gemini-2.0-flash)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_FRAC_THRESHOLD,
        metavar="FRAC",
        help=(
            f"Unmatched-line fraction above which a page is retried "
            f"(default: {DEFAULT_FRAC_THRESHOLD:.0%})"
        ),
    )
    parser.add_argument(
        "--min-unmatched",
        type=int,
        default=DEFAULT_COUNT_MIN,
        dest="min_unmatched",
        metavar="N",
        help=(
            f"Minimum number of unmatched lines required to trigger a retry "
            f"(default: {DEFAULT_COUNT_MIN}). Prevents retrying pages that "
            f"have only 1-2 legitimately unmatchable lines."
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
        "--dry-run",
        action="store_true",
        help="Show which pages would be retried without making any changes",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-image progress output",
    )
    args = parser.parse_args()

    slug = align_ocr.model_slug(args.model)
    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    # Collect images that have an existing aligned JSON to assess
    all_jpgs = sorted(images_root.rglob("*.jpg"))
    images: list[Path] = []
    for p in all_jpgs:
        if p.stem.endswith("_viz"):
            continue
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        left  = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        if (p.parent / f"{p.stem}_{slug}_aligned.json").exists():
            images.append(p)

    if not images:
        print(
            f"No *_{slug}_aligned.json files found under {images_root}",
            file=sys.stderr,
        )
        sys.exit(0)

    total = len(images)
    dry_tag = "  [DRY RUN — no files will be changed]\n" if args.dry_run else ""
    print(
        f"\n{dry_tag}"
        f"Checking {total} page(s) — threshold={args.threshold:.0%}, "
        f"min-unmatched={args.min_unmatched}, model={args.model}",
        file=sys.stderr,
    )

    counts: dict[str, int] = {
        "improved": 0, "already_good": 0,
        "no_improvement": 0, "missing": 0, "failed": 0,
    }
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                try_improve, img, args.model,
                args.threshold, args.min_unmatched,
                args.dry_run, args.quiet,
            ): img
            for img in images
        }
        for future in as_completed(futures):
            image_path = futures[future]
            completed += 1
            try:
                status = future.result()
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                _log(f"Warning: exception for {image_path.name}: {exc}")

            counts[status] += 1

            if not args.quiet:
                slug_out = f"{image_path.stem}_{slug}_aligned.json"
                if status == "no_improvement":
                    _log(
                        f"[{completed:04d}/{total}] No improvement: "
                        f"{image_path.name}"
                    )
                elif status in ("missing", "failed"):
                    _log(
                        f"[{completed:04d}/{total}] {status.upper()}: "
                        f"{image_path.name}"
                    )
                # 'improved' is logged inside try_improve(); 'already_good' is silent

    print(
        f"\nDone. "
        f"{counts['improved']} improved, "
        f"{counts['already_good']} already within threshold, "
        f"{counts['no_improvement']} retried with no improvement"
        + (f", {counts['missing']} missing" if counts["missing"] else "")
        + (f", {counts['failed']} failed" if counts["failed"] else "")
        + ".",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
