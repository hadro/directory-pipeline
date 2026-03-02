#!/usr/bin/env python3
"""Run Tesseract OCR on images downloaded by download_images.py, saving hOCR output.

For each .jpg found in the images directory, runs Tesseract and saves a .hocr
file alongside it. Already-processed images (where .hocr already exists) are
skipped, so the script is safe to re-run.

Multiple images are processed in parallel using --workers (default: 4), which
works well on M-series MacBook Airs. Each worker runs a single Tesseract subprocess
with OMP_THREAD_LIMIT=1 to avoid internal thread contention between workers.

Requires Tesseract to be installed and on PATH:
    brew install tesseract        # macOS
    apt install tesseract-ocr     # Debian/Ubuntu

Usage
-----
    python run_ocr.py images/travelguide
    python run_ocr.py images/greenbooks --lang eng
    python run_ocr.py images/travelguide --workers 6
    python run_ocr.py images/travelguide --quiet
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def process_image(
    image_path: Path,
    lang: str,
    psm: int | None,
    oem: int | None,
    dpi: int | None,
    use_dict: bool,
    quiet: bool,
) -> tuple[str, bool | None]:
    """
    OCR one image. Returns (status, success) where status is one of
    'skipped', 'ok', 'failed'.
    """
    output_base = image_path.parent / (image_path.stem + "_tesseract")
    hocr_path = output_base.with_suffix(".hocr")
    if hocr_path.exists():
        return "skipped", None

    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"  # prevent each worker spawning many threads

    cmd = ["tesseract", str(image_path), str(output_base), "-l", lang]
    if psm is not None:
        cmd += ["--psm", str(psm)]
    if oem is not None:
        cmd += ["--oem", str(oem)]
    if dpi is not None:
        cmd += ["--dpi", str(dpi)]
    if not use_dict:
        # Suppress dictionary correction: Tesseract otherwise "corrects"
        # proper nouns, street names, and abbreviations toward dictionary
        # words, which corrupts NW alignment with the Gemini text.
        cmd += ["-c", "load_system_dawg=0", "-c", "load_freq_dawg=0"]
    cmd += ["hocr", "txt"]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return ("ok" if result.returncode == 0 else "failed"), result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Tesseract OCR on downloaded NYPL images, producing hOCR files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images_dir",
        help="Root images directory to process (e.g. images/travelguide)",
    )
    parser.add_argument(
        "--lang", "-l",
        default="eng",
        help="Tesseract language code (default: eng)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Tesseract page segmentation mode (see: tesseract --help-psm). "
            "Default: Tesseract's built-in default (3 = auto). "
            "Recommended for multi-column pages: 1 (auto with OSD)."
        ),
    )
    parser.add_argument(
        "--oem",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Tesseract OCR engine mode (0=legacy, 1=LSTM, 2=legacy+LSTM, 3=default). "
            "Try --oem 2 for degraded historical scans where LSTM alone misses text."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Image DPI hint for Tesseract. Set if images lack DPI metadata "
            "(300 or 400 are typical for archival scans). Wrong DPI causes "
            "Tesseract to discard small text as noise."
        ),
    )
    parser.add_argument(
        "--dict",
        dest="use_dict",
        action="store_true",
        help=(
            "Re-enable Tesseract dictionary correction (disabled by default). "
            "Dictionary correction is off by default because it silently changes "
            "proper nouns, street names, and abbreviations toward dictionary words, "
            "which corrupts NW alignment with Gemini text."
        ),
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=min(4, os.cpu_count() or 1),
        metavar="N",
        help="Number of parallel Tesseract processes (default: 4)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = parser.parse_args()

    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "Error: tesseract not found on PATH. Install with:\n"
            "  brew install tesseract        # macOS\n"
            "  apt install tesseract-ocr     # Debian/Ubuntu",
            file=sys.stderr,
        )
        sys.exit(1)

    all_jpgs = sorted(images_root.rglob("*.jpg"))
    images = []
    for p in all_jpgs:
        # Skip visualization output files
        if p.stem.endswith("_viz"):
            continue
        # Always include split output files
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        # Skip the original if split files exist (OCR the splits instead)
        left = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        images.append(p)
    if not images:
        print(f"No .jpg files found under {images_root}", file=sys.stderr)
        sys.exit(0)

    # Load per-image PSM recommendations from detect_columns.py output (if present).
    # The global --psm flag always takes precedence over the per-image value.
    psm_map: dict[str, int] = {}
    columns_report = images_root / "columns_report.csv"
    if columns_report.exists():
        with open(columns_report, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = row.get("recommended_psm", "").strip()
                if raw:
                    try:
                        psm_map[row["image"]] = int(raw)
                    except ValueError:
                        pass
        if not args.quiet:
            print(
                f"Loaded per-image PSM from {columns_report.name} ({len(psm_map)} entries).",
                file=sys.stderr,
            )

    total = len(images)
    if not args.quiet:
        print(
            f"Processing {total} image(s) with {args.workers} worker(s)…",
            file=sys.stderr,
        )

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_image, img, args.lang,
                # Global --psm overrides per-image PSM from columns_report.csv
                args.psm if args.psm is not None else psm_map.get(img.name),
                args.oem, args.dpi, args.use_dict, args.quiet,
            ): img
            for img in images
        }
        for future in as_completed(futures):
            image_path = futures[future]
            completed += 1
            try:
                status, _ = future.result()
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                _log(f"Warning: exception processing {image_path}: {exc}")

            counts[status] += 1

            if not args.quiet:
                hocr_name = (image_path.parent / (image_path.stem + "_tesseract.hocr")).name
                if status == "skipped":
                    _log(f"[{completed:04d}/{total}] Skipped (exists): {hocr_name}")
                elif status == "ok":
                    _log(f"[{completed:04d}/{total}] Done: {hocr_name}")
                else:
                    _log(f"[{completed:04d}/{total}] FAILED: {image_path.name}")

    if not args.quiet:
        print(
            f"\nDone. {total} image(s): "
            f"{counts['ok']} processed, {counts['skipped']} skipped, {counts['failed']} failed.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
