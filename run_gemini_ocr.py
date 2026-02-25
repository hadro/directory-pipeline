#!/usr/bin/env python3
"""Run Gemini-powered OCR on images, saving plain-text output alongside each image.

For each .jpg found in the images directory, sends the image to the Gemini API and
saves a .txt file alongside it. Already-processed images (where .txt already exists)
are skipped, so the script is safe to re-run.

The system prompt is read from ocr_prompt.md in the same directory as this script.

Requires a Gemini API key in the GEMINI_API_KEY environment variable.

Usage
-----
    python run_gemini_ocr.py images/travelguide
    python run_gemini_ocr.py images/greenbooks --workers 8
    python run_gemini_ocr.py images/travelguide --model gemini-2.0-flash
    python run_gemini_ocr.py images/travelguide --quiet
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai.types import GenerateContentConfig, Part

DEFAULT_MODEL = "gemini-2.0-flash"
PROMPT_FILE = Path(__file__).parent / "ocr_prompt.md"

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def process_image(
    client: genai.Client,
    image_path: Path,
    model: str,
    system_prompt: str,
) -> tuple[str, bool | None]:
    """
    OCR one image via Gemini. Returns (status, success) where status is one of
    'skipped', 'ok', 'failed'.
    """
    model_slug = model.replace("/", "_")
    txt_path = image_path.parent / f"{image_path.stem}_{model_slug}.txt"
    if txt_path.exists():
        if txt_path.stat().st_size > 0:
            return "skipped", None
        # File is empty — previous run produced no output; delete and retry.
        _log(f"  Re-running (empty output file): {txt_path.name}")
        txt_path.unlink()

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    max_retries = 5
    delay = 10  # seconds; doubles on each 429
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                config=GenerateContentConfig(system_instruction=system_prompt),
                contents=[Part.from_bytes(data=img_bytes, mime_type="image/jpeg")],
            )
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                _log(f"  Rate limited — retrying {image_path.name} in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise

    text = response.text or ""
    txt_path.write_text(text, encoding="utf-8")
    return "ok", True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini OCR on downloaded NYPL images, producing .txt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images_dir",
        help="Root images directory to process (e.g. images/travelguide)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel API requests (default: 4)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # System prompt
    if not PROMPT_FILE.exists():
        print(f"Error: prompt file not found: {PROMPT_FILE}", file=sys.stderr)
        sys.exit(1)
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    # Images
    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
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

    total = len(images)
    if not args.quiet:
        print(
            f"Processing {total} image(s) with {args.workers} worker(s) using {args.model}…",
            file=sys.stderr,
        )

    client = genai.Client(api_key=api_key)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_image, client, img, args.model, system_prompt): img
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
                model_slug = args.model.replace("/", "_")
                txt_name = f"{image_path.stem}_{model_slug}.txt"
                if status == "skipped":
                    _log(f"[{completed:04d}/{total}] Skipped (exists): {txt_name}")
                elif status == "ok":
                    _log(f"[{completed:04d}/{total}] Done: {txt_name}")
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
