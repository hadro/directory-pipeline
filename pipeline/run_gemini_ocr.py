#!/usr/bin/env python3
"""Run Gemini-powered OCR on images, saving plain-text output alongside each image.

For each .jpg found in the images directory, sends the image to the Gemini API and
saves a .txt file alongside it. Already-processed images (where .txt already exists)
are skipped, so the script is safe to re-run.

The system prompt is read from ocr_prompt.md in the same directory as this script.

Requires a Gemini API key in the GEMINI_API_KEY environment variable.

Usage
-----
    python run_gemini_ocr.py output/travelguide
    python run_gemini_ocr.py output/greenbooks --workers 8
    python run_gemini_ocr.py output/travelguide --model gemini-2.0-flash
    python run_gemini_ocr.py output/travelguide --quiet
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()
from google.genai.types import GenerateContentConfig, MediaResolution, Part, ThinkingConfig

DEFAULT_MODEL = "gemini-2.0-flash"
PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "ocr_prompt.md"

# Keywords in the OCR prompt that indicate handwriting → high resolution recommended
_HANDWRITING_KEYWORDS = ("handwrit", "manuscript", "cursive")


def _needs_high_res(prompt_text: str) -> bool:
    lower = prompt_text.lower()
    return any(kw in lower for kw in _HANDWRITING_KEYWORDS)

_DITTO_INSTRUCTION = (
    "\nDitto marks: when you see small raised comma-pairs (printed as '' or 〃, "
    "often misread as 66 or as a quotation mark) used to repeat a value from the "
    "row above, expand them in place — write out the full repeated value rather "
    "than transcribing the mark itself."
)

_print_lock = threading.Lock()


def _load_scope(output_root: Path) -> "set[str] | None":
    """Return the set of filenames to process, or None (= process all).

    Reads included_pages.txt from output_root or output_root.parent.
    Returns None when no file is found (backward-compatible: process everything).
    """
    for d in (output_root.resolve(), output_root.resolve().parent):
        p = d / "included_pages.txt"
        if p.exists():
            lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.startswith("#")]
            if lines:
                return set(lines)
    return None


def _find_prompt(output_root: Path, fallback: Path) -> Path:
    """Return a volume-specific ocr_prompt.md if one exists alongside the images,
    otherwise return the global fallback.

    Checks output_root and output_root.parent so the lookup works whether
    output_root is the item directory or the slug-level directory above it.
    """
    for candidate_dir in (output_root.resolve(), output_root.resolve().parent):
        p = candidate_dir / "ocr_prompt.md"
        if p.exists():
            return p
    return fallback


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def process_image(
    client: genai.Client,
    image_path: Path,
    model: str,
    system_prompt: str,
    media_resolution: "MediaResolution | None" = None,
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
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    media_resolution=media_resolution,
                    thinking_config=ThinkingConfig(thinking_budget=0),
                ),
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
    if not text:
        candidate = response.candidates[0] if response.candidates else None
        if candidate:
            _log(f"  finish_reason: {candidate.finish_reason}  [{image_path.name}]")
            if candidate.safety_ratings:
                _log(f"  safety_ratings: {candidate.safety_ratings}  [{image_path.name}]")
    txt_path.write_text(text, encoding="utf-8")
    return "ok", True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini OCR on downloaded NYPL images, producing .txt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Root images directory to process (e.g. output/travelguide)",
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
        "--prompt-file",
        metavar="PATH",
        help=(
            "Path to a custom OCR system prompt (default: prompts/ocr_prompt.md). "
            "Use generate_prompt.py to produce a volume-specific prompt."
        ),
    )
    parser.add_argument(
        "--expand-dittos",
        dest="expand_dittos",
        action="store_true",
        help=(
            "Expand ditto marks in place rather than transcribing them literally. "
            "Ditto marks (printed as '' or 〃) are common in tabular historical "
            "documents and are often misread as 66. When this flag is set, the model "
            "is instructed to write out the repeated value on each row instead."
        ),
    )
    parser.add_argument(
        "--high-res",
        dest="high_res",
        action="store_true",
        help=(
            "Send images at high media resolution (more detail, higher token cost). "
            "Auto-enabled when the OCR prompt mentions handwriting or manuscripts."
        ),
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

    # Images root (needed for prompt auto-discovery below)
    output_root = Path(args.output_dir)

    # System prompt — explicit flag > volume-specific > global default
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
    else:
        prompt_path = _find_prompt(output_root, PROMPT_FILE)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8")
    if args.expand_dittos:
        system_prompt = system_prompt.rstrip() + _DITTO_INSTRUCTION
    if not args.quiet and prompt_path != PROMPT_FILE:
        print(f"Using volume prompt: {prompt_path}", file=sys.stderr)
    if not args.quiet and args.expand_dittos:
        print("Ditto mark expansion: enabled", file=sys.stderr)

    # Media resolution: explicit flag > auto-detect from prompt > None (API default)
    if args.high_res:
        media_resolution = MediaResolution.MEDIA_RESOLUTION_HIGH
    elif _needs_high_res(system_prompt):
        media_resolution = MediaResolution.MEDIA_RESOLUTION_HIGH
        if not args.quiet:
            print("High media resolution: auto-enabled (handwriting detected in prompt)", file=sys.stderr)
    else:
        media_resolution = None
    if not output_root.exists():
        print(f"Error: directory not found: {output_root}", file=sys.stderr)
        sys.exit(1)

    all_jpgs = sorted(output_root.rglob("*.jpg"))
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
        print(f"No .jpg files found under {output_root}", file=sys.stderr)
        sys.exit(0)

    # Apply scope filter (included_pages.txt) if present
    scope = _load_scope(output_root)
    if scope is not None:
        before = len(images)
        images = [p for p in images if p.name in scope]
        if not args.quiet:
            print(
                f"Scope filter: {len(images)} of {before} pages included"
                f" (from included_pages.txt)",
                file=sys.stderr,
            )

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
            executor.submit(process_image, client, img, args.model, system_prompt, media_resolution): img
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
