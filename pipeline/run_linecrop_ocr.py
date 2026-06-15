#!/usr/bin/env python3
"""Alignment-free OCR fallback: transcribe each Surya line crop individually.

This is the "hard-page" fallback inspired by uvalawlibrary/HTR-ALTO-Pipeline
(see docs/comparison-htr-alto-pipeline.md). Instead of OCR'ing a whole page and
then mapping the text back onto Surya bboxes with Needleman-Wunsch
(``align_ocr.py``), it crops each Surya line and sends that crop to Gemini on its
own. The model only has to *read* one line — position is already known from
segmentation — so there is no alignment step and no class of alignment-drift
failure. Text and bbox are bound together by construction.

When to use it
--------------
The NW path is cheaper (one API call per page) and works well on clean
multi-column directories. This script trades cost (one API call per *line*) for
robustness, so it is meant for the minority of pages where alignment genuinely
fails. The default selection therefore targets *genuine* column merges — pages
whose aligned JSON contains lines carrying two street addresses (the same
heuristic as ``tools/detect_column_merges.py``), which is a far more reliable
trigger than the noisy ``possible_column_merge`` flag (see
docs/comparison-htr-alto-pipeline.md). Use ``--by-flag`` for the older
``needs_review`` / ``possible_column_merge`` selection.

Cost warning
------------
Each line crop costs nearly as much as a full-page OCR call (Gemini bills a
per-image token floor), so cost scales ~linearly with line count: measured ≈ 9×
the calls and ≈ 9× the input tokens on a sparse 9-line page, and ~40× on a
typical ~50-line directory page (see docs/comparison-htr-alto-pipeline.md). This
is affordable *only* because it is scoped to a few flagged pages — a few cents
each. **Do not run ``--all`` on a large volume**: that re-OCRs every line of
every page and can cost 8–100× the entire volume's normal OCR bill for no
benefit on pages that aligned fine. Likewise **do not pass ``--flex`` here**:
Flex's 1–15 min latency is per request, and this stage makes 50–300 requests per
page, so the standard tier (the default) is both faster and the right choice.

Output
------
Writes the same JSON shape as ``align_ocr.py`` so downstream stages
(``extract_entries.py`` etc.) consume it unchanged, with two differences:

* each line's ``confidence`` is ``"crop"`` (a new tier alongside "line"/"manual")
* a top-level ``"method": "linecrop"`` marker is added
* ``surya_text`` (Surya's own transcription of the line) is preserved alongside
  ``gemini_text`` so the two can be compared during evaluation

By default output goes to ``{stem}_{slug}_linecrop.json`` (non-destructive, easy
to diff against the existing ``_aligned.json``). Pass ``--replace`` to overwrite
the canonical ``{stem}_{slug}_aligned.json`` instead.

Usage
-----
    # Re-OCR every flagged page in a collection, non-destructively
    python run_linecrop_ocr.py output/greenbooks

    # Target specific pages by filename stem (repeatable)
    python run_linecrop_ocr.py output/greenbooks --page 0018_..._0017.jp2

    # Process every page regardless of flags, and replace the aligned JSON
    python run_linecrop_ocr.py output/greenbooks --all --replace

    # See which pages would be processed without calling the API
    python run_linecrop_ocr.py output/greenbooks --dry-run

Requires GEMINI_API_KEY (see utils/gemini.get_client).
"""

import argparse
import io
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

from pipeline.align_ocr import (
    _expand_multiline_surya_lines,
    filter_margin_lines,
    filter_short_lines,
    parse_surya,
    resolve_fragment_fn,
    sort_by_reading_order,
)
from pipeline.state import get_ocr_model
from tools.detect_column_merges import _looks_merged
from utils.gemini import flex_http_options, generate_with_retry, get_client
from utils.models import DEFAULT_OCR_MODEL, discover_ocr_slug, model_slug

# A line crop carries no page context, so the prompt is deliberately minimal:
# read exactly what is visible, return nothing else. Sentinel tokens mirror the
# HTR-ALTO machine contract so downstream code can detect gaps explicitly rather
# than guessing on empty strings.
_LINE_PROMPT = (
    "This image is a single cropped line of text from a historical printed "
    "directory or guidebook. Transcribe exactly the text visible in the image. "
    "Return only that text as a single line of plain text — no quotation marks, "
    "no commentary, no Markdown. If the crop is blank, return [blank]. "
    "If text is present but illegible, return [illegible]."
)

# Pixels of padding added around each Surya bbox before cropping. Surya boxes are
# tight; a little slack avoids clipping ascenders/descenders and improves reads.
_CROP_PAD_X = 3
_CROP_PAD_Y = 4

# Crops smaller than this (in either dimension, post-pad) are treated as noise
# and skipped without an API call.
_MIN_CROP_PX = 6

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def _discover_images(output_root: Path) -> list[Path]:
    """Split-aware page discovery — same selection rule as the OCR/align stages.

    Prefers ``_left``/``_right`` split images over their un-split original when
    both exist, and skips ``_viz`` visualization artifacts.
    """
    images: list[Path] = []
    for p in sorted(output_root.rglob("*.jpg")):
        if p.stem.endswith("_viz"):
            continue
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        left = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        images.append(p)
    return images


def _page_is_flagged(aligned_path: Path) -> bool:
    """True if an existing aligned JSON marks the page as needing review.

    The older, broader selection (``--by-flag``). ``possible_column_merge`` is
    noisy (Surya over-segmentation trips it) and ``needs_review`` pages are often
    un-segmentable, so :func:`_page_genuine_merge_count` is the default instead.
    """
    if not aligned_path.exists():
        return False
    try:
        data = json.loads(aligned_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return bool(data.get("needs_review") or data.get("possible_column_merge"))


def _page_genuine_merge_count(aligned_path: Path) -> int:
    """Count aligned lines that look like two concatenated entries (genuine merge).

    Reuses ``tools.detect_column_merges._looks_merged`` (two street addresses on
    one line) — the reliable line-crop trigger. Returns 0 for missing/short pages.
    """
    if not aligned_path.exists():
        return 0
    try:
        data = json.loads(aligned_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    lines = [l.get("gemini_text", "") for l in (data.get("lines") or [])]
    if len(lines) < 8:
        return 0
    median_len = int(statistics.median([len(t) for t in lines]) or 1)
    return sum(1 for t in lines if _looks_merged(t, median_len))


def _crop_to_jpeg(page: "Image.Image", bbox: list[int]) -> "bytes | None":
    """Crop a padded line region from the page and encode it as JPEG bytes.

    Returns None when the (padded, clamped) crop is degenerate.
    """
    w, h = page.size
    x1 = max(0, bbox[0] - _CROP_PAD_X)
    y1 = max(0, bbox[1] - _CROP_PAD_Y)
    x2 = min(w, bbox[2] + _CROP_PAD_X)
    y2 = min(h, bbox[3] + _CROP_PAD_Y)
    if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
        return None
    crop = page.crop((x1, y1, x2, y2))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _transcribe_crop(
    client,
    jpeg_bytes: bytes,
    label: str,
    model: str,
    service_tier: "str | None",
) -> str:
    """Send one line crop to Gemini and return the transcribed text."""
    response = generate_with_retry(
        client,
        model=model,
        config=GenerateContentConfig(
            system_instruction=_LINE_PROMPT,
            temperature=0.0,
            thinking_config=ThinkingConfig(thinking_budget=0),
            http_options=flex_http_options(service_tier),
        ),
        contents=[Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")],
        label=label,
        log=_log,
    )
    return (response.text or "").strip()


def linecrop_image(
    client,
    image_path: Path,
    model: str,
    *,
    workers: int = 2,
    service_tier: "str | None" = None,
    replace: bool = False,
    min_surya_confidence: float = 0.35,
    dry_run: bool = False,
) -> tuple[str, int]:
    """Line-crop OCR one page. Returns (status, n_lines).

    status ∈ {'ok', 'skipped', 'missing', 'empty', 'dry-run'}.
    """
    slug = model_slug(model)
    stem = image_path.stem
    surya_path = image_path.parent / f"{stem}_surya.json"
    out_path = image_path.parent / (
        f"{stem}_{slug}_aligned.json" if replace
        else f"{stem}_{slug}_linecrop.json"
    )

    if not surya_path.exists():
        return "missing", 0
    if out_path.exists() and not replace and not dry_run:
        # Non-destructive default already produced — leave it; re-run with the
        # file removed to regenerate. (replace=True always overwrites.)
        return "skipped", 0

    # Same preprocessing as the NW path so bboxes/reading order match exactly.
    page_bbox, lines, _median = parse_surya(surya_path, min_confidence=min_surya_confidence)
    lines = _expand_multiline_surya_lines(lines)
    img_w, img_h = page_bbox[2], page_bbox[3]
    lines = filter_short_lines(lines)
    lines = filter_margin_lines(lines, img_w)
    lines = sort_by_reading_order(lines, img_w)

    if not lines:
        return "empty", 0
    if dry_run:
        return "dry-run", len(lines)

    fragment, canvas_meta = resolve_fragment_fn(image_path, img_w, img_h)

    with Image.open(image_path) as im:
        page = im.convert("RGB")

    # Crop every line up front (cheap), then transcribe crops in parallel while
    # preserving reading order. Keeping concurrency low (default 2) follows the
    # HTR-ALTO observation that vision APIs throttle sustained parallelism by
    # stalling rather than returning 429.
    crops: list[bytes | None] = [_crop_to_jpeg(page, ln["bbox"]) for ln in lines]

    texts: list[str] = [""] * len(lines)

    def _work(i: int) -> None:
        jpeg = crops[i]
        if jpeg is None:
            texts[i] = "[blank]"
            return
        texts[i] = _transcribe_crop(
            client, jpeg, f"{image_path.name}#L{i}", model, service_tier
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_work, range(len(lines))))

    result_lines: list[dict] = []
    for ln, text in zip(lines, texts):
        bbox = list(ln["bbox"])
        entry: dict = {
            "bbox": bbox,
            "canvas_fragment": fragment(bbox),
            "confidence": "crop",
            "gemini_text": text,
            "surya_text": ln.get("text", ""),
        }
        sc = ln.get("surya_confidence")
        if sc is not None:
            entry["surya_confidence"] = sc
        result_lines.append(entry)

    result = {
        "image": image_path.name,
        "model": model,
        "method": "linecrop",
        "canvas_uri": canvas_meta["canvas_uri"],
        "canvas_width": canvas_meta["canvas_width"],
        "canvas_height": canvas_meta["canvas_height"],
        "coords_from_fallback": canvas_meta["coords_from_fallback"],
        "lines": result_lines,
        "unmatched_gemini": [],
    }
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return "ok", len(result_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alignment-free OCR fallback: transcribe each Surya line crop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output_dir", help="Root images directory (e.g. output/greenbooks)")
    parser.add_argument(
        "--model", "-m", default=None, metavar="MODEL",
        help=f"Gemini model (default: auto-detected, else {DEFAULT_OCR_MODEL}).",
    )
    parser.add_argument(
        "--page", action="append", default=[], metavar="STEM",
        help="Process only the page(s) whose filename stem contains this string "
             "(repeatable). Overrides flag-based selection.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process every page, not just selected ones.",
    )
    parser.add_argument(
        "--by-flag", dest="by_flag", action="store_true",
        help="Select by the aligned JSON's needs_review / possible_column_merge "
             "flags instead of the default genuine-merge heuristic (noisier).",
    )
    parser.add_argument(
        "--min-merged", dest="min_merged", type=int, default=2, metavar="N",
        help="Default selection: process pages with at least N aligned lines that "
             "look like two concatenated entries (default: 2).",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Overwrite the canonical {stem}_{slug}_aligned.json instead of "
             "writing a separate _linecrop.json.",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=2, metavar="N",
        help="Parallel line-crop requests per page (default: 2 — vision APIs "
             "throttle higher sustained parallelism by stalling).",
    )
    parser.add_argument(
        "--flex", action="store_true",
        help="Use Flex inference (service_tier='flex'): ~50%% cheaper, slower.",
    )
    parser.add_argument(
        "--min-surya-confidence", type=float, default=0.35, metavar="THRESHOLD",
        help="Skip Surya lines below this detection confidence (default: 0.35).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the pages (and line counts) that would be processed; no API calls.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    if not output_root.exists():
        print(f"Error: directory not found: {output_root}", file=sys.stderr)
        sys.exit(1)

    model = args.model or get_ocr_model(output_root) or discover_ocr_slug(output_root) or DEFAULT_OCR_MODEL
    slug = model_slug(model)
    print(f"  OCR model: {model}", file=sys.stderr)

    images = _discover_images(output_root)
    if not images:
        print(f"No .jpg files found under {output_root}", file=sys.stderr)
        sys.exit(0)

    # Selection: explicit --page > --all > --by-flag > genuine-merge (default).
    if args.page:
        wanted = list(args.page)
        images = [p for p in images if any(w in p.stem for w in wanted)]
        sel_desc = f"matching --page {wanted}"
    elif args.all:
        sel_desc = "all pages"
    elif args.by_flag:
        images = [
            p for p in images
            if _page_is_flagged(p.parent / f"{p.stem}_{slug}_aligned.json")
        ]
        sel_desc = "flagged pages (needs_review / possible_column_merge)"
    else:
        images = [
            p for p in images
            if _page_genuine_merge_count(p.parent / f"{p.stem}_{slug}_aligned.json") >= args.min_merged
        ]
        sel_desc = f"genuine-merge pages (≥ {args.min_merged} two-address lines)"

    if not images:
        print(f"No pages selected ({sel_desc}).", file=sys.stderr)
        sys.exit(0)

    print(f"Selected {len(images)} page(s): {sel_desc}", file=sys.stderr)
    if args.dry_run:
        print("Dry run — no API calls. Pages and Surya line counts:", file=sys.stderr)

    service_tier = "flex" if args.flex else None
    client = None if args.dry_run else get_client()

    counts: dict[str, int] = {}
    total_lines = 0
    for i, img in enumerate(images, 1):
        try:
            status, n = linecrop_image(
                client, img, model,
                workers=args.workers,
                service_tier=service_tier,
                replace=args.replace,
                min_surya_confidence=args.min_surya_confidence,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            status, n = "failed", 0
            _log(f"  Error on {img.name}: {exc}")
        counts[status] = counts.get(status, 0) + 1
        total_lines += n
        _log(f"[{i:04d}/{len(images)}] {status:8s} {n:4d} lines  {img.name}")

    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"\nDone. {summary}. {total_lines} line(s) transcribed.", file=sys.stderr)


if __name__ == "__main__":
    main()
