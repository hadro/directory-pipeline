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
    python run_gemini_ocr.py output/travelguide --model gemini-2.5-flash
    python run_gemini_ocr.py output/travelguide --quiet
    python run_gemini_ocr.py output/travelguide --flex                      # ~50% cheaper, 1-15 min/req
    python run_gemini_ocr.py output/travelguide --model gemini-3.1-flash-lite --flex
"""

import argparse
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()
from google.genai.types import FinishReason, GenerateContentConfig, MediaResolution, Part, ThinkingConfig

from utils.gemini import flex_http_options, generate_with_retry, get_client
from utils.models import DEFAULT_OCR_MODEL, FALLBACK_MODEL, model_slug

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "ocr_prompt.md"

# Bare-minimum prompt used as a last resort before model escalation.
# Deliberately short — complex prompts can themselves trigger recitation on hard pages.
_SIMPLE_FALLBACK_PROMPT = (
    "Transcribe all text on this page exactly as it appears. "
    "Return plain text only. No commentary, no formatting."
)

# Written to the .txt file when every strategy fails, so downstream stages
# can detect and report the gap rather than silently treating it as blank.
_OCR_FAILED_PLACEHOLDER = "[OCR FAILED: page could not be transcribed after all retries]"

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

_NO_LEADER_INSTRUCTION = (
    "\n\nCRITICAL: Do NOT reproduce dot leaders (sequences of dots, or '. . .' "
    "patterns used as visual spacing between a name and an address). Replace them "
    "with a single space or omit them entirely. For example, transcribe "
    "\"Smith Hotel. . . . . . . . . . 123 Main St\" as \"Smith Hotel 123 Main St\"."
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

    Search order:
    1. output_root itself (item directory passed directly)
    2. output_root.parent (slug-level directory, when called with item subdir)
    3. Any immediate subdirectory of output_root (item subdir when called with
       slug-level directory — covers the case where --select-pages and
       --generate-prompts wrote ocr_prompt.md into the item subdir but the
       OCR stage is invoked with the parent slug directory)
    """
    root = output_root.resolve()
    for candidate_dir in (root, root.parent):
        p = candidate_dir / "ocr_prompt.md"
        if p.exists():
            return p
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir():
            p = subdir / "ocr_prompt.md"
            if p.exists():
                return p
    return fallback


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


# Output quality thresholds
_REPETITION_THRESHOLD = 0.40   # fraction of lines that are identical → runaway loop
_REPETITION_MIN_REPEATS = 20   # minimum line count before repetition check applies
_REPETITION_MIN_LINE_LEN = 4   # ignore lines shorter than this when counting
_LEADER_RUNAWAY_MIN_RUN = 50   # consecutive identical non-word chars → dot-leader runaway
_LEADER_SPACED_MIN_RUN = 50   # spaced pattern like ". . . . ." repeated this many times
_LEADER_MAX_LINE_LEN = 500    # any line longer than this is a runaway (catches spaced variants)

# Temperatures to try in order when output quality checks fail
_RETRY_TEMPERATURES = [0.1, 0.3, 0.7]

# Regex: 200+ consecutive identical non-word, non-space characters (dots, dashes, etc.)
_LEADER_RE = re.compile(r"([^\w\s])\1{%d,}" % (_LEADER_RUNAWAY_MIN_RUN - 1))
# Regex: spaced dot-leader like ". . . . ." — non-word char + space repeated 50+ times
_LEADER_SPACED_RE = re.compile(r"([^\w\s] ){%d,}" % _LEADER_SPACED_MIN_RUN)
# Regex: collapse any run of 10+ dots (with optional spaces) to a canonical 5-dot leader
_LEADER_COLLAPSE_RE = re.compile(r"(\.\s*){10,}")


def _clean_output(text: str) -> str:
    """Apply post-processing cleanup to OCR text before saving.

    Collapses any run of 10+ dots (with optional spaces) down to a canonical
    5-dot leader, preserving any content that follows (e.g. 'r.' or 'r. m.').
    This handles pages where dot-leader runaways survive all temperature retries.
    """
    return _LEADER_COLLAPSE_RE.sub(".....", text)


def _output_issue(text: str) -> str:
    """Return a non-empty description if the text has a detectable quality problem, else ''.

    Checks for two failure modes:
      1. Repetition loop  — a single line makes up >= 40% of all lines (min 20 lines).
      2. Dot-leader runaway — any line contains 200+ consecutive identical non-word chars.
    """
    if not text:
        return ""

    # Dot-leader runaway — consecutive (.....) or spaced (. . . .) or just very long lines
    for line in text.splitlines():
        if _LEADER_RE.search(line):
            return f"leader runaway in line [{line[:50]}…]"
        if _LEADER_SPACED_RE.search(line):
            return f"spaced leader runaway in line [{line[:50]}…]"
        if len(line) > _LEADER_MAX_LINE_LEN:
            return f"line too long ({len(line)} chars): [{line[:50]}…]"

    # Repetition loop
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= _REPETITION_MIN_REPEATS:
        from collections import Counter
        counts = Counter(l for l in lines if len(l) >= _REPETITION_MIN_LINE_LEN)
        if counts:
            top_line, top_count = counts.most_common(1)[0]
            ratio = top_count / len(lines)
            if ratio >= _REPETITION_THRESHOLD:
                return f"repetition loop ({ratio:.0%} of lines are [{top_line[:40]}…])"

    return ""


def _call_gemini(
    client: genai.Client,
    img_bytes: bytes,
    image_name: str,
    model: str,
    system_prompt: str,
    media_resolution: "MediaResolution | None",
    temperature: float,
    service_tier: str | None = None,
):
    """Call Gemini with 429-retry and 503-retry logic. Returns the response object."""
    return generate_with_retry(
        client,
        model=model,
        config=GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            media_resolution=media_resolution,
            thinking_config=ThinkingConfig(thinking_budget=0),
            http_options=flex_http_options(service_tier),
        ),
        contents=[Part.from_bytes(data=img_bytes, mime_type="image/jpeg")],
        label=image_name,
        log=_log,
    )


def process_image(
    client: genai.Client,
    image_path: Path,
    model: str,
    system_prompt: str,
    media_resolution: "MediaResolution | None" = None,
    service_tier: str | None = None,
    fallback_model: str | None = None,
) -> tuple[str, bool | None]:
    """
    OCR one image via Gemini. Returns (status, success) where status is one of
    'skipped', 'ok', 'failed'.
    """
    txt_path = image_path.parent / f"{image_path.stem}_{model_slug(model)}.txt"
    if txt_path.exists():
        if txt_path.stat().st_size > 0:
            return "skipped", None
        # File is empty — previous run produced no output; delete and retry.
        _log(f"  Re-running (empty output file): {txt_path.name}")
        txt_path.unlink()

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = _call_gemini(client, img_bytes, image_path.name, model, system_prompt, media_resolution, temperature=0.0, service_tier=service_tier)

    text = response.text or ""

    # Determine whether this output needs a retry.
    # Empty text: retry only on RECITATION (STOP with no text = genuinely blank page).
    # Non-empty text: retry on repetition loop or dot-leader runaway.
    if not text:
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.finish_reason == FinishReason.RECITATION:
            needs_retry, reason = True, "RECITATION"
        else:
            needs_retry, reason = False, ""
            if candidate:
                _log(f"  finish_reason: {candidate.finish_reason}  [{image_path.name}]")
                if candidate.safety_ratings:
                    _log(f"  safety_ratings: {candidate.safety_ratings}  [{image_path.name}]")
    else:
        issue = _output_issue(text)
        needs_retry, reason = bool(issue), issue

    if needs_retry:
        best_text = text  # track cleanest output seen; start with original (may be empty)
        is_leader_issue = "leader" in reason or "line too long" in reason
        retry_prompt = (system_prompt + _NO_LEADER_INSTRUCTION) if is_leader_issue else system_prompt
        for temp in _RETRY_TEMPERATURES:
            _log(f"  {reason} — retrying at temperature={temp}: {image_path.name}")
            retry_response = _call_gemini(client, img_bytes, image_path.name, model, retry_prompt, media_resolution, temperature=temp)
            retry_text = retry_response.text or ""
            retry_issue = _output_issue(retry_text)
            if retry_text and not retry_issue:
                text = retry_text  # clean output — accept and stop retrying
                break
            if retry_text:
                _log(f"  temperature={temp} still problematic ({retry_issue}): {image_path.name}")
                best_text = retry_text  # imperfect but non-empty; keep as fallback
        else:
            # All temperature retries exhausted without a clean result.
            if best_text:
                text = best_text
            if is_leader_issue:
                text = _clean_output(text)

            # Only attempt deeper fallbacks when we still have no usable text.
            if not text or _output_issue(text):
                # Strategy 1: simple prompt — complex prompts can themselves
                # trigger recitation on visually dense pages.
                for temp in (0.3, 0.7):
                    _log(f"  Trying simple fallback prompt at temperature={temp}: {image_path.name}")
                    r = _call_gemini(
                        client, img_bytes, image_path.name, model,
                        _SIMPLE_FALLBACK_PROMPT, media_resolution, temperature=temp,
                    )
                    simple_text = r.text or ""
                    if simple_text and not _output_issue(simple_text):
                        text = simple_text
                        _log(f"  Simple prompt succeeded at temperature={temp}: {image_path.name}")
                        break
                    if simple_text and len(simple_text) > len(text):
                        text = simple_text  # imperfect but longer — keep as new best

            if not text or _output_issue(text):
                # Strategy 2: model escalation — larger model handles layout
                # complexity that causes recitation in smaller models.
                esc_model = fallback_model
                if esc_model and esc_model != model:
                    _log(f"  Escalating to {esc_model}: {image_path.name}")
                    r = _call_gemini(
                        client, img_bytes, image_path.name, esc_model,
                        _SIMPLE_FALLBACK_PROMPT, media_resolution, temperature=0.0,
                    )
                    esc_text = r.text or ""
                    if esc_text and not _output_issue(esc_text):
                        text = esc_text
                        _log(f"  Model escalation succeeded ({esc_model}): {image_path.name}")
                    elif esc_text:
                        text = esc_text
                        _log(f"  Model escalation: imperfect output ({esc_model}): {image_path.name}")

            if text:
                _log(f"  All retries exhausted — keeping best available output: {image_path.name}")
            else:
                text = _OCR_FAILED_PLACEHOLDER
                _log(f"  All retries exhausted — no output recovered; writing placeholder: {image_path.name}")

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
        default=DEFAULT_OCR_MODEL,
        help=f"Gemini model name (default: {DEFAULT_OCR_MODEL})",
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
        "--flex",
        action="store_true",
        help=(
            "Use Flex inference (service_tier='flex'): ~50%% cheaper, "
            "1–15 min latency per request, best-effort availability."
        ),
    )
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL,
        metavar="MODEL",
        help=(
            f"Model to escalate to when the primary model exhausts all retries "
            f"with no usable output (default: {FALLBACK_MODEL}). "
            "Pass an empty string to disable escalation."
        ),
    )
    parser.add_argument(
        "--sections",
        metavar="PATH",
        help=(
            "Path to sections.txt marking section boundaries. "
            "When provided, each page is OCR'd with the prompt for its section "
            "(ocr_prompt_{label}.md) falling back to ocr_prompt.md if not found."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = parser.parse_args()

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

    service_tier = "flex" if args.flex else None

    # --- Per-page prompt mapping (sections mode) ----------------------------
    img_prompts: dict[Path, str] = {}
    if args.sections:
        from utils.section_utils import load_sections, prompt_for_page

        sections_path = Path(args.sections)
        if not sections_path.exists():
            print(f"Error: sections file not found: {sections_path}", file=sys.stderr)
            sys.exit(1)
        all_names = [p.name for p in images]
        try:
            sections = load_sections(sections_path, all_names)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Resolve slug_dir — where section-specific prompt files live
        root = output_root.resolve()
        slug_dir = root
        for candidate in (root, root.parent):
            if (candidate / "ocr_prompt.md").exists() or any(candidate.glob("ocr_prompt_*.md")):
                slug_dir = candidate
                break

        prompt_cache: dict[str, str] = {}
        for img in images:
            ppath = prompt_for_page(img.name, sections, all_names, slug_dir, "ocr_prompt")
            key = str(ppath)
            if key not in prompt_cache:
                if ppath.exists():
                    text = ppath.read_text(encoding="utf-8")
                else:
                    text = system_prompt  # fall through to generic
                if args.expand_dittos:
                    text = text.rstrip() + _DITTO_INSTRUCTION
                prompt_cache[key] = text
            img_prompts[img] = prompt_cache[key]

        if not args.quiet:
            unique_prompts = len(set(img_prompts.values()))
            print(
                f"Sections mode: {len(sections)} section(s), {unique_prompts} distinct prompt(s)",
                file=sys.stderr,
            )
    else:
        img_prompts = {img: system_prompt for img in images}

    total = len(images)
    if not args.quiet:
        tier_note = " [flex inference]" if service_tier else ""
        print(
            f"Processing {total} image(s) with {args.workers} worker(s) using {args.model}{tier_note}…",
            file=sys.stderr,
        )

    client = get_client()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    completed = 0

    fallback_model = args.fallback_model.strip() or None
    if fallback_model and not args.quiet:
        print(f"Fallback model (on exhaustion): {fallback_model}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_image, client, img, args.model, img_prompts[img], media_resolution, service_tier, fallback_model): img
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
                txt_name = f"{image_path.stem}_{model_slug(args.model)}.txt"
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
