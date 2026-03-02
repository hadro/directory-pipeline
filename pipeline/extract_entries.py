#!/usr/bin/env python3
"""Extract structured directory entries from aligned OCR output using Gemini NER.

Reads *_{model}_aligned.json files in sorted (page) order and calls the Gemini
API to identify and extract structured entries: establishment name, address,
city, state, category, advertisement flag, and phone number.

Cross-page context (last active state / city / category heading) is carried
forward between pages so entries at the top of a page that inherit context
from the prior page are correctly attributed.

Input (per page):
  {stem}_{model_slug}_aligned.json   — Gemini-corrected lines with bounding boxes

Output (per page):
  {stem}_{model_slug}_entries.json   — extracted entries with canvas fragments

Output (per item directory):
  entries_{model_slug}.csv           — flat CSV of all entries across all pages

Usage
-----
    python extract_entries.py images/greenbooks/feb978b0
    python extract_entries.py images/greenbooks/feb978b0 --mode multimodal
    python extract_entries.py images/greenbooks/ --force
    python extract_entries.py images/greenbooks/feb978b0 --dry-run
"""

import argparse
import csv
import difflib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()
from google.genai.types import GenerateContentConfig, Part

DEFAULT_MODEL = "gemini-2.0-flash"
# Fallback used when the primary model appears to have hit its output token limit.
# gemini-2.0-flash caps output at ~8 k tokens; gemini-2.5-flash allows up to 65 k.
FALLBACK_MODEL = "gemini-2.5-flash"
NER_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "ner_prompt.md"

ENTRY_FIELDS = [
    "image", "page",
    "establishment_name", "raw_address", "address_type",
    "city", "state", "category",
    "is_advertisement", "phone", "notes",
    "line_text", "canvas_fragment",
]

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def model_slug(model: str) -> str:
    return model.replace("/", "_")


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

def _call_gemini(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_text: str,
    image_path: Path | None = None,
) -> str:
    """Call Gemini with optional image. Returns raw response text."""
    parts: list = []
    if image_path is not None:
        parts.append(Part.from_bytes(data=image_path.read_bytes(), mime_type="image/jpeg"))
    parts.append(Part.from_text(text=user_text))

    max_retries = 5
    delay = 10
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    max_output_tokens=65536,
                ),
                contents=parts,
            )
            return response.text or ""
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                _log(f"  Rate limited — retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    return ""


def _strip_fence(text: str) -> str:
    """Strip markdown code fences from a model response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


def _parse_json_response(text: str) -> dict | None:
    """Parse a JSON object from a model response, stripping any markdown fences."""
    text = _strip_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract a JSON object if there's surrounding prose
        start = text.find("{")
        end = text.rfind("}") + 1
        if 0 <= start < end:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _recover_partial_json(text: str) -> dict | None:
    """Salvage complete entries from a response truncated by the model's token limit.

    When a page has many entries the model's output can be cut off mid-entry.
    This function uses raw_decode to walk the entries array and collect every
    complete entry that was emitted before the truncation point.

    Returns a dict with "entries" and "page_context" keys (like a normal
    response), or None if no complete entries could be recovered.
    """
    text = _strip_fence(text)
    decoder = json.JSONDecoder()

    # Recover page_context (appears before entries; it's a shallow object)
    page_context: dict = {}
    ctx_pos = text.find('"page_context"')
    if ctx_pos != -1:
        colon = text.find(":", ctx_pos + len('"page_context"'))
        brace = text.find("{", colon) if colon != -1 else -1
        if brace != -1:
            try:
                page_context, _ = decoder.raw_decode(text, brace)
            except json.JSONDecodeError:
                pass

    # Find the entries array
    entries_pos = text.find('"entries"')
    if entries_pos == -1:
        return None
    array_open = text.find("[", entries_pos)
    if array_open == -1:
        return None

    # Walk the array, parsing one complete object at a time
    entries: list[dict] = []
    i = array_open + 1
    while i < len(text):
        ch = text[i]
        if ch in " \t\n\r,":
            i += 1
            continue
        if ch == "]":
            break  # Normal end of array (response was not truncated)
        if ch == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                entries.append(obj)
                i = end
            except json.JSONDecodeError:
                break  # Truncation point — stop here
        else:
            break

    if not entries:
        return None
    return {"entries": entries, "page_context": page_context}


# ---------------------------------------------------------------------------
# Page text and context
# ---------------------------------------------------------------------------

def _normalize_line(text: str) -> str:
    """Collapse dot-leader runs and long whitespace to a single separator.

    Directory pages use long runs of dots or spaces between a name and an
    address (e.g. "Blue Moon Restaurant ........... 1811 Conover Street").
    These inflate output token counts dramatically when the model echoes them
    back verbatim in line_text fields.  Replace any run of 3+ dots or 3+
    spaces (after stripping leading/trailing whitespace) with a single tab,
    which is compact and still separates the two fields visually.
    """
    text = text.strip()
    # Collapse runs of 3+ dots (possibly mixed with spaces) to a tab
    text = re.sub(r'[.\s]{3,}', '\t', text)
    # Collapse any remaining runs of 2+ spaces to a single space
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _page_text_from_aligned(aligned: dict) -> str:
    """Build a plain-text page transcript from an aligned JSON, in line order."""
    lines = [
        _normalize_line(ln.get("gemini_text", ""))
        for ln in aligned.get("lines", [])
        if ln.get("gemini_text", "").strip()
    ]
    # Append any lines the aligner couldn't match to Tesseract (still good text)
    lines += [_normalize_line(t) for t in aligned.get("unmatched_gemini", []) if t.strip()]
    return "\n".join(lines)


def _build_user_message(page_text: str, prior_context: dict) -> str:
    state = prior_context.get("state") or "unknown"
    city = prior_context.get("city") or "unknown"
    category = prior_context.get("category") or "unknown"
    return (
        f"## Prior page context\n"
        f"State: {state}\n"
        f"City: {city}\n"
        f"Category: {category}\n\n"
        f"## Page text\n"
        f"{page_text}\n\n"
        f"Return the JSON extraction."
    )


def _img_path_from_aligned(aligned_path: Path, slug: str) -> Path | None:
    """Reconstruct the source image path from an aligned JSON filename."""
    # Pattern: {stem}_{slug}_aligned.json  →  {stem}.jpg
    suffix = f"_{slug}_aligned.json"
    name = aligned_path.name
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    candidate = aligned_path.parent / f"{stem}.jpg"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Canvas fragment linking
# ---------------------------------------------------------------------------

def _find_fragment(line_text: str, aligned_lines: list[dict]) -> str | None:
    """Find the canvas_fragment for an entry by matching its line_text to aligned lines.

    Tries three strategies in order:
    1. Exact match against gemini_text
    2. Substring match — handles the common Green Book format where the aligned
       line includes the city name and dot leaders before the establishment name,
       e.g. "LAKE GEORGE....Woodbine Cottage, 75 Dieskau Street"
    3. Fuzzy match as a last resort
    """
    if not line_text or not aligned_lines:
        return None
    # Exact match
    for ln in aligned_lines:
        if ln.get("gemini_text", "") == line_text:
            return ln.get("canvas_fragment")
    # Substring match
    lt_lower = line_text.lower()
    for ln in aligned_lines:
        if lt_lower in ln.get("gemini_text", "").lower():
            return ln.get("canvas_fragment")
    # Fuzzy match (handles minor whitespace / OCR differences)
    candidates = [ln.get("gemini_text", "") for ln in aligned_lines]
    matches = difflib.get_close_matches(line_text, candidates, n=1, cutoff=0.6)
    if matches:
        for ln in aligned_lines:
            if ln.get("gemini_text", "") == matches[0]:
                return ln.get("canvas_fragment")
    return None


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def process_page(
    client: genai.Client,
    aligned_path: Path,
    model: str,
    system_prompt: str,
    prior_context: dict,
    mode: str,
    force: bool,
    dry_run: bool,
    aligned_model: str | None = None,
    fallback_model: str | None = FALLBACK_MODEL,
) -> dict:
    """
    Extract entries from one aligned JSON page.

    aligned_model: if set, the model whose slug appears in aligned_path's name.
    The output file is always tagged with the NER model slug.

    Returns:
      {
        "entries": [...],
        "page_context": {...},
        "status": "ok" | "partial_recovery" | "skipped" | "empty" | "error:<msg>" | "parse_error",
      }
    """
    slug = model_slug(model)
    aligned_slug = model_slug(aligned_model) if aligned_model else slug
    suffix = f"_{aligned_slug}_aligned.json"
    stem = aligned_path.name[: -len(suffix)]
    out_path = aligned_path.parent / f"{stem}_{slug}_entries.json"

    # Return cached result if it exists and --force is not set
    if not force and not dry_run and out_path.exists() and out_path.stat().st_size > 0:
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            return {
                "entries": data.get("entries", []),
                "page_context": data.get("page_context", prior_context),
                "status": "skipped",
            }
        except Exception:
            pass  # corrupt file — re-run

    # Load aligned JSON
    try:
        aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"entries": [], "page_context": prior_context, "status": f"error:{exc}"}

    page_text = _page_text_from_aligned(aligned)
    if not page_text.strip():
        return {"entries": [], "page_context": prior_context, "status": "empty"}

    if dry_run:
        return {
            "entries": [],
            "page_context": prior_context,
            "status": f"dry_run ({len(page_text.splitlines())} lines)",
        }

    user_msg = _build_user_message(page_text, prior_context)

    # Resolve image path for multimodal mode
    image_path: Path | None = None
    if mode == "multimodal":
        image_path = _img_path_from_aligned(aligned_path, slug)
        if image_path is None:
            _log(f"    Warning: image not found for {aligned_path.name}; falling back to text-only")

    error_path = aligned_path.parent / f"{stem}_{slug}_entries_error.txt"

    # Call primary model once
    try:
        last_raw = _call_gemini(client, model, system_prompt, user_msg, image_path)
    except Exception as exc:
        return {"entries": [], "page_context": prior_context, "status": f"api_error:{exc}"}
    result = _parse_json_response(last_raw)

    # If the primary model failed, try once with the fallback model.
    # gemini-2.0-flash caps output at ~8 k tokens; the fallback (gemini-2.5-flash)
    # allows up to 65 k, which handles dense pages with 100+ entries.
    if result is None and fallback_model and fallback_model != model:
        _log(
            f"    [{aligned_path.name}] primary model exhausted — retrying once"
            f" with fallback model {fallback_model}"
        )
        try:
            last_raw = _call_gemini(client, fallback_model, system_prompt, user_msg, image_path)
            result = _parse_json_response(last_raw)
        except Exception as exc:
            _log(f"    [{aligned_path.name}] fallback model error: {exc}")

    partial = False
    if result is None:
        # Full parse failed even after retries and fallback. The response is
        # likely truncated. Try to salvage complete entries.
        result = _recover_partial_json(last_raw)
        if result is None:
            # Nothing recoverable — save for debugging and skip this page
            error_path.write_text(last_raw, encoding="utf-8")
            return {"entries": [], "page_context": prior_context, "status": "parse_error"}
        partial = True
        _log(
            f"    [{aligned_path.name}] WARNING: response was truncated; "
            f"recovered {len(result.get('entries', []))} entries (may be incomplete)"
        )

    # Clean up any leftover error file from a prior run
    if error_path.exists():
        error_path.unlink()

    entries = result.get("entries", [])
    new_context = result.get("page_context") or prior_context

    # Link canvas fragments from the aligned JSON.
    # Use line_text if present (legacy), otherwise match on establishment_name.
    aligned_lines = aligned.get("lines", [])
    canvas_uri = aligned.get("canvas_uri", "")
    for entry in entries:
        lt = entry.get("line_text", "") or entry.get("establishment_name", "")
        cf = _find_fragment(lt, aligned_lines)
        entry["canvas_fragment"] = cf or canvas_uri  # fall back to full canvas
        entry["image"] = aligned.get("image", "")

    # Write per-page output
    output = {
        "image": aligned.get("image", ""),
        "model": model,
        "mode": mode,
        "prior_context": prior_context,
        "page_context": new_context,
        "entries": entries,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"entries": entries, "page_context": new_context, "status": "partial_recovery" if partial else "ok"}


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

def process_item(
    client: genai.Client,
    item_dir: Path,
    model: str,
    system_prompt: str,
    mode: str,
    force: bool,
    dry_run: bool,
    quiet: bool,
    aligned_model: str | None = None,
    fallback_model: str | None = FALLBACK_MODEL,
) -> list[dict]:
    """
    Process all aligned JSON pages in an item directory in page order.
    Returns all extracted entries across all pages.

    aligned_model: if set, read *_{aligned_slug}_aligned.json files but write
    output tagged with the NER model slug.  Useful when OCR was run with one
    model and NER is run with another.
    fallback_model: if set, used when the primary model appears to have hit its
    output token limit (all retries produce unparseable responses).
    """
    slug = model_slug(model)
    aligned_slug = model_slug(aligned_model) if aligned_model else slug
    aligned_files = sorted(item_dir.glob(f"*_{aligned_slug}_aligned.json"))
    if not aligned_files:
        return []

    all_entries: list[dict] = []
    context: dict = {"state": "", "city": "", "category": ""}

    # Resume: load persisted context from a previous run
    context_file = item_dir / f"extraction_context_{slug}.json"
    if not force and context_file.exists():
        try:
            context = json.loads(context_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not quiet:
        _log(f"  {item_dir.name}: {len(aligned_files)} page(s), mode={mode}")

    aligned_suffix = f"_{aligned_slug}_aligned.json"
    for i, aligned_path in enumerate(aligned_files, 1):
        # Auto-retry pages whose last run produced a parse error, even without --force
        stem = aligned_path.name[: -len(aligned_suffix)]
        error_path = aligned_path.parent / f"{stem}_{slug}_entries_error.txt"
        page_force = force
        if not force and error_path.exists():
            _log(f"    [{aligned_path.name}] prior parse-error file detected — forcing retry")
            page_force = True

        result = process_page(
            client, aligned_path, model, system_prompt,
            context, mode, page_force, dry_run, aligned_model,
            fallback_model=fallback_model,
        )
        status = result["status"]
        n = len(result["entries"])
        ctx = result.get("page_context", context)

        if not quiet:
            ctx_str = f"{ctx.get('state','?')} > {ctx.get('city','?')} > {ctx.get('category','?')}"
            _log(f"  [{i:03d}/{len(aligned_files)}] {aligned_path.name}  {status}  {n} entries  [{ctx_str}]")

        context = ctx
        all_entries.extend(result.get("entries", []))

        # Persist context so runs can be resumed
        if not dry_run:
            context_file.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")

    return all_entries


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(entries: list[dict], out_path: Path) -> None:
    """Write a flat CSV of all entries."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTRY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for e in entries:
            writer.writerow({k: e.get(k, "") for k in ENTRY_FIELDS})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured entries from aligned OCR using Gemini NER.",
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
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Gemini model for NER (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--aligned-model",
        default=None,
        metavar="MODEL",
        help=(
            "Model whose slug appears in the *_aligned.json filenames "
            "(default: same as --model). Use this when OCR was run with a "
            "different model than the one doing NER, e.g. "
            "--model gemini-2.5-flash --aligned-model gemini-2.0-flash"
        ),
    )
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL,
        metavar="MODEL",
        help=(
            "Model to try when the primary model fails all parse attempts "
            "(typically because it hit its output token limit on dense pages). "
            f"Default: {FALLBACK_MODEL}. Pass empty string to disable."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["text-only", "multimodal"],
        default="text-only",
        help=(
            "text-only: send corrected text only; "
            "multimodal: also send the page image (default: text-only)"
        ),
    )
    parser.add_argument(
        "--prompt", "-p",
        default=str(NER_PROMPT_FILE),
        metavar="FILE",
        help=f"NER system prompt file (default: {NER_PROMPT_FILE})",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-run even if output files already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files and show what would be run, without calling the API",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-page progress output",
    )
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # NER prompt
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"Error: NER prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    aligned_model = args.aligned_model
    slug = model_slug(args.model)
    aligned_slug = model_slug(aligned_model) if aligned_model else slug
    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key) if not args.dry_run else None  # type: ignore[assignment]

    # Discover item directories: either the given dir itself (if it has aligned
    # JSONs directly) or its immediate subdirectories.
    slug_pattern = f"*_{aligned_slug}_aligned.json"
    direct = list(images_root.glob(slug_pattern))
    if direct:
        item_dirs = [images_root]
    else:
        item_dirs = [
            d for d in sorted(images_root.iterdir())
            if d.is_dir() and list(d.glob(slug_pattern))
        ]

    if not item_dirs:
        print(f"No *_{aligned_slug}_aligned.json files found under {images_root}", file=sys.stderr)
        sys.exit(1)

    fallback_model = args.fallback_model or None  # empty string → disabled

    print(
        f"\nExtracting entries: {len(item_dirs)} item dir(s), model={args.model}"
        + (f" (aligned by {aligned_model})" if aligned_model else "")
        + (f", fallback={fallback_model}" if fallback_model and fallback_model != args.model else "")
        + f", mode={args.mode}"
        + (" [DRY RUN]" if args.dry_run else ""),
        file=sys.stderr,
    )

    for item_dir in item_dirs:
        entries = process_item(
            client, item_dir, args.model, system_prompt,
            args.mode, args.force, args.dry_run, args.quiet, aligned_model,
            fallback_model=fallback_model,
        )
        if not args.dry_run:
            csv_path = item_dir / f"entries_{slug}.csv"
            write_csv(entries, csv_path)
            print(
                f"  → {len(entries)} entries total → {csv_path}",
                file=sys.stderr,
            )

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
