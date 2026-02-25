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
    python extract_entries.py images/greenbooks/feb978b0 --model gemini-2.0-flash
    python extract_entries.py images/greenbooks/feb978b0 --model gemini-2.0-flash \\
        --mode multimodal
    python extract_entries.py images/greenbooks/ --model gemini-2.0-flash --force
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

from google import genai
from google.genai.types import GenerateContentConfig, Part

DEFAULT_MODEL = "gemini-2.0-flash"
NER_PROMPT_FILE = Path(__file__).parent / "ner_prompt.md"

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
                    max_output_tokens=8192,
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


def _parse_json_response(text: str) -> dict | None:
    """Parse a JSON object from a model response, stripping any markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Strip opening fence (possibly ```json) and closing fence
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
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
) -> dict:
    """
    Extract entries from one aligned JSON page.

    Returns:
      {
        "entries": [...],
        "page_context": {...},
        "status": "ok" | "skipped" | "empty" | "error:<msg>" | "parse_error",
      }
    """
    slug = model_slug(model)
    suffix = f"_{slug}_aligned.json"
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

    # Call API
    try:
        raw = _call_gemini(client, model, system_prompt, user_msg, image_path)
    except Exception as exc:
        return {"entries": [], "page_context": prior_context, "status": f"api_error:{exc}"}

    result = _parse_json_response(raw)
    if result is None:
        # Save raw response for debugging
        (aligned_path.parent / f"{stem}_{slug}_entries_error.txt").write_text(
            raw, encoding="utf-8"
        )
        return {"entries": [], "page_context": prior_context, "status": "parse_error"}

    entries = result.get("entries", [])
    new_context = result.get("page_context", prior_context)

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

    return {"entries": entries, "page_context": new_context, "status": "ok"}


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
) -> list[dict]:
    """
    Process all aligned JSON pages in an item directory in page order.
    Returns all extracted entries across all pages.
    """
    slug = model_slug(model)
    aligned_files = sorted(item_dir.glob(f"*_{slug}_aligned.json"))
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

    for i, aligned_path in enumerate(aligned_files, 1):
        result = process_page(
            client, aligned_path, model, system_prompt,
            context, mode, force, dry_run,
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

    slug = model_slug(args.model)
    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key) if not args.dry_run else None  # type: ignore[assignment]

    # Discover item directories: either the given dir itself (if it has aligned
    # JSONs directly) or its immediate subdirectories.
    slug_pattern = f"*_{slug}_aligned.json"
    direct = list(images_root.glob(slug_pattern))
    if direct:
        item_dirs = [images_root]
    else:
        item_dirs = [
            d for d in sorted(images_root.iterdir())
            if d.is_dir() and list(d.glob(slug_pattern))
        ]

    if not item_dirs:
        print(f"No *_{slug}_aligned.json files found under {images_root}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nExtracting entries: {len(item_dirs)} item dir(s), model={args.model}, mode={args.mode}"
        f"{' [DRY RUN]' if args.dry_run else ''}",
        file=sys.stderr,
    )

    for item_dir in item_dirs:
        entries = process_item(
            client, item_dir, args.model, system_prompt,
            args.mode, args.force, args.dry_run, args.quiet,
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
