#!/usr/bin/env python3
"""Generate volume-specific OCR and NER prompts by having Gemini analyze sample pages.

Makes two Gemini calls with the same sample images — one to write a tailored OCR
transcription system prompt, one to write a tailored NER entry-extraction system
prompt — and saves both as Markdown files in prompts/.

The generated prompts can be passed to run_gemini_ocr.py (--prompt-file) and
extract_entries.py (--prompt-file) respectively, or used as starting points for
manual editing.

Usage
-----
    # Generate both prompts from a selection file:
    python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \\
        --selection path/to/selection.txt

    # Generate both from an explicit page list:
    python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \\
        --pages 0005_58019060.jpg 0012_58019067.jpg 0023_58019078.jpg 0041_58019096.jpg

    # OCR prompt only:
    python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \\
        --selection selection.txt --ocr-only

    # NER prompt only:
    python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \\
        --selection selection.txt --ner-only

    # Custom output paths (default is {slug_dir}/ocr_prompt.md and ner_prompt.md):
    python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \\
        --selection selection.txt \\
        --ocr-out output/the_travelers_guide_e088efa0/ocr_prompt.md \\
        --ner-out output/the_travelers_guide_e088efa0/ner_prompt.md

    # Use a specific model:
    python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \\
        --selection selection.txt --model gemini-3-flash-preview
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "gemini-3-flash-preview"

# Locate sibling prompts/ directory (two levels up from this file)
_PIPELINE_ROOT = Path(__file__).parent.parent
_DEFAULT_OCR_PROMPT = _PIPELINE_ROOT / "prompts" / "ocr_prompt.md"
_DEFAULT_NER_PROMPT = _PIPELINE_ROOT / "prompts" / "ner_prompt.md"


# ---------------------------------------------------------------------------
# Meta-prompts
# ---------------------------------------------------------------------------

_OCR_META_PROMPT = """\
You are calibrating an OCR transcription pipeline for a digitized historical print directory.

I am sharing {n} sample page images from this collection. Study them carefully — \
pay attention to the typography, layout, page hierarchy, recurring abbreviations, \
and any structural conventions — then write a tailored OCR system prompt that will \
help a large language model transcribe pages from this specific volume accurately.

Your prompt should describe and instruct on all of the following that apply:

1. **Publication identity** — Name, publication era, geographic scope, and the \
community or audience it serves. One or two sentences that orient the model.

2. **Page structure and hierarchy** — The nesting of headings, sub-headings, \
categories, and entries. Use concrete examples from what you see (e.g. \
STATE → CITY → CATEGORY → one-line entries, or CITY → street listing with \
occupants, etc.).

3. **Column layout** — Typical number of columns. Note exceptions (e.g., \
full-width headings, index pages, advertisements).

4. **Typographic and formatting conventions** — Common abbreviations \
(e.g. St., Ave., Cor., Bds., N.W.), punctuation styles, whether entries \
span multiple lines, use of parentheses, em-dashes, etc.

5. **Special content** — Advertisements, illustrations, index pages, maps, \
or other non-directory content, and how to handle each.

6. **Transcription instructions** — Reading order (column-by-column vs. \
row-by-row), how to handle degraded or blurred text, what margin content \
to include or exclude, whether to preserve blank lines between sections.

7. **Output format** — Plain text only, one printed line per output line, \
no markdown, no commentary. Restate this clearly in the prompt.

Write the prompt in the second person ("You are transcribing…"). \
The output should be ready to paste directly into a pipeline as a system prompt — \
no preamble, no section headers of your own, no explanation outside the prompt text itself.
"""

_NER_META_PROMPT = """\
You are calibrating a named entity recognition pipeline for a digitized historical print directory.

I am sharing {n} sample page images from this collection. Study them carefully — \
pay attention to the heading hierarchy, entry format, category types present, address \
conventions, continuation markers, and any unusual structural features.

Then write a tailored NER system prompt that will help a language model extract \
structured entries from the transcribed text of pages from this volume.

The pipeline gives the NER model the OCR transcription of one page at a time, along \
with the last known state/city/category context carried forward from the prior page. \
The model returns a JSON object with one entry per business or listing.

Here is the existing NER prompt template used for the Negro Motorist Green Book. \
Use it as your structural guide — preserve the JSON schema and output format exactly, \
but adapt all descriptions, examples, category lists, and rules to match what you \
observe in the sample pages:

--- BEGIN TEMPLATE ---
{ner_template}
--- END TEMPLATE ---

Adaptation guidelines:
- If this publication does not have a state-level heading hierarchy, remove STATE \
  from the context tracking and entry schema description accordingly.
- Update the category mapping to reflect the heading types actually present in \
  this volume. Map each to the closest canonical value from this fixed set \
  (required for downstream compatibility): \
  formal_accommodations, informal_accommodations, eating_drinking, \
  barber_beauty, service_station, other.
- If entries span multiple lines or have a format different from a simple \
  "Name, Address" one-liner, describe that format clearly.
- Replace all Green Book–specific examples with terminology and formatting \
  drawn from this specific volume.
- Preserve the JSON field names exactly (establishment_name, raw_address, \
  address_type, city, state, category, is_advertisement, phone, notes) \
  for compatibility with downstream scripts.
- Update the continuation marker description if this volume uses different \
  conventions (e.g. "—cont." instead of "Contd.").

Write the prompt in the second person ("You are a structured data extractor…"). \
Output only the adapted prompt — no preamble, no explanation, no section headers \
of your own.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_item_dir(root: Path, filenames: list[str]) -> Path:
    """Given a slug-or-item directory and a list of image filenames,
    return the directory that contains those files."""
    if any((root / f).exists() for f in filenames):
        return root
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and any((sub / f).exists() for f in filenames):
            return sub
    raise FileNotFoundError(
        f"Could not find image files {filenames[:2]}… under {root}"
    )


def _load_selection(selection_path: Path) -> list[str]:
    """Read a selection.txt — one filename per non-blank, non-comment line."""
    lines = [
        l.strip() for l in selection_path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    if not lines:
        raise ValueError(f"Selection file is empty: {selection_path}")
    return lines


def _slug_from_dir(root: Path) -> str:
    """Derive a short slug from the images directory path."""
    parts = root.resolve().parts
    for part in reversed(parts):
        if not part.startswith("images"):
            return part
    return root.name


def _load_images(
    item_dir: Path,
    image_names: list[str],
    quiet: bool,
) -> list:
    """Load image files and return a list of genai Part objects."""
    from google.genai.types import Part
    parts = []
    for fname in image_names:
        img_path = item_dir / fname
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not quiet:
            print(f"  Loading {img_path.name}", file=sys.stderr)
        parts.append(Part.from_bytes(data=img_path.read_bytes(), mime_type="image/jpeg"))
    return parts


def _call_gemini(client, model: str, meta: str, image_parts: list, quiet: bool) -> str:
    """Send meta-prompt + images to Gemini and return the response text."""
    from google.genai.types import GenerateContentConfig
    response = client.models.generate_content(
        model=model,
        config=GenerateContentConfig(temperature=0.2),
        contents=[meta] + image_parts,
    )
    return (response.text or "").strip()


def _save(text: str, path: Path, quiet: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    if not quiet:
        print(f"  Saved → {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate volume-specific OCR and NER prompts using Gemini "
            "and sample page images."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Slug directory (output/{slug}/) or item directory containing the selected images",
    )

    sel_group = parser.add_mutually_exclusive_group(required=True)
    sel_group.add_argument(
        "--selection", "-s",
        metavar="FILE",
        help="Path to selection.txt produced by select_sample_pages.py (one filename per line)",
    )
    sel_group.add_argument(
        "--pages", "-p",
        nargs="+",
        metavar="FILENAME",
        help="Explicit list of image filenames to use as samples",
    )

    parser.add_argument(
        "--model", "-m",
        default=_DEFAULT_MODEL,
        help=f"Gemini model to use (default: {_DEFAULT_MODEL})",
    )

    # Which prompts to generate
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--ocr-only",
        action="store_true",
        help="Generate only the OCR transcription prompt (skip NER)",
    )
    mode.add_argument(
        "--ner-only",
        action="store_true",
        help="Generate only the NER entry-extraction prompt (skip OCR)",
    )

    # Output paths
    parser.add_argument(
        "--ocr-out",
        metavar="PATH",
        help=(
            "Output path for the OCR prompt "
            "(default: {images_slug_dir}/ocr_prompt.md)"
        ),
    )
    parser.add_argument(
        "--ner-out",
        metavar="PATH",
        help=(
            "Output path for the NER prompt "
            "(default: {images_slug_dir}/ner_prompt.md)"
        ),
    )
    parser.add_argument(
        "--ner-template",
        metavar="PATH",
        default=str(_DEFAULT_NER_PROMPT),
        help=(
            f"NER prompt template to adapt (default: {_DEFAULT_NER_PROMPT}). "
            "Shown to Gemini as the structural guide."
        ),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print generated prompts to stdout only — do not save files",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # --- Resolve paths -------------------------------------------------------
    root = Path(args.output_dir).resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    if args.selection:
        sel_path = Path(args.selection)
        if not sel_path.exists():
            print(f"Error: selection file not found: {sel_path}", file=sys.stderr)
            sys.exit(1)
        image_names = _load_selection(sel_path)
    else:
        image_names = args.pages

    if not (4 <= len(image_names) <= 8):
        print(
            f"Warning: {len(image_names)} image(s) selected "
            "(recommend 4–8 for best results).",
            file=sys.stderr,
        )

    # Resolve item_dir early so we can derive the slug dir for default output paths.
    try:
        item_dir = _find_item_dir(root, image_names)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Prompts live in the slug-level directory (parent of the item dir), so
    # run_gemini_ocr.py / extract_entries.py can find them automatically
    # without needing an explicit --prompt-file flag.
    slug_dir = item_dir.parent

    ocr_out = Path(args.ocr_out) if args.ocr_out else slug_dir / "ocr_prompt.md"
    ner_out = Path(args.ner_out) if args.ner_out else slug_dir / "ner_prompt.md"

    # --- Set up Gemini client + load images (shared) -------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    if not args.quiet:
        print(f"Loading {len(image_names)} image(s)…", file=sys.stderr)
    try:
        image_parts = _load_images(item_dir, image_names, args.quiet)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- OCR prompt ----------------------------------------------------------
    if not args.ner_only:
        if not args.quiet:
            print(f"\nGenerating OCR prompt ({args.model})…", file=sys.stderr)
        try:
            ocr_meta = _OCR_META_PROMPT.format(n=len(image_names))
            ocr_text = _call_gemini(client, args.model, ocr_meta, image_parts, args.quiet)
        except Exception as exc:
            print(f"Error generating OCR prompt: {exc}", file=sys.stderr)
            sys.exit(1)

        if not ocr_text:
            print("Error: Gemini returned an empty OCR prompt.", file=sys.stderr)
            sys.exit(1)

        if args.print_only:
            print("=== OCR PROMPT ===")
            print(ocr_text)
        else:
            _save(ocr_text, ocr_out, args.quiet)

    # --- NER prompt ----------------------------------------------------------
    if not args.ocr_only:
        ner_template_path = Path(args.ner_template)
        if not ner_template_path.exists():
            print(f"Error: NER template not found: {ner_template_path}", file=sys.stderr)
            sys.exit(1)
        ner_template_text = ner_template_path.read_text(encoding="utf-8")

        if not args.quiet:
            print(f"\nGenerating NER prompt ({args.model})…", file=sys.stderr)
        try:
            ner_meta = _NER_META_PROMPT.format(
                n=len(image_names),
                ner_template=ner_template_text,
            )
            ner_text = _call_gemini(client, args.model, ner_meta, image_parts, args.quiet)
        except Exception as exc:
            print(f"Error generating NER prompt: {exc}", file=sys.stderr)
            sys.exit(1)

        if not ner_text:
            print("Error: Gemini returned an empty NER prompt.", file=sys.stderr)
            sys.exit(1)

        if args.print_only:
            print("\n=== NER PROMPT ===")
            print(ner_text)
        else:
            _save(ner_text, ner_out, args.quiet)

    # --- Usage hints ---------------------------------------------------------
    if not args.print_only and not args.quiet:
        print(file=sys.stderr)
        if not args.ner_only:
            print(
                f"OCR prompt saved to {ocr_out}\n"
                f"  run_gemini_ocr.py will use it automatically for this volume,\n"
                f"  or pass --prompt-file {ocr_out} to use it explicitly.",
                file=sys.stderr,
            )
        if not args.ocr_only:
            print(
                f"NER prompt saved to {ner_out}\n"
                f"  extract_entries.py will use it automatically for this volume,\n"
                f"  or pass --prompt {ner_out} to use it explicitly.",
                file=sys.stderr,
            )

    # Print generated text(s) to stdout for immediate review
    if not args.print_only:
        if not args.ner_only:
            print(ocr_text)
        if not args.ocr_only:
            if not args.ner_only:
                print("\n---\n")
            print(ner_text)


if __name__ == "__main__":
    main()
