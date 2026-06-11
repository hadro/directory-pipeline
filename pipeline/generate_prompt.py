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

    # Multi-section volume (city directory with alphabetical/street/business sections):
    python pipeline/generate_prompt.py output/tulsa_1921/ \\
        --sections output/tulsa_1921/sections.txt
    # Generates ocr_prompt_alphabetical.md, ner_prompt_alphabetical.md, etc.
    # plus ocr_prompt.md / ner_prompt.md as fallbacks from all sampled pages.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils.gemini import get_client
from utils.models import DEFAULT_PROMPT_MODEL

_DITTO_INSTRUCTION = (
    "\n\n8. **Ditto marks** — This volume uses ditto marks (small raised comma-pairs, "
    "sometimes printed as '' or 〃, often misread by OCR as 66 or as quotation marks) "
    "to repeat a value from the row above. Instruct the model to expand them in place: "
    "write out the full repeated value on each line rather than transcribing the mark itself."
)

# Appended to the OCR meta-prompt when pixel analysis detects a two-column layout,
# to ensure the generated prompt explicitly forbids merging columns onto one line.
_COLUMN_INJECTION = (
    "\n\nIMPORTANT — pixel analysis of the sample pages confirms a two-column layout. "
    "Your generated prompt MUST include this instruction verbatim, placed immediately "
    "after the column-layout sentence:\n\n"
    '  "When the page has two columns, text in the left column and text in the right '
    "column at the same vertical position are two separate output lines. Never place "
    'entries from both columns on the same output line."'
)

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

3. **Column layout** — Typical number of columns on content pages. If pages are \
two-column, say so explicitly and include this instruction: "When the page has two \
columns, text in the left column and text in the right column at the same vertical \
position are two separate output lines. Never place entries from both columns on the \
same output line." Note exceptions (full-width headings, index pages, advertisements).

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
You are calibrating a named entity recognition pipeline for a digitized historical document.

I am sharing {n} sample page images from this collection. Study them carefully — \
pay attention to the heading hierarchy, entry format, recurring patterns, continuation \
markers, and any unusual structural features.

Then write a tailored NER system prompt that will help a language model extract \
structured entries from the transcribed text of pages from this volume.

The pipeline works as follows: the NER model receives the OCR transcription of one \
page at a time, along with context carried forward from the prior page (the last active \
heading values). The model returns a JSON object. **The only fixed structural \
requirement is this JSON envelope:**

{{
  "page_context": {{ ... }},
  "entries": [ ... ]
}}

Write a prompt that specifies:

1. **Document identity** — what this document is and what kind of records it contains \
   (one to two sentences).

2. **Heading hierarchy and context** — the nesting of headings and what each level \
   represents. Define the `page_context` fields: the heading values that must be \
   tracked and carried between pages. Use field names that match the actual content \
   (e.g. `state`/`city`/`category` for a place directory; `insect_type`/`chapter` \
   for a natural history volume; whatever fits this document). If headings continue \
   across pages (e.g., "OHIO-continued") try to normalize these to their regular form.\

3. **Entry schema** — the specific fields each entry should contain, named to match \
   the actual content. Choose field names that are clear and meaningful for this \
   document type. Do not force a schema from a different document type onto this one. \
   Always include the heading hierarchy and context elements within the entries \
   themselves.

4. **Extraction rules** — 4–6 rules specific to this volume: what counts as an entry, \
   what to skip (page numbers, headings, decorative elements), how continuation \
   markers work, whether entries can span pages, etc. Always include a rule about \
   mid-page heading transitions: when a new heading appears mid-page, entries after \
   it inherit the *new* context — not the prior-page context. The prior-page context \
   only applies to entries before the first heading change on the current page.

5. **Output format** — instruct the model to return only valid JSON with no markdown \
   code fences and no explanatory text.

Here is a reference prompt from a different collection type. Use it as a structural \
example only — do not copy its field names, category lists, or document-specific \
content. Adapt everything to match what you observe in the sample pages:

--- BEGIN REFERENCE ---
{ner_template}
--- END REFERENCE ---

Write the prompt in the second person ("You are a structured data extractor…"). \
Output only the tailored prompt — no preamble, no explanation, no section headers \
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


def _sample_section_pages(
    section: dict,
    all_files: list[str],
    item_dir: Path,
    n: int = 3,
) -> list[str]:
    """Pick n representative filenames spread evenly through a section's page range.

    Only returns pages that actually exist on disk.
    """
    indices = section["page_indices"]
    if not indices:
        return []
    if len(indices) <= n:
        candidates = indices
    else:
        step = (len(indices) - 1) / (n - 1) if n > 1 else 0
        candidates = [indices[round(i * step)] for i in range(n)]
    result = []
    for idx in candidates:
        fname = Path(all_files[idx]).name
        if (item_dir / fname).exists():
            result.append(fname)
    return result


def _detect_two_columns(image_paths: list[Path]) -> bool:
    """Return True if the majority of sample pages appear to be two-column layout.

    Delegates to detect_columns.analyze_image(), which uses recursive valley
    prominence on the vertical dark-pixel projection.  Uses a conservative
    gutter_threshold of 0.45 (only deep, unambiguous gutters count) to avoid
    false positives from single-column layouts with leader dots or binding edges.

    A strict majority of usable pages must be detected as 2-column.
    """
    from pipeline.detect_columns import analyze_image

    two_col = 0
    usable = 0
    for p in image_paths[:6]:
        try:
            result = analyze_image(p, margin_frac=0.10, gutter_threshold=0.45)
            usable += 1
            if result["num_columns"] >= 2:
                two_col += 1
        except Exception:
            pass
    return usable > 0 and two_col > max(1, usable // 2)


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

    sel_group = parser.add_mutually_exclusive_group(required=False)
    sel_group.add_argument(
        "--selection", "-s",
        metavar="FILE",
        help="Path to selection.txt produced by select_pages.py (one filename per line)",
    )
    sel_group.add_argument(
        "--pages", "-p",
        nargs="+",
        metavar="FILENAME",
        help="Explicit list of image filenames to use as samples",
    )
    parser.add_argument(
        "--sections",
        metavar="PATH",
        help=(
            "Path to sections.txt marking structural section boundaries. "
            "When provided, auto-samples pages per section and generates "
            "per-section prompt files (ocr_prompt_{label}.md, ner_prompt_{label}.md) "
            "plus fallback ocr_prompt.md / ner_prompt.md from all sampled pages. "
            "--selection / --pages are optional when this flag is used."
        ),
    )

    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_PROMPT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_PROMPT_MODEL})",
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
        "--expand-dittos",
        dest="expand_dittos",
        action="store_true",
        help=(
            "Instruct the generated OCR prompt to expand ditto marks in place "
            "rather than transcribing them literally. Ditto marks ('' or 〃) are "
            "common in tabular historical documents and are often misread as 66. "
            "Off by default — verbatim transcription is the standard behavior."
        ),
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-generate prompts even if output files already exist",
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

    # Validate that at least one image source is supplied
    if not args.sections and not args.selection and not args.pages:
        parser.error("one of --selection, --pages, or --sections is required")

    # =========================================================================
    # SECTIONS MODE — auto-sample per section and generate per-section prompts
    # =========================================================================
    if args.sections:
        from utils.section_utils import load_sections

        sections_path = Path(args.sections)
        if not sections_path.exists():
            print(f"Error: sections file not found: {sections_path}", file=sys.stderr)
            sys.exit(1)

        # Discover images using the same logic as run_gemini_ocr.py:
        # include _left/_right splits in place of the original spread file.
        _all_jpgs = sorted(
            p for p in root.rglob("*.jpg")
            if not p.stem.endswith("_viz")
        )
        all_imgs: list[Path] = []
        for _p in _all_jpgs:
            if _p.stem.endswith("_left") or _p.stem.endswith("_right"):
                all_imgs.append(_p)
                continue
            _left = _p.with_name(f"{_p.stem}_left.jpg")
            _right = _p.with_name(f"{_p.stem}_right.jpg")
            if _left.exists() and _right.exists():
                continue
            all_imgs.append(_p)
        if not all_imgs:
            print(f"Error: no .jpg images found under {root}", file=sys.stderr)
            sys.exit(1)

        # item_dir = the directory that actually contains the images
        item_dir = all_imgs[0].parent
        slug_dir = root if root == item_dir else item_dir.parent

        all_file_names = [p.name for p in all_imgs]

        try:
            sections = load_sections(sections_path, all_file_names)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if not sections:
            print("Error: sections.txt is empty or contains no valid entries.", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"Found {len(sections)} section(s): {', '.join(s['label'] for s in sections)}", file=sys.stderr)

        # --- Set up Gemini client + NER template ----------------------------
        client = get_client()

        ner_template_path = Path(args.ner_template)
        if not ner_template_path.exists():
            print(f"Error: NER template not found: {ner_template_path}", file=sys.stderr)
            sys.exit(1)
        ner_template_text = ner_template_path.read_text(encoding="utf-8")

        # --- Use manually selected pages if available, else auto-sample --------
        # Check for selection.txt (explicit flag takes priority, then auto-discover)
        manually_selected: list[str] = []
        if args.selection:
            sel_path = Path(args.selection)
            if sel_path.exists():
                manually_selected = _load_selection(sel_path)
                if not args.quiet:
                    print(f"Using manually selected pages from {sel_path}", file=sys.stderr)
        else:
            for candidate in (slug_dir / "selection.txt", item_dir / "selection.txt"):
                if candidate.exists():
                    manually_selected = _load_selection(candidate)
                    if not args.quiet:
                        print(f"Using manually selected pages from {candidate}", file=sys.stderr)
                    break

        # Build a set-of-indices for each section so we can group manual selections
        section_index_sets = {
            sec["label"]: set(sec["page_indices"]) for sec in sections
        }
        file_index = {n: i for i, n in enumerate(all_file_names)}

        def _pages_for_section(sec: dict) -> list[str]:
            """Return pages for this section: manually selected if available, else auto-sampled."""
            if manually_selected:
                in_section = [
                    n for n in manually_selected
                    if file_index.get(n, -1) in section_index_sets[sec["label"]]
                ]
                if in_section:
                    return in_section
            return _sample_section_pages(sec, all_file_names, item_dir, n=3)

        all_sampled_names: list[str] = []

        # --- Per-section prompt generation ----------------------------------
        for sec in sections:
            label = sec["label"]
            sec_names = _pages_for_section(sec)
            if not sec_names:
                print(f"  Warning: no images found for section '{label}', skipping.", file=sys.stderr)
                continue

            all_sampled_names.extend(sec_names)

            ocr_sec_out = slug_dir / f"ocr_prompt_{label}.md"
            ner_sec_out = slug_dir / f"ner_prompt_{label}.md"

            if not args.force and not args.print_only:
                sec_ocr_needed = not args.ner_only and not ocr_sec_out.exists()
                sec_ner_needed = not args.ocr_only and not ner_sec_out.exists()
                if not sec_ocr_needed and not sec_ner_needed:
                    if not args.quiet:
                        print(
                            f"  [{label}] prompts already exist, skipping. Use --force to regenerate.",
                            file=sys.stderr,
                        )
                    continue

            if not args.quiet:
                print(
                    f"\n[{label}] Sampling {len(sec_names)} page(s): {', '.join(sec_names)}",
                    file=sys.stderr,
                )

            try:
                sec_parts = _load_images(item_dir, sec_names, args.quiet)
            except FileNotFoundError as e:
                print(f"  Error loading images for section '{label}': {e}", file=sys.stderr)
                continue

            # OCR prompt for this section
            if not args.ner_only:
                if not args.quiet:
                    print(f"  [{label}] Generating OCR prompt…", file=sys.stderr)
                try:
                    ocr_meta = _OCR_META_PROMPT.format(n=len(sec_names))
                    if args.expand_dittos:
                        ocr_meta = ocr_meta.rstrip() + _DITTO_INSTRUCTION
                    if _detect_two_columns([item_dir / n for n in sec_names]):
                        ocr_meta = ocr_meta.rstrip() + _COLUMN_INJECTION
                    ocr_text = _call_gemini(client, args.model, ocr_meta, sec_parts, args.quiet)
                except Exception as exc:
                    print(f"  Error generating OCR prompt for '{label}': {exc}", file=sys.stderr)
                    continue
                if ocr_text and not args.print_only:
                    _save(ocr_text, ocr_sec_out, args.quiet)
                elif args.print_only:
                    print(f"=== OCR PROMPT [{label}] ===\n{ocr_text}\n")

            # NER prompt for this section
            if not args.ocr_only:
                if not args.quiet:
                    print(f"  [{label}] Generating NER prompt…", file=sys.stderr)
                try:
                    ner_meta = _NER_META_PROMPT.format(
                        n=len(sec_names),
                        ner_template=ner_template_text,
                    )
                    ner_text = _call_gemini(client, args.model, ner_meta, sec_parts, args.quiet)
                except Exception as exc:
                    print(f"  Error generating NER prompt for '{label}': {exc}", file=sys.stderr)
                    continue
                if ner_text and not args.print_only:
                    _save(ner_text, ner_sec_out, args.quiet)
                elif args.print_only:
                    print(f"=== NER PROMPT [{label}] ===\n{ner_text}\n")

        # --- Fallback all-pages prompts from combined samples ----------------
        # Deduplicate while preserving order
        seen: set[str] = set()
        combined_names = [n for n in all_sampled_names if not (n in seen or seen.add(n))]

        if combined_names:
            ocr_out = Path(args.ocr_out) if args.ocr_out else slug_dir / "ocr_prompt.md"
            ner_out = Path(args.ner_out) if args.ner_out else slug_dir / "ner_prompt.md"
            fallback_needed = (
                args.force
                or args.print_only
                or (not args.ner_only and not ocr_out.exists())
                or (not args.ocr_only and not ner_out.exists())
            )
            if fallback_needed:
                if not args.quiet:
                    print(
                        f"\nGenerating fallback prompts from {len(combined_names)} combined sample(s)…",
                        file=sys.stderr,
                    )
                try:
                    combined_parts = _load_images(item_dir, combined_names, args.quiet)
                except FileNotFoundError as e:
                    print(f"Error loading combined images: {e}", file=sys.stderr)
                    sys.exit(1)

                if not args.ner_only:
                    ocr_meta = _OCR_META_PROMPT.format(n=len(combined_names))
                    if args.expand_dittos:
                        ocr_meta = ocr_meta.rstrip() + _DITTO_INSTRUCTION
                    if _detect_two_columns([item_dir / n for n in combined_names]):
                        ocr_meta = ocr_meta.rstrip() + _COLUMN_INJECTION
                    try:
                        ocr_text = _call_gemini(client, args.model, ocr_meta, combined_parts, args.quiet)
                        if ocr_text and not args.print_only:
                            _save(ocr_text, ocr_out, args.quiet)
                    except Exception as exc:
                        print(f"Error generating fallback OCR prompt: {exc}", file=sys.stderr)

                if not args.ocr_only:
                    ner_meta = _NER_META_PROMPT.format(
                        n=len(combined_names), ner_template=ner_template_text
                    )
                    try:
                        ner_text = _call_gemini(client, args.model, ner_meta, combined_parts, args.quiet)
                        if ner_text and not args.print_only:
                            _save(ner_text, ner_out, args.quiet)
                    except Exception as exc:
                        print(f"Error generating fallback NER prompt: {exc}", file=sys.stderr)

        return  # sections mode is complete

    # =========================================================================
    # STANDARD MODE — single selection or explicit page list
    # =========================================================================
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

    # --- Skip if prompts already exist (unless --force or --print-only) ------
    if not args.force and not args.print_only:
        ocr_needed = not args.ner_only and not ocr_out.exists()
        ner_needed = not args.ocr_only and not ner_out.exists()
        if not ocr_needed and not ner_needed:
            if not args.quiet:
                print(
                    f"Skipping --generate-prompts: prompts already exist "
                    f"({ocr_out.name}, {ner_out.name}). Use --force to regenerate.",
                    file=sys.stderr,
                )
            sys.exit(0)

    # --- Set up Gemini client + load images (shared) -------------------------
    client = get_client()

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
            if args.expand_dittos:
                ocr_meta = ocr_meta.rstrip() + _DITTO_INSTRUCTION
            if _detect_two_columns([item_dir / n for n in image_names]):
                if not args.quiet:
                    print(
                        "  Detected two-column layout — injecting column instruction.",
                        file=sys.stderr,
                    )
                ocr_meta = ocr_meta.rstrip() + _COLUMN_INJECTION
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
