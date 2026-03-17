#!/usr/bin/env python3
"""Export aligned OCR output as IIIF Annotation Pages (W3C Web Annotation).

Reads *_{model}_aligned.json files from an item directory and writes a
corresponding *_{model}_annotations.json for each page — a W3C Annotation
Page directly loadable by Mirador, Universal Viewer, Clover, and other
IIIF-compatible viewers.

If *_{model}_entries.json sidecars are also present, generates a second set
of entry-level annotation files (*_{model}_entry_annotations.json) with the
structured data (name, address, city, state, category) as annotation bodies.

Annotation format — line-level transcription:
  motivation: "supplementing"
  body:       TextualBody (Gemini-corrected text, format=text/plain)
  target:     canvas_uri#xywh=x,y,w,h

Annotation format — entry-level:
  motivation: "describing"
  body:       TextualBody (name — address, city, state, format=text/plain)
  target:     canvas_fragment from the first matching line

The W3C spec allows annotation pages without @id fields (omit --base-url to
produce self-contained files with no network dependencies).  Adding --base-url
lets viewers like Mirador reload annotations from a known URL.

Usage
-----
    python pipeline/iiif/export_annotations.py output/green_book_1940_feb978b0/uuid/
    python pipeline/iiif/export_annotations.py output/green_book_1940_feb978b0/uuid/ \\
        --model gemini-2.0-flash \\
        --base-url https://example.org/annotations
    python pipeline/iiif/export_annotations.py output/green_book_1940_feb978b0/uuid/ \\
        --no-entries
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from iiif.label_utils import (  # noqa: E402
    parse_ner_label_fields,
    build_entry_label,
    handle_missing_ner_prompt,
)


# ---------------------------------------------------------------------------
# Annotation builders
# ---------------------------------------------------------------------------

def _line_annotation(ann_id: str, canvas_fragment: str, text: str, lang: str) -> dict:
    ann: dict = {
        "type":       "Annotation",
        "motivation": "supplementing",
        "body": {
            "type":     "TextualBody",
            "value":    text,
            "format":   "text/plain",
            "language": lang,
        },
        "target": canvas_fragment,
    }
    if ann_id:
        ann["id"] = ann_id
    return ann


def _entry_annotation(ann_id: str, canvas_fragment: str, entry: dict,
                      name_fields: list[str] | None = None,
                      addr_fields: list[str] | None = None) -> dict:
    """Build a describing annotation for a structured directory entry."""
    label = build_entry_label(entry, name_fields, addr_fields)

    ann: dict = {
        "type":       "Annotation",
        "motivation": "describing",
        "body": {
            "type":   "TextualBody",
            "value":  label,
            "format": "text/plain",
        },
        "target": canvas_fragment,
    }
    if ann_id:
        ann["id"] = ann_id
    return ann


def _make_page(page_id: str, items: list[dict]) -> dict:
    page: dict = {
        "@context": "http://www.w3.org/ns/anno.jsonld",
        "type":     "AnnotationPage",
        "items":    items,
    }
    if page_id:
        page["id"] = page_id
    return page


# ---------------------------------------------------------------------------
# Per-file export
# ---------------------------------------------------------------------------

def export_page(
    aligned_path: Path,
    base_url: str,
    lang: str,
    include_entries: bool,
    force: bool,
    quiet: bool,
    ner_prompt_path: Path | None = None,
) -> tuple[int, int]:
    """Export one aligned JSON to annotation page(s). Returns (n_lines, n_entries)."""
    try:
        data = json.loads(aligned_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  SKIP {aligned_path.name}: {exc}", file=sys.stderr)
        return 0, 0

    # aligned filename: {stem}_{model_slug}_aligned.json
    base_stem = aligned_path.stem.removesuffix("_aligned")

    # ---- Line-level annotations ----------------------------------------
    ann_path = aligned_path.parent / f"{base_stem}_annotations.json"
    n_lines = 0
    if force or not ann_path.exists():
        page_id = f"{base_url}/{base_stem}_annotations" if base_url else ""
        items = []
        for i, line in enumerate(data.get("lines", [])):
            cf   = line.get("canvas_fragment", "")
            text = line.get("gemini_text", "")
            if not cf or not text:
                continue
            ann_id = f"{page_id}/line/{i}" if page_id else ""
            items.append(_line_annotation(ann_id, cf, text, lang))
        ann_path.write_text(
            json.dumps(_make_page(page_id, items), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        n_lines = len(items)

    # ---- Entry-level annotations ----------------------------------------
    n_entries = 0
    if include_entries:
        # entries sidecar: {stem}_{model_slug}_entries.json
        entry_sidecar = aligned_path.parent / f"{base_stem}_entries.json"
        if entry_sidecar.exists():
            entry_ann_path = aligned_path.parent / f"{base_stem}_entry_annotations.json"
            if force or not entry_ann_path.exists():
                try:
                    raw = json.loads(entry_sidecar.read_text(encoding="utf-8"))
                    entries = (
                        raw if isinstance(raw, list)
                        else raw.get("entries", [])
                    )
                except Exception:
                    entries = []

                resolved_ner = ner_prompt_path or (aligned_path.parent / "ner_prompt.md")
                name_fields, addr_fields = parse_ner_label_fields(resolved_ner)

                page_id = f"{base_url}/{base_stem}_entry_annotations" if base_url else ""
                items = []
                for i, entry in enumerate(entries):
                    cf = entry.get("canvas_fragment", "")
                    if not cf:
                        continue
                    ann_id = f"{page_id}/entry/{i}" if page_id else ""
                    items.append(_entry_annotation(ann_id, cf, entry,
                                                   name_fields or None, addr_fields or None))
                entry_ann_path.write_text(
                    json.dumps(_make_page(page_id, items), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                n_entries = len(items)

    if not quiet:
        print(
            f"  {aligned_path.name}: {n_lines} line annotations"
            + (f", {n_entries} entry annotations" if include_entries else ""),
            file=sys.stderr,
        )
    return n_lines, n_entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export aligned OCR output as IIIF Annotation Pages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("item_dir",
        help="Item directory (or images root) containing *_aligned.json files")
    parser.add_argument("--model", "-m", default="gemini-2.0-flash", metavar="MODEL",
        help="Model slug used in aligned JSON filenames (default: gemini-2.0-flash)")
    parser.add_argument("--base-url", default="", metavar="URL",
        help="Base URL for annotation IDs "
             "(e.g. https://example.org/annotations). "
             "Omit for ID-less pages (valid per W3C spec; no network dependency).")
    parser.add_argument("--lang", default="en", metavar="LANG",
        help="BCP 47 language tag for line annotation bodies (default: en)")
    parser.add_argument("--no-entries", action="store_true",
        help="Skip entry-level annotation export (line-level transcription only)")
    parser.add_argument("--ner-prompt", default=None, metavar="PATH",
        help="Path to a ner_prompt.md for this collection type.  "
             "If omitted, looks for ner_prompt.md in the item directory.")
    parser.add_argument("--force", "-f", action="store_true",
        help="Re-export even if output files already exist")
    parser.add_argument("--quiet", "-q", action="store_true",
        help="Suppress per-file progress output")
    args = parser.parse_args()

    item_dir = Path(args.item_dir)
    if not item_dir.is_dir():
        print(f"Error: not a directory: {item_dir}", file=sys.stderr)
        sys.exit(1)

    slug = args.model.replace("/", "_")
    aligned_files = sorted(item_dir.rglob(f"*_{slug}_aligned.json"))
    if not aligned_files:
        print(
            f"No *_{slug}_aligned.json files found under {item_dir}.\n"
            f"Run --align-ocr first, or check --model matches the slug used.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Exporting annotation pages for {len(aligned_files)} file(s)…",
        file=sys.stderr,
    )

    ner_prompt_path: Path | None = None
    if args.ner_prompt:
        ner_prompt_path = Path(args.ner_prompt)
        if not ner_prompt_path.is_file():
            print(f"Error: --ner-prompt path not found: {ner_prompt_path}", file=sys.stderr)
            sys.exit(1)
    elif not args.no_entries:
        candidate = item_dir / "ner_prompt.md"
        if candidate.is_file():
            ner_prompt_path = candidate

    if ner_prompt_path and not args.no_entries:
        name_fields, addr_fields = parse_ner_label_fields(ner_prompt_path)
        if not args.quiet:
            print(
                f"NER prompt: {ner_prompt_path}\n"
                f"  label fields  — name: {name_fields}, addr: {addr_fields}",
                file=sys.stderr,
            )
    elif not args.no_entries:
        handle_missing_ner_prompt()

    total_lines   = 0
    total_entries = 0
    for p in aligned_files:
        n_lines, n_entries = export_page(
            p,
            base_url=args.base_url,
            lang=args.lang,
            include_entries=not args.no_entries,
            force=args.force,
            quiet=args.quiet,
            ner_prompt_path=ner_prompt_path,
        )
        total_lines   += n_lines
        total_entries += n_entries

    entry_msg = (
        f", {total_entries} entry annotation(s)" if not args.no_entries else ""
    )
    print(
        f"\nDone. {total_lines} line annotation(s){entry_msg} "
        f"across {len(aligned_files)} page(s).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
