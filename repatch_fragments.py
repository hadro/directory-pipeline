#!/usr/bin/env python3
"""Re-lookup canvas_fragment coordinates in entries.json from updated aligned.json files.

After re-running align_ocr --force, entries.json files have stale fragment
coordinates.  This script walks all *_entries.json files, looks up each
entry's establishment_name in the corresponding *_aligned.json, and updates
the canvas_fragment in-place.

Usage
-----
    python repatch_fragments.py images/green_book_1962_9ab2e8f0/ --model gemini-2.0-flash
    python repatch_fragments.py images/green_book_1962_9ab2e8f0/ --model gemini-2.0-flash --dry-run
"""

import argparse
import difflib
import json
import sys
from pathlib import Path


def _find_fragment(line_text: str, aligned_lines: list[dict]) -> str | None:
    """Find canvas_fragment by matching line_text against aligned lines.

    Three strategies (same as extract_entries.py):
    1. Exact match against gemini_text
    2. Substring match
    3. Fuzzy match (cutoff 0.6)
    """
    if not line_text or not aligned_lines:
        return None
    # Exact
    for ln in aligned_lines:
        if ln.get("gemini_text", "") == line_text:
            return ln.get("canvas_fragment")
    # Substring
    lt_lower = line_text.lower()
    for ln in aligned_lines:
        if lt_lower in ln.get("gemini_text", "").lower():
            return ln.get("canvas_fragment")
    # Fuzzy
    candidates = [ln.get("gemini_text", "") for ln in aligned_lines]
    matches = difflib.get_close_matches(line_text, candidates, n=1, cutoff=0.6)
    if matches:
        for ln in aligned_lines:
            if ln.get("gemini_text", "") == matches[0]:
                return ln.get("canvas_fragment")
    return None


def _is_real_fragment(frag: str | None) -> bool:
    return bool(frag and "#xywh=" in frag)


def repatch_file(
    entries_path: Path,
    aligned_path: Path,
    dry_run: bool,
) -> tuple[int, int]:
    """Repatch one entries.json.  Returns (updated, total)."""
    entries_data = json.loads(entries_path.read_text(encoding="utf-8"))
    aligned_data = json.loads(aligned_path.read_text(encoding="utf-8"))
    aligned_lines = aligned_data.get("lines", [])

    entries = entries_data.get("entries", [])
    updated = 0
    for entry in entries:
        name = entry.get("establishment_name", "")
        if not name:
            continue
        new_frag = _find_fragment(name, aligned_lines)
        old_frag = entry.get("canvas_fragment")
        if new_frag and new_frag != old_frag:
            if not dry_run:
                entry["canvas_fragment"] = new_frag
            updated += 1

    if not dry_run and updated:
        entries_path.write_text(
            json.dumps(entries_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return updated, len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images_dir",
        help="Item images directory or parent of item directories",
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        metavar="MODEL",
        help="Model slug used for aligned/entries files (e.g. gemini-2.0-flash)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files",
    )
    args = parser.parse_args()

    slug = args.model.replace("/", "_")
    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    entries_files = sorted(images_root.rglob(f"*_{slug}_entries.json"))
    if not entries_files:
        print(f"No *_{slug}_entries.json files found under {images_root}", file=sys.stderr)
        sys.exit(0)

    dry_tag = " [DRY RUN]" if args.dry_run else ""
    print(f"{dry_tag}Repatching {len(entries_files)} entries files …", file=sys.stderr)

    total_updated = 0
    total_entries = 0
    for ef in entries_files:
        stem = ef.name.replace(f"_{slug}_entries.json", "")
        af = ef.parent / f"{stem}_{slug}_aligned.json"
        if not af.exists():
            print(f"  MISSING aligned: {af.name}", file=sys.stderr)
            continue
        updated, total = repatch_file(ef, af, args.dry_run)
        total_updated += updated
        total_entries += total
        if updated:
            tag = "[dry]" if args.dry_run else "     "
            print(f"  {tag} {ef.name}: {updated}/{total} fragments updated", file=sys.stderr)

    print(
        f"\nDone. {total_updated}/{total_entries} fragment(s) updated across "
        f"{len(entries_files)} file(s).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
