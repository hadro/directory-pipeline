#!/usr/bin/env python3
"""Patch canvas_fragment values in per-page entries cache files and the volume CSV.

Reads *_aligned.json files and matches each cached entry's text fields against
aligned lines to find the correct #xywh= canvas fragment — without making any
Gemini API calls.

Usage:
    python pipeline/patch_canvas_fragments.py output/my_volume/ --aligned-model gemini-2.0-flash
    python pipeline/patch_canvas_fragments.py output/collection/ --aligned-model gemini-2.0-flash
"""

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path


def _find_fragment(query: str, aligned_lines: list[dict]) -> str | None:
    """Match query text against aligned lines; return canvas_fragment or None."""
    if not query or not aligned_lines:
        return None
    q = query.strip()
    # 1. Exact
    for ln in aligned_lines:
        if ln.get("gemini_text", "").strip() == q:
            return ln.get("canvas_fragment")
    # 2. Substring
    q_lower = q.lower()
    for ln in aligned_lines:
        if q_lower in ln.get("gemini_text", "").lower():
            return ln.get("canvas_fragment")
    # 3. Fuzzy
    candidates = [ln.get("gemini_text", "") for ln in aligned_lines]
    matches = difflib.get_close_matches(q, candidates, n=1, cutoff=0.6)
    if matches:
        for ln in aligned_lines:
            if ln.get("gemini_text") == matches[0]:
                return ln.get("canvas_fragment")
    return None


def _best_fragment(entry: dict, aligned_lines: list[dict]) -> str | None:
    """Try entry fields in priority order to find a matching aligned line."""
    # Fields most likely to appear verbatim in aligned text, in priority order
    for field in ("firm_name", "name", "business_name", "subject", "city", "address"):
        val = entry.get(field)
        if val:
            frag = _find_fragment(str(val), aligned_lines)
            if frag:
                return frag
    return None


def patch_item_dir(item_dir: Path, aligned_model_slug: str, dry_run: bool = False) -> dict:
    aligned_suffix = f"_{aligned_model_slug}_aligned.json"
    aligned_files = sorted(item_dir.glob(f"*{aligned_suffix}"))
    if not aligned_files:
        return {"patched": 0, "skipped": 0, "no_match": 0}

    # Build map: page_stem → aligned_lines
    page_lines: dict[str, list[dict]] = {}
    for af in aligned_files:
        stem = af.name[: -len(aligned_suffix)]
        try:
            data = json.loads(af.read_text(encoding="utf-8"))
            page_lines[stem] = data.get("lines", [])
        except Exception:
            pass

    patched = skipped = no_match = 0

    # Patch per-page cache files
    for cache_path in sorted(item_dir.glob("*_entries.json")):
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("mode") != "text-only":
            skipped += 1
            continue

        # Derive page stem from image field or filename
        image = data.get("image", "")
        page_stem = Path(image).stem if image else cache_path.stem.split("_entries")[0]
        # Strip NER model slug from stem if present (e.g. "0017_5142941")
        # cache filename: {page_stem}_{ner_slug}_entries.json
        # page_stem in page_lines uses aligned_model_slug naming: "0017_5142941"
        lines = page_lines.get(page_stem)
        if not lines:
            # Try matching by 4-digit prefix
            prefix = page_stem[:4] if len(page_stem) >= 4 else page_stem
            lines = next(
                (v for k, v in page_lines.items() if k.startswith(prefix)), None
            )
        if not lines:
            no_match += 1
            continue

        changed = False
        for entry in data.get("entries", []):
            frag = _best_fragment(entry, lines)
            if frag and frag != entry.get("canvas_fragment"):
                entry["canvas_fragment"] = frag
                changed = True
                patched += 1

        if changed and not dry_run:
            data["mode"] = "aligned-patched"
            cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"patched": patched, "skipped": skipped, "no_match": no_match}


def rebuild_csv(item_dir: Path) -> int:
    """Re-assemble the volume CSV from patched per-page cache files."""
    cache_files = sorted(item_dir.glob("*_entries.json"))
    if not cache_files:
        return 0

    # Detect NER slug from first cache filename: {stem}_{ner_slug}_entries.json
    ner_slug_pat = re.compile(r"_(gemini[-\w.]+)_entries\.json$")
    ner_slug = None
    for cf in cache_files:
        m = ner_slug_pat.search(cf.name)
        if m:
            ner_slug = m.group(1)
            break
    if not ner_slug:
        return 0

    csv_path = item_dir / f"entries_{ner_slug}.csv"

    all_entries: list[dict] = []
    fieldnames: list[str] = []
    for cf in cache_files:
        try:
            data = json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in data.get("entries", []):
            all_entries.append(entry)
            for k in entry:
                if k not in fieldnames:
                    fieldnames.append(k)

    if not all_entries:
        return 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_entries)

    return len(all_entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Item dir or collection dir containing item subdirs")
    parser.add_argument("--aligned-model", default="gemini-2.0-flash",
                        help="Model slug used in *_aligned.json filenames (default: gemini-2.0-flash)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing files")
    args = parser.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    aligned_slug = args.aligned_model.replace("/", "_")
    uuid_pat = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

    # Collect item dirs: either root itself, or any subdir containing aligned files
    item_dirs: list[Path] = []
    if any(root.glob(f"*{aligned_slug}_aligned.json")):
        item_dirs = [root]
    else:
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and any(sub.glob(f"*{aligned_slug}_aligned.json")):
                item_dirs.append(sub)

    if not item_dirs:
        print(f"No item directories found under {root}", file=sys.stderr)
        sys.exit(1)

    total_patched = total_no_match = 0
    for item_dir in item_dirs:
        stats = patch_item_dir(item_dir, aligned_slug, dry_run=args.dry_run)
        label = item_dir.name
        print(f"{label}: patched={stats['patched']} no_match={stats['no_match']} skipped={stats['skipped']}")
        total_patched += stats["patched"]
        total_no_match += stats["no_match"]

        if not args.dry_run and stats["patched"] > 0:
            n = rebuild_csv(item_dir)
            print(f"  → rebuilt CSV: {n} entries")

    print(f"\nTotal: {total_patched} fragments patched, {total_no_match} pages with no aligned data")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
