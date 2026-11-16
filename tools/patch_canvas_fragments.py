#!/usr/bin/env python3
"""Patch canvas_fragment values in per-page entries cache files and the volume CSV.

Reads *_aligned.json files and matches each cached entry's text fields against
aligned lines to find the correct #xywh= canvas fragment — without making any
Gemini API calls.

Usage:
    python tools/patch_canvas_fragments.py output/my_volume/ --dry-run
    python tools/patch_canvas_fragments.py output/collection/ --aligned-model gemini-3.1-flash-lite

--aligned-model defaults to the model slug most common among the aligned files
under the directory. That reproduces the old gemini-2.0-flash default on the
Green Books collection root, and finds current runs' files too. Pass it
explicitly where a tree mixes models.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.entry_binding import bind_entries  # noqa: E402
from utils.models import DEFAULT_OCR_MODEL, discover_ocr_slug, model_slug  # noqa: E402


def patch_item_dir(item_dir: Path, aligned_model_slug: str, dry_run: bool = False) -> dict:
    aligned_suffix = f"_{aligned_model_slug}_aligned.json"
    aligned_files = sorted(item_dir.glob(f"*{aligned_suffix}"))
    if not aligned_files:
        return {"patched": 0, "skipped": 0, "no_match": 0, "unplaced": 0}

    # Build map: page_stem → (aligned_lines, canvas_width). The width feeds the
    # edge-bleed guard, which rejects the narrow sliver of the facing page that
    # spread scans leave at the extreme left or right edge.
    page_lines: dict[str, tuple[list[dict], int]] = {}
    for af in aligned_files:
        stem = af.name[: -len(aligned_suffix)]
        try:
            data = json.loads(af.read_text(encoding="utf-8"))
            page_lines[stem] = (data.get("lines", []), data.get("canvas_width") or 0)
        except Exception:
            pass

    patched = skipped = no_match = unplaced = 0

    # Patch per-page cache files
    for cache_path in sorted(item_dir.glob("*_entries.json")):
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Every extraction mode is patchable: the mode records how the NER call
        # was made (with the page image or from text alone), not where the boxes
        # came from, and a page with no aligned data is already caught by the
        # no_match branch below. The old list omitted "multimodal", which is two
        # thirds of the corpus and every Green Book volume — this tool silently
        # skipped all of them and reported success.
        if data.get("mode") not in ("text-only", "multimodal", "aligned", "aligned-patched"):
            skipped += 1
            continue

        # Derive page stem from image field or filename
        image = data.get("image", "")
        page_stem = Path(image).stem if image else cache_path.stem.split("_entries")[0]
        # Strip NER model slug from stem if present (e.g. "0017_5142941")
        # cache filename: {page_stem}_{ner_slug}_entries.json
        # page_stem in page_lines uses aligned_model_slug naming: "0017_5142941"
        page = page_lines.get(page_stem)
        if not page:
            # Try matching by 4-digit prefix
            prefix = page_stem[:4] if len(page_stem) >= 4 else page_stem
            page = next(
                (v for k, v in page_lines.items() if k.startswith(prefix)), None
            )
        if not page or not page[0]:
            no_match += 1
            continue
        lines, canvas_width = page

        # One call for the whole page, not one per entry: the assignment is
        # one-to-one, so the strongest agreement settles before a weaker claim
        # can take its line. Patching entry-by-entry let several entries share
        # a box.
        entries = data.get("entries", [])
        changed = False
        for entry, binding in zip(entries, bind_entries(entries, lines,
                                                        canvas_width=canvas_width)):
            if binding.line_index is None:
                # Leave whatever is there rather than overwriting a good box
                # with a guess; an entry this tool cannot place is one for review.
                unplaced += 1
                continue
            frag = lines[binding.line_index].get("canvas_fragment")
            if frag and frag != entry.get("canvas_fragment"):
                entry["canvas_fragment"] = frag
                changed = True
                patched += 1

        if changed and not dry_run:
            data["mode"] = "aligned-patched"
            cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"patched": patched, "skipped": skipped, "no_match": no_match,
            "unplaced": unplaced}


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
    parser.add_argument("--aligned-model", default=None,
                        help="Model slug used in *_aligned.json filenames (default: the slug "
                             "most common among aligned files under output_dir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing files")
    args = parser.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    aligned_slug = model_slug(
        args.aligned_model or discover_ocr_slug(root, recursive=True) or DEFAULT_OCR_MODEL
    )
    if not args.aligned_model:
        print(f"Aligned model: {aligned_slug} (most common on disk; --aligned-model to override)")

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

    total_patched = total_no_match = total_unplaced = 0
    for item_dir in item_dirs:
        stats = patch_item_dir(item_dir, aligned_slug, dry_run=args.dry_run)
        label = item_dir.name
        print(f"{label}: patched={stats['patched']} unplaced={stats['unplaced']}"
              f" no_match={stats['no_match']} skipped={stats['skipped']}")
        total_patched += stats["patched"]
        total_no_match += stats["no_match"]
        total_unplaced += stats["unplaced"]

        if not args.dry_run and stats["patched"] > 0:
            n = rebuild_csv(item_dir)
            print(f"  → rebuilt CSV: {n} entries")

    print(f"\nTotal: {total_patched} fragments patched, {total_unplaced} entries left"
          f" unplaced (no line matched strongly enough), {total_no_match} pages"
          f" with no aligned data")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
