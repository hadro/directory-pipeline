#!/usr/bin/env python3
"""Add a human-readable page label column to an extracted entries CSV.

Entries CSVs identify their source page only by the `image` filename
(`0001_p16445coll4:27074.jpg`) and by `canvas_fragment`, neither of which reads
as a page in the explorer UI. This adds a short `page` column derived from the
volume's manifest.json, which `explore_entries.py` picks up automatically as a
facet (checkbox filter + bar chart) because its cardinality is low.

The label comes from the canvas label's trailing "Page N" where present,
otherwise the 1-based canvas position. Historical directories usually carry a
printed page number that differs from the scan position — pass --printed-offset
to show both, e.g. an offset of 28 renders canvas 674 as "p. 646 (scan 674)".

Usage:
    python tools/add_page_column.py output/my-vol/entries_gemini-3.1-flash-lite.csv
    python tools/add_page_column.py output/my-vol/entries_gemini-3.1-flash-lite.csv \\
        --printed-offset 28
    python tools/add_page_column.py output/my-vol/entries_*.csv --column source_page

Rewrites each CSV in place (a .bak copy is kept unless --no-backup).
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import iiif_utils  # noqa: E402

_PAGE_RE = re.compile(r"Page\s+(\d+)\s*$")


def build_page_labels(manifest_path: Path, printed_offset: int | None) -> dict[str, str]:
    """Map image_id → page label, in canvas order.

    Both the image_id (what download_images puts in the filename) and the label
    come off the same iter_canvases dict, so a manifest whose canvases are not
    all yielded — iter_canvases skips any that lack an image body or service —
    cannot slide the labels out of step with the images.

    Positions count yielded canvases, matching how download_images numbers the
    files, so a "Page N" fallback lines up with the CSV's image column.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    for position, canvas in enumerate(iiif_utils.iter_canvases(manifest), start=1):
        match = _PAGE_RE.search(canvas.get("label", ""))
        scan = int(match.group(1)) if match else position
        if printed_offset is not None:
            labels[canvas["image_id"]] = f"p. {scan - printed_offset} (scan {scan})"
        else:
            labels[canvas["image_id"]] = f"Page {scan}"
    return labels


def image_id_from_filename(image: str) -> str:
    """'0001_p16445coll4:27074.jpg' → 'p16445coll4:27074'.

    Split-spread halves ('..._left.jpg' / '..._right.jpg') resolve to the
    canvas they were cropped from, so both halves carry that scan's label.
    Mirrors align_ocr._extract_image_id.
    """
    stem = Path(image).stem
    for suffix in ("_left", "_right"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.split("_", 1)[1] if "_" in stem else stem


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add a page label column to entries CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("csv_files", nargs="+", help="entries CSV(s) to rewrite in place")
    ap.add_argument("--manifest", help="manifest.json (default: alongside the CSV)")
    ap.add_argument("--column", default="page", help="Column name to add (default: page)")
    ap.add_argument("--after", default="subsection",
                    help="Insert after this column when present (default: subsection)")
    ap.add_argument("--printed-offset", type=int,
                    help="Scan number minus printed page number, e.g. 28 → 'p. 646 (scan 674)'")
    ap.add_argument("--no-backup", action="store_true", help="Do not write a .bak copy")
    args = ap.parse_args()

    for csv_arg in args.csv_files:
        csv_path = Path(csv_arg)
        if not csv_path.exists():
            sys.exit(f"Error: {csv_path} not found.")
        manifest_path = Path(args.manifest) if args.manifest else csv_path.parent / "manifest.json"
        if not manifest_path.exists():
            sys.exit(f"Error: manifest not found at {manifest_path} — pass --manifest.")

        labels = build_page_labels(manifest_path, args.printed_offset)

        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"{csv_path.name}: empty, skipped.", file=sys.stderr)
            continue
        if "image" not in rows[0]:
            sys.exit(f"Error: {csv_path.name} has no 'image' column to derive pages from.")

        fields = [c for c in rows[0] if c != args.column]
        insert_at = fields.index(args.after) + 1 if args.after in fields else 0
        fields.insert(insert_at, args.column)

        unmatched: set[str] = set()
        for row in rows:
            image_id = image_id_from_filename(row.get("image", ""))
            label = labels.get(image_id, "")
            if not label and image_id:
                unmatched.add(image_id)
            row[args.column] = label

        if not args.no_backup:
            shutil.copy2(csv_path, csv_path.with_suffix(csv_path.suffix + ".bak"))
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        filled = sum(1 for r in rows if r[args.column])
        distinct = sorted({r[args.column] for r in rows if r[args.column]})
        print(f"{csv_path.name}: {filled}/{len(rows)} rows labelled — {', '.join(distinct)}",
              file=sys.stderr)
        if unmatched:
            print(f"  Warning: no manifest canvas for image id(s): {', '.join(sorted(unmatched))}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
