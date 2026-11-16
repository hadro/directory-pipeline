#!/usr/bin/env python3
"""Collate the entries the binder refused to place, across a volume or collection.

``utils.entry_binding`` gives an entry no box when no aligned line agrees with it
strongly enough, and ``extract_entries`` records those in each page's
``*_entries.json`` under ``unplaced_entries``. That is the right shape for the
extractor — per page, next to the data — and the wrong shape for a person, who
wants one ranked list for a whole volume.

This walks the per-page files and writes that list, ordered by how close the
nearest miss came, so the entries most likely to be a threshold problem rather
than a genuine absence sort to the top.

An unplaced entry is not necessarily a defect. Three things land here:

  * the line exists but the alignment never boxed it — the entry is real, the
    box is missing upstream, and re-aligning the page is the fix
  * the OCR never read the line, so there is nothing to bind to
  * the NER invented an entry, or duplicated one, and there is correctly no line

Only the first is repairable by binding, which is why this is a review queue and
not an error report. The `nearest_score` column is the triage signal: entries
scoring just under the floor usually have a real line; entries at 0.00 usually
do not.

Usage:
    python tools/collate_review_queue.py output/greenbooks/
    python tools/collate_review_queue.py output/greenbooks/ --min-score 0.2
    python tools/collate_review_queue.py output/ --csv review_queue.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ENTRIES_RE = re.compile(r"_(gemini[-\w.]+)_entries\.json$")


def collect(root: Path, min_score: float = 0.0) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob("*_entries.json")):
        if not ENTRIES_RE.search(path.name):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        unplaced = data.get("unplaced_entries")
        if not unplaced:
            continue
        image = data.get("image", "")
        for item in unplaced:
            if item.get("nearest_score", 0.0) < min_score:
                continue
            rows.append({
                "volume": path.parent.name,
                "image": image,
                "aligned_model": data.get("aligned_model", ""),
                "entry_index": item.get("entry_index"),
                "entry_text": item.get("entry_text", ""),
                "nearest_score": item.get("nearest_score", 0.0),
                "nearest_line": item.get("nearest_line", ""),
                "nearest_fragment": item.get("nearest_fragment", ""),
            })
    # Closest misses first — most likely to be a threshold call rather than an
    # entry with no line on the page at all.
    rows.sort(key=lambda r: (-r["nearest_score"], r["volume"], r["image"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path, help="Volume or collection directory")
    ap.add_argument("--csv", type=Path, help="Write the queue here instead of stdout")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="Only show entries whose nearest miss scored at least this")
    ap.add_argument("--limit", type=int, default=40,
                    help="Rows to print when writing to stdout (default: %(default)s)")
    args = ap.parse_args()

    if not args.output_dir.is_dir():
        print(f"error: {args.output_dir} is not a directory", file=sys.stderr)
        return 2

    rows = collect(args.output_dir, args.min_score)
    if not rows:
        print("Nothing unplaced. Either every entry found a line, or this tree was "
              "extracted before the binder recorded a review queue.")
        return 0

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        near = sum(1 for r in rows if r["nearest_score"] > 0)
        print(f"wrote {args.csv}: {len(rows)} unplaced entries "
              f"({near} with a near-miss line, {len(rows) - near} with none)")
        return 0

    print(f"{len(rows)} unplaced entries (showing {min(args.limit, len(rows))}):\n")
    print(f"{'score':>6}  {'image':<22} {'entry':<34} nearest line")
    for r in rows[:args.limit]:
        print(f"{r['nearest_score']:>6.2f}  {r['image'][:22]:<22} "
              f"{r['entry_text'][:34]:<34} {r['nearest_line'][:44]}")
    if len(rows) > args.limit:
        print(f"\n… {len(rows) - args.limit} more. Use --csv to write the full queue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
