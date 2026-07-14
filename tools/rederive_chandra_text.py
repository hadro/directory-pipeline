#!/usr/bin/env python3
"""Re-derive Chandra plain-text (.txt) from the saved native layout (.html).

The local Chandra OCR stage saves two artifacts per page: the consumed plain
text ``{stem}_{slug}.txt`` and the full native layout ``{stem}_{slug}.html``
(data-bbox / data-label / <br/> / <h2>). Because the rich HTML is captured, any
improvement to the HTML→text parsing — figure filtering today, column anchors or
entry-boundary handling tomorrow — can be re-applied to existing pages WITHOUT
re-running the (~4 min/page) model. This tool does exactly that: it rewrites each
.txt from its sibling .html using the current parsing logic in run_local_ocr.

Usage
-----
    # Re-derive every Chandra page under a volume (recursive):
    python tools/rederive_chandra_text.py output/<slug>/

    # Different engine slug, or preview without writing:
    python tools/rederive_chandra_text.py output/<slug>/ --slug chandra-ocr-2 --dry-run
"""

import argparse
import sys
from pathlib import Path

# Reuse the exact parsing the live stage uses, so re-derivation can never drift
# from generation-time output.
from pipeline.run_local_ocr import _strip_label_blocks, _to_plain_lines


def rederive(output_root: Path, slug: str, dry_run: bool) -> int:
    suffix = f"_{slug}.html"
    html_files = sorted(p for p in output_root.rglob(f"*{suffix}"))
    if not html_files:
        print(f"no *{suffix} files under {output_root}", file=sys.stderr)
        return 1
    changed = 0
    for html_path in html_files:
        txt_path = html_path.with_name(html_path.name[: -len(suffix)] + f"_{slug}.txt")
        new_text = _to_plain_lines(_strip_label_blocks(html_path.read_text(encoding="utf-8")))
        old_text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else None
        if new_text == old_text:
            continue
        changed += 1
        if dry_run:
            print(f"would update {txt_path.name}")
        else:
            txt_path.write_text(new_text, encoding="utf-8")
    verb = "would change" if dry_run else "rewrote"
    print(f"{verb} {changed} / {len(html_files)} .txt file(s) from saved .html")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("directory", type=Path, help="output dir to scan (recursive)")
    ap.add_argument("--slug", default="chandra-ocr-2", help="engine slug (default chandra-ocr-2)")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args(argv)
    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 2
    return rederive(args.directory, args.slug, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
