#!/usr/bin/env python3
"""Slice a IIIF manifest down to a contiguous run of canvases.

Large multi-hundred-page volumes (city directories, phone books) often contain a
single section worth extracting.  This writes a synthetic manifest holding only
the canvases you want, so `--download` fetches those pages and nothing else.

Canvas identity is preserved: every canvas keeps its original `@id`/`id`, image
service URL, and dimensions, so `canvas_fragment` values in the extracted CSV
still resolve against the source repository's viewer.

Selection is by canvas position (`--from`/`--to`, 1-based, inclusive) or by a
substring of the canvas URI (`--from-id`/`--to-id`) — useful when you have
CONTENTdm/repository page URLs rather than page numbers.

Usage:
    # By page position
    python tools/slice_manifest.py https://example.org/iiif/vol/manifest.json \\
        --from 674 --to 677 --slug london-1841-booksellers

    # By canvas-URI substring (e.g. CONTENTdm item ids from the page URLs)
    python tools/slice_manifest.py https://example.org/iiif/vol/manifest.json \\
        --from-id 27074 --to-id 27077 --slug london-1841-booksellers

Writes output/{slug}/manifest.json.  Then run the pipeline against that file:

    python main.py output/{slug}/manifest.json --slug {slug} \\
        --download --gemini-ocr --extract-entries --explore
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import iiif_utils  # noqa: E402

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/ld+json, application/json, */*",
}


def fetch_manifest(src: str) -> dict:
    """Load a manifest from a local path or URL (following redirects)."""
    local = Path(src)
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    req = urllib.request.Request(src, headers=_BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def canvas_list(manifest: dict) -> tuple[list, str]:
    """Return (canvases, container_key) for v2 or v3, so edits write back correctly."""
    if iiif_utils.manifest_version(manifest) == 2:
        return manifest["sequences"][0]["canvases"], "v2"
    return manifest["items"], "v3"


def canvas_id(canvas: dict) -> str:
    return canvas.get("@id") or canvas.get("id", "")


def canvas_label(canvas: dict) -> str:
    label = canvas.get("label", "")
    if isinstance(label, dict):  # v3 language map
        values = next(iter(label.values()), [])
        return values[0] if values else ""
    return str(label)


def resolve_index(canvases: list, needle: str, what: str) -> int:
    """Return the index of the single canvas whose URI contains needle."""
    hits = [i for i, c in enumerate(canvases) if needle in canvas_id(c)]
    if not hits:
        sys.exit(f"Error: no canvas URI contains {needle!r} ({what}).")
    if len(hits) > 1:
        ids = ", ".join(canvas_id(canvases[i]) for i in hits[:5])
        sys.exit(f"Error: {needle!r} ({what}) matches {len(hits)} canvases: {ids}…")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Slice a IIIF manifest to a contiguous canvas range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("manifest", help="IIIF manifest URL or local .json path")
    ap.add_argument("--slug",
                    help="Output slug; manifest is written to output/{slug}/manifest.json. "
                         "Required unless --list.")
    ap.add_argument("--from", dest="start", type=int,
                    help="First canvas position, 1-based inclusive")
    ap.add_argument("--to", dest="end", type=int,
                    help="Last canvas position, 1-based inclusive")
    ap.add_argument("--from-id", help="Substring of the first canvas's URI (e.g. a page id)")
    ap.add_argument("--to-id", help="Substring of the last canvas's URI")
    ap.add_argument("--output", help="Override output path (default output/{slug}/manifest.json)")
    ap.add_argument("--list", action="store_true",
                    help="Print every canvas position, id, and label, then exit")
    args = ap.parse_args()

    print(f"Fetching manifest: {args.manifest}", file=sys.stderr)
    manifest = fetch_manifest(args.manifest)
    canvases, version = canvas_list(manifest)
    print(f"{len(canvases)} canvas(es) in source manifest ({version}).", file=sys.stderr)

    if args.list:
        for i, c in enumerate(canvases, 1):
            print(f"{i}\t{canvas_id(c)}\t{canvas_label(c)}")
        return

    if not args.slug and not args.output:
        sys.exit("Error: --slug is required (or --output), unless you pass --list.")

    if args.from_id:
        start = resolve_index(canvases, args.from_id, "--from-id")
    elif args.start:
        start = args.start - 1
    else:
        sys.exit("Error: give --from or --from-id.")

    if args.to_id:
        end = resolve_index(canvases, args.to_id, "--to-id")
    elif args.end:
        end = args.end - 1
    else:
        end = start

    if not 0 <= start <= end < len(canvases):
        sys.exit(f"Error: range {start + 1}–{end + 1} is outside 1–{len(canvases)}.")

    selected = canvases[start:end + 1]

    # Rebuild the container in place; drop `structures` (its ranges reference
    # canvases we just removed, which breaks strict viewers).
    if version == "v2":
        manifest["sequences"][0]["canvases"] = selected
    else:
        manifest["items"] = selected
    manifest.pop("structures", None)

    label = manifest.get("label", "")
    if isinstance(label, str) and label:
        manifest["label"] = f"{label} — pages {start + 1}–{end + 1}"

    out_path = Path(args.output) if args.output else Path("output") / args.slug / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(
        f"\nSelected canvases {start + 1}–{end + 1} ({len(selected)} page(s)):",
        file=sys.stderr,
    )
    for i, c in enumerate(selected, start + 1):
        print(f"  {i}  {canvas_id(c)}  {canvas_label(c)}", file=sys.stderr)
    print(f"\nWrote {out_path}", file=sys.stderr)
    slug = args.slug or out_path.parent.name
    print(
        f"\nNext:\n  python main.py {out_path} --slug {slug} "
        f"--download --gemini-ocr --extract-entries --explore",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
