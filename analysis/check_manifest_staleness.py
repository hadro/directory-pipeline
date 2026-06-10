"""
Compare cached manifest.json files against their live source IIIF manifests to
detect canvas-count or ordering drift that would cause wrong canvasIndex
deep-links.

Usage:
    python analysis/check_manifest_staleness.py <collection_dir> [--update-stale]

Example:
    python analysis/check_manifest_staleness.py \
        output/green_books_and_related/the_green_book_9ea5d5b0

Options:
    --update-stale   Overwrite stale manifest.json files with the live version.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def canvas_ids(manifest: dict) -> list[str]:
    canvases = manifest.get("items") or (
        manifest.get("sequences", [{}])[0].get("canvases", [])
    )
    ids = []
    for c in canvases:
        ids.append(c.get("id") or c.get("@id") or "")
    return ids


def manifest_source_url(manifest: dict) -> str:
    """The manifest's own canonical URL (IIIF v3 ``id`` / v2 ``@id``).

    Every manifest we cache records where it came from, so staleness checks
    work against any provider (LoC, IA, CONTENTdm, NYPL, …) without
    reconstructing a provider-specific API URL.
    """
    return manifest.get("id") or manifest.get("@id") or ""


def fetch_live(manifest_url: str, timeout: int = 20) -> dict | None:
    try:
        req = urllib.request.Request(manifest_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"    [fetch error] {e}", file=sys.stderr)
        return None


def first_image_id(ids: list[str]) -> str:
    """Extract the bare image ID from the first canvas URL."""
    if not ids:
        return ""
    # e.g. https://iiif.nypl.org/iiif/3/5206006/full/!760,760/0/default.jpg
    parts = ids[0].rstrip("/").split("/")
    for i, p in enumerate(parts):
        if p.isdigit() and i > 0:
            return p
    return ids[0]


def canvas_offset(cached_ids: list[str], live_ids: list[str]) -> int | None:
    """
    Return how many canvases the live manifest has REMOVED from the front
    relative to the cached manifest (positive = live starts later).
    Returns None if the first live canvas isn't found in the cached list at all.
    """
    if not live_ids or not cached_ids:
        return None
    first_live = live_ids[0]
    try:
        return cached_ids.index(first_live)
    except ValueError:
        return None


def check_collection(collection_dir: Path, update_stale: bool) -> None:
    manifest_paths = sorted(collection_dir.glob("*/manifest.json"))
    if not manifest_paths:
        sys.exit(f"No manifest.json files found under {collection_dir}")

    stale: list[Path] = []
    ok: list[str] = []

    print(f"Checking {len(manifest_paths)} manifests against live source manifests …\n")

    for mp in manifest_paths:
        name = mp.parent.name
        cached = json.loads(mp.read_text(encoding="utf-8"))
        cached_ids = canvas_ids(cached)
        source_url = manifest_source_url(cached)

        # Derive title from label field
        label_raw = cached.get("label", {})
        if isinstance(label_raw, dict):
            title = next(iter(label_raw.values()), [""])[0] if label_raw else ""
        else:
            title = str(label_raw)

        print(f"  {name}  ({title[:55]})")
        print(f"    cached : {len(cached_ids)} canvases, first={first_image_id(cached_ids)}")

        if not source_url:
            print("    live   : no source manifest URL recorded in cache — skipping\n")
            continue

        live = fetch_live(source_url)
        time.sleep(0.3)  # be polite to the source server

        if live is None:
            print("    live   : FETCH FAILED — skipping\n")
            continue

        live_ids = canvas_ids(live)
        print(f"    live   : {len(live_ids)} canvases, first={first_image_id(live_ids)}")

        count_diff = len(cached_ids) - len(live_ids)
        offset = canvas_offset(cached_ids, live_ids)

        if cached_ids == live_ids:
            print("    status : OK — identical\n")
            ok.append(name)
        elif offset is not None and offset != 0:
            print(
                f"    status : STALE — cached has {offset} extra canvas(es) at front "
                f"(canvasIndex will be off by {offset})"
            )
            stale.append(mp)
            if update_stale:
                mp.write_text(json.dumps(live, ensure_ascii=False, indent=2), encoding="utf-8")
                print("             → manifest.json updated with live version")
            print()
        elif count_diff != 0:
            print(
                f"    status : CHANGED — count differs by {count_diff} "
                f"(cached {len(cached_ids)} vs live {len(live_ids)}); "
                "first canvas matches, ordering may still be correct"
            )
            stale.append(mp)
            if update_stale:
                mp.write_text(json.dumps(live, ensure_ascii=False, indent=2), encoding="utf-8")
                print("             → manifest.json updated with live version")
            print()
        else:
            print("    status : OK — same count, same first canvas\n")
            ok.append(name)

    print("─" * 60)
    print(f"OK     : {len(ok)}")
    print(f"Stale  : {len(stale)}")
    if stale:
        print("\nStale volumes:")
        for p in stale:
            print(f"  {p.parent.name}")
        if not update_stale:
            print(
                "\nRe-run with --update-stale to overwrite stale manifests, "
                "then rebuild the explorer."
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Check cached IIIF manifests for staleness.")
    p.add_argument("collection_dir", type=Path)
    p.add_argument(
        "--update-stale",
        action="store_true",
        help="Overwrite stale manifest.json files with the live version",
    )
    args = p.parse_args()

    collection_dir = args.collection_dir.resolve()
    if not collection_dir.is_dir():
        sys.exit(f"Not a directory: {collection_dir}")

    check_collection(collection_dir, args.update_stale)


if __name__ == "__main__":
    main()
