#!/usr/bin/env python3
"""Write a standard 4-column pipeline CSV from any IIIF manifest URL.

Handles:
  - IIIF Presentation v2 and v3 item manifests  → one CSV row
  - IIIF Collection manifests (v2 sc:Collection / v3 Collection)
      → one row per child manifest (no sub-fetching required; labels are
        read from the collection manifest itself)

Output schema (same as nypl_collection_csv.py / loc_collection_csv.py):
    item_id, item_title, iiif_manifest_url, microform

Usage
-----
    python sources/iiif_manifest_csv.py <manifest_url> --output output/slug/slug.csv
"""

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.iiif_utils import manifest_item_id


def _fetch_manifest(url: str) -> dict:
    """Fetch and parse a IIIF manifest JSON from any public URL."""
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/ld+json, application/json, */*"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _extract_label(manifest: dict) -> str:
    """Return a human-readable label from a v2 or v3 manifest/collection entry."""
    label = manifest.get("label", "")
    if not label:
        return ""

    # v3: {"en": ["The title"], "none": ["..."]}
    if isinstance(label, dict):
        for lang in ("en", "none"):
            vals = label.get(lang, [])
            if vals:
                return str(vals[0]).strip()
        # Any language
        for vals in label.values():
            if isinstance(vals, list) and vals:
                return str(vals[0]).strip()
        return ""

    # v2: plain string
    if isinstance(label, str):
        return label.strip()

    # v2: list of {"@value": "...", "@language": "en"} dicts
    if isinstance(label, list):
        for item in label:
            if isinstance(item, dict):
                v = item.get("@value") or item.get("value") or ""
                if v:
                    return str(v).strip()
            elif item:
                return str(item).strip()

    return ""


def rows_from_manifest(url: str, manifest: dict) -> list[dict]:
    """Return one or more CSV rows for a manifest URL.

    For IIIF Collections, returns one row per child manifest.
    For item manifests, returns a single row.
    Child manifest labels are read inline from the collection manifest (no
    sub-fetches required).
    """
    mtype = manifest.get("@type") or manifest.get("type") or ""
    is_collection = mtype in ("sc:Collection", "Collection")

    if is_collection:
        rows = []
        # v2: manifest["manifests"] array
        children = manifest.get("manifests") or []
        # v3: manifest["items"] array filtered to type == "Manifest"
        if not children:
            for item in manifest.get("items", []):
                t = item.get("@type") or item.get("type") or ""
                if t in ("sc:Manifest", "Manifest"):
                    children.append(item)

        for child in children:
            child_url = str(child.get("@id") or child.get("id") or "").strip()
            if not child_url:
                continue
            child_id = manifest_item_id(child_url)
            child_label = _extract_label(child) or child_id
            rows.append({
                "item_id": child_id,
                "item_title": child_label,
                "iiif_manifest_url": child_url,
                "microform": "",
            })
        return rows

    # Single item manifest
    item_id = manifest_item_id(url)
    title = _extract_label(manifest) or item_id
    return [{
        "item_id": item_id,
        "item_title": title,
        "iiif_manifest_url": url,
        "microform": "",
    }]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a 4-column pipeline CSV from any IIIF manifest URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "manifest_url",
        help="IIIF manifest URL (Presentation v2 or v3, item manifest or collection)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Output CSV path (e.g. output/slug/slug.csv)",
    )
    args = parser.parse_args()

    print(f"Fetching manifest: {args.manifest_url}", file=sys.stderr)
    try:
        manifest = _fetch_manifest(args.manifest_url)
    except Exception as exc:
        print(f"Error fetching manifest: {exc}", file=sys.stderr)
        sys.exit(1)

    rows = rows_from_manifest(args.manifest_url, manifest)
    if not rows:
        print("No manifest entries found in the fetched manifest.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["item_id", "item_title", "iiif_manifest_url", "microform"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mtype = manifest.get("@type") or manifest.get("type") or "Manifest"
    print(
        f"Wrote {len(rows)} row(s) to {out_path}  [{mtype}]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
