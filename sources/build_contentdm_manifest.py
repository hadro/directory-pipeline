#!/usr/bin/env python3
"""Build a synthetic IIIF Presentation v3 manifest from a CONTENTdm compound object.

Fetches the page structure via the CONTENTdm API, retrieves image dimensions
from each page's info.json, and writes a standards-compliant manifest.json.

Usage:
    python analysis/build_contentdm_manifest.py \\
        --server cdm16063.contentdm.oclc.org \\
        --collection p15020coll12 \\
        --pointer 2857 \\
        --label "Polk-Hoffhine Directory Co.'s Tulsa City Directory 1921" \\
        --output output/tulsa_1921/manifest.json

    # Using the library's own IIIF proxy (recommended — Level 2, works for PDF-backed items):
    python analysis/build_contentdm_manifest.py \\
        --iiif-base https://digitalcollections.tulsalibrary.org/iiif/2 \\
        --pointer 2857 \\
        --label "Polk-Hoffhine Directory Co.'s Tulsa City Directory 1921" \\
        --output output/tulsa_1921/manifest.json
"""

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def get_compound_pages(server: str, collection: str, pointer: str) -> list[dict]:
    """Fetch all child page pointers and titles from the CONTENTdm compound object API."""
    url = (
        f"https://{server}/digital/bl/dmwebservices/index.php"
        f"?q=dmGetCompoundObjectInfo/{collection}/{pointer}/json"
    )
    data = fetch_json(url)

    pages = []
    def _walk(node):
        if "page" in node:
            for p in node["page"]:
                _walk(p)
        elif "pageptr" in node:
            pages.append({
                "pointer": str(node["pageptr"]),
                "title": node.get("pagetitle", ""),
            })
        # Top-level nodes list
        if "node" in node:
            nodes = node["node"]
            if isinstance(nodes, dict):
                nodes = [nodes]
            for n in nodes:
                _walk(n)

    _walk(data)

    # Fallback: some CONTENTdm versions use a flat "page" list at top level
    if not pages and "page" in data:
        raw = data["page"]
        if isinstance(raw, dict):
            raw = [raw]
        for p in raw:
            pages.append({
                "pointer": str(p.get("pageptr", p.get("pointer", ""))),
                "title": p.get("pagetitle", p.get("title", "")),
            })

    return pages


def get_item_metadata(server: str, collection: str, pointer: str) -> dict:
    url = (
        f"https://{server}/digital/bl/dmwebservices/index.php"
        f"?q=dmGetItemInfo/{collection}/{pointer}/json"
    )
    return fetch_json(url)


def fetch_info_json(service_id: str) -> tuple[int, int]:
    """Return (width, height) from the IIIF Image API info.json."""
    url = f"{service_id}/info.json"
    try:
        data = fetch_json(url)
        return int(data.get("width", 0)), int(data.get("height", 0))
    except Exception as e:
        print(f"  Warning: info.json failed for {service_id.rsplit('/', 1)[-1]}: {e}", file=sys.stderr)
        return 0, 0


def _service_id(iiif_base: str, collection: str, pointer: str) -> str:
    """Build the IIIF Image Service base URL for a single page.

    Two URL schemes are supported:
      Legacy CDM path:   https://{server}/digital/iiif/{collection}/{pointer}
      Colon-id scheme:   https://{host}/iiif/2/{collection}:{pointer}   ← preferred
    Pass the full iiif_base (e.g. 'https://digitalcollections.tulsalibrary.org/iiif/2')
    to use the colon-id scheme; pass a bare server hostname to use the legacy scheme.
    """
    if iiif_base.startswith("http"):
        return f"{iiif_base}/{collection}:{pointer}"
    return f"https://{iiif_base}/digital/iiif/{collection}/{pointer}"


def build_manifest(
    iiif_base: str,
    collection: str,
    root_pointer: str,
    label: str,
    pages: list[dict],
    metadata: dict,
    fetch_dims: bool = True,
    default_width: int = 4546,
    default_height: int = 3689,
) -> dict:
    manifest_id = f"{_service_id(iiif_base, collection, root_pointer)}/manifest.json"

    # Fetch dimensions in parallel
    dims: dict[str, tuple[int, int]] = {}
    if fetch_dims:
        print(f"Fetching info.json for {len(pages)} pages...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(fetch_info_json, _service_id(iiif_base, collection, p["pointer"])): p["pointer"]
                for p in pages
            }
            done = 0
            for fut in as_completed(futures):
                ptr = futures[fut]
                w, h = fut.result()
                dims[ptr] = (w or default_width, h or default_height)
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(pages)} done", file=sys.stderr)
    else:
        for p in pages:
            dims[p["pointer"]] = (default_width, default_height)

    # Build canvases
    canvases = []
    for i, page in enumerate(pages):
        ptr = page["pointer"]
        w, h = dims.get(ptr, (default_width, default_height))
        canvas_id = f"{manifest_id}/canvas/{i}"
        image_service = _service_id(iiif_base, collection, ptr)

        canvas = {
            "id": canvas_id,
            "type": "Canvas",
            "label": {"none": [page["title"] or f"Page {i + 1}"]},
            "width": w,
            "height": h,
            "items": [
                {
                    "id": f"{canvas_id}/page",
                    "type": "AnnotationPage",
                    "items": [
                        {
                            "id": f"{canvas_id}/page/image",
                            "type": "Annotation",
                            "motivation": "painting",
                            "target": canvas_id,
                            "body": {
                                "id": f"{image_service}/full/max/0/default.jpg",
                                "type": "Image",
                                "format": "image/jpeg",
                                "width": w,
                                "height": h,
                                "service": [
                                    {
                                        "id": image_service,
                                        "type": "ImageService2",
                                        "profile": "level1",
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        canvases.append(canvas)

    # Descriptive metadata from item record
    meta_pairs = []
    for key, label_text in [
        ("title", "Title"),
        ("date", "Date"),
        ("publi", "Publisher"),
        ("descri", "Description"),
        ("rights", "Rights"),
        ("callnu", "Call Number"),
    ]:
        val = metadata.get(key, "")
        if val and not isinstance(val, dict):
            meta_pairs.append({
                "label": {"en": [label_text]},
                "value": {"en": [str(val)]},
            })

    manifest = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": manifest_id,
        "type": "Manifest",
        "label": {"en": [label]},
        "metadata": meta_pairs,
        "rights": "https://creativecommons.org/publicdomain/zero/1.0/",
        "requiredStatement": {
            "label": {"en": ["Attribution"]},
            "value": {"en": ["Tulsa City-County Library, Central Library Local History Collection"]},
        },
        "items": canvases,
    }

    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="cdm16063.contentdm.oclc.org",
                   help="CDM API server hostname (used for compound object + metadata API calls)")
    p.add_argument("--iiif-base", default=None,
                   help="IIIF Image API base URL for image services. "
                        "If a full URL (e.g. https://example.org/iiif/2), uses "
                        "{iiif-base}/{collection}:{pointer} scheme. "
                        "If omitted, falls back to --server with legacy path scheme.")
    p.add_argument("--collection", default="p15020coll12")
    p.add_argument("--pointer", default="2857",
                   help="CONTENTdm pointer for the compound object (default: 2857 = 1921 dir)")
    p.add_argument("--label", default="Polk-Hoffhine Directory Co.'s Tulsa City Directory 1921")
    p.add_argument("--output", default="output/tulsa_1921/manifest.json")
    p.add_argument("--no-fetch-dims", action="store_true",
                   help="Skip info.json fetches; use default dimensions (faster but less accurate)")
    args = p.parse_args()

    print(f"Fetching compound object structure for pointer {args.pointer}...", file=sys.stderr)
    pages = get_compound_pages(args.server, args.collection, args.pointer)
    if not pages:
        print(f"Error: no pages found for pointer {args.pointer}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(pages)} pages.", file=sys.stderr)

    print("Fetching item metadata...", file=sys.stderr)
    metadata = get_item_metadata(args.server, args.collection, args.pointer)

    iiif_base = args.iiif_base or args.server
    manifest = build_manifest(
        iiif_base=iiif_base,
        collection=args.collection,
        root_pointer=args.pointer,
        label=args.label,
        pages=pages,
        metadata=metadata,
        fetch_dims=not args.no_fetch_dims,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(pages)}-canvas manifest to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
