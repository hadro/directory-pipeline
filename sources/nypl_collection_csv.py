#!/usr/bin/env python3
"""NYPL Digital Collections → CSV extractor.

For a given collection (by UUID or Digital Collections URL), outputs a CSV with:
  - container_uuid   : UUID of the sub-collection/container holding the item
                       (empty string when items sit directly in the top-level collection)
  - item_uuid        : UUID of the item (e.g. an individual volume or photograph)
  - iiif_manifest_url: IIIF Presentation manifest URL for the item
  - image_id         : imageID of the item's first capture (used for image access)

Authentication
--------------
You need an NYPL API token. Sign up at https://api.repo.nypl.org/sign_up
Pass it via --token or the NYPL_API_TOKEN environment variable.

Usage
-----
    python main.py <uuid-or-url> --token <api-token>
    python main.py <uuid-or-url> --token <api-token> --output out.csv

    # from a Digital Collections URL:
    python main.py https://digitalcollections.nypl.org/collections/634f3af0-c607-012f-9f2d-58d385a7bc34 \\
        --token abc123 --output travelguide.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.repo.nypl.org/api/v2"

# NYPL IIIF Presentation manifest URL pattern (one manifest per item/volume)
IIIF_MANIFEST_TEMPLATE = (
    "https://api-collections.nypl.org/manifests/{uuid}"
    "?manifest=https://api-collections.nypl.org/manifests/{uuid}"
)

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def extract_uuid(text: str) -> str | None:
    """Return the first UUID found in text (handles both raw UUIDs and URLs)."""
    m = UUID_RE.search(text)
    return m.group(0) if m else None


def is_item_url(text: str) -> bool:
    """Return True if the input looks like a Digital Collections item URL."""
    return "/items/" in text


def _as_list(obj) -> list:
    """Normalize API values that may be a single dict or a list of dicts."""
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]


def _total_pages(data: dict) -> int:
    """Extract totalPages from an API response's nested request block."""
    req = data.get("nyplAPI", {}).get("request", {})
    inner = req.get("request", {})
    val = inner.get("totalPages", 1)
    if isinstance(val, dict):
        val = val.get("$", 1)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 1


class NYPLClient:
    """Thin wrapper around the NYPL Digital Collections API v2."""

    def __init__(
        self,
        token: str,
        request_delay: float = 0.15,
        cache_dir: "Path | None" = None,
    ):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f'Token token="{token}"'
        self.delay = request_delay
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_get(self, key: str) -> dict | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _cache_put(self, key: str, data: dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / f"{key}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{API_BASE}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.delay)
        return resp.json()

    def get_collection_page(
        self, uuid: str, page: int = 1, per_page: int = 100
    ) -> dict:
        """One page of child items and containers for a collection UUID."""
        key = f"collection_{uuid}_p{page}_n{per_page}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        data = self._get(f"/collections/{uuid}", {"page": page, "per_page": per_page})
        self._cache_put(key, data)
        return data

    def get_captures_page(
        self, item_uuid: str, page: int = 1, per_page: int = 1
    ) -> dict:
        """One page of captures (individual images/pages) for an item UUID."""
        key = f"item_{item_uuid}_p{page}_n{per_page}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        data = self._get(
            f"/items/item_details/{item_uuid}", {"page": page, "per_page": per_page}
        )
        self._cache_put(key, data)
        return data



def iter_collection_children(client: NYPLClient, uuid: str, per_page: int = 100):
    """
    Yield (kind, child_dict) for every direct child of a collection.

    kind is 'item' or 'container'.
    Handles API pagination automatically.

    The NYPL API uses 'item' for items and either 'container' or 'collection'
    for sub-collections depending on the collection type.
    """
    page = 1
    while True:
        data = client.get_collection_page(uuid, page=page, per_page=per_page)
        response = data.get("nyplAPI", {}).get("response", {})

        for item in _as_list(response.get("item")):
            yield "item", item

        # Sub-collections appear under 'container' or 'collection' depending on collection type
        for container in _as_list(response.get("container")) + _as_list(response.get("collection")):
            yield "container", container

        if page >= _total_pages(data):
            break
        page += 1


def _extract_str(obj) -> str:
    """Unwrap a {'$': 'value'} node or return the value as-is."""
    if isinstance(obj, dict):
        return obj.get("$", "")
    return str(obj) if obj else ""


def get_first_capture_info(client: NYPLClient, item_uuid: str) -> tuple[str, str, int, str, str, bool]:
    """
    Return (image_id, rights_statement_uri, total_images, date_issued, item_title, microform)
    for item_uuid.

    All values come from the same API call.
    Returns ('', '', 0, '', '', False) on failure.

    microform is True if "microform" or "microfilm" appears anywhere in the API response.

    The imageID can be used to build image access URLs, e.g.:
        https://iiif-prod.nypl.org/index.php?id={image_id}&t=w
    """
    try:
        data = client.get_captures_page(item_uuid, page=1, per_page=1)
        response = data.get("nyplAPI", {}).get("response", {})

        # Check entire raw response for microform/microfilm keywords
        raw_text = json.dumps(data).lower()
        microform = "microform" in raw_text or "microfilm" in raw_text
        total_images = int(_extract_str(response.get("numResults", "0")) or "0")

        mods = response.get("mods") or {}

        date_issued = ""
        for oi in _as_list(mods.get("originInfo")):
            di = oi.get("dateIssued") if isinstance(oi, dict) else None
            candidate = _extract_str(_as_list(di)[0] if isinstance(di, list) else di)
            if candidate:
                date_issued = candidate
                break

        item_title = ""
        for ti in _as_list(mods.get("titleInfo")):
            candidate = _extract_str(ti.get("title") if isinstance(ti, dict) else None)
            if candidate:
                item_title = candidate
                break

        for key in ("sibling_captures", "imm_captures", "root_captures"):
            node = response.get(key, {})
            captures = _as_list(node.get("capture") if isinstance(node, dict) else None)
            if captures:
                capture = captures[0]
                image_id = _extract_str(capture.get("imageID", ""))
                rights_uri = _extract_str(capture.get("rightsStatementURI", ""))
                return image_id, rights_uri, total_images, date_issued, item_title, microform
    except requests.HTTPError as exc:
        print(
            f"  Warning: HTTP {exc.response.status_code} fetching captures "
            f"for {item_uuid}",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"  Warning: could not fetch captures for {item_uuid}: {exc}",
            file=sys.stderr,
        )
    return "", "", 0, "", "", False


def process_collection(
    client: NYPLClient,
    collection_uuid: str,
    writer: csv.DictWriter,
    per_page: int = 100,
    verbose: bool = True,
) -> int:
    """
    Recursively walk the collection hierarchy, writing a CSV row for every item found.

    Returns the total number of rows written.
    """
    rows_written = 0

    def emit(container_uuid: str, item: dict) -> None:
        nonlocal rows_written
        item_uuid = item.get("uuid", "")
        title = ""
        mods = item.get("mods", {})
        if isinstance(mods, dict):
            title_info = mods.get("titleInfo", {})
            if isinstance(title_info, dict):
                title = title_info.get("title", "")

        if verbose:
            label = title or item_uuid
            print(f"  Fetching captures for: {label} ({item_uuid})", file=sys.stderr)

        _image_id, _rights_uri, _total_images, _date_issued, item_title, microform = get_first_capture_info(client, item_uuid)
        writer.writerow(
            {
                "item_id": item_uuid,
                "item_title": item_title,
                "iiif_manifest_url": IIIF_MANIFEST_TEMPLATE.format(uuid=item_uuid),
                "microform": microform,
            }
        )
        rows_written += 1

    def walk(uuid: str, parent_uuid: str) -> None:
        """Recursively descend containers; emit a row for each item."""
        for kind, child in iter_collection_children(client, uuid, per_page):
            if kind == "item":
                emit(uuid, child)
            elif kind == "container":
                container_uuid = child.get("uuid", "")
                if verbose:
                    title = (child.get("mods") or {}).get("titleInfo", {}).get("title", container_uuid)
                    print(f"Entering container: {title} ({container_uuid})", file=sys.stderr)
                walk(container_uuid, uuid)

    walk(collection_uuid, collection_uuid)
    return rows_written


def process_single_item(
    client: NYPLClient,
    item_uuid: str,
    writer: csv.DictWriter,
    verbose: bool = True,
) -> int:
    """Fetch capture info for a single item UUID and write one CSV row."""
    if verbose:
        print(f"  Fetching captures for item: {item_uuid}", file=sys.stderr)
    _image_id, _rights_uri, _total_images, _date_issued, item_title, microform = get_first_capture_info(client, item_uuid)
    writer.writerow(
        {
            "item_id": item_uuid,
            "item_title": item_title,
            "iiif_manifest_url": IIIF_MANIFEST_TEMPLATE.format(uuid=item_uuid),
            "microform": microform,
        }
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export NYPL Digital Collections item metadata to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "collection",
        help="Collection or item UUID, or Digital Collections URL",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("NYPL_API_TOKEN", ""),
        help=(
            "NYPL API token (or set NYPL_API_TOKEN env var). "
            "Get one at https://api.repo.nypl.org/sign_up"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default="-",
        help="Output CSV file path (default: stdout)",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        metavar="N",
        help="Items to request per API page (default: 100)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for caching API responses (JSON). "
            "Defaults to <output-stem>_cache/ next to the output CSV. "
            "Pass 'none' to disable caching."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error(
            "No API token provided. Use --token or set NYPL_API_TOKEN."
        )

    target_uuid = extract_uuid(args.collection)
    if not target_uuid:
        parser.error(f"Could not find a UUID in: {args.collection!r}")

    item_mode = is_item_url(args.collection)

    if not args.quiet:
        kind_label = "Item" if item_mode else "Collection"
        print(f"{kind_label} UUID: {target_uuid}", file=sys.stderr)

    # Resolve output path first so we can derive the default cache dir from it
    if args.output == "-":
        out_file = sys.stdout
        out_path = "-"
        should_close = False
    else:
        out_path = args.output
        if not os.path.dirname(out_path):
            os.makedirs("collection_csv", exist_ok=True)
            out_path = os.path.join("collection_csv", out_path)
        out_file = open(out_path, "w", newline="", encoding="utf-8")
        should_close = True

    # Resolve cache directory
    if args.cache_dir and args.cache_dir.lower() == "none":
        cache_dir = None
    elif args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif out_path != "-":
        # Default: <output-stem>_cache/ next to the CSV
        p = Path(out_path)
        cache_dir = p.parent / f"{p.stem}_cache"
    else:
        cache_dir = None  # stdout — no default cache

    if cache_dir and not args.quiet:
        print(f"API cache: {cache_dir}", file=sys.stderr)

    client = NYPLClient(token=args.token, cache_dir=cache_dir)
    fieldnames = ["item_id", "item_title", "iiif_manifest_url", "microform"]

    try:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        if item_mode:
            total = process_single_item(
                client,
                target_uuid,
                writer,
                verbose=not args.quiet,
            )
        else:
            total = process_collection(
                client,
                target_uuid,
                writer,
                per_page=args.per_page,
                verbose=not args.quiet,
            )
        if not args.quiet:
            dest = out_path if args.output != "-" else "stdout"
            print(f"\nWrote {total} row(s) to {dest}.", file=sys.stderr)
    finally:
        if should_close:
            out_file.close()


if __name__ == "__main__":
    main()
