#!/usr/bin/env python3
"""Internet Archive Digital Collections → CSV extractor.

For a given Internet Archive item or collection URL, outputs a CSV that is
fully compatible with download_images.py (same column schema as the NYPL
and LoC equivalents).

No API token or authentication is required for publicly accessible items.

The item-vs-collection distinction is detected automatically using the IA
metadata API (mediatype == "collection").  Collections are enumerated using
the Archive.org Scrape API and paginated automatically using cursors.
Sub-collections within a collection are skipped; only concrete items are
exported.

The IIIF manifest URL is derived from the item identifier:
    https://iiif.archive.org/iiif/{identifier}/manifest.json

This CSV feeds directly into the existing download → OCR → align pipeline:
    python ia_collection_csv.py https://archive.org/details/durstoldyorklibrary \\
        --output durstoldyorklibrary.csv
    python main.py collections.txt --download --tesseract --gemini-ocr …

Usage
-----
    # Export a single item:
    python ia_collection_csv.py https://archive.org/details/ldpd_11290437_000/

    # Export a collection:
    python ia_collection_csv.py https://archive.org/details/durstoldyorklibrary

    # Write to a named file (default: collection_csv/{slug}.csv):
    python ia_collection_csv.py https://archive.org/details/durstoldyorklibrary \\
        --output collection_csv/durstoldyorklibrary.csv
"""

import argparse
import csv
import os
import re
import sys
import time

import requests

IA_METADATA_API = "https://archive.org/metadata/{identifier}"
IA_SCRAPE_API   = "https://archive.org/services/search/v1/scrape"
IIIF_MANIFEST   = "https://iiif.archive.org/iiif/{identifier}/manifest.json"

API_DELAY          = 0.3   # seconds between requests
MAX_RETRIES        = 5
RETRYABLE_STATUSES = {429, 503}

FIELDNAMES = [
    "item_id",
    "item_title",
    "iiif_manifest_url",
    "microform",
]

IDENTIFIER_RE = re.compile(r"archive\.org/details/([^/?#]+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_identifier(url: str) -> str | None:
    """Extract the IA identifier from an archive.org/details/… URL."""
    m = IDENTIFIER_RE.search(url)
    return m.group(1).rstrip("/") if m else None


def _manifest_url(identifier: str) -> str:
    return IIIF_MANIFEST.format(identifier=identifier)


def _get_with_retry(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    label: str = "",
) -> requests.Response:
    """GET with exponential backoff on 429/503."""
    for attempt in range(MAX_RETRIES + 1):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code not in RETRYABLE_STATUSES or attempt == MAX_RETRIES:
            resp.raise_for_status()
            return resp
        retry_after = resp.headers.get("Retry-After", "")
        try:
            wait = float(retry_after)
        except ValueError:
            wait = 2 ** (attempt + 1)
        ctx = f" ({label})" if label else ""
        print(
            f"  HTTP {resp.status_code}{ctx} — waiting {wait:.0f}s "
            f"({attempt + 1}/{MAX_RETRIES})…",
            file=sys.stderr,
        )
        time.sleep(wait)
    resp.raise_for_status()
    return resp  # type: ignore[return-value]


def _fetch_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    label: str = "",
) -> dict:
    resp = _get_with_retry(session, url, params=params, label=label)
    time.sleep(API_DELAY)
    return resp.json()


def _str_val(v) -> str:
    """Unwrap a list-of-one or coerce to str."""
    if isinstance(v, list):
        return str(v[0]).strip() if v else ""
    return str(v).strip() if v else ""


def _is_collection(metadata_response: dict) -> bool:
    return metadata_response.get("metadata", {}).get("mediatype", "") == "collection"


def _is_microform(raw: object) -> bool:
    t = str(raw).lower()
    return "microfilm" in t or "microform" in t


def _make_slug(title: str, identifier: str) -> str:
    """Build a filesystem-safe slug from a title and IA identifier."""
    # IA identifiers can be long; cap the prefix to stay readable
    id_part = identifier[:20]
    if title:
        sanitized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
        if sanitized:
            return f"{sanitized}_{id_part}"
    return id_part


def _default_stem(session: requests.Session, identifier: str, meta_data: dict) -> str:
    """Derive a sensible CSV filename stem from an IA identifier and cached metadata."""
    title = _str_val(meta_data.get("metadata", {}).get("title", ""))
    return _make_slug(title, identifier)


# ---------------------------------------------------------------------------
# Collection export
# ---------------------------------------------------------------------------

def iter_collection_items(
    session: requests.Session,
    collection_id: str,
    verbose: bool = True,
):
    """
    Yield one raw item dict per item in an IA collection via the Scrape API.
    Sub-collections (mediatype == "collection") are skipped.
    """
    params: dict = {
        "q": f"collection:{collection_id}",
        "fields": "identifier,title,mediatype",
        "count": 100,
    }
    page = 1
    while True:
        if verbose:
            print(f"  Fetching page {page}…", file=sys.stderr)
        try:
            data = _fetch_json(
                session, IA_SCRAPE_API, params=params, label=f"page {page}"
            )
        except requests.HTTPError as exc:
            print(
                f"  Warning: HTTP {exc.response.status_code} on page {page} — stopping.",
                file=sys.stderr,
            )
            break

        for item in data.get("items", []):
            if item.get("mediatype") == "collection":
                continue  # skip nested sub-collections
            yield item

        cursor = data.get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor
        page += 1


def process_collection(
    session: requests.Session,
    collection_id: str,
    writer: csv.DictWriter,
    verbose: bool = True,
) -> int:
    """Export all items from an IA collection. Returns row count."""
    rows = 0
    for item in iter_collection_items(session, collection_id, verbose):
        identifier = item.get("identifier", "")
        if not identifier:
            continue
        title = _str_val(item.get("title", ""))
        microform = _is_microform(item)
        writer.writerow({
            "item_id":           identifier,
            "item_title":        title,
            "iiif_manifest_url": _manifest_url(identifier),
            "microform":         microform,
        })
        rows += 1
        if verbose:
            label = title[:60] if title else identifier
            print(f"    {identifier}: {label}", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Single-item export
# ---------------------------------------------------------------------------

def process_single_item(
    session: requests.Session,
    identifier: str,
    writer: csv.DictWriter,
    meta_data: dict,
    verbose: bool = True,
) -> int:
    """Write one CSV row for a single IA item. Returns 1 on success."""
    meta = meta_data.get("metadata", {})
    title = _str_val(meta.get("title", ""))
    microform = _is_microform(meta_data)
    writer.writerow({
        "item_id":           identifier,
        "item_title":        title,
        "iiif_manifest_url": _manifest_url(identifier),
        "microform":         microform,
    })
    if verbose:
        print(f"    {identifier}: {title[:60]}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export Internet Archive item metadata to CSV for use with download_images.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help=(
            "An Internet Archive item or collection URL "
            "(https://archive.org/details/{identifier})."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output CSV file path.  If no directory component is given, written "
            "to collection_csv/.  Defaults to collection_csv/{slug}.csv derived "
            "from the identifier and title.  Pass '-' to write to stdout."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    source = args.source.strip()
    identifier = _extract_identifier(source)
    if not identifier:
        parser.error(
            "Expected an Internet Archive URL containing /details/{identifier}.\n"
            f"Got: {source}"
        )

    session = requests.Session()
    session.headers["Accept"] = "application/json"

    # Fetch metadata once — used for both item/collection detection and title
    if not args.quiet:
        print(f"IA source: {source}", file=sys.stderr)
        print(f"  Checking metadata for {identifier}…", file=sys.stderr)

    try:
        meta_data = _fetch_json(
            session,
            IA_METADATA_API.format(identifier=identifier),
            label=identifier,
        )
    except requests.HTTPError as exc:
        print(
            f"Error: HTTP {exc.response.status_code} fetching metadata for {identifier}",
            file=sys.stderr,
        )
        sys.exit(1)

    is_coll = _is_collection(meta_data)
    kind = "collection" if is_coll else "item"
    if not args.quiet:
        print(f"  Type: {kind}", file=sys.stderr)

    # Resolve output path
    output = args.output
    if output is None:
        stem = _default_stem(session, identifier, meta_data)
        output = f"{stem}.csv"

    if output == "-":
        out_file = sys.stdout
        out_path = "-"
        should_close = False
    else:
        out_path = output
        if not os.path.dirname(out_path):
            os.makedirs("collection_csv", exist_ok=True)
            out_path = os.path.join("collection_csv", out_path)
        out_file = open(out_path, "w", newline="", encoding="utf-8")
        should_close = True

    try:
        writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        if is_coll:
            total = process_collection(
                session, identifier, writer, verbose=not args.quiet,
            )
        else:
            if not args.quiet:
                print(f"  Exporting item {identifier}…", file=sys.stderr)
            total = process_single_item(
                session, identifier, writer, meta_data, verbose=not args.quiet,
            )

        if not args.quiet:
            dest = out_path if output != "-" else "stdout"
            print(f"\nWrote {total} row(s) to {dest}.", file=sys.stderr)
    finally:
        if should_close:
            out_file.close()


if __name__ == "__main__":
    main()
