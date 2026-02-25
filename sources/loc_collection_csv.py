#!/usr/bin/env python3
"""Library of Congress Digital Collections → CSV extractor.

For a given LoC collection URL or single item URL, outputs a CSV that is
fully compatible with download_images.py (same column schema as the NYPL
equivalent, nypl_collection_csv.py).

No API token or authentication is required.  The LoC JSON API is publicly
accessible for all digitized items.

Collection pages are paginated automatically (100 items per request by
default).  Only items with digitized=true are included.  Sub-collection
entries and non-item results are skipped automatically.

The IIIF manifest URL is derived predictably from the item URL:
    https://www.loc.gov/item/{id}/manifest.json

This CSV feeds directly into the existing download → OCR → align pipeline:
    python loc_collection_csv.py https://www.loc.gov/collections/civil-war-maps/ \\
        --output civil-war-maps.csv
    python main.py collections.txt --download --tesseract --gemini-ocr …
        (or point --download directly at the generated CSV)

Usage
-----
    # Export a full collection:
    python loc_collection_csv.py https://www.loc.gov/collections/civil-war-maps/

    # Export a single item:
    python loc_collection_csv.py https://www.loc.gov/item/01015253/

    # Write to a named file (default: stdout):
    python loc_collection_csv.py https://www.loc.gov/collections/civil-war-maps/ \\
        --output collection_csv/civil-war-maps.csv
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import requests

API_DELAY = 0.3   # seconds between requests — conservative for a public server
MAX_RETRIES = 5
RETRYABLE_STATUSES = {429, 503}

ITEM_ID_RE = re.compile(r"/item/([^/?#]+)/?")

FIELDNAMES = [
    "item_id",
    "item_title",
    "iiif_manifest_url",
    "microform",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_item_id(url: str) -> str | None:
    """
    Extract the LoC item ID from an item URL or id string.
        'http://www.loc.gov/item/01015253/'  → '01015253'
        'https://www.loc.gov/item/sn83045462/' → 'sn83045462'
    """
    m = ITEM_ID_RE.search(url)
    return m.group(1).strip("/") if m else None


def _is_item_url(url: str) -> bool:
    return "/item/" in url


def _is_collection_url(url: str) -> bool:
    return "/collections/" in url


def _manifest_url(item_id: str) -> str:
    return f"https://www.loc.gov/item/{item_id}/manifest.json"


def _get_with_retry(
    session: requests.Session, url: str, label: str = ""
) -> requests.Response:
    """GET with exponential backoff on 429/503."""
    for attempt in range(MAX_RETRIES + 1):
        resp = session.get(url, timeout=30)
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


def _fetch_json(session: requests.Session, url: str, label: str = "") -> dict:
    resp = _get_with_retry(session, url, label=label)
    time.sleep(API_DELAY)
    return resp.json()


def _str_val(v) -> str:
    """Unwrap a list-of-one or return the value as a string."""
    if isinstance(v, list):
        return str(v[0]).strip() if v else ""
    return str(v).strip() if v else ""


def _make_slug(title: str, item_id: str) -> str:
    """
    Build a filesystem-safe slug from a title and item ID, mirroring main.py.
        ('The Brooklyn city directory', '01015253')
            → 'the_brooklyn_city_directory_01015253'
    """
    id_suffix = item_id[:12]  # LoC IDs are short; keep in full
    if title:
        sanitized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
        if sanitized:
            return f"{sanitized}_{id_suffix}"
    return id_suffix


def _fetch_title(session: requests.Session, item_id: str) -> str:
    """Quick best-effort fetch of an item title for slug generation."""
    try:
        data = _fetch_json(session, f"https://www.loc.gov/item/{item_id}/?fo=json")
        return _str_val(data.get("item", {}).get("title", ""))
    except Exception:  # noqa: BLE001
        return ""


def _default_stem(url: str, session: requests.Session | None = None) -> str:
    """
    Derive a sensible CSV filename stem from a LoC URL.
        /item/01015253/              → 'the_brooklyn_city_directory_01015253'
        /collections/civil-war-maps/ → 'civil-war-maps'
    When session is None the title fetch is skipped (falls back to ID only).
    """
    if _is_item_url(url):
        item_id = _extract_item_id(url) or "loc-item"
        title = _fetch_title(session, item_id) if session else ""
        return _make_slug(title, item_id)
    m = re.search(r"/collections/([^/?#]+)", url)
    return m.group(1).rstrip("/") if m else "loc-collection"


def _is_microform(raw_text: str) -> bool:
    t = raw_text.lower()
    return "microform" in t or "microfilm" in t


def _make_row(item_id: str, title: str, microform: bool) -> dict:
    return {
        "item_id": item_id,
        "item_title": title,
        "iiif_manifest_url": _manifest_url(item_id),
        "microform": microform,
    }


# ---------------------------------------------------------------------------
# Collection export
# ---------------------------------------------------------------------------

def iter_collection_items(
    session: requests.Session,
    collection_url: str,
    per_page: int = 100,
    verbose: bool = True,
):
    """
    Yield one raw result dict per digitized item in a LoC collection.
    Follows the pagination.next URL automatically until exhausted.
    """
    base = collection_url.rstrip("/")
    url: str | None = f"{base}/?fo=json&c={per_page}&sp=1"
    page = 1
    while url:
        if verbose:
            print(f"  Fetching page {page}…", file=sys.stderr)
        try:
            data = _fetch_json(session, url, label=f"page {page}")
        except requests.HTTPError as exc:
            print(
                f"  Warning: HTTP {exc.response.status_code} on page {page} — stopping.",
                file=sys.stderr,
            )
            break

        for item in data.get("results", []):
            # Only keep digitized items with a /item/ URL
            if not item.get("digitized", False):
                continue
            if not _is_item_url(item.get("id", "")):
                continue
            yield item

        url = data.get("pagination", {}).get("next")
        page += 1


def process_collection(
    session: requests.Session,
    collection_url: str,
    writer: csv.DictWriter,
    per_page: int = 100,
    verbose: bool = True,
) -> int:
    """Export all digitized items from a LoC collection. Returns row count."""
    # Extract a short container ID from the collection URL slug
    slug_match = re.search(r"/collections/([^/?#]+)", collection_url)
    container_id = slug_match.group(1).rstrip("/") if slug_match else ""

    rows = 0
    for item in iter_collection_items(session, collection_url, per_page, verbose):
        item_id = _extract_item_id(item.get("id", ""))
        if not item_id:
            continue

        title = _str_val(item.get("title", ""))
        microform = _is_microform(str(item))
        writer.writerow(_make_row(item_id, title, microform))
        rows += 1
        if verbose:
            label = title[:60] if title else item_id
            print(f"    {item_id}: {label}", file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# Single-item export
# ---------------------------------------------------------------------------

def process_single_item(
    session: requests.Session,
    item_url: str,
    writer: csv.DictWriter,
    verbose: bool = True,
) -> int:
    """Fetch full item metadata and export one CSV row. Returns 1 on success."""
    item_id = _extract_item_id(item_url)
    if not item_id:
        print(f"  Warning: could not extract item ID from: {item_url}", file=sys.stderr)
        return 0

    api_url = f"https://www.loc.gov/item/{item_id}/?fo=json"
    if verbose:
        print(f"  Fetching item {item_id}…", file=sys.stderr)

    try:
        data = _fetch_json(session, api_url, label=item_id)
    except requests.HTTPError as exc:
        print(
            f"  Warning: HTTP {exc.response.status_code} fetching {api_url}",
            file=sys.stderr,
        )
        return 0

    item = data.get("item", {})

    title = _str_val(item.get("title", ""))
    microform = _is_microform(str(data))
    writer.writerow(_make_row(item_id, title, microform))
    if verbose:
        label = title[:60] if title else item_id
        print(f"    {item_id}: {label} ({total} image(s))", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export Library of Congress item metadata to CSV for use with download_images.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help=(
            "A LoC collection URL (https://www.loc.gov/collections/{slug}/) "
            "or a single item URL (https://www.loc.gov/item/{id}/)."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output CSV file path. If no directory is given, written to collection_csv/. "
            "Defaults to collection_csv/{slug}.csv derived from the URL. "
            "Pass '-' to write to stdout."
        ),
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        metavar="N",
        help="Items per API page for collection export (default: 100; max: 150)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    source = args.source.strip()
    if not source.startswith("http"):
        parser.error(f"Expected a URL starting with https://www.loc.gov/…, got: {source!r}")
    if not _is_item_url(source) and not _is_collection_url(source):
        parser.error(
            "URL must be a LoC collection (/collections/…) or item (/item/…) URL.\n"
            f"Got: {source}"
        )

    session = requests.Session()
    session.headers["Accept"] = "application/json"

    # Resolve output path — default to collection_csv/{slug}.csv
    output = args.output
    if output is None:
        output = f"{_default_stem(source, session)}.csv"

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

    if not args.quiet:
        kind = "item" if _is_item_url(source) else "collection"
        print(f"LoC {kind}: {source}", file=sys.stderr)

    try:
        writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        if _is_item_url(source):
            total = process_single_item(session, source, writer, verbose=not args.quiet)
        else:
            total = process_collection(
                session, source, writer,
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
