#!/usr/bin/env python3
"""Download full-resolution images from IIIF manifests.

Two input modes
---------------
CSV mode (default):
    Reads a collection CSV produced by nypl_collection_csv.py or
    loc_collection_csv.py and downloads every item's images.

    Images are saved to:
        images/<csv-stem>/<item_id>/<page>_<image_id>.jpg

Manifest mode (--manifest):
    Downloads a single IIIF manifest directly — no CSV required.
    Accepts any public IIIF Presentation v2 or v3 manifest URL.

    Images are saved to:
        images/<item_id>/<page>_<image_id>.jpg

    where <item_id> is derived from the manifest URL, or overridden
    with --output-dir.

In both modes, already-downloaded files are skipped (safe to re-run).

On HTTP 429 or 503, the script retries automatically with exponential
backoff, respecting the Retry-After header when provided.

If a server returns HTTP 400 for a requested size, it likely means the
width exceeds the server's limit.  Use --width to request a smaller size.

Usage
-----
    # CSV mode:
    python download_images.py collection_csv/travelguide.csv
    python download_images.py collection_csv/travelguide.csv --width 2048

    # Manifest mode:
    python download_images.py --manifest https://www.loc.gov/item/01015253/manifest.json
    python download_images.py --manifest https://example.org/iiif/item/manifest.json \\
        --output-dir images/my-item --width 2048
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

import iiif_utils

DEFAULT_WIDTH = 2048
DEFAULT_DELAY = 0.5   # ~2 requests/second — conservative for institutional servers
MAX_RETRIES = 5
RETRYABLE_STATUSES = {429, 503}

# Fallback IIIF endpoint used when the primary service returns HTTP 403
# (which usually means the requested size exceeds what the server permits).
FALLBACK_IIIF_BASE = "https://iiif.nypl.org/iiif/3"
FALLBACK_SIZE = "!760,760"
FALLBACK_403_THRESHOLD = 3   # switch after this many consecutive primary 403s


def get_with_retry(
    session: requests.Session,
    url: str,
    timeout: float,
    label: str = "",
    **kwargs,
) -> requests.Response:
    """
    GET url, retrying on 429/503 up to MAX_RETRIES times.
    Respects the Retry-After response header (seconds); falls back to
    exponential backoff (2, 4, 8, 16, 32 s) if the header is absent or unparseable.
    Raises requests.HTTPError on any other non-2xx status or after all retries exhausted.
    """
    for attempt in range(MAX_RETRIES + 1):
        resp = session.get(url, timeout=timeout, **kwargs)
        if resp.status_code not in RETRYABLE_STATUSES or attempt == MAX_RETRIES:
            resp.raise_for_status()
            return resp

        # Parse Retry-After (seconds integer or float; ignore HTTP-date form)
        retry_after = resp.headers.get("Retry-After", "")
        try:
            wait = float(retry_after)
        except ValueError:
            wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32

        context = f" ({label})" if label else ""
        print(
            f"  HTTP {resp.status_code}{context} — waiting {wait:.0f}s then retrying "
            f"({attempt + 1}/{MAX_RETRIES})…",
            file=sys.stderr,
        )
        time.sleep(wait)

    # Unreachable, but satisfies type checkers
    resp.raise_for_status()
    return resp  # type: ignore[return-value]


def _build_loc_manifest(session: requests.Session, manifest_url: str) -> "dict | None":
    """
    Build a synthetic IIIF Presentation v3 manifest from the LoC item JSON API.

    www.loc.gov/item/{id}/manifest.json is blocked by Cloudflare for non-browser
    clients.  The item JSON API (?fo=json) is not blocked and exposes the same
    IIIF Image API service URLs hosted on tile.loc.gov, which are also accessible.

    Returns None if the URL doesn't look like a LoC item manifest or if no
    IIIF service URLs are found in the item data.
    """
    if "loc.gov/item/" not in manifest_url:
        return None
    item_id = iiif_utils.manifest_item_id(manifest_url)
    resp = get_with_retry(
        session,
        f"https://www.loc.gov/item/{item_id}/?fo=json",
        timeout=30,
        label="loc-item",
    )
    data = resp.json()
    items = []
    for resource in data.get("resources", []):
        for page_files in resource.get("files", []):
            for f in page_files:
                info_url = f.get("info", "")
                if info_url.endswith("/info.json"):
                    svc_id = info_url[: -len("/info.json")]
                    n = len(items) + 1
                    canvas_id = f"https://www.loc.gov/item/{item_id}/canvas/{n}"
                    native_w = f.get("width", 0)
                    items.append({
                        "id": canvas_id,
                        "type": "Canvas",
                        "width": native_w,
                        "height": f.get("height", 0),
                        "items": [{
                            "id": f"{canvas_id}/page",
                            "type": "AnnotationPage",
                            "items": [{
                                "id": f"{canvas_id}/annotation",
                                "type": "Annotation",
                                "motivation": "painting",
                                "body": {
                                    "id": f"{svc_id}/full/max/0/default.jpg",
                                    "type": "Image",
                                    "service": [{"id": svc_id, "type": "ImageService3", "maxWidth": native_w}],
                                },
                                "target": canvas_id,
                            }],
                        }],
                    })
                    break  # one IIIF service per page
    if not items:
        return None
    return {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": manifest_url,
        "type": "Manifest",
        "label": {"en": [item_id]},
        "items": items,
    }


def fetch_manifest(
    session: requests.Session,
    manifest_url: str,
    cache_path: "Path | None" = None,
) -> dict:
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    try:
        resp = get_with_retry(session, manifest_url, timeout=30, label="manifest")
        data = resp.json()
    except requests.HTTPError as exc:
        if exc.response.status_code == 403:
            # LoC manifest endpoints are blocked by Cloudflare; fall back to
            # building a synthetic manifest from the item JSON API.
            data = _build_loc_manifest(session, manifest_url)
            if data is None:
                raise
        else:
            raise
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def iter_canvas_images(manifest: dict, width: int) -> list[tuple[str, str, int | None]]:
    """
    Return ordered list of (image_id, url, max_width) for all canvases.
    Handles both IIIF Presentation v2 and v3 via iiif_utils.
    Caps the requested width at the service's maxWidth when advertised, so that
    servers which only hold a fixed native resolution (e.g. LoC tile servers)
    are not asked to upscale from lower-quality tiles.
    """
    result = []
    for c in iiif_utils.iter_canvases(manifest):
        effective_width = min(width, c["max_width"]) if c["max_width"] else width
        result.append((c["image_id"], iiif_utils.image_url(c["service_id"], effective_width), c["max_width"]))
    return result


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    delay: float,
) -> bool:
    """Download url to dest. Returns True if downloaded, False if skipped."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = get_with_retry(session, url, timeout=60, label=dest.name, stream=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    time.sleep(delay)
    return True


def _download_item_images(
    canvas_images: list,
    item_dir: Path,
    session: requests.Session,
    width: int,
    delay: float,
    quiet: bool,
) -> tuple[int, int]:
    """Download canvas images for one item. Returns (n_downloaded, n_skipped)."""
    total_downloaded = 0
    total_skipped = 0
    consecutive_primary_403 = 0

    for i, (image_id, primary_url, max_width) in enumerate(canvas_images, start=1):
        dest = item_dir / f"{i:04d}_{image_id}.jpg"
        fallback = (
            f"{FALLBACK_IIIF_BASE}/{image_id}/full/{FALLBACK_SIZE}/0/default.jpg"
        )

        primary_ok = False
        need_fallback = consecutive_primary_403 >= FALLBACK_403_THRESHOLD

        if not need_fallback:
            try:
                downloaded = download_file(session, primary_url, dest, delay)
                consecutive_primary_403 = 0
                primary_ok = True
                if downloaded:
                    total_downloaded += 1
                    if not quiet:
                        print(f"    [{i:04d}] Downloaded → {dest}", file=sys.stderr)
                else:
                    total_skipped += 1
                    if not quiet:
                        print(f"    [{i:04d}] Skipped (already exists)", file=sys.stderr)
            except requests.HTTPError as exc:
                status = exc.response.status_code
                if status == 403:
                    consecutive_primary_403 += 1
                    need_fallback = True
                    if consecutive_primary_403 >= FALLBACK_403_THRESHOLD:
                        print(
                            f"    [{i:04d}] HTTP 403 "
                            f"(×{consecutive_primary_403}) — "
                            f"switching to fallback URL for remaining pages.",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"    [{i:04d}] HTTP 403 on primary — trying fallback URL.",
                            file=sys.stderr,
                        )
                elif status == 400:
                    msg = (
                        f"    [{i:04d}] Warning: HTTP 400 — server rejected "
                        f"width={width}px (likely exceeds server limit)."
                    )
                    if max_width:
                        msg += f" Manifest reports max width: {max_width}px. Try: --width {max_width}"
                    else:
                        msg += " Try a smaller --width, e.g. --width 2048"
                    print(msg, file=sys.stderr)
                else:
                    print(
                        f"    [{i:04d}] Warning: HTTP {status} — {primary_url}",
                        file=sys.stderr,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"    [{i:04d}] Warning: {exc} — {primary_url}", file=sys.stderr)

        if need_fallback and not primary_ok:
            try:
                downloaded = download_file(session, fallback, dest, delay)
                if downloaded:
                    total_downloaded += 1
                    if not quiet:
                        print(
                            f"    [{i:04d}] Downloaded (fallback) → {dest}",
                            file=sys.stderr,
                        )
                else:
                    total_skipped += 1
                    if not quiet:
                        print(f"    [{i:04d}] Skipped (already exists)", file=sys.stderr)
            except requests.HTTPError as exc:
                print(
                    f"    [{i:04d}] Warning: fallback HTTP {exc.response.status_code}"
                    f" — {fallback}",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    [{i:04d}] Warning: fallback failed: {exc}", file=sys.stderr)

    return total_downloaded, total_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download IIIF images for items listed in a collection CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=None,
        help="Path to the collection CSV (e.g. collection_csv/travelguide.csv)",
    )
    parser.add_argument(
        "--manifest", "-M",
        default=None,
        metavar="URL",
        help=(
            "Download a single IIIF manifest directly — no CSV required. "
            "Accepts any public IIIF Presentation v2 or v3 manifest URL."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=(
            "Directory to write images into. "
            "Defaults to images/<csv-stem>/ (CSV mode) or images/<item-id>/ (manifest mode)."
        ),
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=DEFAULT_WIDTH,
        metavar="PX",
        help=f"Image width in pixels (default: {DEFAULT_WIDTH}). Height is auto-scaled.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        metavar="SECS",
        help=f"Seconds to wait between requests (default: {DEFAULT_DELAY}, ~2 req/s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a previously started download, skipping files that already exist. "
            "Without this flag the script exits if the output directory already contains images."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # Validate: exactly one input mode must be specified
    if args.manifest and args.csv_file:
        print("Error: provide either a CSV file or --manifest, not both.", file=sys.stderr)
        sys.exit(1)
    if not args.manifest and not args.csv_file:
        parser.print_help(sys.stderr)
        sys.exit(1)

    session = requests.Session()
    session.headers["Accept"] = "application/json"

    # ── Manifest mode ──────────────────────────────────────────────────────
    if args.manifest:
        item_id = iiif_utils.manifest_item_id(args.manifest)
        out_root = Path(args.output_dir) if args.output_dir else Path("images") / item_id

        if out_root.exists():
            existing = list(out_root.rglob("*.jpg"))
            if existing:
                if not args.resume:
                    print(
                        f"Error: {out_root} already contains {len(existing)} image(s).\n"
                        f"  Use --resume to continue/skip existing files, or delete the "
                        f"directory to start fresh.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                elif not args.quiet:
                    print(
                        f"Resuming — {len(existing)} image(s) already in {out_root} will be skipped.",
                        file=sys.stderr,
                    )

        manifest_cache = out_root / "manifest.json"
        if not args.quiet:
            if manifest_cache.exists():
                print(f"Loading manifest from cache…", file=sys.stderr)
            else:
                print(f"Fetching manifest: {args.manifest}", file=sys.stderr)

        try:
            manifest = fetch_manifest(session, args.manifest, cache_path=manifest_cache)
            time.sleep(args.delay)
        except requests.HTTPError as exc:
            print(f"Error: HTTP {exc.response.status_code} fetching manifest.", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: could not fetch manifest: {exc}", file=sys.stderr)
            sys.exit(1)

        canvas_images = iter_canvas_images(manifest, args.width)
        if not canvas_images:
            print("No images found in manifest.", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"{len(canvas_images)} page(s) to download.", file=sys.stderr)

        total_downloaded, total_skipped = _download_item_images(
            canvas_images, out_root, session, args.width, args.delay, args.quiet,
        )

        if not args.quiet:
            print(
                f"\nDone. {total_downloaded} downloaded, {total_skipped} skipped.",
                file=sys.stderr,
            )
        return

    # ── CSV mode ───────────────────────────────────────────────────────────
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = Path("images") / csv_path.stem

    # Guard against accidentally re-downloading a completed collection.
    if out_root.exists():
        existing = list(out_root.rglob("*.jpg"))
        if existing:
            if not args.resume:
                print(
                    f"Error: {out_root} already contains {len(existing)} image(s).\n"
                    f"  Use --resume to continue/skip existing files, or delete the "
                    f"directory to start fresh.",
                    file=sys.stderr,
                )
                sys.exit(1)
            elif not args.quiet:
                print(
                    f"Resuming — {len(existing)} image(s) already in {out_root} will be skipped.",
                    file=sys.stderr,
                )

    total_downloaded = 0
    total_skipped = 0
    items_processed = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        item_id = row.get("item_id", "").strip()
        manifest_url = row.get("iiif_manifest_url", "").strip()
        item_title = row.get("item_title", "").strip() or item_id

        if not manifest_url or not item_id:
            continue

        item_dir = out_root / item_id
        manifest_cache = item_dir / "manifest.json"

        if not args.quiet:
            print(f"\n[{items_processed + 1}/{len(rows)}] {item_title} ({item_id})",
                  file=sys.stderr)
            if manifest_cache.exists():
                print(f"  Loading manifest from cache…", file=sys.stderr)
            else:
                print(f"  Fetching manifest…", file=sys.stderr)

        try:
            manifest = fetch_manifest(session, manifest_url, cache_path=manifest_cache)
            time.sleep(args.delay)
        except requests.HTTPError as exc:
            print(f"  Warning: HTTP {exc.response.status_code} fetching manifest, skipping.",
                  file=sys.stderr)
            items_processed += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not fetch manifest: {exc}, skipping.", file=sys.stderr)
            items_processed += 1
            continue

        canvas_images = iter_canvas_images(manifest, args.width)
        if not canvas_images:
            if not args.quiet:
                print("  No images found in manifest.", file=sys.stderr)
            items_processed += 1
            continue

        if not args.quiet:
            print(f"  {len(canvas_images)} page(s) to download.", file=sys.stderr)

        dl, sk = _download_item_images(
            canvas_images, item_dir, session, args.width, args.delay, args.quiet,
        )
        total_downloaded += dl
        total_skipped += sk
        items_processed += 1

    if not args.quiet:
        print(
            f"\nDone. {items_processed} item(s) processed: "
            f"{total_downloaded} downloaded, {total_skipped} skipped.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
