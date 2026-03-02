#!/usr/bin/env python3
"""Geocode entries from an entries CSV and write a geocoded CSV.

Reads  entries_{slug}.csv
Writes entries_{slug}_geocoded.csv  — all original fields plus:
    lat            float | empty
    lon            float | empty
    geocode_level  "address" | "city" | ""   (empty = no geocode found)

Geocoding strategy per entry:
    1. raw_address present + GOOGLE_MAPS_API_KEY set
       → Google Maps Geocoding API  (address + city + state)
    2. Google fails, address missing, or no API key
       → Nominatim  (city + state centroid)
    3. Both fail
       → lat/lon left blank

All results are cached in geocache.json (same file map_entries.py uses),
keyed by the query string.  Only new queries hit the network; re-runs
over the same data are instant.

Usage
-----
    python geocode_entries.py images/green_book_1962_9ab2e8f0/ --model gemini-2.0-flash
    python geocode_entries.py path/to/entries_gemini-2.0-flash.csv
    GOOGLE_MAPS_API_KEY=... python geocode_entries.py ...
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from geopy.geocoders import Nominatim, GoogleV3

load_dotenv()
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict, path: Path) -> None:
    path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _city_key(city: str, state: str) -> str:
    city  = city.strip().title()
    state = state.strip().split(",")[0].strip().title()
    return f"{city}, {state}"


def _address_key(address: str, city: str, state: str) -> str:
    city  = city.strip().title()
    state = state.strip().split(",")[0].strip().title()
    return f"{address.strip()}, {city}, {state}"


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def _geocode_google(
    keys: list[str],
    cache: dict,
    api_key: str,
    delay: float,
) -> None:
    """Geocode address-level keys via Google Maps, updating cache in-place."""
    needed = [k for k in keys if k not in cache]
    if not needed:
        return

    google = GoogleV3(api_key=api_key)
    n_resolved = 0
    print(
        f"Google geocoding {len(needed)} address(es) …",
        file=sys.stderr,
    )
    for i, key in enumerate(needed, 1):
        result = None
        try:
            loc = google.geocode(key, exactly_one=True, timeout=10)
            if loc:
                result = [round(loc.latitude, 7), round(loc.longitude, 7)]
                n_resolved += 1
        except Exception as exc:
            print(f"  [{i}/{len(needed)}] Google error: {exc}", file=sys.stderr)
        cache[key] = result
        if i % 100 == 0:
            print(f"  [{i}/{len(needed)}] {n_resolved} resolved so far …",
                  file=sys.stderr)
        time.sleep(delay)

    print(
        f"  Google: {n_resolved}/{len(needed)} resolved.",
        file=sys.stderr,
    )


def _geocode_nominatim(
    keys: list[str],
    cache: dict,
    delay: float,
) -> None:
    """Geocode city-level keys via Nominatim, updating cache in-place."""
    needed = [k for k in keys if k not in cache]
    if not needed:
        return

    geolocator = Nominatim(user_agent="greenbook-geocode/1.0")
    n_resolved = 0
    print(
        f"Nominatim geocoding {len(needed)} city/state pair(s) …",
        file=sys.stderr,
    )
    for i, key in enumerate(needed, 1):
        result = None
        for attempt in range(3):
            try:
                loc = geolocator.geocode(key, exactly_one=True, timeout=10)
                if loc:
                    result = [round(loc.latitude, 7), round(loc.longitude, 7)]
                    n_resolved += 1
                break
            except (GeocoderTimedOut, GeocoderServiceError):
                time.sleep(delay * (attempt + 1))
        if result is None:
            print(f"  [{i}/{len(needed)}] NOT FOUND: {key!r}", file=sys.stderr)
        cache[key] = result
        time.sleep(delay)

    print(
        f"  Nominatim: {n_resolved}/{len(needed)} resolved.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Main geocoding logic
# ---------------------------------------------------------------------------

def geocode_rows(
    rows: list[dict],
    cache_path: Path,
    google_api_key: str | None,
    google_delay: float = 0.05,
    nominatim_delay: float = 1.1,
) -> list[dict]:
    """Geocode every row.  Returns rows with lat, lon, geocode_level added."""
    cache = _load_cache(cache_path)

    # Collect keys that need geocoding
    address_keys: list[str] = []
    city_keys:    list[str] = []

    for row in rows:
        addr  = (row.get("raw_address") or "").strip()
        city  = row.get("city", "")
        state = row.get("state", "")
        if addr and google_api_key:
            address_keys.append(_address_key(addr, city, state))
        city_keys.append(_city_key(city, state))

    # Deduplicate
    address_keys = list(dict.fromkeys(address_keys))
    city_keys    = list(dict.fromkeys(city_keys))

    # Geocode new keys
    if google_api_key:
        _geocode_google(address_keys, cache, google_api_key, google_delay)
    _geocode_nominatim(city_keys, cache, nominatim_delay)
    _save_cache(cache, cache_path)

    # Annotate rows
    out = []
    counts = {"address": 0, "city": 0, "none": 0}

    for row in rows:
        addr  = (row.get("raw_address") or "").strip()
        city  = row.get("city", "")
        state = row.get("state", "")

        coords = None
        level  = ""

        if addr and google_api_key:
            coords = cache.get(_address_key(addr, city, state))
            if coords:
                level = "address"

        if not coords:
            coords = cache.get(_city_key(city, state))
            if coords:
                level = "city"

        if coords:
            counts[level] += 1
        else:
            counts["none"] += 1

        out.append({
            **row,
            "lat":           coords[0] if coords else "",
            "lon":           coords[1] if coords else "",
            "geocode_level": level,
        })

    total = len(out)
    print(
        f"\nResults: {counts['address']} address-level, "
        f"{counts['city']} city-level, "
        f"{counts['none']} unresolved  "
        f"(total {total})",
        file=sys.stderr,
    )
    return out


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _find_csv(path: Path, slug: str) -> Path:
    if path.is_file():
        return path
    hits = sorted(path.rglob(f"entries_{slug}.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"No entries_{slug}.csv found under {path}. "
        f"Run extract_entries.py first."
    )


def write_geocoded_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geocode entries CSV and write results with lat/lon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help="Path to an entries CSV file, or a directory containing one.",
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.0-flash",
        metavar="MODEL",
        help="Model slug used in the CSV filename (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--out", "-o",
        metavar="FILE",
        help=(
            "Output geocoded CSV path "
            "(default: entries_{slug}_geocoded.csv next to the input CSV)"
        ),
    )
    parser.add_argument(
        "--cache",
        metavar="FILE",
        help="Geocode cache JSON path (default: geocache.json next to the CSV)",
    )
    parser.add_argument(
        "--google-delay",
        type=float,
        default=0.05,
        metavar="SEC",
        help="Delay between Google API requests in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--nominatim-delay",
        type=float,
        default=1.1,
        metavar="SEC",
        help="Delay between Nominatim requests in seconds (default: 1.1)",
    )
    args = parser.parse_args()

    google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if google_api_key:
        print("Google Maps API key found — will use for address geocoding.",
              file=sys.stderr)
    else:
        print(
            "GOOGLE_MAPS_API_KEY not set — falling back to Nominatim "
            "city-level geocoding for all entries.",
            file=sys.stderr,
        )

    slug       = args.model.replace("/", "_")
    csv_path   = _find_csv(Path(args.source), slug)
    out_path   = (
        Path(args.out) if args.out
        else csv_path.parent / csv_path.name.replace(
            f"_{slug}.csv", f"_{slug}_geocoded.csv"
        )
    )
    cache_path = (
        Path(args.cache) if args.cache
        else csv_path.parent / "geocache.json"
    )

    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"Loaded {len(rows)} entries from {csv_path.name}", file=sys.stderr)

    geocoded = geocode_rows(
        rows, cache_path, google_api_key,
        google_delay=args.google_delay,
        nominatim_delay=args.nominatim_delay,
    )

    write_geocoded_csv(geocoded, out_path)
    print(f"Saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
