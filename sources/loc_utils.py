"""Library of Congress helpers for URL normalization, title fetching, and slug generation."""

import re
import sys

import requests


def _resource_url_to_item_url(resource_url: str) -> str:
    """Convert a LoC /resource/ URL to its canonical /item/ URL via the JSON API.

    https://www.loc.gov/resource/rbc0001.2026batch96169559/?sp=74
        → https://www.loc.gov/item/rbc0001.2026batch96169559/

    The resource JSON response includes a 'related_items' or 'item' field that
    points back to the catalog item.  Falls back to substituting /resource/ with
    /item/ in the URL path, which works for many LoC identifiers.
    """
    if "/resource/" not in resource_url:
        return resource_url
    # Strip ?sp= page parameter before querying the API
    base = re.sub(r"\?.*", "", resource_url).rstrip("/")
    try:
        resp = requests.get(f"{base}?fo=json", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # The JSON 'item' key usually contains a dict with an 'id' or 'link' field
        item = data.get("item") or {}
        link = item.get("id") or item.get("link") or ""
        if link and "/item/" in link:
            # Normalise to https and strip trailing query/fragment
            link = re.sub(r"^http:", "https:", link.split("?")[0].split("#")[0])
            return link.rstrip("/") + "/"
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not resolve LoC resource URL {resource_url}: {exc}", file=sys.stderr)
    # Fallback: swap /resource/ → /item/ in the path
    return re.sub(r"/resource/", "/item/", resource_url.split("?")[0]).rstrip("/") + "/"


def _fetch_loc_title(item_id: str) -> str:
    """Best-effort fetch of a LoC item title from the public JSON API."""
    try:
        resp = requests.get(
            f"https://www.loc.gov/item/{item_id}/?fo=json",
            timeout=15,
        )
        resp.raise_for_status()
        v = resp.json().get("item", {}).get("title", "")
        if isinstance(v, list):
            return str(v[0]).strip() if v else ""
        return str(v).strip() if v else ""
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not fetch LoC title for {item_id}: {exc}", file=sys.stderr)
        return ""


def _make_loc_slug(title: str, item_id: str) -> str:
    """Build a slug from a LoC title and item ID.

    Mirrors _make_slug() in loc_collection_csv.py so both scripts
    produce the same slug for the same item.

    ('The Brooklyn city directory', '01015253')
        → 'the_brooklyn_city_directory_01015253'
    """
    id_suffix = item_id[:12]
    if title:
        sanitized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
        if sanitized:
            return f"{sanitized}_{id_suffix}"
    return id_suffix


def loc_slug(url: str) -> str:
    """Derive a filesystem-safe slug from a LoC URL.

    For item URLs, fetches the item title from the LoC JSON API so the
    slug matches what loc_collection_csv.py produces by default:
        /item/01015253/              → 'the_brooklyn_city_directory_01015253'
        /collections/civil-war-maps/ → 'civil-war-maps'

    For Chronicling America issue URLs (/item/{lccn}/{date}/ed-{edition}/)
    the date and edition are appended so each issue gets a unique slug:
        /item/sn83030313/1847-05-01/ed-1/ → 'the_new_york_herald_..._sn83030313_1847_05_01_ed1'
    """
    m = re.search(r"/collections/([^/?#]+)", url)
    if m:
        return m.group(1).rstrip("/")
    # Chronicling America: /item/{lccn}/{date}/ed-{edition}/
    m = re.search(r"/(?:item|resource)/([^/?#]+)/(\d{4}-\d{2}-\d{2})/ed-(\d+)", url)
    if m:
        lccn, date, edition = m.group(1), m.group(2), m.group(3)
        title = _fetch_loc_title(lccn)
        base = _make_loc_slug(title, lccn)
        date_slug = date.replace("-", "_")
        return f"{base}_{date_slug}_ed{edition}"
    m = re.search(r"/(?:item|resource)/([^/?#]+)", url)
    if m:
        item_id = m.group(1).rstrip("/")
        title = _fetch_loc_title(item_id)
        return _make_loc_slug(title, item_id)
    return re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")[:40]
