"""NYPL-specific helpers for slug generation and title fetching via the NYPL API."""

import re

import requests

API_BASE = "https://api.repo.nypl.org/api/v2"


def _ensure_list(obj) -> list:
    """Coerce *obj* to a list: None → [], scalar → [scalar], list → list."""
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]


def _unwrap_text(obj) -> str:
    """Unwrap a {'$': 'value'} MODS node or coerce to str."""
    if isinstance(obj, dict):
        return str(obj.get("$", ""))
    return str(obj) if obj else ""


def fetch_title(session: requests.Session, uuid: str, item: bool) -> str:
    """Best-effort fetch of a human-readable title for a UUID from the NYPL API.

    Returns an empty string on any failure — callers fall back to UUID-only slugs.
    """
    try:
        if item:
            resp = session.get(
                f"{API_BASE}/items/item_details/{uuid}",
                params={"page": 1, "per_page": 1},
                timeout=15,
            )
            resp.raise_for_status()
            mods = (
                resp.json().get("nyplAPI", {}).get("response", {}).get("mods") or {}
            )
            for ti in _ensure_list(mods.get("titleInfo")):
                t = _unwrap_text(ti.get("title") if isinstance(ti, dict) else None)
                if t:
                    return t
        else:
            resp = session.get(
                f"{API_BASE}/collections/{uuid}",
                params={"page": 1, "per_page": 1},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("nyplAPI", {}).get("response", {})
            # Try top-level keys first
            candidates = [
                data.get("title"),
                data.get("heading"),
                (data.get("collection") or {}).get("title"),
            ]
            for candidate in candidates:
                t = _unwrap_text(candidate)
                if t:
                    return t
            # Fall back to mods structure (some collection responses embed mods)
            mods = data.get("mods") or {}
            for ti in _ensure_list(mods.get("titleInfo")):
                t = _unwrap_text(ti.get("title") if isinstance(ti, dict) else None)
                if t:
                    return t
            # Last resort: item_details endpoint — always has full mods with title
            resp2 = session.get(
                f"{API_BASE}/items/item_details/{uuid}",
                params={"page": 1, "per_page": 1},
                timeout=15,
            )
            resp2.raise_for_status()
            mods2 = (
                resp2.json().get("nyplAPI", {}).get("response", {}).get("mods") or {}
            )
            for ti in _ensure_list(mods2.get("titleInfo")):
                t = _unwrap_text(ti.get("title") if isinstance(ti, dict) else None)
                if t:
                    return t
    except Exception as exc:  # noqa: BLE001
        import sys
        print(f"Warning: could not fetch title for {uuid}: {exc}", file=sys.stderr)
    return ""


def make_slug(title: str, uuid: str) -> str:
    """Build a filesystem-safe slug: {title_words}_{uuid8}.

    Falls back to just {uuid8} if no usable title is available.
    """
    uuid8 = uuid.replace("-", "")[:8]
    if title:
        sanitized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
        if sanitized:
            return f"{sanitized}_{uuid8}"
    return uuid8
