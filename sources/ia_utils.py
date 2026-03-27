"""Internet Archive helpers for identifier extraction, metadata fetching, and slug generation."""

import re
import sys

import requests

_IA_IDENTIFIER_RE = re.compile(r"archive\.org/details/([^/?#]+)")
_IA_METADATA_API  = "https://archive.org/metadata/{identifier}"


def _extract_ia_identifier(url: str) -> str | None:
    m = _IA_IDENTIFIER_RE.search(url)
    return m.group(1).rstrip("/") if m else None


def _fetch_ia_info(identifier: str) -> tuple[str, str]:
    """Return (title, kind) for an IA identifier.

    kind is "ia-collection" or "ia-item".  Returns ("", "ia-item") on failure.
    """
    try:
        resp = requests.get(
            _IA_METADATA_API.format(identifier=identifier),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("metadata", {})
        title = meta.get("title", "")
        if isinstance(title, list):
            title = title[0] if title else ""
        mediatype = meta.get("mediatype", "")
        kind = "ia-collection" if mediatype == "collection" else "ia-item"
        return str(title).strip(), kind
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not fetch IA metadata for {identifier}: {exc}", file=sys.stderr)
        return "", "ia-item"


def _make_ia_slug(title: str, identifier: str) -> str:
    """Build a filesystem-safe slug from an IA title and identifier."""
    id_part = identifier[:20]
    if title:
        sanitized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
        if sanitized:
            return f"{sanitized}_{id_part}"
    return id_part
