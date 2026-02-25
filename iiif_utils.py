"""IIIF Presentation manifest parsing — versions 2 and 3.

Both download_images.py and align_ocr.py need to walk IIIF manifests to find
image service URLs, canvas IDs, and dimensions.  This module provides a single
version-agnostic interface for both.

IIIF Presentation v3 canvas path
---------------------------------
  manifest["items"]
    → canvas["items"][0]["items"][0]["body"]["service"][0]["id"]

IIIF Presentation v2 canvas path
---------------------------------
  manifest["sequences"][0]["canvases"]
    → canvas["images"][0]["resource"]["service"]["@id"]
  (service may be a dict, a list, or a plain string URL)

Image API URL construction
--------------------------
  {service_id}/full/{width},/0/default.jpg
works for both IIIF Image API v2 and v3.
"""

from __future__ import annotations

from typing import Iterator


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------

def manifest_version(manifest: dict) -> int:
    """Return 2 or 3 based on @context, defaulting to 3."""
    ctx = manifest.get("@context", "")
    if isinstance(ctx, list):
        ctx = " ".join(str(c) for c in ctx)
    if "/presentation/2" in ctx or (
        not manifest.get("items") and manifest.get("sequences")
    ):
        return 2
    return 3


# ---------------------------------------------------------------------------
# Service URL extraction (handles dict / list / plain string)
# ---------------------------------------------------------------------------

def _svc_id(service) -> str:
    """
    Extract the IIIF Image Service base URL from a service value that may be:
      - a dict  with "id" or "@id"
      - a list  of such dicts
      - a plain string URL
    Returns '' if nothing usable is found.
    """
    if isinstance(service, list):
        service = service[0] if service else {}
    if isinstance(service, dict):
        return str(service.get("id") or service.get("@id") or "").rstrip("/")
    if isinstance(service, str):
        return service.rstrip("/")
    return ""


def _svc_max_width(service) -> int | None:
    """Return maxWidth from a service object/list if the server advertises it."""
    if isinstance(service, list):
        service = service[0] if service else {}
    if isinstance(service, dict):
        v = service.get("maxWidth")
        return int(v) if v is not None else None
    return None


# ---------------------------------------------------------------------------
# Per-version canvas iterators
# ---------------------------------------------------------------------------

def _iter_v3(manifest: dict) -> Iterator[dict]:
    for canvas in manifest.get("items", []):
        canvas_id = str(canvas.get("id", ""))
        canvas_w = int(canvas.get("width") or 0)
        canvas_h = int(canvas.get("height") or 0)
        try:
            body = canvas["items"][0]["items"][0]["body"]
        except (KeyError, IndexError, TypeError):
            continue
        service = body.get("service")
        if not service:
            continue
        svc_id = _svc_id(service)
        if not svc_id:
            continue
        yield {
            "canvas_id": canvas_id,
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
            "service_id": svc_id,
            "image_id": svc_id.rsplit("/", 1)[-1],
            "max_width": _svc_max_width(service),
        }


def _iter_v2(manifest: dict) -> Iterator[dict]:
    for sequence in manifest.get("sequences", []):
        for canvas in sequence.get("canvases", []):
            canvas_id = str(canvas.get("@id", ""))
            canvas_w = int(canvas.get("width") or 0)
            canvas_h = int(canvas.get("height") or 0)
            images = canvas.get("images", [])
            if not images:
                continue
            resource = images[0].get("resource", {})
            service = resource.get("service")
            if not service:
                # Some v2 resources embed the image URL directly without a service
                # block; skip them — there's no service ID to build a URL from.
                continue
            svc_id = _svc_id(service)
            if not svc_id:
                continue
            yield {
                "canvas_id": canvas_id,
                "canvas_width": canvas_w,
                "canvas_height": canvas_h,
                "service_id": svc_id,
                "image_id": svc_id.rsplit("/", 1)[-1],
                "max_width": _svc_max_width(service),
            }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def iter_canvases(manifest: dict) -> Iterator[dict]:
    """
    Yield one dict per canvas in the manifest, handling both v2 and v3.

    Each dict contains:
        canvas_id     str       canvas URI  (@id in v2, id in v3)
        canvas_width  int       canvas width in pixels  (0 if unknown)
        canvas_height int       canvas height in pixels (0 if unknown)
        service_id    str       IIIF Image Service base URL (no trailing slash)
        image_id      str       last path segment of service_id
        max_width     int|None  server-advertised max width, if any
    """
    if manifest_version(manifest) == 2:
        yield from _iter_v2(manifest)
    else:
        yield from _iter_v3(manifest)


def image_url(service_id: str, width: int) -> str:
    """
    Construct a IIIF Image API download URL for the requested pixel width.
    Works for both Image API v2 and v3.
    """
    return f"{service_id}/full/{width},/0/default.jpg"


def manifest_item_id(manifest_url: str) -> str:
    """
    Extract a usable item identifier from a manifest URL for use as a
    directory or file name.

        'https://www.loc.gov/item/01015253/manifest.json' → '01015253'
        'https://api-collections.nypl.org/manifests/abc-def-123' → 'abc-def-123'
    """
    url = manifest_url.rstrip("/")
    for suffix in ("/manifest.json", "/manifest"):
        if url.lower().endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rsplit("/", 1)[-1] or "manifest"
