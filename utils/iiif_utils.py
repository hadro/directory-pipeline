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

import json
import urllib.request
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


# ---------------------------------------------------------------------------
# Viewer-ready manifest normalization
# ---------------------------------------------------------------------------

def normalize_for_viewer(
    manifest: dict,
    hosted_manifest_url: str,
    *,
    session=None,
) -> tuple[dict, dict[str, str]]:
    """Normalize a raw NYPL (or similar) IIIF Presentation v3 manifest so that
    strict viewers (e.g. Clover IIIF) can render it.

    Problems fixed
    --------------
    * ``manifest["id"]`` — replaced with *hosted_manifest_url* (the URL at which
      this file will actually be served; many NYPL manifests point to the
      originating API endpoint instead).
    * Canvas IDs — NYPL manifests use image-API URLs as canvas IDs, making
      them indistinguishable from the image body.  Replaced with proper
      ``{hosted_manifest_url}/canvas/{n}`` URIs.
    * AnnotationPage / Annotation IDs within the manifest — rekeyed to match
      the new canvas IDs.
    * ``motivation: "painting"`` — added to every painting annotation that
      lacks it (required by IIIF Presentation 3).
    * Image body ``width`` / ``height`` / ``format`` — populated from the IIIF
      Image API ``info.json`` endpoint when missing.
    * Canvas ``width`` / ``height`` — updated to match image dimensions when
      they differ (NYPL manifests often declare 2560×2560 for all canvases).
    * ``structures`` canvas references — updated to the new canvas IDs.
    * Top-level ``services`` — removed (NYPL search services are dead and some
      viewers block on fetching them).

    Parameters
    ----------
    manifest:
        A parsed IIIF Presentation v3 manifest dict.  Modified **in-place**
        and also returned.
    hosted_manifest_url:
        The URL at which *manifest* will be served, e.g.
        ``"https://example.org/iiif/manifest.json"``.
    session:
        Optional ``requests.Session`` for info.json fetches.  If *None*,
        ``urllib.request`` is used.

    Returns
    -------
    (manifest, old_to_new)
        *manifest* is the normalised dict.
        *old_to_new* maps each original canvas ID → new canvas ID so callers
        can update external annotation files via
        :func:`update_annotation_targets`.
    """
    hosted_manifest_url = hosted_manifest_url.rstrip("/")
    manifest["id"] = hosted_manifest_url

    # Remove dead/blocking services
    manifest.pop("services", None)

    # Build old→new canvas ID map and patch each canvas
    old_to_new: dict[str, str] = {}
    for seq, canvas in enumerate(manifest.get("items", [])):
        old_id: str = canvas.get("id", "")
        new_canvas_id = f"{hosted_manifest_url}/canvas/{seq}"
        old_to_new[old_id] = new_canvas_id
        canvas["id"] = new_canvas_id

        # Fetch real image dimensions from info.json when body is missing them
        try:
            body = canvas["items"][0]["items"][0]["body"]
        except (KeyError, IndexError, TypeError):
            body = None

        if body:
            svc_id = _svc_id(body.get("service", ""))
            if svc_id and (not body.get("width") or not body.get("height")):
                dims = _fetch_image_dims(svc_id, session=session)
                if dims:
                    body["width"], body["height"] = dims
                    canvas["width"], canvas["height"] = dims
            elif body.get("width") and body.get("height"):
                # Correct canvas dims even if body already has them
                canvas["width"] = body["width"]
                canvas["height"] = body["height"]
            if not body.get("format"):
                body["format"] = "image/jpeg"

        # Fix annotation page / annotation IDs and targets within the canvas
        for ap_idx, ann_page in enumerate(canvas.get("items", [])):
            ann_page["id"] = f"{new_canvas_id}/page/{ap_idx}"
            for ann_idx, ann in enumerate(ann_page.get("items", [])):
                ann["id"] = f"{new_canvas_id}/page/{ap_idx}/annotation/{ann_idx}"
                ann.setdefault("motivation", "painting")
                ann["target"] = new_canvas_id

    # Update structures canvas references
    def _fix_node(node):
        if isinstance(node, dict):
            if node.get("type") == "Canvas" and node.get("id") in old_to_new:
                node["id"] = old_to_new[node["id"]]
            for v in node.values():
                _fix_node(v)
        elif isinstance(node, list):
            for item in node:
                _fix_node(item)

    _fix_node(manifest.get("structures", []))

    return manifest, old_to_new


def _fetch_image_dims(
    service_id: str,
    *,
    session=None,
) -> tuple[int, int] | None:
    """Fetch width/height from a IIIF Image API info.json.  Returns None on failure."""
    url = service_id.rstrip("/") + "/info.json"
    try:
        if session is not None:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            info = resp.json()
        else:
            with urllib.request.urlopen(url, timeout=15) as resp:
                info = json.loads(resp.read())
        w = int(info.get("width", 0))
        h = int(info.get("height", 0))
        return (w, h) if w and h else None
    except Exception:
        return None


def update_annotation_targets(
    ann_data: dict,
    canvas_map: dict[str, str],
) -> dict:
    """Rewrite canvas ID references in an annotation page dict.

    Annotation targets take the form ``{canvas_id}`` or
    ``{canvas_id}#xywh=x,y,w,h``.  This replaces the canvas ID prefix using
    *canvas_map* (old → new), preserving any ``#fragment`` suffix.

    Modifies *ann_data* in-place and returns it.
    """
    for ann in ann_data.get("items", []):
        target = ann.get("target", "")
        if not target:
            continue
        # Split off any fragment (#xywh=…)
        fragment = ""
        if "#" in target:
            target, fragment = target.split("#", 1)
            fragment = "#" + fragment
        new_canvas = canvas_map.get(target)
        if new_canvas:
            ann["target"] = new_canvas + fragment
    return ann_data


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
