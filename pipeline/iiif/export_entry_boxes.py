#!/usr/bin/env python3
"""Export entry-level IIIF annotations with category-colored bounding boxes.

Reads *_{model}_entries.json files and produces *_{model}_box_annotations.json
containing W3C Annotation Pages with SVG bounding boxes — one annotation per
entry, color-coded by category to match the visualize_entries.py output.

Each annotation uses motivation "commenting" with two selectors:
  • FragmentSelector (xywh) — tells the viewer which region to target
  • SvgSelector         — draws the filled, stroked rectangle

With --update-manifest, each canvas in manifest.json gains an "annotations"
property referencing its box annotation page, so IIIF viewers (Mirador, UV)
load the colored boxes automatically on open.  Requires --base-url.

The manifest canvas width/height are also corrected to match the natural image
dimensions (fetched from the IIIF image service info.json), because Mirador
maps annotation xywh coordinates directly to image pixel space.  NYPL manifests
declare canvases as 2560×2560 (square) but the actual images are portrait
(e.g. 3316×4513); using canvas coords in a square canvas places boxes in the
wrong position.

Color coding (matches visualize_entries.py palette)
----------------------------------------------------
  blue     — formal_accommodations   (Hotels, Motels)
  teal     — informal_accommodations (Tourist Homes)
  red      — eating_drinking         (Restaurants, Bars)
  purple   — barber_beauty           (Beauty Parlors, Barbers)
  amber    — service_station         (Service Stations)
  grey     — other
Advertisements receive a stroke twice as thick as regular entries.

Usage
-----
    python pipeline/iiif/export_entry_boxes.py output/green_book_1947_xxx/uuid/
    python pipeline/iiif/export_entry_boxes.py output/green_book_1947_xxx/uuid/ \\
        --model gemini-2.0-flash
    python pipeline/iiif/export_entry_boxes.py output/green_book_1947_xxx/uuid/ \\
        --base-url https://example.org/annotations \\
        --update-manifest
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import iiif_utils
from iiif.label_utils import (
    parse_ner_label_fields,
    build_entry_label,
    handle_missing_ner_prompt,
)


# ---------------------------------------------------------------------------
# Category palette — (fill rgba, stroke rgb, base stroke-width)
# Matches visualize_entries.py _PALETTE exactly.
# ---------------------------------------------------------------------------

_PALETTE: dict[str, tuple[str, str, int]] = {
    "formal_accommodations":   ("rgba(50,110,220,0.24)",  "rgb(40,90,200)",   3),
    "informal_accommodations": ("rgba(0,185,170,0.24)",   "rgb(0,155,140)",   3),
    "eating_drinking":         ("rgba(210,55,45,0.24)",   "rgb(185,35,25)",   3),
    "barber_beauty":           ("rgba(175,55,210,0.24)",  "rgb(150,35,185)",  3),
    "service_station":         ("rgba(195,148,0,0.24)",   "rgb(165,125,0)",   3),
    "other":                   ("rgba(110,110,130,0.24)", "rgb(85,85,105)",   3),
}
_DEFAULT_STYLE: tuple[str, str, int] = ("rgba(130,130,130,0.24)", "rgb(100,100,100)", 3)

_CATEGORY_LABELS: dict[str, str] = {
    "formal_accommodations":   "Hotels / Motels",
    "informal_accommodations": "Tourist Homes",
    "eating_drinking":         "Restaurants / Bars",
    "barber_beauty":           "Barber / Beauty",
    "service_station":         "Service Stations",
    "other":                   "Other",
}


# ---------------------------------------------------------------------------
# IIIF image service helpers
# ---------------------------------------------------------------------------

# Module-level cache so repeated calls for the same service are free.
_info_cache: dict[str, tuple[int, int]] = {}


def _get_natural_dims(canvas_id: str) -> tuple[int, int]:
    """Return (width, height) of the natural image by fetching info.json.

    Parses the IIIF Image API service base from *canvas_id* (an IIIF Image URL
    of the form ``https://host/prefix/identifier/region/size/rotation/quality.fmt``),
    then fetches ``{service_base}/info.json``.  Results are cached in-process.
    Returns (0, 0) on any error.
    """
    parts = urlparse(canvas_id)
    path_parts = parts.path.split("/")
    # IIIF Image API path: /{prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{fmt}
    # Service base = scheme + host + first 4 path segments (index 0-3, where 0 is "")
    if len(path_parts) < 5:
        return 0, 0
    service_base = f"{parts.scheme}://{parts.netloc}" + "/".join(path_parts[:4])
    if service_base in _info_cache:
        return _info_cache[service_base]
    info_url = f"{service_base}/info.json"
    try:
        with urllib.request.urlopen(info_url, timeout=15) as resp:
            info = json.loads(resp.read())
        w = int(info.get("width") or 0)
        h = int(info.get("height") or 0)
    except Exception as exc:
        print(f"  Warning: could not fetch {info_url}: {exc}", file=sys.stderr)
        w, h = 0, 0
    _info_cache[service_base] = (w, h)
    return w, h


# ---------------------------------------------------------------------------
# SVG + annotation builders
# ---------------------------------------------------------------------------

def _svg_rect(
    x: int, y: int, w: int, h: int,
    canvas_w: int, canvas_h: int,
    fill: str, stroke: str, stroke_width: int,
    is_ad: bool = False,
) -> str:
    """Return an inline SVG string with one colored rectangle."""
    sw = stroke_width * 2 if is_ad else stroke_width
    vb = f' viewBox="0 0 {canvas_w} {canvas_h}"' if canvas_w and canvas_h else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg"{vb}>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Entry body text (box-specific: appends category label)
# ---------------------------------------------------------------------------

def _entry_body_text(
    entry: dict,
    name_fields: list[str] | None = None,
    addr_fields: list[str] | None = None,
) -> str:
    """One-line human-readable label for an entry, with category appended.

    Delegates core name/address/city-state formatting to :func:`build_entry_label`
    from :mod:`label_utils`, then appends the Green Book category label if present.
    """
    label = build_entry_label(entry, name_fields, addr_fields)
    cat_label = _CATEGORY_LABELS.get(entry.get("category", ""), "")
    if cat_label:
        return f"{label} — [{cat_label}]" if label else f"[{cat_label}]"
    return label


def _box_annotation(ann_id: str, canvas_id: str, x: int, y: int, w: int, h: int,
                     canvas_w: int, canvas_h: int, entry: dict,
                     name_fields: list[str] | None = None,
                     addr_fields: list[str] | None = None) -> dict:
    cat = entry.get("category", "other")
    fill, stroke, sw = _PALETTE.get(cat, _DEFAULT_STYLE)
    is_ad = str(entry.get("is_advertisement", "")).strip().lower() in ("true", "1", "yes")

    ann: dict = {
        "type": "Annotation",
        "motivation": "commenting",
        "body": {
            "type": "TextualBody",
            "value": _entry_body_text(entry, name_fields, addr_fields),
            "format": "text/plain",
        },
        # Use the simplest possible target format: plain canvas URI fragment.
        # Viewers (Mirador 3, UV) handle this most reliably.  The SvgSelector
        # approach caused silent rendering failures in Mirador 3.3.
        "target": f"{canvas_id}#xywh={x},{y},{w},{h}",
    }
    if ann_id:
        ann["id"] = ann_id
    return ann


def _make_page(page_id: str, items: list[dict]) -> dict:
    page: dict = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "type": "AnnotationPage",
        "items": items,
    }
    if page_id:
        page["id"] = page_id
    return page


# ---------------------------------------------------------------------------
# Per-file export
# ---------------------------------------------------------------------------

def export_boxes(
    entries_path: Path,
    base_url: str,
    force: bool,
    quiet: bool,
    ner_prompt_path: Path | None = None,
) -> tuple[int, str | None, int, int]:
    """
    Export one *_entries.json to a *_box_annotations.json Annotation Page.

    Coordinates in canvas_fragment are in the manifest's declared canvas pixel
    space (non-uniform, e.g. 2560×2560 for NYPL square canvases).  Mirador maps
    annotation xywh directly to image pixel space, so we convert:
        nat_x = round(canvas_x * nat_w / canvas_w)
        nat_y = round(canvas_y * nat_h / canvas_h)
    using the natural image dimensions fetched from the IIIF image service.

    Returns (n_annotations, canvas_id | None, nat_w, nat_h).
    canvas_id and nat dimensions are needed so the caller can update the manifest.
    """
    out_path = Path(str(entries_path).replace("_entries.json", "_box_annotations.json"))

    if out_path.exists() and not force:
        if not quiet:
            print(f"  [skip] {out_path.name}", file=sys.stderr)
        return 0, None, 0, 0

    # Load entries
    try:
        raw = json.loads(entries_path.read_text(encoding="utf-8"))
        entries = raw if isinstance(raw, list) else raw.get("entries", [])
    except Exception as exc:
        print(f"  SKIP {entries_path.name}: {exc}", file=sys.stderr)
        return 0, None, 0, 0

    # Canvas dimensions from sibling *_aligned.json
    aligned_path = Path(str(entries_path).replace("_entries.json", "_aligned.json"))
    canvas_w, canvas_h = 0, 0
    if aligned_path.exists():
        try:
            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
            canvas_w = int(aligned.get("canvas_width") or 0)
            canvas_h = int(aligned.get("canvas_height") or 0)
        except Exception:
            pass

    # Natural image dimensions are fetched lazily from the IIIF image service
    # when the first canvas_fragment is encountered.
    nat_w, nat_h = 0, 0

    def _to_natural_coords(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        """Convert non-uniform canvas pixel coords to natural image pixel coords.

        Mirador 3 maps annotation xywh directly to image pixel space regardless
        of the canvas dimensions declared in the manifest.  The correct coords
        are: nat_x = canvas_x * (nat_w / canvas_w), and similarly for y/w/h.
        """
        if not (canvas_w and canvas_h and nat_w and nat_h):
            return x, y, w, h
        return (
            round(x * nat_w / canvas_w),
            round(y * nat_h / canvas_h),
            max(1, round(w * nat_w / canvas_w)),
            max(1, round(h * nat_h / canvas_h)),
        )

    # Parse NER prompt for this collection to determine label field priorities.
    # Prefer an explicit override path; fall back to a sibling ner_prompt.md.
    resolved_ner = ner_prompt_path or (entries_path.parent / "ner_prompt.md")
    name_fields, addr_fields = parse_ner_label_fields(resolved_ner)

    # Annotation page ID
    base_stem = entries_path.stem.removesuffix("_entries")
    page_id = f"{base_url}/{base_stem}_box_annotations.json" if base_url else ""

    items: list[dict] = []
    first_canvas_id: str | None = None

    for i, entry in enumerate(entries):
        cf = entry.get("canvas_fragment", "")
        if "#xywh=" not in cf:
            continue
        canvas_id, xywh = cf.split("#xywh=", 1)
        parts = xywh.split(",")
        if len(parts) < 4:
            continue
        try:
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            continue

        if first_canvas_id is None:
            first_canvas_id = canvas_id
            nat_w, nat_h = _get_natural_dims(canvas_id)

        x, y, w, h = _to_natural_coords(x, y, w, h)
        if w <= 0 or h <= 0:
            continue

        ann_id = f"{page_id}/entry/{i}" if page_id else ""
        items.append(
            _box_annotation(ann_id, canvas_id, x, y, w, h, nat_w, nat_h, entry,
                            name_fields or None, addr_fields or None)
        )

    out_path.write_text(
        json.dumps(_make_page(page_id, items), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not quiet:
        print(
            f"  {entries_path.name}: {len(items)} box annotation(s)"
            + (f"  [nat {nat_w}×{nat_h}]" if nat_w else ""),
            file=sys.stderr,
        )

    return len(items), first_canvas_id, nat_w, nat_h


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------

def update_manifest(
    manifest_path: Path,
    canvas_to_ann_url: dict[str, str],
    canvas_to_dims: dict[str, tuple[int, int]],
    quiet: bool,
) -> int:
    """
    Add an 'annotations' reference to each canvas that has a box annotation page,
    and update each canvas's width/height to the natural image dimensions so that
    Mirador's annotation coordinate space matches the image pixel space.

    Only operates on IIIF v3 manifests (those with 'items' not 'sequences').
    Backs up the original to manifest_bak.json before writing.
    Returns the number of canvases updated.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  Error reading manifest: {exc}", file=sys.stderr)
        return 0

    if iiif_utils.manifest_version(manifest) != 3:
        print(
            "  Warning: manifest update only supported for IIIF Presentation v3.",
            file=sys.stderr,
        )
        return 0

    updated = 0
    for canvas in manifest.get("items", []):
        canvas_id = canvas.get("id", "")
        ann_url = canvas_to_ann_url.get(canvas_id)
        if not ann_url:
            continue
        # Remove any previous box annotation ref added by this script
        existing = canvas.get("annotations", [])
        existing = [a for a in existing if "_box_annotations" not in a.get("id", "")]
        existing.append({"id": ann_url, "type": "AnnotationPage"})
        canvas["annotations"] = existing
        # Update canvas dimensions to natural image pixel dimensions so that
        # annotation xywh coordinates align with the image content.
        nat_w, nat_h = canvas_to_dims.get(canvas_id, (0, 0))
        if nat_w and nat_h:
            canvas["width"] = nat_w
            canvas["height"] = nat_h
        updated += 1

    if updated == 0:
        if not quiet:
            print("  No canvas IDs matched — manifest unchanged.", file=sys.stderr)
        return 0

    # Backup then overwrite
    bak_path = manifest_path.with_name("manifest_bak.json")
    bak_path.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not quiet:
        print(
            f"  Updated {updated} canvas(es) in manifest.json  (backup → {bak_path.name})",
            file=sys.stderr,
        )
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export entry-level IIIF annotations with colored bounding boxes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "item_dir",
        help="Item directory containing *_entries.json files",
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.0-flash",
        metavar="MODEL",
        help="Model slug in the entries filenames (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--base-url",
        default="",
        metavar="URL",
        help=(
            "Base URL for annotation page IDs and manifest references "
            "(e.g. https://example.org/annotations). "
            "Required for --update-manifest."
        ),
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help=(
            "Add 'annotations' references into manifest.json so viewers "
            "auto-load the boxes. Requires --base-url."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "After updating manifest.json, normalize it for strict IIIF viewers "
            "(Clover, etc.): fix canvas IDs, add motivation:painting, fetch real "
            "image dimensions, fix structures, remove dead services. "
            "The hosted manifest URL is derived as <base-url>/manifest.json. "
            "Also updates annotation file targets to match the new canvas IDs. "
            "Implies --update-manifest."
        ),
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-export even if output files already exist",
    )
    parser.add_argument(
        "--ner-prompt",
        default=None,
        metavar="PATH",
        help=(
            "Path to a ner_prompt.md to use for all files in this run.  "
            "If omitted the script looks for ner_prompt.md in each file's "
            "directory, then falls back to built-in field-name defaults."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")

    if args.normalize:
        args.update_manifest = True

    if (args.update_manifest or args.normalize) and not args.base_url:
        print(
            "Error: --update-manifest requires --base-url so the manifest "
            "can reference annotation pages by absolute URI.",
            file=sys.stderr,
        )
        sys.exit(1)

    item_dir = Path(args.item_dir)
    if not item_dir.is_dir():
        print(f"Error: not a directory: {item_dir}", file=sys.stderr)
        sys.exit(1)

    slug = args.model.replace("/", "_")
    entries_files = sorted(item_dir.rglob(f"*_{slug}_entries.json"))

    if not entries_files:
        print(
            f"No *_{slug}_entries.json files found under {item_dir}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Resolve NER prompt path and log / confirm once ───────────────────────
    ner_prompt_path: Path | None = None

    if args.ner_prompt:
        ner_prompt_path = Path(args.ner_prompt)
        if not ner_prompt_path.is_file():
            print(f"Error: --ner-prompt path not found: {ner_prompt_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Look for a sibling ner_prompt.md in the item dir itself.
        candidate = item_dir / "ner_prompt.md"
        if candidate.is_file():
            ner_prompt_path = candidate

    if ner_prompt_path:
        name_fields, addr_fields = parse_ner_label_fields(ner_prompt_path)
        if not args.quiet:
            print(
                f"NER prompt: {ner_prompt_path}\n"
                f"  label fields  — name: {name_fields}, addr: {addr_fields}",
                file=sys.stderr,
            )
    else:
        handle_missing_ner_prompt()

    print(
        f"Exporting box annotations for {len(entries_files)} entries file(s)…",
        file=sys.stderr,
    )

    total = 0
    # canvas_id → absolute annotation page URL (for manifest update)
    canvas_to_ann_url: dict[str, str] = {}
    # canvas_id → (nat_w, nat_h) (for manifest canvas dimension update)
    canvas_to_dims: dict[str, tuple[int, int]] = {}

    for entries_path in entries_files:
        n, canvas_id, nat_w, nat_h = export_boxes(
            entries_path,
            base_url=args.base_url,
            force=args.force,
            quiet=args.quiet,
            ner_prompt_path=ner_prompt_path,
        )
        total += n
        if canvas_id and args.base_url:
            base_stem = entries_path.stem.removesuffix("_entries")
            ann_url = f"{args.base_url}/{base_stem}_box_annotations.json"
            canvas_to_ann_url[canvas_id] = ann_url
        if canvas_id and nat_w and nat_h:
            canvas_to_dims[canvas_id] = (nat_w, nat_h)

    print(
        f"\nDone. {total} box annotation(s) across {len(entries_files)} page(s).",
        file=sys.stderr,
    )

    if args.update_manifest:
        manifest_paths = list(item_dir.rglob("manifest.json"))
        if not manifest_paths:
            print("  Warning: no manifest.json found — skipping manifest update.", file=sys.stderr)
        for mp in manifest_paths:
            update_manifest(mp, canvas_to_ann_url, canvas_to_dims, args.quiet)

            if args.normalize:
                hosted_manifest_url = f"{args.base_url}/manifest.json"
                if not args.quiet:
                    print(
                        f"  Normalizing manifest for viewer: {hosted_manifest_url}",
                        file=sys.stderr,
                    )
                manifest = json.loads(mp.read_text(encoding="utf-8"))
                manifest, canvas_map = iiif_utils.normalize_for_viewer(
                    manifest, hosted_manifest_url
                )
                mp.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                # Update annotation file targets to use the new canvas IDs
                ann_files = list(mp.parent.glob("*_box_annotations.json"))
                for ann_path in ann_files:
                    try:
                        ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
                        iiif_utils.update_annotation_targets(ann_data, canvas_map)
                        ann_path.write_text(
                            json.dumps(ann_data, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        print(f"  Warning: could not update {ann_path.name}: {exc}", file=sys.stderr)
                if not args.quiet:
                    print(
                        f"  Updated targets in {len(ann_files)} annotation file(s).",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    main()
