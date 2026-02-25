#!/usr/bin/env python3
"""Visualize extract_entries.py output by drawing bounding boxes on images.

For each *_entries.json file found, opens the corresponding image, draws
colour-coded bounding boxes per entry (coloured by category), and saves a
*_entries_viz.jpg next to it.

Colour coding
-------------
  blue    — formal_accommodations  (Hotels, Motels)
  teal    — informal_accommodations (Tourist Homes, Boarding Houses)
  red     — eating_drinking        (Restaurants, Bars, Taverns)
  purple  — barber_beauty          (Beauty Parlors, Barber Shops)
  amber   — service_station        (Service Stations, Garages)
  grey    — other                  (Drug Stores, Undertakers, etc.)

Advertisements get a thicker outline (3 px vs 1 px).
Entries with no spatial coordinates (canvas_fragment has no #xywh) are
listed in the right margin in their category colour.

Usage
-----
    python visualize_entries.py images/greenbooks/item_uuid/
    python visualize_entries.py images/greenbooks/item_uuid/0039_123_gemini-2.0-flash_entries.json
    python visualize_entries.py images/greenbooks/ --model gemini-2.0-flash
    python visualize_entries.py images/greenbooks/ --force
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow is required. Install with: uv add pillow", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Category colour palette  (fill alpha ~60, outline alpha 255)
# ---------------------------------------------------------------------------

_PALETTE: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "formal_accommodations":   ((50,  110, 220,  60), (40,   90, 200, 255)),
    "informal_accommodations": ((0,   185, 170,  60), (0,   155, 140, 255)),
    "eating_drinking":         ((210,  55,  45,  60), (185,  35,  25, 255)),
    "barber_beauty":           ((175,  55, 210,  60), (150,  35, 185, 255)),
    "service_station":         ((195, 148,   0,  60), (165, 125,   0, 255)),
    "other":                   ((110, 110, 130,  60), ( 85,  85, 105, 255)),
}
_DEFAULT_PALETTE = ((130, 130, 130, 60), (100, 100, 100, 255))

_CATEGORY_LABELS = {
    "formal_accommodations":   "Hotels/Motels",
    "informal_accommodations": "Tourist Homes",
    "eating_drinking":         "Restaurants/Bars",
    "barber_beauty":           "Beauty/Barber",
    "service_station":         "Service Stations",
    "other":                   "Other",
}

BOX_WIDTH_NORMAL = 1
BOX_WIDTH_AD     = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.ImageFont:
    for face in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(face, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _parse_xywh(fragment: str) -> tuple[int, int, int, int] | None:
    """Extract (x, y, w, h) from a IIIF canvas fragment string, or None."""
    m = re.search(r"#xywh=(\d+),(\d+),(\d+),(\d+)", fragment)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _scale_xywh(
    xywh: tuple[int, int, int, int],
    canvas_w: int,
    canvas_h: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Scale canvas-space xywh to image pixel space."""
    x, y, w, h = xywh
    sx = img_w / canvas_w if canvas_w else 1.0
    sy = img_h / canvas_h if canvas_h else 1.0
    return round(x * sx), round(y * sy), round(w * sx), round(h * sy)


def _canvas_dims_from_aligned(entries_path: Path) -> tuple[int, int]:
    """
    Read canvas_width / canvas_height from the sibling *_aligned.json.
    Returns (0, 0) if the file is absent or the keys are missing.
    """
    aligned_path = Path(str(entries_path).replace("_entries.json", "_aligned.json"))
    if not aligned_path.exists():
        return 0, 0
    try:
        data = json.loads(aligned_path.read_text(encoding="utf-8"))
        cw = data.get("canvas_width") or 0
        ch = data.get("canvas_height") or 0
        return int(cw), int(ch)
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# Core visualizer
# ---------------------------------------------------------------------------

def visualize(json_path: Path, show_text: bool = True, force: bool = False) -> bool:
    """
    Draw entry boxes on the image paired with json_path.
    Returns True on success, False if the image is missing.
    """
    out_path = json_path.with_name(json_path.name.replace("_entries.json", "_entries_viz.jpg"))
    if out_path.exists() and not force:
        print(f"  skipped (exists): {out_path.name}", file=sys.stderr)
        return True

    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_path = json_path.parent / data.get("image", "")
    if not img_path.exists():
        # entries JSON stores image name in first entry's "image" field
        first = next(iter(data.get("entries", [])), None)
        if first:
            img_path = json_path.parent / first.get("image", "")
    if not img_path.exists():
        print(f"  missing image: {img_path}", file=sys.stderr)
        return False

    img = Image.open(img_path).convert("RGBA")
    iw, ih = img.size

    canvas_w, canvas_h = _canvas_dims_from_aligned(json_path)
    if not canvas_w or not canvas_h:
        # Fall back: assume canvas == image
        canvas_w, canvas_h = iw, ih

    entries = data.get("entries", [])

    # Split entries into those with spatial coords and those without
    positioned: list[tuple[dict, tuple[int, int, int, int]]] = []
    unpositioned: list[dict] = []

    for entry in entries:
        frag = entry.get("canvas_fragment", "")
        xywh = _parse_xywh(frag) if frag else None
        if xywh:
            positioned.append((entry, xywh))
        else:
            unpositioned.append(entry)

    # ── draw bounding boxes on a transparent overlay ────────────────────────
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    category_counts: dict[str, int] = {}

    for entry, xywh in positioned:
        cat = entry.get("category", "other") or "other"
        fill, outline = _PALETTE.get(cat, _DEFAULT_PALETTE)
        category_counts[cat] = category_counts.get(cat, 0) + 1

        is_ad = bool(entry.get("is_advertisement"))
        width = BOX_WIDTH_AD if is_ad else BOX_WIDTH_NORMAL

        ix, iy, iw_box, ih_box = _scale_xywh(xywh, canvas_w, canvas_h, iw, ih)
        x1, y1, x2, y2 = ix, iy, ix + iw_box, iy + ih_box
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=width)

    # ── composite overlay onto image ─────────────────────────────────────────
    result = Image.alpha_composite(img, overlay).convert("RGB")

    # ── text labels ───────────────────────────────────────────────────────────
    if show_text:
        td = ImageDraw.Draw(result)
        font_entry = _load_font(max(9, iw // 160))

        for entry, xywh in positioned:
            cat = entry.get("category", "other") or "other"
            _, outline = _PALETTE.get(cat, _DEFAULT_PALETTE)
            outline_rgb = outline[:3]

            name = entry.get("establishment_name", "") or ""
            if len(name) > 35:
                name = name[:33] + "…"

            ix, iy, _, _ = _scale_xywh(xywh, canvas_w, canvas_h, iw, ih)
            # White drop-shadow then category-coloured text
            td.text((ix + 1, iy + 1), name, fill=(255, 255, 255, 210), font=font_entry)
            td.text((ix,     iy    ), name, fill=outline_rgb,           font=font_entry)

        # ── right-margin list for unpositioned entries ────────────────────────
        if unpositioned:
            font_margin = _load_font(max(9, iw // 170))
            margin_x = iw - max(200, iw // 5)
            margin_y = 8
            lh = max(12, iw // 130)

            # semi-transparent background strip
            strip_h = lh * (len(unpositioned) + 1) + 16
            td.rectangle(
                [margin_x - 4, 0, iw, strip_h],
                fill=(255, 255, 255, 200),
            )
            td.text(
                (margin_x, margin_y),
                f"No bbox ({len(unpositioned)}):",
                fill=(80, 80, 80),
                font=font_margin,
            )
            margin_y += lh
            for entry in unpositioned:
                cat = entry.get("category", "other") or "other"
                _, outline = _PALETTE.get(cat, _DEFAULT_PALETTE)
                name = (entry.get("establishment_name") or "")[:30]
                td.text((margin_x, margin_y), name, fill=outline[:3], font=font_margin)
                margin_y += lh

        # ── legend ───────────────────────────────────────────────────────────
        font_leg = _load_font(max(11, iw // 130))
        lh_leg = max(14, iw // 100)
        pad = 8
        swatch = lh_leg - 2

        ctx = data.get("page_context", {})
        ctx_str = (
            f"{ctx.get('state','?')} › {ctx.get('city','?')} › {ctx.get('category','?')}"
        )
        header_lines = [
            f"model: {data.get('model', '?')}",
            f"entries: {len(entries)}  (ad: {sum(1 for e in entries if e.get('is_advertisement'))},"
            f" no-bbox: {len(unpositioned)})",
            f"context: {ctx_str}",
        ]
        cat_lines = [
            (cat, _CATEGORY_LABELS.get(cat, cat), category_counts.get(cat, 0))
            for cat in _PALETTE
            if category_counts.get(cat, 0) > 0
        ]

        total_lines = len(header_lines) + len(cat_lines)
        legend_w = max(300, iw // 4)
        legend_h = pad + total_lines * lh_leg + len(header_lines) * 2 + pad

        td.rectangle([0, 0, legend_w, legend_h], fill=(255, 255, 255, 220))

        y = pad
        for line in header_lines:
            td.text((pad, y), line, fill=(60, 60, 60), font=font_leg)
            y += lh_leg + 2

        for cat, label, count in cat_lines:
            fill_c, outline_c = _PALETTE[cat]
            # draw colour swatch
            td.rectangle(
                [pad, y + 1, pad + swatch, y + swatch + 1],
                fill=fill_c[:3],
                outline=outline_c[:3],
                width=1,
            )
            td.text(
                (pad + swatch + 4, y),
                f"{label}: {count}",
                fill=outline_c[:3],
                font=font_leg,
            )
            y += lh_leg

    result.save(out_path, "JPEG", quality=88)
    print(f"  saved: {out_path.name}", file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize extract_entries.py bounding boxes on images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Directory to scan recursively, or a single *_entries.json file.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        metavar="MODEL",
        help="Only visualize JSON files for this model (substring match).",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Draw boxes only; skip text labels.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-generate _entries_viz.jpg files even if they already exist.",
    )
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file() and target.suffix == ".json":
        jsons = [target]
    elif target.is_dir():
        jsons = sorted(target.rglob("*_entries.json"))
    else:
        print(f"Error: not a directory or .json file: {target}", file=sys.stderr)
        sys.exit(1)

    if args.model:
        jsons = [j for j in jsons if args.model in j.name]

    if not jsons:
        print("No *_entries.json files found.", file=sys.stderr)
        sys.exit(0)

    print(f"Visualizing {len(jsons)} file(s)…", file=sys.stderr)
    ok = failed = 0
    for j in jsons:
        print(f"[{ok + failed + 1:04d}/{len(jsons)}] {j.name}", file=sys.stderr)
        if visualize(j, show_text=not args.no_text, force=args.force):
            ok += 1
        else:
            failed += 1

    print(
        f"\nDone. {ok} visualized" + (f", {failed} failed." if failed else "."),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
