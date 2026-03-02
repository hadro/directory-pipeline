#!/usr/bin/env python3
"""Visualize align_ocr.py output by drawing bounding boxes on images.

For each *_aligned.json file found, opens the corresponding image, draws
color-coded bounding boxes, and saves a *_viz.jpg next to it:

  green  — word confidence: per-word boxes, Tesseract aligned to Gemini
  orange — line confidence: one box for the whole line (word alignment weak)
  red    — unmatched Gemini lines (no Tesseract coordinates, listed in margin)

Usage
-----
    # Visualize all aligned JSONs in an item directory:
    python visualize_alignment.py images/greenbooks/item_uuid/

    # Visualize a single JSON:
    python visualize_alignment.py images/greenbooks/item_uuid/0039_123_gemini-2.0-flash_aligned.json

    # Filter by model:
    python visualize_alignment.py images/greenbooks/ --model gemini-2.0-flash

    # Re-generate (skip existing _viz.jpg files by default):
    python visualize_alignment.py images/greenbooks/ --force
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow is required. Install with: uv add pillow", file=sys.stderr)
    sys.exit(1)

# ─── colour palette ───────────────────────────────────────────────────────────
WORD_FILL    = (0,   180,  50,  55)   # green, translucent fill
WORD_OUTLINE = (0,   160,  40, 255)   # green, solid outline
LINE_FILL    = (255, 140,   0,  55)   # orange, translucent fill
LINE_OUTLINE = (220, 110,   0, 255)   # orange, solid outline
BOX_WIDTH    = 2                      # outline stroke width (pixels)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a small bitmap font; fall back gracefully if none available."""
    for face in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(face, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def visualize(json_path: Path, show_text: bool = True, force: bool = False) -> bool:
    """
    Draw alignment boxes on the image paired with json_path.

    Returns True on success, False if the image is missing.
    """
    out_path = json_path.with_name(json_path.name.replace("_aligned.json", "_viz.jpg"))
    if out_path.exists() and not force:
        print(f"  skipped (exists): {out_path.name}", file=sys.stderr)
        return True

    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_path = json_path.parent / data["image"]
    if not img_path.exists():
        print(f"  missing image: {img_path}", file=sys.stderr)
        return False

    img = Image.open(img_path).convert("RGBA")
    iw, ih = img.size

    # ── draw bounding boxes on a transparent overlay ────────────────────────
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    word_count = line_count = 0

    for entry in data.get("lines", []):
        conf = entry.get("confidence", "none")

        if conf == "word":
            word_count += len(entry.get("words", []))
            for word in entry.get("words", []):
                x1, y1, x2, y2 = word["bbox"]
                draw.rectangle([x1, y1, x2, y2], fill=WORD_FILL, outline=WORD_OUTLINE, width=BOX_WIDTH)
        elif conf == "line":
            line_count += 1
            x1, y1, x2, y2 = entry["bbox"]
            draw.rectangle([x1, y1, x2, y2], fill=LINE_FILL, outline=LINE_OUTLINE, width=BOX_WIDTH)

    # ── composite and convert back to RGB for JPEG output ───────────────────
    result = Image.alpha_composite(img, overlay).convert("RGB")

    # ── optional text labels ─────────────────────────────────────────────────
    if show_text:
        td = ImageDraw.Draw(result)
        font_small = _load_font(max(10, iw // 150))

        for entry in data.get("lines", []):
            conf = entry.get("confidence", "none")
            text = entry.get("gemini_text", "")
            if not text:
                continue

            if conf == "word" and entry.get("words"):
                # label at the top-left of the first word box
                x1, y1 = entry["words"][0]["bbox"][:2]
            elif conf == "line":
                x1, y1 = entry["bbox"][:2]
            else:
                continue

            # tiny white drop-shadow, then coloured text
            color = (0, 140, 30) if conf == "word" else (180, 90, 0)
            td.text((x1 + 1, y1 + 1), text, fill=(255, 255, 255, 200), font=font_small)
            td.text((x1,     y1    ), text, fill=color,                font=font_small)

        # ── legend ───────────────────────────────────────────────────────────
        unmatched = data.get("unmatched_gemini", [])
        legend_lines = [
            f"word-confidence boxes: {word_count}",
            f"line-confidence boxes: {line_count}",
            f"unmatched Gemini lines: {len(unmatched)}",
            f"model: {data.get('model', '?')}",
        ]
        font_leg = _load_font(max(12, iw // 120))
        pad = 8
        lh = max(14, iw // 100)
        lw = max(len(s) for s in legend_lines) * (lh // 2)
        box_x2 = pad + lw + pad
        box_y2 = pad + lh * len(legend_lines) + pad
        td.rectangle([0, 0, box_x2, box_y2], fill=(255, 255, 255, 230))
        for i, line in enumerate(legend_lines):
            color = (0, 130, 20) if "word" in line else (160, 80, 0) if "line" in line else (80, 80, 80)
            td.text((pad, pad + i * lh), line, fill=color, font=font_leg)

    result.save(out_path, "JPEG", quality=88)
    print(f"  saved: {out_path.name}", file=sys.stderr)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize align_ocr.py bounding boxes on images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Directory to scan recursively, or a single *_aligned.json file.",
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
        help="Draw boxes only; skip Gemini text labels.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-generate _viz.jpg files even if they already exist.",
    )
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file() and target.suffix == ".json":
        jsons = [target]
    elif target.is_dir():
        jsons = sorted(target.rglob("*_aligned.json"))
    else:
        print(f"Error: not a directory or .json file: {target}", file=sys.stderr)
        sys.exit(1)

    if args.model:
        jsons = [j for j in jsons if args.model in j.name]

    if not jsons:
        print("No *_aligned.json files found.", file=sys.stderr)
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
