#!/usr/bin/env python3
"""Export aligned OCR JSON to ALTO v3 XML — standard OCR coordinate markup.

For each ``{stem}_{model}_aligned.json`` produced by ``align_ocr.py``, write a
sibling ``{stem}_{model}.alto.xml`` in ALTO v3 format. ALTO is the LOC standard for OCR text-with-coordinates, so the
output drops straight into Solr, IIIF Content Search, Mirador, and Universal
Viewer with no custom glue — the pipeline already has everything ALTO needs
(line text + bounding boxes).

Coordinates
-----------
Each line's ``canvas_fragment`` (``…#xywh=x,y,w,h``) is already scaled into the
IIIF natural-image pixel space that Mirador maps annotations to, so those coords
are used directly and the ALTO ``Page`` dimensions are the aligned JSON's
``canvas_width``/``canvas_height``. When a line has no ``canvas_fragment`` (no
IIIF manifest was available), the raw Surya ``bbox`` is used and the page size is
derived from the maximum line extent.

Word boxes
----------
ALTO ``String`` elements are word-level. Since the source gives line-level boxes,
each line is split into words by proportional character-width allocation (the
technique from docs/comparison-htr-alto-pipeline.md): word width ∝ character
count, with a fixed inter-word gap. Boxes land on the correct word essentially
always — good enough for search highlighting. Pass ``--line-strings`` to instead
emit one ``String`` per whole line (skip word estimation).

Usage
-----
    python export_alto.py output/greenbooks
    python export_alto.py output/greenbooks --model gemini-2.0-flash
    python export_alto.py output/greenbooks --force --line-strings
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from utils.models import DEFAULT_OCR_MODEL, discover_ocr_slug, model_slug
from pipeline.state import get_ocr_model

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v3#"

# Inter-word gap as a fraction of the per-character unit width (matches the
# HTR-ALTO word-estimation heuristic).
_WORD_GAP_UNITS = 0.8


def _line_xywh(line: dict) -> "tuple[int, int, int, int] | None":
    """Return (x, y, w, h) for a line, preferring canvas-space coords.

    Uses the ``canvas_fragment`` xywh (scaled to IIIF natural pixel space) when
    present, else the raw Surya ``bbox``. Returns None if neither is usable.
    """
    frag = line.get("canvas_fragment") or ""
    if "xywh=" in frag:
        try:
            x, y, w, h = (int(round(float(v))) for v in frag.split("xywh=")[-1].split(","))
            return x, y, w, h
        except (ValueError, TypeError):
            pass
    bbox = line.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))
    return None


def _split_words(
    x: int, y: int, w: int, h: int, text: str
) -> "list[tuple[str, int, int, int, int]]":
    """Split a line box into per-word boxes by proportional character width.

    Returns (word, wx, wy, ww, wh) tuples. Word width is proportional to the
    word's character count; words are separated by ``_WORD_GAP_UNITS`` of the
    per-character unit. All words inherit the line's y/height.
    """
    words = text.split()
    if not words:
        return []
    n_gaps = len(words) - 1
    total_units = sum(len(wd) for wd in words) + _WORD_GAP_UNITS * n_gaps
    unit = w / total_units if total_units else w
    out: list[tuple[str, int, int, int, int]] = []
    cursor = float(x)
    for i, wd in enumerate(words):
        ww = max(1, round(len(wd) * unit))
        out.append((wd, int(round(cursor)), y, ww, h))
        cursor += ww + (_WORD_GAP_UNITS * unit if i < n_gaps else 0)
    return out


def _page_dimensions(data: dict, lines: list[dict]) -> "tuple[int, int]":
    """Resolve ALTO Page WIDTH/HEIGHT: canvas size if known, else max extent."""
    cw, ch = data.get("canvas_width"), data.get("canvas_height")
    if cw and ch:
        return int(cw), int(ch)
    max_x = max_y = 0
    for ln in lines:
        xywh = _line_xywh(ln)
        if xywh:
            x, y, w, h = xywh
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
    return max_x or 1, max_y or 1


def build_alto(data: dict, line_strings: bool = False) -> "ET.ElementTree":
    """Build an ALTO v3 ElementTree from one aligned-JSON dict."""
    lines = [ln for ln in (data.get("lines") or []) if _line_xywh(ln)]
    page_w, page_h = _page_dimensions(data, lines)

    ET.register_namespace("", ALTO_NS)
    alto = ET.Element(f"{{{ALTO_NS}}}alto")

    desc = ET.SubElement(alto, f"{{{ALTO_NS}}}Description")
    ET.SubElement(desc, f"{{{ALTO_NS}}}MeasurementUnit").text = "pixel"
    sii = ET.SubElement(desc, f"{{{ALTO_NS}}}sourceImageInformation")
    ET.SubElement(sii, f"{{{ALTO_NS}}}fileName").text = data.get("image", "")
    proc = ET.SubElement(desc, f"{{{ALTO_NS}}}OCRProcessing", ID="OCR_1")
    step = ET.SubElement(proc, f"{{{ALTO_NS}}}ocrProcessingStep")
    sw = ET.SubElement(step, f"{{{ALTO_NS}}}processingSoftware")
    ET.SubElement(sw, f"{{{ALTO_NS}}}softwareName").text = "directory-pipeline"
    ET.SubElement(sw, f"{{{ALTO_NS}}}softwareVersion").text = str(data.get("model", ""))

    layout = ET.SubElement(alto, f"{{{ALTO_NS}}}Layout")
    page = ET.SubElement(
        layout, f"{{{ALTO_NS}}}Page",
        ID="P1", PHYSICAL_IMG_NR="1",
        WIDTH=str(page_w), HEIGHT=str(page_h),
    )
    ps = ET.SubElement(
        page, f"{{{ALTO_NS}}}PrintSpace",
        HPOS="0", VPOS="0", WIDTH=str(page_w), HEIGHT=str(page_h),
    )
    block = ET.SubElement(
        ps, f"{{{ALTO_NS}}}TextBlock",
        ID="B1", HPOS="0", VPOS="0", WIDTH=str(page_w), HEIGHT=str(page_h),
    )

    for li, ln in enumerate(lines, 1):
        x, y, w, h = _line_xywh(ln)
        text = (ln.get("gemini_text") or "").strip()
        tl = ET.SubElement(
            block, f"{{{ALTO_NS}}}TextLine",
            ID=f"L{li}", HPOS=str(x), VPOS=str(y), WIDTH=str(w), HEIGHT=str(h),
        )
        words = (
            [(text, x, y, w, h)] if (line_strings or " " not in text)
            else _split_words(x, y, w, h, text)
        )
        for wi, (word, wx, wy, ww, wh) in enumerate(words, 1):
            if wi > 1:
                ET.SubElement(tl, f"{{{ALTO_NS}}}SP")
            ET.SubElement(
                tl, f"{{{ALTO_NS}}}String",
                ID=f"L{li}_S{wi}",
                HPOS=str(wx), VPOS=str(wy), WIDTH=str(ww), HEIGHT=str(wh),
                CONTENT=word,
            )

    tree = ET.ElementTree(alto)
    ET.indent(tree, space="  ")
    return tree


def export_image(aligned_path: Path, force: bool, line_strings: bool) -> str:
    """Write the ALTO sibling for one aligned JSON. Returns a status string."""
    out_path = aligned_path.with_name(
        aligned_path.name.replace("_aligned.json", ".alto.xml")
    )
    if out_path.exists() and not force:
        return "skipped"
    data = json.loads(aligned_path.read_text(encoding="utf-8"))
    if not (data.get("lines")):
        return "empty"
    tree = build_alto(data, line_strings=line_strings)
    tree.write(out_path, encoding="unicode", xml_declaration=True)
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export aligned OCR JSON to ALTO v3 XML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output_dir", help="Root images directory (e.g. output/greenbooks)")
    parser.add_argument(
        "--model", "-m", default=None, metavar="MODEL",
        help=f"Model slug of the aligned JSON to export (default: auto-detected, else {DEFAULT_OCR_MODEL}).",
    )
    parser.add_argument(
        "--line-strings", action="store_true",
        help="Emit one String per whole line instead of estimating word boxes.",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing ALTO files.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-file output.")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    if not output_root.exists():
        print(f"Error: directory not found: {output_root}", file=sys.stderr)
        sys.exit(1)

    model = args.model or get_ocr_model(output_root) or discover_ocr_slug(output_root) or DEFAULT_OCR_MODEL
    slug = model_slug(model)
    aligned = sorted(output_root.rglob(f"*_{slug}_aligned.json"))
    if not aligned:
        print(f"No *_{slug}_aligned.json files under {output_root}", file=sys.stderr)
        sys.exit(0)

    if not args.quiet:
        print(f"Exporting ALTO v3 for {len(aligned)} page(s) (model={slug})…", file=sys.stderr)

    counts: dict[str, int] = {}
    for i, p in enumerate(aligned, 1):
        try:
            status = export_image(p, args.force, args.line_strings)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            print(f"  Error on {p.name}: {exc}", file=sys.stderr)
        counts[status] = counts.get(status, 0) + 1
        if not args.quiet:
            print(f"[{i:04d}/{len(aligned)}] {status:8s} {p.name.replace('_aligned.json', '.alto.xml')}", file=sys.stderr)

    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"\nDone. {summary}.", file=sys.stderr)


if __name__ == "__main__":
    main()
