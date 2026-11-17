#!/usr/bin/env python3
"""Detect text lines with Indra OCR and write them where alignment reads Surya's.

An experiment, not a pipeline stage: does Datalab's CPU line engine give better
line boxes than Surya on the dense pages where Surya under-segments? (Evidence
and method: docs/evidence/2026-11-17-indra-lines/.)

Indra's ``Line(box, text, conf)`` is the same shape as a Surya line, so each
page is written as ``{stem}_surya.json`` in Surya's schema (image_width,
image_height, lines[bbox, text, confidence]) and align_ocr, extract_entries and
the explorer run on it unchanged. Indra-only fields sit under ``detector`` and
``blocks``, which parse_surya ignores.

Nothing is written into the volume itself. Pages go to a MIRROR of it, by default
``output/_indra_lines/<same relative path>/``, which the collection registry
already excludes (``_*``):

  - page images are symlinked (large, never written by alignment or extraction);
  - each page's Gemini OCR text and the directory's small metadata files
    (manifest.json, pipeline_state.json, sections.txt, included_pages.txt,
    selection.txt, prompts) are COPIED, because later stages write some of them
    and a write through a symlink would land in the real volume.

Then, to see a whole volume on Indra's lines:

    python tools/indra_lines.py output/tulsa_1921/
    python pipeline/align_ocr.py output/_indra_lines/tulsa_1921/
    python pipeline/extract_entries.py output/_indra_lines/tulsa_1921/   # costs NER calls

Pages are read upright (``turn=False``), as Surya reads them: Indra's
orientation model would otherwise return boxes in a turned frame for any page it
judged sideways. A page Indra reads as vertical text is reported.

Per-page timings go to ``indra_run.json`` at the mirror root: Indra's own stage
timings and the wall time including image load.

Requires the ``indra`` extra (ONNX Runtime; no torch, no GPU):
    uv sync --inexact --extra indra
The first run downloads Indra's models (about 76 MB) from Hugging Face.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MIRROR = Path("output") / "_indra_lines"
RUN_LOG = "indra_run.json"

# Copied (never linked) from each directory between the volume root and a page.
_META_GLOBS = (
    "manifest.json",
    "pipeline_state.json",
    "sections.txt",
    "included_pages.txt",
    "selection.txt",
    "*prompt*.md",
)
# A page's own text files that are not Gemini OCR: Surya's (replaced here) and
# NER error dumps (extraction output).
_NOT_OCR_TEXT = ("_surya.txt", "_entries_error.txt")


# ---------------------------------------------------------------------------
# Page selection — the same rule as run_surya_ocr.py
# ---------------------------------------------------------------------------

def select_images(volume: Path, exclude: Path | None = None) -> list[Path]:
    """The .jpg pages Surya would run on, under *volume*.

    Same rule as run_surya_ocr.py: skip ``_viz`` renders, and where a page has
    been split into ``_left``/``_right``, take the halves instead of the whole.
    Also skips anything under a ``_``-prefixed directory (backups, quarantine,
    this tool's own mirror) and under *exclude*.
    """
    out = []
    excl = exclude.resolve() if exclude else None
    for p in sorted(volume.rglob("*.jpg")):
        rel_dirs = p.relative_to(volume).parts[:-1]
        if any(d.startswith("_") for d in rel_dirs):
            continue
        if excl and (p.resolve() == excl or excl in p.resolve().parents):
            continue
        if p.stem.endswith("_viz"):
            continue
        if not p.stem.endswith(("_left", "_right")):
            if (p.with_name(f"{p.stem}_left.jpg").exists()
                    and p.with_name(f"{p.stem}_right.jpg").exists()):
                continue
        out.append(p)
    return out


def mirror_root_for(volume: Path, mirror_root: Path, output_root: Path) -> Path:
    """Where *volume* is mirrored: its path under output/ is kept, so a
    collection's nesting survives; a volume outside output/ goes by name."""
    v = volume.resolve()
    try:
        rel = v.relative_to(output_root.resolve())
    except ValueError:
        rel = Path(v.name)
    if not rel.parts:
        sys.exit(f"Error: point this at a volume or collection, not at {output_root} itself.")
    return mirror_root / rel


# ---------------------------------------------------------------------------
# Indra page → Surya-schema record
# ---------------------------------------------------------------------------

def _clamp_box(box, width: int, height: int) -> list[int] | None:
    x0, y0, x1, y1 = (int(round(float(v))) for v in box[:4])
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def page_record(page, width: int, height: int, version: str) -> dict:
    """Surya's ``*_surya.json`` schema from an Indra ``Page``.

    ``confidence`` is Indra's line confidence (mean CTC peak probability),
    which is not on Surya's scale. A line under Indra's MIN_CONF (0.5) keeps its
    box and comes back with empty text; it is kept, as Indra returns it.
    """
    lines, dropped = [], 0
    for ln in page.lines:
        bbox = _clamp_box(ln.box, width, height)
        if bbox is None:
            dropped += 1
            continue
        lines.append({
            "bbox": bbox,
            "text": ln.text or "",
            "confidence": round(float(ln.conf), 4),
        })
    blocks = []
    for b in getattr(page, "blocks", None) or []:
        bbox = _clamp_box(b.get("bbox") or (0, 0, 0, 0), width, height)
        if bbox is not None:
            blocks.append({"label": b.get("label", ""), "bbox": bbox, "text": b.get("text", "")})
    timings = {k: round(float(v), 1) for k, v in (getattr(page, "timings", None) or {}).items()}
    return {
        "image_width": width,
        "image_height": height,
        "lines": lines,
        "detector": {
            "name": "indra-ocr",
            "version": version,
            "turn": False,
            "rotation": getattr(page, "rotation", 0),
            "skew": getattr(page, "skew", 0.0),
            "vertical": bool(getattr(page, "vertical", False)),
            "source": getattr(page, "source", "ocr"),
            "note": getattr(page, "note", ""),
            "timings_ms": timings,
            "degenerate_boxes_dropped": dropped,
        },
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Mirror inputs
# ---------------------------------------------------------------------------

def _copy_if_newer(src: Path, dst: Path) -> None:
    if dst.is_symlink():
        dst.unlink()
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)


def mirror_inputs(img: Path, volume: Path, mirror: Path) -> Path:
    """Link the page image and copy its OCR text and metadata into the mirror.
    Returns the mirrored image path."""
    rel = img.relative_to(volume)
    dst_dir = mirror / rel.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    link = dst_dir / img.name
    if not link.exists() and not link.is_symlink():
        link.symlink_to(img.resolve())

    for f in img.parent.glob(f"{img.stem}_*.txt"):
        if f.name.endswith(_NOT_OCR_TEXT):
            continue
        _copy_if_newer(f, dst_dir / f.name)

    # Metadata from every directory between the volume root and this page.
    src_dir, dst = img.parent, dst_dir
    while True:
        for pattern in _META_GLOBS:
            for f in src_dir.glob(pattern):
                if f.is_file():
                    _copy_if_newer(f, dst / f.name)
        if src_dir.resolve() == volume.resolve():
            break
        src_dir, dst = src_dir.parent, dst.parent
    return link


def _targets(paths: list[Path]) -> list[tuple[Path, list[Path]]]:
    """(volume, pages) per argument. A directory is a volume with every page
    selected (an empty list); image files are grouped by their directory, which
    is then treated as the volume, so each lands at its path under output/."""
    out: dict[Path, list[Path]] = {}
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".jpg":
            out.setdefault(p.parent, []).append(p)
        else:
            out.setdefault(p, [])
    return list(out.items())


def _load_run_log(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volumes", nargs="+", type=Path,
                    help="Volume or collection directories (as given to --surya-ocr), or "
                         "single page images (each is mirrored at its path under output/)")
    ap.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR,
                    help=f"Where mirrors are written (default: {DEFAULT_MIRROR})")
    ap.add_argument("--output-root", type=Path, default=Path("output"),
                    help="Root whose layout the mirror keeps (default: output)")
    ap.add_argument("--threads", type=int, default=4,
                    help="ONNX Runtime threads (default 4, as in Datalab's benchmark)")
    ap.add_argument("--det-long-side", type=int, metavar="PX",
                    help="Long side the page is scaled to for line detection (Indra's "
                         "default: 1024, which loses small type on dense directory pages)")
    ap.add_argument("--no-split", action="store_true",
                    help="Keep each detected line whole. Indra otherwise cuts a line at gaps "
                         "other lines' edges align with, which on a directory page separates "
                         "name from address, and alignment pairs one line with one box")
    ap.add_argument("--pages", metavar="GLOB",
                    help="Only image filenames matching this glob, e.g. '0012_*'")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N pages per volume (0: all)")
    ap.add_argument("--force", action="store_true",
                    help="Re-detect pages whose mirrored _surya.json exists")
    ap.add_argument("--quiet", "-q", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        from PIL import Image

        import indra_ocr
        from indra_ocr import ModelError, OrtOCR, model_dir
    except ImportError as exc:
        sys.exit(f"Error: {exc}\nInstall with:  uv sync --inexact --extra indra")

    from utils.image_utils import image_is_complete

    if not args.quiet:
        print(f"Loading Indra OCR {indra_ocr.__version__} models…", file=sys.stderr)
    try:
        ocr = OrtOCR(model_dir(), threads=args.threads)
    except ModelError as exc:
        sys.exit(f"Error: could not load Indra's models: {exc}\n"
                 "The first run downloads them from Hugging Face.")
    if args.det_long_side:
        ocr.long_side = args.det_long_side
    if args.no_split:
        import indra_ocr.pipeline as indra_pipeline
        # shortcut: patches Indra 1.x internals (SPLIT_GAP, gutter_cuts); recheck on an upgrade.
        if not (hasattr(indra_pipeline, "SPLIT_GAP") and hasattr(ocr, "gutter_cuts")):
            sys.exit(f"Error: --no-split does not support Indra {indra_ocr.__version__}")
        indra_pipeline.SPLIT_GAP = float("inf")
        ocr.gutter_cuts = lambda boxes, rec: [[] for _ in rec]

    failed_any = False
    for volume, chosen in _targets(args.volumes):
        if not volume.is_dir():
            print(f"Error: not a directory or image: {volume}", file=sys.stderr)
            failed_any = True
            continue
        mirror = mirror_root_for(volume, args.mirror_root, args.output_root)
        if volume.resolve() == mirror.resolve() or mirror.resolve() in volume.resolve().parents:
            print(f"Error: {volume} is inside the mirror {mirror}", file=sys.stderr)
            failed_any = True
            continue
        images = chosen or select_images(volume, exclude=args.mirror_root)
        if args.pages:
            images = [p for p in images if fnmatch.fnmatch(p.name, args.pages)]
        if args.limit:
            images = images[: args.limit]
        if not images:
            print(f"{volume}: no pages selected", file=sys.stderr)
            continue

        mirror.mkdir(parents=True, exist_ok=True)
        log_path = mirror / RUN_LOG
        log = _load_run_log(log_path)
        log.update({
            "source": str(volume),
            "indra_version": indra_ocr.__version__,
            "threads": args.threads,
            "det_long_side": getattr(ocr, "long_side", None),
            "split": not args.no_split,
            "turn": False,
            "machine": f"{platform.system()} {platform.machine()} ({os.cpu_count()} cpus)",
            "python": platform.python_version(),
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        pages_log = log.setdefault("pages", {})

        done = skipped = failed = 0
        if not args.quiet:
            print(f"{volume} → {mirror}: {len(images)} page(s)", file=sys.stderr)
        for i, img in enumerate(images, 1):
            dst_img = mirror_inputs(img, volume, mirror)
            out_json = dst_img.parent / f"{img.stem}_surya.json"
            key = str(img.relative_to(volume))
            if out_json.exists() and not args.force:
                skipped += 1
                continue
            if not image_is_complete(img):
                print(f"[{i:04d}] FAILED: corrupt or truncated image: {img.name} "
                      "(re-run the download to fetch it again)", file=sys.stderr)
                failed += 1
                continue
            t0 = time.perf_counter()
            try:
                with Image.open(img) as pil:
                    width, height = pil.size
                    gray = np.ascontiguousarray(np.asarray(pil.convert("L"), dtype=np.uint8))
                page = ocr(gray, turn=False)
            except Exception as exc:  # noqa: BLE001 — one bad page must not end the run
                print(f"[{i:04d}] FAILED {img.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed += 1
                continue
            wall_ms = (time.perf_counter() - t0) * 1000
            if getattr(page, "source", "ocr") == "error":
                print(f"[{i:04d}] FAILED {img.name}: {page.note}", file=sys.stderr)
                failed += 1
                continue
            rec = page_record(page, width, height, indra_ocr.__version__)
            if rec["detector"]["vertical"]:
                print(f"[{i:04d}] WARNING {img.name}: Indra read this page as vertical "
                      "text; its lines are columns", file=sys.stderr)
            out_json.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            out_json.with_suffix(".txt").write_text(
                "\n".join(ln["text"] for ln in rec["lines"]), encoding="utf-8")
            empty = sum(1 for ln in rec["lines"] if not ln["text"].strip())
            pages_log[key] = {
                "wall_ms": round(wall_ms, 1),
                "indra_ms": rec["detector"]["timings_ms"].get("total_ms"),
                "lines": len(rec["lines"]),
                "empty_text": empty,
                "width": width,
                "height": height,
            }
            done += 1
            if not args.quiet:
                print(f"[{i:04d}/{len(images)}] {img.name}: {len(rec['lines'])} lines "
                      f"({empty} without text), {wall_ms / 1000:.2f}s", file=sys.stderr)

        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        walls = sorted(p["wall_ms"] for p in pages_log.values() if p.get("wall_ms"))
        median = f", median {walls[len(walls) // 2] / 1000:.2f}s/page" if walls else ""
        print(f"{volume}: {done} detected, {skipped} already done, {failed} failed{median}. "
              f"Timings: {log_path}", file=sys.stderr)
        failed_any |= bool(failed)

    if failed_any:
        sys.exit(1)


if __name__ == "__main__":
    main()
