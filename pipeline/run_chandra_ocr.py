#!/usr/bin/env python3
"""Run Chandra OCR on images, saving plain-text output alongside each image.

Chandra (datalab-to/chandra-ocr-2) is a 5B vision-language model that produces
layout-aware Markdown output. This script converts that Markdown to plain text
compatible with align_ocr.py, saving {stem}_chandra-ocr-2.txt alongside each image.

Already-processed images are skipped, so the script is safe to re-run.

Requires:
    pip install chandra-ocr[hf]   (HuggingFace inference — works on Colab T4)
    # or
    pip install chandra-ocr[vllm] (vLLM server — faster but needs H100 80GB)

GPU strongly recommended (T4 16 GB fits the 5B BF16 model with headroom).

To use the output with align_ocr.py, pass --model chandra-ocr-2 explicitly:
    python pipeline/align_ocr.py output/dir --model chandra-ocr-2

Usage
-----
    python pipeline/run_chandra_ocr.py output/travelguide
    python pipeline/run_chandra_ocr.py output/travelguide --method hf
    python pipeline/run_chandra_ocr.py output/travelguide --batch-size 4
    python pipeline/run_chandra_ocr.py output/travelguide --quiet
"""

import argparse
import re
import sys
import time
from pathlib import Path

MODEL_ID   = "datalab-to/chandra-ocr-2"
MODEL_SLUG = "chandra-ocr-2"


# ---------------------------------------------------------------------------
# Markdown → plain text
# ---------------------------------------------------------------------------

# Table separator rows like |---|---|:  all chars are -, |, :, or space
_TABLE_SEP_RE = re.compile(r"^\|?[\s|:\-]+\|?$")


def _strip_markdown(text: str) -> str:
    """Convert Chandra's Markdown output to plain text for align_ocr.py.

    Rules:
    - Table separator rows (|---|---|) are dropped.
    - Table data rows: pipe delimiters stripped, cells joined with two spaces.
    - Heading markers (# / ## / …) stripped from the start of the line.
    - Inline bold/italic (**, *, __, _) stripped.
    - Code fences (```) dropped.
    - Horizontal rules (---, ===) dropped.
    - Empty lines preserved (align_ocr.py ignores them; keeping them avoids
      accidentally merging lines that Chandra separated with a blank row).
    """
    out = []
    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Drop code fences
        if line.startswith("```"):
            continue

        # Drop table separator rows
        if "|" in line and _TABLE_SEP_RE.match(line) and "-" in line:
            continue

        # Drop standalone horizontal rules
        if re.match(r"^[-=]{3,}$", line):
            continue

        # Table data rows — strip pipe delimiters, join cells with two spaces
        if "|" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = [c for c in cells if c]
            line = "  ".join(cells)

        # Strip heading markers
        line = re.sub(r"^#{1,6}\s+", "", line)

        # Strip inline bold / italic (non-greedy to avoid eating whole paragraphs)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"\*(.+?)\*",     r"\1", line)
        line = re.sub(r"__(.+?)__",     r"\1", line)
        line = re.sub(r"_(.+?)_",       r"\1", line)

        out.append(line)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Scope / image discovery helpers (mirrors run_gemini_ocr.py)
# ---------------------------------------------------------------------------

def _load_scope(output_root: Path) -> "set[str] | None":
    """Return filenames to process from included_pages.txt, or None (= all)."""
    for d in (output_root.resolve(), output_root.resolve().parent):
        p = d / "included_pages.txt"
        if p.exists():
            lines = [
                ln.strip()
                for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            if lines:
                return set(lines)
    return None


def _collect_images(output_root: Path) -> list[Path]:
    """Return the list of .jpg images to process, respecting split-spread logic."""
    images: list[Path] = []
    for p in sorted(output_root.rglob("*.jpg")):
        if p.stem.endswith("_viz"):
            continue
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        left  = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        images.append(p)
    return images


# ---------------------------------------------------------------------------
# Per-image OCR
# ---------------------------------------------------------------------------

def process_image(image_path: Path, model, method: str) -> str:
    """OCR one image. Returns 'skipped', 'ok', or 'failed'."""
    txt_path = image_path.parent / f"{image_path.stem}_{MODEL_SLUG}.txt"

    if txt_path.exists():
        if txt_path.stat().st_size > 0:
            return "skipped"
        txt_path.unlink()  # empty from a previous run — retry

    try:
        from PIL import Image as PILImage

        img = PILImage.open(image_path).convert("RGB")

        if method == "vllm":
            from chandra.model import InferenceManager
            from chandra.model.schema import BatchInputItem

            result = model.generate([
                BatchInputItem(image=img, prompt_type="ocr_layout")
            ])[0]
            raw_md = result.markdown or result.raw or ""
        else:
            # HuggingFace transformers
            from chandra.model.hf import generate_hf
            from chandra.model.schema import BatchInputItem
            from chandra.output import parse_markdown

            result = generate_hf(
                [BatchInputItem(image=img, prompt_type="ocr_layout")],
                model,
            )[0]
            raw_md = parse_markdown(result.raw) if result.raw else (result.markdown or "")

        plain = _strip_markdown(raw_md)
        txt_path.write_text(plain, encoding="utf-8")
        return "ok"

    except Exception as exc:  # noqa: BLE001
        print(f"  Error processing {image_path.name}: {exc}", file=sys.stderr)
        return "failed"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(method: str):
    """Load and return the model object (expensive; call once)."""
    if method == "vllm":
        from chandra.model import InferenceManager
        print("Loading Chandra via vLLM…", file=sys.stderr)
        return InferenceManager(method="vllm")

    # HuggingFace
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading Chandra ({MODEL_ID}) via HuggingFace…", file=sys.stderr)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.1f}s.", file=sys.stderr)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chandra OCR on downloaded images, producing plain-text .txt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Root images directory to process (e.g. output/travelguide)",
    )
    parser.add_argument(
        "--method",
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend: 'hf' (HuggingFace, default) or 'vllm'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Images per inference batch (default: 1). Increase on high-VRAM GPUs "
            "for modest throughput gains; memory scales with batch size."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    if not output_root.exists():
        print(f"Error: directory not found: {output_root}", file=sys.stderr)
        sys.exit(1)

    images = _collect_images(output_root)
    if not images:
        print(f"No .jpg files found under {output_root}", file=sys.stderr)
        sys.exit(0)

    scope = _load_scope(output_root)
    if scope is not None:
        before = len(images)
        images = [p for p in images if p.name in scope]
        if not args.quiet:
            print(
                f"Scope filter: {len(images)} of {before} pages included"
                f" (from included_pages.txt)",
                file=sys.stderr,
            )

    total = len(images)
    if not args.quiet:
        print(
            f"Processing {total} image(s) with Chandra ({MODEL_SLUG}, method={args.method})…",
            file=sys.stderr,
        )

    model = load_model(args.method)

    counts = {"ok": 0, "skipped": 0, "failed": 0}

    # Process in batches (batch_size > 1 passes multiple images to generate_hf at once)
    batch_size = max(1, args.batch_size)
    completed = 0

    i = 0
    while i < total:
        batch_paths = images[i : i + batch_size]
        i += batch_size

        if batch_size == 1:
            path = batch_paths[0]
            completed += 1
            status = process_image(path, model, args.method)
            counts[status] += 1
            if not args.quiet:
                txt_name = f"{path.stem}_{MODEL_SLUG}.txt"
                if status == "skipped":
                    print(f"[{completed:04d}/{total}] Skipped (exists): {txt_name}", file=sys.stderr)
                elif status == "ok":
                    print(f"[{completed:04d}/{total}] Done: {txt_name}", file=sys.stderr)
                else:
                    print(f"[{completed:04d}/{total}] FAILED: {path.name}", file=sys.stderr)
        else:
            # Batch path: check skips individually, run OCR on the rest together
            to_run: list[Path] = []
            for path in batch_paths:
                txt_path = path.parent / f"{path.stem}_{MODEL_SLUG}.txt"
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    counts["skipped"] += 1
                    completed += 1
                    if not args.quiet:
                        print(f"[{completed:04d}/{total}] Skipped (exists): {txt_path.name}", file=sys.stderr)
                else:
                    to_run.append(path)

            if not to_run:
                continue

            try:
                from PIL import Image as PILImage
                from chandra.model.schema import BatchInputItem

                pil_images = [PILImage.open(p).convert("RGB") for p in to_run]
                batch_items = [
                    BatchInputItem(image=img, prompt_type="ocr_layout")
                    for img in pil_images
                ]

                if args.method == "vllm":
                    results = model.generate(batch_items)
                    raws = [r.markdown or r.raw or "" for r in results]
                else:
                    from chandra.model.hf import generate_hf
                    from chandra.output import parse_markdown
                    results = generate_hf(batch_items, model)
                    raws = [
                        parse_markdown(r.raw) if r.raw else (r.markdown or "")
                        for r in results
                    ]

                for path, raw_md in zip(to_run, raws):
                    completed += 1
                    txt_path = path.parent / f"{path.stem}_{MODEL_SLUG}.txt"
                    plain = _strip_markdown(raw_md)
                    txt_path.write_text(plain, encoding="utf-8")
                    counts["ok"] += 1
                    if not args.quiet:
                        print(f"[{completed:04d}/{total}] Done: {txt_path.name}", file=sys.stderr)

            except Exception as exc:  # noqa: BLE001
                for path in to_run:
                    completed += 1
                    counts["failed"] += 1
                    print(f"[{completed:04d}/{total}] FAILED: {path.name} — {exc}", file=sys.stderr)

    if not args.quiet:
        print(
            f"\nDone. {total} image(s): "
            f"{counts['ok']} processed, {counts['skipped']} skipped, {counts['failed']} failed.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
