#!/usr/bin/env python3
"""Run Surya OCR on images, saving line-level bounding boxes and text as JSON.

For each .jpg found in the images directory, runs Surya's recognition pipeline
and saves:
  {stem}_surya.json  — line bboxes, text, and confidence
  {stem}_surya.txt   — plain text (for compare_ocr.py)

Already-processed images (where _surya.json already exists) are skipped, so
the script is safe to re-run.

Surya processes images in batches for GPU/MPS efficiency.  Reduce --batch-size
if you hit out-of-memory errors.

Requires surya-ocr:
    pip install surya-ocr
    uv add surya-ocr

Output JSON schema per image:
    {
        "image_width": 2048,
        "image_height": 3000,
        "lines": [
            {
                "bbox": [x1, y1, x2, y2],
                "text": "The line text",
                "confidence": 0.95
            }
        ]
    }

Usage
-----
    python run_surya_ocr.py images/greenbooks
    python run_surya_ocr.py images/travelguide --batch-size 8
    python run_surya_ocr.py images/travelguide --quiet
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Suppress Surya's per-batch tqdm bars before any surya import
#os.environ.setdefault("SURYA_DISABLE_TQDM", "true")
#os.environ.setdefault("DISABLE_TQDM", "true")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Surya OCR on downloaded images, producing per-image JSON and txt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images_dir",
        help="Root images directory to process (e.g. images/greenbooks)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        metavar="N",
        help="Images per Surya inference batch (default: 4; reduce if OOM)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = parser.parse_args()

    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    # Load Surya models
    if not args.quiet:
        print("Loading Surya models…", file=sys.stderr)
    try:
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
    except ImportError as exc:
        print(
            f"Error: {exc}\n"
            "Install with:  pip install surya-ocr  (or: uv add surya-ocr)",
            file=sys.stderr,
        )
        sys.exit(1)

    t_load = time.monotonic()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(FoundationPredictor())
    if not args.quiet:
        print(f"Models loaded in {time.monotonic() - t_load:.1f}s.\n", file=sys.stderr)

    # Collect images to process (mirrors run_ocr.py split-aware logic)
    all_jpgs = sorted(images_root.rglob("*.jpg"))
    images_to_run: list[Path] = []
    skipped = 0
    for p in all_jpgs:
        if p.stem.endswith("_viz"):
            continue
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            if (p.parent / f"{p.stem}_surya.json").exists():
                skipped += 1
            else:
                images_to_run.append(p)
            continue
        left  = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue  # OCR the splits instead
        if (p.parent / f"{p.stem}_surya.json").exists():
            skipped += 1
        else:
            images_to_run.append(p)

    if not images_to_run and skipped == 0:
        print(f"No .jpg files found under {images_root}", file=sys.stderr)
        sys.exit(0)

    if not args.quiet:
        print(
            f"Processing {len(images_to_run)} image(s) ({skipped} already done)…",
            file=sys.stderr,
        )

    if not images_to_run:
        print("Nothing to do.", file=sys.stderr)
        sys.exit(0)

    from PIL import Image

    ok_count = 0
    fail_count = 0
    total = len(images_to_run)
    batch_size = args.batch_size
    n_batches = (total + batch_size - 1) // batch_size
    start_time = time.monotonic()

    for batch_start in range(0, total, batch_size):
        batch_paths = images_to_run[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        batch_end = min(batch_start + batch_size, total)

        try:
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
        except Exception as exc:
            for p in batch_paths:
                if not args.quiet:
                    print(
                        f"[{batch_start + 1:04d}/{total}] FAILED loading {p.name}: {exc}",
                        file=sys.stderr,
                    )
                fail_count += 1
            continue

        if not args.quiet:
            print(
                f"  [batch {batch_num}/{n_batches}] Running inference on images "
                f"{batch_start + 1}–{batch_end}…",
                file=sys.stderr,
            )

        t_batch = time.monotonic()
        try:
            ocr_results = rec_predictor(
                pil_images,
                det_predictor=det_predictor,
                sort_lines=True,
            )
        except Exception as exc:
            for i, p in enumerate(batch_paths):
                if not args.quiet:
                    print(
                        f"[{batch_start + i + 1:04d}/{total}] FAILED inference {p.name}: {exc}",
                        file=sys.stderr,
                    )
                fail_count += 1
            continue

        batch_elapsed = time.monotonic() - t_batch

        for i, (image_path, pil_img, result) in enumerate(
            zip(batch_paths, pil_images, ocr_results)
        ):
            idx = batch_start + i + 1
            try:
                w, h = pil_img.size
                lines = [
                    {
                        "bbox": [int(v) for v in ln.bbox],
                        "text": ln.text,
                        "confidence": round(float(getattr(ln, "confidence", 1.0)), 4),
                    }
                    for ln in result.text_lines
                ]
                output = {"image_width": w, "image_height": h, "lines": lines}
                json_path = image_path.parent / f"{image_path.stem}_surya.json"
                txt_path  = image_path.parent / f"{image_path.stem}_surya.txt"
                json_path.write_text(
                    json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                txt_path.write_text(
                    "\n".join(ln["text"] for ln in lines), encoding="utf-8"
                )
                if not args.quiet:
                    done = ok_count + fail_count + 1
                    elapsed = time.monotonic() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
                    print(
                        f"[{idx:04d}/{total}] Done: {json_path.name}"
                        f" ({len(lines)} lines,"
                        f" {batch_elapsed / len(batch_paths):.1f}s/img,"
                        f" ~{eta_str} left)",
                        file=sys.stderr,
                    )
                ok_count += 1
            except Exception as exc:
                if not args.quiet:
                    print(
                        f"[{idx:04d}/{total}] FAILED saving {image_path.name}: {exc}",
                        file=sys.stderr,
                    )
                fail_count += 1

    if not args.quiet:
        total_elapsed = time.monotonic() - start_time
        rate = ok_count / total_elapsed if total_elapsed > 0 and ok_count > 0 else 0
        print(
            f"\nDone in {total_elapsed:.1f}s ({rate:.2f} img/s). "
            f"{total} image(s): {ok_count} processed, "
            f"{skipped} already done, {fail_count} failed.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
