#!/usr/bin/env python3
"""Run local (free, on-device) OCR on images, saving plain-text output alongside each image.

A drop-in alternative to ``run_gemini_ocr.py`` for users who want to OCR large
volumes without paying per call — designed to run on Apple Silicon (e.g. a
MacBook Air M2) overnight. It writes the same ``{stem}_{slug}.txt`` filename
contract, so ``align_ocr.py`` / ``extract_entries.py`` consume its output with
zero downstream changes: Surya still provides the bounding-box geometry, this
stage just provides the text instead of Gemini.

Two engines, selected with ``--engine``:

  chandra (default)  Datalab's Chandra OCR 2 — a layout-aware VLM (Qwen3.5-based,
                     same vendor as Surya) run locally via ``mlx-vlm``. Best
                     quality on dense, multi-column historical print; follows
                     the OCR prompt's reading-order instruction. Slow on an M2
                     but fine for overnight batches. Needs the 8-bit MLX weights
                     (default: mlx-community/chandra-ocr-2-8bit-mlx).

  vision             Apple's native Vision OCR via ``ocrmac``. Near-instant and
                     near-zero memory, but a traditional OCR engine (weaker on
                     degraded scans) that returns text with no semantic column
                     reading order — we reconstruct reading order spatially from
                     its bounding boxes using the same ``sort_by_reading_order``
                     helper the aligner uses.

Both engines are Apple-Silicon / macOS-only. Install with::

    uv sync --extra local-ocr

Usage
-----
    python -m pipeline.run_local_ocr output/some_directory
    python -m pipeline.run_local_ocr output/some_directory --engine vision
    python -m pipeline.run_local_ocr output/some_directory --engine chandra \
        --model mlx-community/chandra-ocr-2-8bit-mlx

Validation recipe (Phase 1 — eyeball alignment vs the Gemini baseline)::

    python -m pipeline.run_local_ocr output/<slug>/ --engine chandra
    python -m pipeline.align_ocr     output/<slug>/ --model chandra-ocr-2 --force
    # then compare *_chandra-ocr-2_aligned.json vs the Gemini *_aligned.json
"""

import argparse
import functools
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# Reuse the Gemini stage's prompt-discovery, scope, and output-quality helpers so
# both backends behave identically (same blank-page handling, same quality gates,
# same dot-leader cleanup, same failure placeholder).
from pipeline.align_ocr import sort_by_reading_order
from pipeline.run_gemini_ocr import (
    PROMPT_FILE,
    _DITTO_INSTRUCTION,
    _OCR_FAILED_PLACEHOLDER,
    _clean_output,
    _find_prompt,
    _load_scope,
    _output_issue,
)
from utils.image_utils import is_blank_page
from utils.models import DEFAULT_LOCAL_OCR_MODEL, LOCAL_OCR_SLUGS

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


# Chandra defaults to markdown/HTML/JSON output; we want the same plain-text,
# one-line-per-printed-line contract the aligner expects. Steer it explicitly.
_CHANDRA_PLAINTEXT_GUARD = (
    "\n\nIMPORTANT: Return plain text only — no markdown, no HTML, no tables, "
    "no bullet points, and no code fences. Output each printed line as a single "
    "line of plain text, in the correct reading order."
)


# --------------------------------------------------------------------------- #
# Markdown / HTML → plain lines (defensive cleanup for VLM output)
# --------------------------------------------------------------------------- #
_HTML_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_HTML_BLOCK_CLOSE_RE = re.compile(r"</\s*(p|div|tr|li|h[1-6])\s*>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_FENCE_RE = re.compile(r"^\s*```.*$")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_MD_BULLET_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)")
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|`)(.+?)\1")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}.*$")


def _to_plain_lines(text: str) -> str:
    """Strip markdown / HTML residue a VLM may emit, leaving plain text lines.

    Deliberately conservative: it only removes formatting *scaffolding* (heading
    hashes, list bullets, emphasis wrappers, code fences, HTML tags, and pipe
    table pipes), never the words/punctuation themselves — over-aggressive
    stripping would corrupt names and addresses (dot-leaders, abbreviations).
    """
    if not text:
        return text
    # Normalise HTML line breaks to newlines, then drop remaining tags.
    text = _HTML_BR_RE.sub("\n", text)
    text = _HTML_BLOCK_CLOSE_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub("", text)

    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if _MD_FENCE_RE.match(line):
            continue  # drop ``` fences entirely
        if _TABLE_SEP_RE.match(line) and line.count("-") >= 2 and "|" in line:
            continue  # drop |---|---| table separator rows
        line = _MD_HEADING_RE.sub("", line)
        line = _MD_BULLET_RE.sub("", line)
        line = _MD_EMPHASIS_RE.sub(r"\2", line)
        # Pipe table row: "| a | b |" → "a b"
        if line.count("|") >= 2:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            line = " ".join(c for c in cells if c)
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Engine: Chandra via mlx-vlm
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _load_chandra(checkpoint: str):
    """Load the Chandra MLX model once and cache it (model load is the slow part)."""
    try:
        from mlx_vlm import load
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit(
            "mlx-vlm is required for --engine chandra. Install it with:\n"
            "    uv sync --extra local-ocr\n"
            "(Apple Silicon only.)"
        ) from e

    config = None
    try:
        from mlx_vlm.utils import load_config

        config = load_config(checkpoint)
    except Exception:  # noqa: BLE001 - older/newer mlx-vlm may not expose load_config
        config = None

    model, processor = load(checkpoint)
    if config is None:
        config = getattr(model, "config", None)
    return model, processor, config


def _chandra_ocr(image_path: Path, prompt: str, checkpoint: str, max_tokens: int) -> str:
    """OCR one page with Chandra (mlx-vlm). Returns plain text in reading order."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _load_chandra(checkpoint)
    formatted = apply_chat_template(processor, config, prompt, num_images=1)

    # mlx-vlm's generate signature has shifted across releases (temp vs
    # temperature; return type str vs object). Be defensive: pin the working
    # version once validated, but don't hard-fail on a kwarg rename here.
    gen_kwargs = dict(max_tokens=max_tokens, verbose=False)
    try:
        result = generate(model, processor, formatted, [str(image_path)], temperature=0.0, **gen_kwargs)
    except TypeError:
        try:
            result = generate(model, processor, formatted, [str(image_path)], temp=0.0, **gen_kwargs)
        except TypeError:
            result = generate(model, processor, formatted, [str(image_path)], **gen_kwargs)

    text = result if isinstance(result, str) else getattr(result, "text", str(result))
    return _to_plain_lines(text)


# --------------------------------------------------------------------------- #
# Engine: Apple Vision via ocrmac
# --------------------------------------------------------------------------- #
def _vision_ocr(image_path: Path) -> str:
    """OCR one page with Apple Vision (ocrmac), reconstructing column reading order.

    Apple Vision returns observations with normalized bottom-left-origin bboxes
    and no reading order. We convert each to a top-left-origin pixel bbox (Surya's
    convention) and feed them through ``sort_by_reading_order`` — the exact helper
    the aligner uses — so the OCR reading order matches the bbox reading order.
    """
    try:
        from ocrmac import ocrmac
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit(
            "ocrmac is required for --engine vision. Install it with:\n"
            "    uv sync --extra local-ocr\n"
            "(macOS only.)"
        ) from e

    width, height = Image.open(image_path).size
    annotations = ocrmac.OCR(str(image_path), recognition_level="accurate").recognize()

    observations: list[dict] = []
    for text, _confidence, bbox in annotations:
        nx, ny, nw, nh = bbox  # normalized, origin bottom-left
        x1 = nx * width
        x2 = (nx + nw) * width
        # Flip Y: Vision's y grows upward; Surya/align expect y growing downward.
        y_top = (1.0 - (ny + nh)) * height
        y_bottom = (1.0 - ny) * height
        observations.append(
            {
                "bbox": [round(x1), round(y_top), round(x2), round(y_bottom)],
                "text": text,
            }
        )

    ordered = sort_by_reading_order(observations, width)
    return "\n".join(o["text"] for o in ordered)


def ocr_page(engine: str, image_path: Path, prompt: str, checkpoint: str, max_tokens: int) -> str:
    if engine == "chandra":
        return _chandra_ocr(image_path, prompt, checkpoint, max_tokens)
    if engine == "vision":
        return _vision_ocr(image_path)
    raise ValueError(f"unknown engine: {engine!r}")


# --------------------------------------------------------------------------- #
# Per-image driver (mirrors run_gemini_ocr.process_image)
# --------------------------------------------------------------------------- #
def process_image(
    image_path: Path,
    engine: str,
    slug: str,
    system_prompt: str,
    checkpoint: str,
    max_tokens: int,
) -> str:
    """OCR one image. Returns a status string: 'skipped' | 'blank' | 'ok' | 'failed'."""
    txt_path = image_path.parent / f"{image_path.stem}_{slug}.txt"
    if txt_path.exists():
        if txt_path.stat().st_size > 0:
            return "skipped"
        _log(f"  Re-running (empty output file): {txt_path.name}")
        txt_path.unlink()

    # Blank pages waste compute and invite hallucination — skip without writing,
    # so re-runs just re-check cheaply (matches the Gemini stage's behaviour).
    if is_blank_page(image_path):
        return "blank"

    try:
        text = ocr_page(engine, image_path, system_prompt, checkpoint, max_tokens)
    except SystemExit:
        raise  # missing-dependency: abort the whole run, not just this page
    except Exception as exc:  # noqa: BLE001
        _log(f"  Engine error on {image_path.name}: {exc}")
        text = ""

    if text and _output_issue(text):
        text = _clean_output(text)

    if not text.strip():
        text = _OCR_FAILED_PLACEHOLDER
        _log(f"  No text recovered; writing placeholder: {image_path.name}")
        txt_path.write_text(text, encoding="utf-8")
        return "failed"

    txt_path.write_text(text, encoding="utf-8")
    return "ok"


def _discover_images(output_root: Path) -> list[Path]:
    """Same image-discovery rules as run_gemini_ocr: skip viz, prefer split halves."""
    images: list[Path] = []
    for p in sorted(output_root.rglob("*.jpg")):
        if p.stem.endswith("_viz"):
            continue
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        left = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        images.append(p)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local on-device OCR (Chandra MLX or Apple Vision), producing .txt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Root images directory to process (e.g. output/some_directory)",
    )
    parser.add_argument(
        "--engine",
        choices=("chandra", "vision"),
        default="chandra",
        help=(
            "OCR engine: 'chandra' (Datalab VLM via mlx-vlm, best quality, slow) "
            "or 'vision' (Apple Vision via ocrmac, fast baseline). Default: chandra."
        ),
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_LOCAL_OCR_MODEL,
        metavar="CHECKPOINT",
        help=(
            f"Chandra MLX checkpoint to load (default: {DEFAULT_LOCAL_OCR_MODEL}). "
            "Ignored for --engine vision. Output filename always uses the stable "
            "slug for the engine, so downstream discovery is unaffected by quant."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        metavar="N",
        help="Max generated tokens per page for Chandra (default: 8192).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel workers. Default depends on engine: 1 for chandra (a single "
            "shared MLX model is best driven serially) and 4 for vision."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="Custom OCR prompt for Chandra (default: prompts/ocr_prompt.md, or a volume-specific one).",
    )
    parser.add_argument(
        "--expand-dittos",
        dest="expand_dittos",
        action="store_true",
        help="Instruct Chandra to expand ditto marks ('' or 〃) in place rather than transcribe them literally.",
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

    engine = args.engine
    slug = LOCAL_OCR_SLUGS[engine]
    checkpoint = args.model
    workers = args.workers if args.workers is not None else (1 if engine == "chandra" else 4)

    # Build the system prompt (Chandra only; Apple Vision ignores prompts).
    system_prompt = ""
    if engine == "chandra":
        if args.prompt_file:
            prompt_path = Path(args.prompt_file)
        else:
            prompt_path = _find_prompt(output_root, PROMPT_FILE)
        if not prompt_path.exists():
            print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
            sys.exit(1)
        system_prompt = prompt_path.read_text(encoding="utf-8").rstrip()
        if args.expand_dittos:
            system_prompt += _DITTO_INSTRUCTION
        system_prompt += _CHANDRA_PLAINTEXT_GUARD
        if not args.quiet and prompt_path != PROMPT_FILE:
            print(f"Using volume prompt: {prompt_path}", file=sys.stderr)

    images = _discover_images(output_root)
    if not images:
        print(f"No .jpg files found under {output_root}", file=sys.stderr)
        sys.exit(0)

    scope = _load_scope(output_root)
    if scope is not None:
        before = len(images)
        images = [p for p in images if p.name in scope]
        if not args.quiet:
            print(
                f"Scope filter: {len(images)} of {before} pages included (from included_pages.txt)",
                file=sys.stderr,
            )

    total = len(images)
    if not args.quiet:
        engine_note = checkpoint if engine == "chandra" else "Apple Vision"
        print(
            f"Processing {total} image(s) with {workers} worker(s) using {engine} [{engine_note}]…",
            file=sys.stderr,
        )

    counts = {"ok": 0, "skipped": 0, "blank": 0, "failed": 0}
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_image, img, engine, slug, system_prompt, checkpoint, args.max_tokens): img
            for img in images
        }
        for future in as_completed(futures):
            image_path = futures[future]
            completed += 1
            try:
                status = future.result()
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                _log(f"Warning: exception processing {image_path}: {exc}")
            counts[status] += 1

            if not args.quiet:
                txt_name = f"{image_path.stem}_{slug}.txt"
                if status == "skipped":
                    _log(f"[{completed:04d}/{total}] Skipped (exists): {txt_name}")
                elif status == "blank":
                    _log(f"[{completed:04d}/{total}] Blank page — no OCR: {image_path.name}")
                elif status == "ok":
                    _log(f"[{completed:04d}/{total}] Done: {txt_name}")
                else:
                    _log(f"[{completed:04d}/{total}] FAILED: {image_path.name}")

    if not args.quiet:
        print(
            f"\nDone. {total} image(s): "
            f"{counts['ok']} processed, {counts['skipped']} skipped, "
            f"{counts['blank']} blank, {counts['failed']} failed.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
