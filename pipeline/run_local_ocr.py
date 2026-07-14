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
                     (default: jwindle47/chandra-ocr-2-8bit-mlx, pinned revision).

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
        --model jwindle47/chandra-ocr-2-8bit-mlx

Validation recipe (Phase 1 — eyeball alignment vs the Gemini baseline)::

    python -m pipeline.run_local_ocr output/<slug>/ --engine chandra
    python -m pipeline.align_ocr     output/<slug>/ --model chandra-ocr-2 --force
    # then compare *_chandra-ocr-2_aligned.json vs the Gemini *_aligned.json
"""

import argparse
import functools
import html
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
from utils.models import (
    DEFAULT_LOCAL_OCR_MODEL,
    DEFAULT_LOCAL_OCR_REVISION,
    LOCAL_OCR_SLUGS,
)

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)



# --------------------------------------------------------------------------- #
# Markdown / HTML → plain lines (defensive cleanup for VLM output)
# --------------------------------------------------------------------------- #
_HTML_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
# Closing tags that end a printed line. td/th included so multi-column blocks
# Chandra renders as an HTML <table> (e.g. a two-column advertisement index)
# split into one line per cell instead of jamming both columns' entries — and
# their adjacent words — onto a single line.
_HTML_BLOCK_CLOSE_RE = re.compile(r"</\s*(p|div|tr|td|th|li|h[1-6])\s*>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_FENCE_RE = re.compile(r"^\s*```.*$")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_MD_BULLET_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)")
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|`)(.+?)\1")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}.*$")

# Decorative horizontal rules a VLM may emit for printed dividers (e.g. a long
# run of "~~~~" or "----"). Drop them: they carry no text and would otherwise
# survive as a junk line that pollutes alignment and NER. The char set excludes
# "." so address dot-leaders are never mistaken for a rule.
_RULE_CHARS = frozenset("~-=_*—–─━‾¯")


def _is_rule_line(line: str) -> bool:
    s = line.strip().replace(" ", "")
    return len(s) >= 3 and not any(c.isalnum() for c in s) and all(c in _RULE_CHARS for c in s)


# Chandra (mlx-vlm) does not always stop at the <|im_end|> turn boundary: after
# the real transcription it can re-open an `assistant` turn and partially repeat
# the page. Everything from the first turn boundary on is spurious, so truncate
# there. Must run on the raw output *before* _to_plain_lines, because that strips
# <|im_end|>-style tokens (they look like HTML tags) and would hide the boundary.
_CHAT_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
_ROLE_LINE_RE = re.compile(r"^\s*(?:assistant|user|system)\s*$", re.IGNORECASE)


def _truncate_chat_continuation(text: str) -> str:
    m = _CHAT_TOKEN_RE.search(text)
    if m:
        text = text[: m.start()]
    out: list[str] = []
    for line in text.splitlines():
        if _ROLE_LINE_RE.match(line):
            break
        out.append(line)
    return "\n".join(out)


# Chandra wraps illustrations/figures in data-label="Image" blocks and fills them
# with a VLM-generated description (prose + an <img alt="…"> caption) rather than a
# transcription of printed words — e.g. "An illustration of a counter show case…".
# Drop these blocks when deriving the plain text; keep all real-content labels
# (Text, Caption, *-Header, Table). The full HTML is still saved as a sidecar, so
# nothing is lost. Assumes Chandra's flat block layout (no <div> nested in a
# <div>), which holds for its ocr_layout output.
_DROP_LABELS = ("Image",)
_DROP_BLOCK_RE = re.compile(
    r'<div\b[^>]*\bdata-label="(?:' + "|".join(_DROP_LABELS) + r')"[^>]*>.*?</div>',
    re.IGNORECASE | re.DOTALL,
)


def _strip_label_blocks(html: str) -> str:
    """Remove figure/illustration blocks (see _DROP_LABELS) from Chandra HTML."""
    return _DROP_BLOCK_RE.sub("", html)


def _to_plain_lines(text: str) -> str:
    """Strip markdown / HTML residue a VLM may emit, leaving plain text lines.

    Deliberately conservative: it only removes formatting *scaffolding* (heading
    hashes, list bullets, emphasis wrappers, code fences, HTML tags, and pipe
    table pipes), never the words/punctuation themselves — over-aggressive
    stripping would corrupt names and addresses (dot-leaders, abbreviations).
    """
    if not text:
        return text
    # Normalise HTML line breaks to newlines, then drop remaining tags. Decode
    # entities (&amp; → &, &#39; → ') only after tag removal so we never recreate
    # a tag from escaped angle brackets.
    text = _HTML_BR_RE.sub("\n", text)
    text = _HTML_BLOCK_CLOSE_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub("", text)
    # Drop a dangling, unclosed tag fragment a truncated generation may leave
    # (e.g. a trailing "<div data-bbox=\"…" with no closing ">").
    text = re.sub(r"<[^>]*$", "", text)
    text = html.unescape(text)

    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if _MD_FENCE_RE.match(line):
            continue  # drop ``` fences entirely
        if _is_rule_line(line):
            continue  # drop decorative horizontal rules (~~~~, ----, ____)
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

    # Pin the default mirror to a verified commit; a user-supplied --model repo
    # is loaded at its main revision (revision=None == main for mlx-vlm).
    revision = (
        DEFAULT_LOCAL_OCR_REVISION if checkpoint == DEFAULT_LOCAL_OCR_MODEL else None
    )

    config = None
    try:
        from mlx_vlm.utils import load_config

        config = load_config(checkpoint, revision=revision) if revision else load_config(checkpoint)
    except Exception:  # noqa: BLE001 - older/newer mlx-vlm may not expose load_config
        config = None

    model, processor = load(checkpoint, revision=revision)
    if config is None:
        config = getattr(model, "config", None)
    return model, processor, config


def _chandra_eos_ids(processor) -> list[int]:
    """Token ids that should end generation. The checkpoint's configured eos is
    only <|endoftext|> (end of document), so generation runs past the <|im_end|>
    turn boundary into a spurious second `assistant` turn — burning the whole
    token budget and re-emitting the page. Adding <|im_end|> as a stop token ends
    generation right after the transcription: faster, and no spurious turn to
    clean up. Derived from the tokenizer so it's not a brittle hardcoded id."""
    tok = getattr(processor, "tokenizer", processor)
    fn = getattr(tok, "convert_tokens_to_ids", None)
    if fn is None:
        return []
    ids = []
    for t in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = fn(t)
        except Exception:  # noqa: BLE001 - tokenizer API varies across versions
            continue
        if isinstance(tid, int) and tid >= 0 and tid not in ids:
            ids.append(tid)
    return ids


def _chandra_ocr(
    image_path: Path, prompt: str, checkpoint: str, max_tokens: int
) -> tuple[str, str]:
    """OCR one page with Chandra (mlx-vlm). Returns (plain_text, native_html),
    both in reading order; native_html keeps Chandra's layout markup."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _load_chandra(checkpoint)

    # Chandra 2 is Qwen3.5-based: its chat template defaults to emitting a
    # chain-of-thought <think> block. Left on, the model spends the token budget
    # "thinking", never terminates its turn cleanly, and degenerates into
    # repeating the page (and leaking the literal "assistant" role marker).
    # enable_thinking=False injects an empty <think></think> so it goes straight
    # to the transcription. Fall back if a processor version rejects the kwarg.
    try:
        formatted = apply_chat_template(
            processor, config, prompt, num_images=1, enable_thinking=False
        )
    except TypeError:
        formatted = apply_chat_template(processor, config, prompt, num_images=1)

    # Greedy (temp=0.0) decoding is repetition-prone on long OCR; a mild
    # repetition penalty further guards against the loop. mlx-vlm's generate
    # signature has shifted across releases (temp vs temperature; repetition_
    # penalty may be unsupported), so apply each kwarg defensively.
    base = dict(max_tokens=max_tokens, verbose=False)
    eos_ids = _chandra_eos_ids(processor)
    if eos_ids:
        base["eos_tokens"] = eos_ids
    for extra in (
        dict(temperature=0.0, repetition_penalty=1.1),
        dict(temp=0.0, repetition_penalty=1.1),
        dict(temperature=0.0),
        dict(temp=0.0),
        {},
    ):
        try:
            result = generate(model, processor, formatted, [str(image_path)], **base, **extra)
            break
        except TypeError:
            continue

    raw = result if isinstance(result, str) else getattr(result, "text", str(result))
    # native = de-looped output with Chandra's HTML layout (data-bbox/data-label,
    # <br/> entry wraps, <h2> headers) intact. Returned alongside the plain text
    # so the rich structure can be saved once and re-parsed later (figure
    # filtering, column anchors, entry boundaries) without re-running inference.
    native = _truncate_chat_continuation(raw)
    # .txt drops figure-description blocks; .html keeps the full native layout.
    return _to_plain_lines(_strip_label_blocks(native)), native


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


def ocr_page(
    engine: str, image_path: Path, prompt: str, checkpoint: str, max_tokens: int
) -> tuple[str, str | None]:
    """Returns (plain_text, native_output). native_output is Chandra's raw layout
    HTML (saved as a sidecar for richer post-processing) or None for engines that
    have no richer form than their plain text."""
    if engine == "chandra":
        return _chandra_ocr(image_path, prompt, checkpoint, max_tokens)
    if engine == "vision":
        return _vision_ocr(image_path), None
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

    native: str | None = None
    try:
        text, native = ocr_page(engine, image_path, system_prompt, checkpoint, max_tokens)
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
    # Save the engine's native layout output (Chandra HTML) as a sidecar so its
    # structure can be re-parsed later without re-running inference. Additive: no
    # other stage reads it, and the .txt contract is unchanged.
    if native:
        (image_path.parent / f"{image_path.stem}_{slug}.html").write_text(
            native, encoding="utf-8"
        )
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
        # NB: we deliberately do NOT append an anti-markdown/HTML guard. Chandra
        # is trained to emit HTML with data-bbox blocks; suppressing that fights
        # the model and destabilises generation (image captioning, repetition).
        # Let it emit its native format and let _to_plain_lines strip the tags.
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
