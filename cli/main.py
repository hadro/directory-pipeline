#!/usr/bin/env python3
"""Directory pipeline CLI.

Usage
-----
    pipeline run      URL   [flags]   # download → OCR → extract → explore
    pipeline guided   URL   [flags]   # human-in-loop with page selection + review
    pipeline ingest   URL   [flags]   # download IIIF images
    pipeline calibrate DIR  [flags]   # select sample pages + generate prompts
    pipeline ocr      DIR   [flags]   # Surya OCR + Gemini OCR + alignment
    pipeline extract  DIR   [flags]   # NER extraction + explorer
    pipeline review   DIR             # interactive alignment review (Flask UI)
    pipeline postprocess DIR [flags]  # fix + combine + explorer (post-extraction)

Examples
--------
    pipeline run https://archive.org/details/ldpd_11290437_000/
    pipeline run https://www.loc.gov/collections/civil-war-maps/
    pipeline guided https://archive.org/details/... --sections sections.txt
    pipeline ocr output/my_vol/ --model gemini-2.0-flash --workers 8 --flex
    pipeline extract output/my_vol/ --mode multimodal
    pipeline postprocess output/green_books_and_related/ --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_MAIN = str(_ROOT / "main.py")
_POSTPROCESS = str(_ROOT / "pipeline" / "postprocess.py")


def _exec(args: list[str]) -> None:
    result = subprocess.run([sys.executable] + args, check=False)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Per-subcommand parsers
# ---------------------------------------------------------------------------

def _parser_run() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline run",
        description=(
            "One-shot automated pipeline: download → Gemini OCR → "
            "extract entries → build explorer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="LoC/IA URL, IIIF manifest URL, or pre-built CSV path")
    p.add_argument("--model", "-m", metavar="MODEL", default=None,
                   help="Gemini model for OCR + NER (default: gemini-2.0-flash for OCR, gemini-3.1-flash-lite for NER)")
    p.add_argument("--ner-prompt", metavar="FILE", default=None,
                   help="Custom NER system prompt (reuse from a prior calibrated volume)")
    p.add_argument("--ocr-prompt", metavar="FILE", default=None,
                   help="Custom OCR system prompt")
    p.add_argument("--mode", choices=["text-only", "multimodal"], default=None,
                   help="Entry extraction mode (default: text-only)")
    p.add_argument("--flex", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Gemini Flex inference (~50%% cheaper, 1–15 min latency). "
                        "On by default. Pass --no-flex for time-sensitive runs.")
    p.add_argument("--workers", "-w", type=int, metavar="N", default=None,
                   help="Parallel workers for OCR stages")
    p.add_argument("--sections", metavar="PATH", default=None,
                   help="Path to sections.txt marking structural boundaries in a volume")
    p.add_argument("--slug", metavar="SLUG", default=None,
                   help="Override the auto-generated output folder name")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p


def _run(args: argparse.Namespace) -> None:
    cmd = [_MAIN, args.source, "--extract"]
    if args.model:       cmd += ["--ocr-model", args.model]
    if args.ner_prompt:  cmd += ["--ner-prompt", args.ner_prompt]
    if args.ocr_prompt:  cmd += ["--ocr-prompt", args.ocr_prompt]
    if args.mode:        cmd += ["--mode", args.mode]
    if args.flex:        cmd.append("--flex")
    if args.workers:     cmd += ["--workers", str(args.workers)]
    if args.sections:    cmd += ["--sections", args.sections]
    if args.slug:        cmd += ["--slug", args.slug]
    if args.dry_run:     cmd.append("--dry-run")
    _exec(cmd)


def _parser_guided() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline guided",
        description=(
            "Human-in-loop pipeline: download → select pages → Surya OCR → "
            "Gemini OCR → align → review alignment → extract → geocode → map.\n\n"
            "Pauses at --select-pages and --review-alignment for user input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="LoC/IA URL, IIIF manifest URL, or pre-built CSV path")
    p.add_argument("--model", "-m", metavar="MODEL", default=None,
                   help="Gemini model for OCR + NER")
    p.add_argument("--ner-prompt", metavar="FILE", default=None,
                   help="Custom NER system prompt (reuse from a calibrated volume)")
    p.add_argument("--ocr-prompt", metavar="FILE", default=None,
                   help="Custom OCR system prompt")
    p.add_argument("--sections", metavar="PATH", default=None,
                   help="Path to sections.txt marking structural boundaries")
    p.add_argument("--workers", "-w", type=int, metavar="N", default=None,
                   help="Parallel workers (default: 8 in guided mode)")
    p.add_argument("--flex", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Gemini Flex inference (~50%% cheaper, 1–15 min latency). "
                        "On by default. Pass --no-flex for time-sensitive runs.")
    p.add_argument("--slug", metavar="SLUG", default=None,
                   help="Override the auto-generated output folder name")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p


def _guided(args: argparse.Namespace) -> None:
    cmd = [_MAIN, args.source, "--guided"]
    if args.model:       cmd += ["--ocr-model", args.model]
    if args.ner_prompt:  cmd += ["--ner-prompt", args.ner_prompt]
    if args.ocr_prompt:  cmd += ["--ocr-prompt", args.ocr_prompt]
    if args.sections:    cmd += ["--sections", args.sections]
    if args.workers:     cmd += ["--workers", str(args.workers)]
    if args.flex:        cmd.append("--flex")
    if args.slug:        cmd += ["--slug", args.slug]
    if args.dry_run:     cmd.append("--dry-run")
    _exec(cmd)


def _parser_ingest() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline ingest",
        description="Download IIIF images for a collection or item.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="LoC/IA URL, IIIF manifest URL, or pre-built CSV path")
    p.add_argument("--width", type=int, metavar="PX", default=None,
                   help="Image download width in pixels")
    p.add_argument("--detect-spreads", action="store_true",
                   help="Detect double-page spreads after downloading")
    p.add_argument("--split-spreads", action="store_true",
                   help="Split detected spreads into left/right pages")
    p.add_argument("--slug", metavar="SLUG", default=None,
                   help="Override the auto-generated output folder name")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p


def _ingest(args: argparse.Namespace) -> None:
    cmd = [_MAIN, args.source, "--download"]
    if args.detect_spreads: cmd.append("--detect-spreads")
    if args.split_spreads:  cmd.append("--split-spreads")
    if args.width:          cmd += ["--width", str(args.width)]
    if args.slug:           cmd += ["--slug", args.slug]
    if args.dry_run:        cmd.append("--dry-run")
    _exec(cmd)


def _parser_calibrate() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline calibrate",
        description=(
            "One-time setup for a new collection type: select representative sample\n"
            "pages, then have Gemini generate tailored OCR + NER prompts from them.\n\n"
            "Run this once per collection type, then reuse the generated prompts\n"
            "for all subsequent volumes with --ner-prompt and --ocr-prompt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="URL or output directory for the volume")
    p.add_argument("--sections", metavar="PATH", default=None,
                   help="Path to sections.txt marking structural boundaries (e.g. "
                        "alphabetical / street / business sections)")
    p.add_argument("--model", "-m", metavar="MODEL", default=None,
                   help="Gemini model to use for prompt generation")
    p.add_argument("--ocr-only", action="store_true",
                   help="Generate only the OCR transcription prompt (skip NER)")
    p.add_argument("--ner-only", action="store_true",
                   help="Generate only the NER entry-extraction prompt (skip OCR)")
    p.add_argument("--no-open", action="store_true",
                   help="Generate the page-selector HTML without opening a browser")
    p.add_argument("--selection", metavar="PATH", default=None,
                   help="Explicit path to an existing selection.txt (skips --select-pages)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p


def _calibrate(args: argparse.Namespace) -> None:
    cmd = [_MAIN, args.source, "--select-pages", "--generate-prompts"]
    if args.sections:   cmd += ["--sections", args.sections]
    if args.model:      cmd += ["--prompt-model", args.model]
    if args.ocr_only:   cmd.append("--ocr-only")
    if args.ner_only:   cmd.append("--ner-only")
    if args.no_open:    cmd.append("--no-open")
    if args.selection:  cmd += ["--selection", args.selection]
    if args.dry_run:    cmd.append("--dry-run")
    _exec(cmd)


def _parser_ocr() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline ocr",
        description=(
            "Run Surya OCR (line bboxes) + Gemini OCR (text) on downloaded images,\n"
            "then align Gemini text to Surya bboxes with Needleman-Wunsch.\n\n"
            "Requires images downloaded by `pipeline ingest` or `pipeline run`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="Output directory (e.g. output/my_vol/)")
    p.add_argument("--model", "-m", metavar="MODEL", default=None,
                   help="Gemini model for OCR + alignment (default: gemini-2.0-flash)")
    p.add_argument("--ocr-prompt", metavar="FILE", default=None,
                   help="Custom OCR system prompt file")
    p.add_argument("--sections", metavar="PATH", default=None,
                   help="Path to sections.txt for per-section prompt routing")
    p.add_argument("--workers", "-w", type=int, metavar="N", default=None,
                   help="Parallel workers for Gemini OCR")
    p.add_argument("--flex", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Gemini Flex inference (~50%% cheaper, 1–15 min latency). "
                        "On by default. Pass --no-flex for time-sensitive runs.")
    p.add_argument("--high-res", action="store_true",
                   help="Use high media resolution for Gemini (higher token cost)")
    p.add_argument("--expand-dittos", action="store_true",
                   help="Expand ditto marks in tabular documents")
    p.add_argument("--no-align", action="store_true",
                   help="Skip the alignment step (run Surya + Gemini OCR only)")
    p.add_argument("--surya-only", action="store_true",
                   help="Run only Surya OCR (skip Gemini OCR and alignment)")
    p.add_argument("--gemini-only", action="store_true",
                   help="Run only Gemini OCR (skip Surya and alignment)")
    p.add_argument("--force", action="store_true",
                   help="Force re-processing of already-aligned files")
    p.add_argument("--min-surya-confidence", type=float, metavar="THRESHOLD", default=None,
                   help="Skip Surya lines below this confidence (0.0–1.0, default: 0.35)")
    p.add_argument("--batch-size", type=int, metavar="N", default=None,
                   help="Images per Surya inference batch (reduce if OOM)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p


def _ocr(args: argparse.Namespace) -> None:
    if args.surya_only:
        stages = ["--surya-ocr"]
    elif args.gemini_only:
        stages = ["--gemini-ocr"]
    elif args.no_align:
        stages = ["--surya-ocr", "--gemini-ocr"]
    else:
        stages = ["--surya-ocr", "--gemini-ocr", "--align-ocr"]

    cmd = [_MAIN, args.source] + stages
    if args.model:                  cmd += ["--ocr-model", args.model]
    if args.ocr_prompt:             cmd += ["--ocr-prompt", args.ocr_prompt]
    if args.sections:               cmd += ["--sections", args.sections]
    if args.workers:                cmd += ["--workers", str(args.workers)]
    if args.flex:                   cmd.append("--flex")
    if args.high_res:               cmd.append("--high-res")
    if args.expand_dittos:          cmd.append("--expand-dittos")
    if args.force:                  cmd.append("--force")
    if args.min_surya_confidence is not None:
        cmd += ["--min-surya-confidence", str(args.min_surya_confidence)]
    if args.batch_size:             cmd += ["--batch-size", str(args.batch_size)]
    if args.dry_run:                cmd.append("--dry-run")
    _exec(cmd)


def _parser_extract() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline extract",
        description=(
            "Extract structured entries from aligned OCR output and build an\n"
            "interactive HTML explorer.\n\n"
            "Requires aligned JSON files from `pipeline ocr`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="Output directory (e.g. output/my_vol/)")
    p.add_argument("--model", "-m", metavar="MODEL", default=None,
                   help="Gemini model for NER (default: gemini-3.1-flash-lite)")
    p.add_argument("--ner-prompt", metavar="FILE", default=None,
                   help="Custom NER system prompt (reuse from a calibrated volume)")
    p.add_argument("--mode", choices=["text-only", "multimodal"], default=None,
                   help="Extraction mode: text-only (default) or multimodal "
                        "(sends images to Gemini alongside text)")
    p.add_argument("--sections", metavar="PATH", default=None,
                   help="Path to sections.txt for context reset at section boundaries")
    p.add_argument("--workers", "-w", type=int, metavar="N", default=None,
                   help="Parallel workers for NER extraction")
    p.add_argument("--geocode", action="store_true",
                   help="Geocode extracted addresses to lat/lon")
    p.add_argument("--map", action="store_true",
                   help="Generate an interactive HTML map of geocoded entries")
    p.add_argument("--no-explore", action="store_true",
                   help="Skip building the HTML explorer")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p


def _extract(args: argparse.Namespace) -> None:
    stages = ["--extract-entries"]
    if not args.no_explore: stages.append("--explore")
    if args.geocode:        stages.append("--geocode")
    if args.map:            stages.append("--map")

    cmd = [_MAIN, args.source] + stages
    if args.model:      cmd += ["--ocr-model", args.model]
    if args.ner_prompt: cmd += ["--ner-prompt", args.ner_prompt]
    if args.mode:       cmd += ["--mode", args.mode]
    if args.sections:   cmd += ["--sections", args.sections]
    if args.workers:    cmd += ["--workers", str(args.workers)]
    if args.dry_run:    cmd.append("--dry-run")
    _exec(cmd)


def _parser_review() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline review",
        description=(
            "Launch the interactive Flask alignment review UI.\n\n"
            "Opens a local web server to inspect and correct misaligned entries.\n"
            "Access at http://localhost:5000 (or via ngrok/Colab proxy)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="Output directory (e.g. output/my_vol/)")
    p.add_argument("--model", "-m", metavar="MODEL", default=None,
                   help="Model slug to review (auto-detected if omitted)")
    return p


def _review(args: argparse.Namespace) -> None:
    cmd = [_MAIN, args.source, "--review-alignment"]
    if args.model: cmd += ["--ocr-model", args.model]
    _exec(cmd)


def _parser_postprocess() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline postprocess",
        description=(
            "Run post-extraction cleanup in sequence:\n"
            "  1. fix_entries   — normalize unicode, categories, flag QA issues\n"
            "  2. combine_volumes — merge per-volume CSVs into one combined CSV\n"
            "  3. explore_entries — build interactive HTML explorer\n\n"
            "The model used during extraction is read from pipeline_state.json\n"
            "automatically — no --model flag needed for most runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("directory", type=Path,
                   help="Output slug directory (e.g. output/green_books_and_related/)")
    p.add_argument("--no-combine", action="store_true",
                   help="Skip the combine step (for single-volume directories)")
    p.add_argument("--model", "-m", metavar="MODEL", default=None,
                   help="Override the auto-detected model slug")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without writing any files")
    return p


def _postprocess(args: argparse.Namespace) -> None:
    cmd = [_POSTPROCESS, str(args.directory)]
    if args.no_combine: cmd.append("--no-combine")
    if args.model:      cmd += ["--model", args.model]
    if args.dry_run:    cmd.append("--dry-run")
    _exec(cmd)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SUBPARSERS = {
    "run":         (_parser_run,         _run),
    "guided":      (_parser_guided,      _guided),
    "ingest":      (_parser_ingest,      _ingest),
    "calibrate":   (_parser_calibrate,   _calibrate),
    "ocr":         (_parser_ocr,         _ocr),
    "extract":     (_parser_extract,     _extract),
    "review":      (_parser_review,      _review),
    "postprocess": (_parser_postprocess, _postprocess),
}

_DESCRIPTIONS = {
    "run":         "Download, OCR, extract, build explorer (one-shot automated)",
    "guided":      "Human-in-loop: page selection + alignment review + full pipeline",
    "ingest":      "Download IIIF images for a collection or item",
    "calibrate":   "Select sample pages + generate OCR/NER prompts (once per collection type)",
    "ocr":         "Run Surya OCR + Gemini OCR + align bboxes",
    "extract":     "Extract structured entries + build HTML explorer",
    "review":      "Interactive Flask UI to correct alignment mismatches",
    "postprocess": "Fix entries, combine volumes, build explorer (post-extraction)",
}


def main() -> None:
    # Top-level: show subcommand list or dispatch.
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        col = max(len(k) for k in _DESCRIPTIONS) + 2
        print("Subcommands:")
        for name, desc in _DESCRIPTIONS.items():
            print(f"  {name:<{col}}{desc}")
        print(f"\nRun `pipeline <subcommand> --help` for per-subcommand options.")
        sys.exit(0)

    subcommand = sys.argv[1]

    if subcommand not in _SUBPARSERS:
        # Unknown — pass everything to main.py (full legacy flag passthrough).
        _exec([_MAIN] + sys.argv[1:])

    make_parser, handler = _SUBPARSERS[subcommand]
    parser = make_parser()
    args = parser.parse_args(sys.argv[2:])
    handler(args)


if __name__ == "__main__":
    main()
