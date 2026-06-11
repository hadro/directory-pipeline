#!/usr/bin/env python3
"""Digital Collections pipeline orchestrator (Library of Congress, Internet Archive, IIIF).

Reads a list of collection URLs from a text file (or a single URL passed
directly) and runs the requested pipeline stages in order for each.
A human-readable slug is derived automatically and used as the base name for
every output file and directory, so all stages stay in sync:

    output/{slug}/{slug}.csv
    output/{slug}/

Stages run in this fixed order regardless of the order flags appear on the
command line:

  --loc-csv         sources/loc_collection_csv.py   → output/{slug}/{slug}.csv  (LoC only)
  --ia-csv          sources/ia_collection_csv.py    → output/{slug}/{slug}.csv  (Internet Archive only)
  --iiif-csv        sources/iiif_manifest_csv.py    → output/{slug}/{slug}.csv  (any IIIF manifest or collection)
  --download        pipeline/download_images.py       → output/{slug}/
  --detect-spreads  pipeline/detect_spreads.py        (double-page spread detection)
  --split-spreads   pipeline/split_spreads.py         (split spreads into left/right pages)
  --surya-detect    pipeline/surya_detect.py          (Surya neural column detection → columns_report.csv)
  --detect-columns  pipeline/detect_columns.py        (pixel-projection column detection → columns_report.csv)
  --surya-ocr       pipeline/run_surya_ocr.py         (Surya OCR → *_surya.json line bboxes + *_surya.txt)
  --gemini-ocr      pipeline/run_gemini_ocr.py        (Gemini OCR)
  --compare-ocr     pipeline/compare_ocr.py           (side-by-side model comparison; accepts "surya" token)
  --align-ocr       pipeline/align_ocr.py             (NW alignment of Gemini text to Surya bboxes)
  --visualize       pipeline/visualize_alignment.py   (draw alignment boxes on images → *_viz.jpg)
  --review-alignment pipeline/review_alignment.py     (interactive UI to correct unmatched entries → *_aligned.json)
  --extract-entries pipeline/extract_entries.py        (extract structured entries from aligned OCR)
  --geocode         pipeline/geo/geocode_entries.py    (geocode entries to lat/lon)
  --map             pipeline/geo/map_entries.py        (generate interactive HTML map)

  --select-pages        pipeline/select_pages.py  (interactive browser UI for picking sample pages and scoping entry pages; opens HTML in browser)
  --generate-prompts    pipeline/generate_prompt.py  (Gemini generates volume-specific OCR + NER prompts from sample pages)

  --guided              shorthand for --download --select-pages --surya-ocr --gemini-ocr
                          --align-ocr --review-alignment --extract-entries --explore
                          (the --extract pipeline plus human-in-the-loop stages;
                          defaults --batch-size to 4 and --workers to 8;
                          run --generate-prompts first for new collection types;
                          add --geocode --map for materials with addresses)

Key model flags:
  --ocr-model MODEL     Gemini model for OCR and downstream stages (each stage uses its own default if omitted)
  --prompt-model MODEL  Gemini model for --generate-prompts (default: gemini-3-flash-preview)

Usage
-----
    # Library of Congress collection or item:
    python main.py https://www.loc.gov/collections/civil-war-maps/ \\
        --loc-csv --download --gemini-ocr
    # Internet Archive item or collection:
    python main.py https://archive.org/details/ldpd_11290437_000/ \\
        --ia-csv --download --gemini-ocr
    python main.py https://archive.org/details/durstoldyorklibrary \\
        --ia-csv --download --gemini-ocr

    # Any IIIF manifest URL (no API key needed):
    python main.py https://dcmny.org/do/<uuid>/metadata/iiifmanifest/default.jsonld \\
        --download --gemini-ocr
    python main.py https://example.org/iiif/collection.json \\
        --iiif-csv --download --gemini-ocr   # IIIF Collection: enumerates all items

    # Pre-built CSV (any source):
    python main.py output/my_items/my_items.csv --download --gemini-ocr

    # Multiple models / parallel workers:
    python main.py collections.txt --download \\
        --compare-ocr --models gemini-2.0-flash gemini-1.5-flash \\
        --workers 8
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).parent

from utils.iiif_utils import manifest_item_id as _iiif_manifest_item_id
from sources.loc_utils import _resource_url_to_item_url, loc_slug
from sources.ia_utils import _extract_ia_identifier, _fetch_ia_info, _make_ia_slug
from pipeline.state import write_state, record_stage
from pipeline.stages import STAGES, STAGE_BY_NAME, build_declarative_args
from utils.models import DEFAULT_OCR_MODEL

# Stage order, flags, and declarative argv specs live in pipeline/stages.py —
# the single registry shared with app.py. Stages whose argv needs real logic
# (source validation, file discovery, multi-run expansion) are handled
# explicitly in build_stage_args() below.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_targets(source: str) -> list[str]:
    """
    Return a list of URL/UUID strings.
    If source is an existing file, read non-blank, non-comment lines from it.
    CSV files (.csv) are treated as a pre-built collection CSV, not parsed line-by-line.
    Otherwise treat source itself as the single target.
    """
    p = Path(source)
    if p.exists() and p.is_file():
        if p.suffix.lower() in (".csv", ".json"):
            return [source]  # pre-built CSV or local manifest — use directly
        return [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [source]


def is_loc_url(text: str) -> bool:
    return "loc.gov" in text


def is_ia_url(text: str) -> bool:
    return "archive.org" in text


def _is_generic_iiif_url(text: str) -> bool:
    """True for any HTTP(S) URL that is not a LoC or IA URL."""
    return (
        text.startswith("http")
        and not is_loc_url(text)
        and not is_ia_url(text)
    )




def run_stage(
    script: str,
    stage_args: list[str],
    dry_run: bool = False,
    interactive: bool = False,
) -> bool:
    """
    Run a pipeline stage as a subprocess, streaming output directly to the
    terminal.  Returns True if the script exited with code 0.
    In dry-run mode, prints the command that would be run without executing it.
    When interactive=True, Ctrl+C is treated as "user finished this stage" rather
    than aborting the whole pipeline (used for select_pages, review_alignment).

    Stages are launched as modules (python -m pipeline.foo) so package imports
    resolve from the repo root or any installed environment.
    """
    module = script.removesuffix(".py").replace("/", ".")
    cmd = [sys.executable, "-m", module] + stage_args
    if dry_run:
        print(f"    [dry run] $ {' '.join(str(a) for a in cmd)}", file=sys.stderr)
        return True
    print(f"    $ {' '.join(str(a) for a in cmd)}", file=sys.stderr)
    if interactive:
        # Use Popen directly so that Ctrl+C (SIGINT) is forwarded to the subprocess
        # but does NOT kill it via subprocess.run()'s internal process.kill().
        # select_pages / review_alignment handle Ctrl+C themselves (advancing items
        # or shutting down gracefully).  We keep waiting until the subprocess exits.
        proc = subprocess.Popen(cmd)
        while proc.poll() is None:
            try:
                proc.wait()
            except KeyboardInterrupt:
                # Subprocess received the same Ctrl+C and handles it; keep waiting.
                pass
        if proc.returncode != 0:
            print(
                f"  Warning: {script} exited with code {proc.returncode}",
                file=sys.stderr,
            )
        return proc.returncode == 0
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"  Warning: {script} exited with code {result.returncode}",
            file=sys.stderr,
        )
    return result.returncode == 0


def build_stage_args(
    stage: str,
    source: str,
    slug: str,
    parsed: argparse.Namespace,
    dry_run: bool = False,
) -> list[str] | None:
    """
    Build the argv list for one pipeline stage.
    Returns None to skip the stage (prints a reason before returning).
    In dry-run mode, existence checks are bypassed so every selected stage
    shows the command it would run.
    """
    csv_path = Path("output") / slug / f"{slug}.csv"
    # When source is an existing local directory deeper than output/{slug},
    # use it directly so stages operate on the specific volume, not the whole collection.
    output_dir = Path(source) if Path(source).is_dir() else Path("output") / slug

    def _resolve_sections() -> str | None:
        """Return an absolute path for --sections, or None if not provided.

        Resolution order for a relative path:
          1. As-is (relative to CWD — e.g. output/tulsa_1921/sections.txt)
          2. Relative to output_dir (e.g. sections.txt → output/tulsa_1921/sections.txt)
        """
        raw = getattr(parsed, "sections", None)
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute():
            return str(p)
        if p.exists():
            return str(p.resolve())
        candidate = output_dir / p
        if candidate.exists():
            return str(candidate.resolve())
        return str(p.resolve())  # let the tool produce the "not found" error

    def _require_images() -> bool:
        if dry_run:
            return True
        if not output_dir.exists():
            print(
                f"    Skipping: images directory not found ({output_dir}). "
                "Run --download first.",
                file=sys.stderr,
            )
            return False
        return True

    # Declarative stages: argv comes from the registry spec (pipeline/stages.py).
    sd = STAGE_BY_NAME[stage]
    if sd.declarative:
        if not _require_images():
            return None
        return build_declarative_args(
            sd, output_dir, parsed, ctx={"sections": _resolve_sections()}
        )

    if stage == "loc_csv":
        if source.lower().endswith(".csv"):
            print(
                "    Skipping: --loc-csv is not applicable when source is a CSV file.",
                file=sys.stderr,
            )
            return None
        if not is_loc_url(source):
            print(
                "    Skipping: --loc-csv requires a loc.gov URL.",
                file=sys.stderr,
            )
            return None
        return [source, "--output", str(Path("output") / slug / f"{slug}.csv")]

    if stage == "ia_csv":
        if source.lower().endswith(".csv"):
            print(
                "    Skipping: --ia-csv is not applicable when source is a CSV file.",
                file=sys.stderr,
            )
            return None
        if not is_ia_url(source):
            print(
                "    Skipping: --ia-csv requires an archive.org URL. "
                "Use --loc-csv for loc.gov sources.",
                file=sys.stderr,
            )
            return None
        return [source, "--output", str(Path("output") / slug / f"{slug}.csv")]

    if stage == "iiif_csv":
        if source.lower().endswith(".csv"):
            print(
                "    Skipping: --iiif-csv is not applicable when source is a CSV file.",
                file=sys.stderr,
            )
            return None
        if not _is_generic_iiif_url(source):
            print(
                "    Skipping: --iiif-csv requires a generic IIIF manifest URL "
                "(not LoC or IA). Use --loc-csv or --ia-csv for those sources.",
                file=sys.stderr,
            )
            return None
        return [source, "--output", str(Path("output") / slug / f"{slug}.csv")]

    if stage == "download":
        actual_csv = Path(source) if source.lower().endswith(".csv") else csv_path
        # For generic IIIF URLs with no CSV yet, use direct --manifest mode
        # (unless --iiif-csv is also requested in the same run, which will create the CSV first)
        iiif_csv_in_run = getattr(parsed, "iiif_csv", False)
        _is_local_json = source.lower().endswith(".json") and Path(source).exists()
        if (_is_generic_iiif_url(source) or _is_local_json) and not iiif_csv_in_run and (dry_run or not actual_csv.exists()):
            a = ["--manifest", source, "--output-dir", str(output_dir), "--resume"]
            if parsed.width is not None:
                a += ["--width", str(parsed.width)]
            return a
        if not dry_run and not actual_csv.exists():
            print(
                f"    Skipping: collection CSV not found ({actual_csv}). "
                "Run --iiif-csv, --loc-csv, or --ia-csv first.",
                file=sys.stderr,
            )
            return None
        a = [str(actual_csv), "--resume"]
        if parsed.width is not None:
            a += ["--width", str(parsed.width)]
        return a

    if stage == "compare_ocr":
        if not _require_images():
            return None
        a = [str(output_dir), "--models"] + list(parsed.models)
        if parsed.workers is not None:
            a += ["--workers", str(parsed.workers)]
        if getattr(parsed, "skip_empty_rerun", False):
            a += ["--skip-empty-rerun"]
        if getattr(parsed, "high_res", False):
            a += ["--high-res"]
        return a

    if stage == "detect_spreads":
        if not _require_images():
            return None
        a = [str(output_dir)]
        if csv_path.exists():
            a += ["--csv", str(csv_path)]
        if parsed.threshold is not None:
            a += ["--threshold", str(parsed.threshold)]
        return a

    if stage == "split_spreads":
        spreads_report = output_dir / "spreads_report.csv"
        if not dry_run and not spreads_report.exists():
            print(
                f"    Skipping: spreads_report.csv not found ({spreads_report}). "
                "Run --detect-spreads first.",
                file=sys.stderr,
            )
            return None
        a = [str(spreads_report)]
        if getattr(parsed, "force", False):
            a += ["--force"]
        return a

    if stage == "select_pages":
        if not _require_images():
            return None
        # When the source is a specific item subdirectory (one level deeper than
        # output/slug), pass it directly so only that item is shown in the UI
        # rather than all siblings starting from the first.
        src_path = Path(source)
        if src_path.is_dir():
            src_resolved = src_path.resolve()
            slug_dir = output_dir.resolve()
            if src_resolved != slug_dir and src_resolved.parent == slug_dir:
                a = [str(src_path)]
                if getattr(parsed, "no_open", False):
                    a += ["--no-open"]
                return a
        a = [str(output_dir)]
        if getattr(parsed, "no_open", False):
            a += ["--no-open"]
        if _resolve_sections():
            a += ["--sections", _resolve_sections()]
        return a

    if stage == "generate_prompts":
        if not _require_images():
            return None
        selection = getattr(parsed, "selection", None)
        sections = _resolve_sections()

        def _make_prompt_args(item_dir: Path, sel_path: Path | None) -> list[str]:
            if sections:
                # Sections mode: no --selection needed; --sections drives sampling
                a = [str(output_dir), "--sections", sections]
            else:
                a = [str(item_dir), "--selection", str(sel_path),
                     "--ocr-out", str(item_dir / "ocr_prompt.md"),
                     "--ner-out", str(item_dir / "ner_prompt.md")]
            if getattr(parsed, "prompt_model", None):
                a += ["--model", parsed.prompt_model]
            if getattr(parsed, "ocr_only", False):
                a += ["--ocr-only"]
            elif getattr(parsed, "ner_only", False):
                a += ["--ner-only"]
            if getattr(parsed, "expand_dittos", False):
                a += ["--expand-dittos"]
            return a

        if sections:
            # Sections mode bypasses the selection.txt requirement
            return [_make_prompt_args(output_dir, None)]

        if selection:
            # Explicit --selection: single run, prompts go next to the selection file
            sel_path = Path(selection)
            item_dir = sel_path.parent if sel_path.parent != output_dir else output_dir
            return [_make_prompt_args(item_dir, sel_path)]

        if dry_run:
            return [_make_prompt_args(output_dir, output_dir / "selection.txt")]

        # Auto-discover: collect every subdir (and slug dir itself) with selection.txt.
        # If multiple subdirs have selection.txt files, generate prompts for each.
        runs = []
        if (output_dir / "selection.txt").exists():
            runs.append(_make_prompt_args(output_dir, output_dir / "selection.txt"))
        if output_dir.exists():
            for sub in sorted(output_dir.iterdir()):
                if sub.is_dir() and (sub / "selection.txt").exists():
                    runs.append(_make_prompt_args(sub, sub / "selection.txt"))
        if not runs:
            print(
                "    Skipping --generate-prompts: no selection.txt found. "
                "Run --select-pages first to create selection.txt in each volume.",
                file=sys.stderr,
            )
            return None
        return runs  # list[list[str]] — one run per item with a selection

    return None


def preflight_requirements(enabled: "set[str]", find_spec=None) -> "tuple[list, str]":
    """Check optional-extra dependencies for the enabled stages.

    Returns (missing_stage_defs, install_cmd). install_cmd is a single
    `uv sync` invocation naming every extra this run needs *plus* any extra
    whose package is already installed: uv sync is exact, so syncing one
    extra removes the others' packages — per-stage hints would send users
    ping-ponging between `--extra gpu` and `--extra geo` forever.
    """
    if find_spec is None:
        import importlib.util
        find_spec = importlib.util.find_spec
    missing = [sd for sd in STAGES
               if sd.name in enabled and sd.requires
               and find_spec(sd.requires) is None]
    if not missing:
        return [], ""
    extras: list[str] = []
    for sd in STAGES:
        if not sd.requires or "--extra" not in sd.install_hint:
            continue
        extra = sd.install_hint.split("--extra", 1)[1].split()[0]
        if extra in extras:
            continue
        if sd.name in enabled or find_spec(sd.requires) is not None:
            extras.append(extra)
    return missing, "uv sync " + " ".join(f"--extra {e}" for e in extras)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run digital collections pipeline stages (Library of Congress, Internet Archive, IIIF).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help=(
            "A text file of URLs/UUIDs (one per line), "
            "a single LoC or Internet Archive URL, a generic IIIF manifest URL, "
            "an existing output directory, or a pre-built collection CSV file."
        ),
    )

    # --- Stage flags ---
    stages = parser.add_argument_group(
        "pipeline stages",
        "Select one or more stages to run. They always execute in the order listed.",
    )
    stages.add_argument(
        "--loc-csv",
        dest="loc_csv",
        action="store_true",
        help="Export Library of Congress item metadata to output/{slug}/{slug}.csv (loc.gov URLs only)",
    )
    stages.add_argument(
        "--ia-csv",
        dest="ia_csv",
        action="store_true",
        help="Export Internet Archive item metadata to output/{slug}/{slug}.csv (archive.org URLs only)",
    )
    stages.add_argument(
        "--iiif-csv",
        dest="iiif_csv",
        action="store_true",
        help=(
            "Export metadata from any IIIF manifest to output/{slug}/{slug}.csv. "
            "Accepts any public IIIF Presentation v2 or v3 manifest URL. "
            "For IIIF Collection manifests, writes one row per child manifest. "
            "Use for institutions not supported by --loc-csv or --ia-csv."
        ),
    )
    stages.add_argument(
        "--download",
        dest="download",
        action="store_true",
        help="Download IIIF images to output/{slug}/",
    )
    stages.add_argument(
        "--surya-detect",
        dest="surya_detect",
        action="store_true",
        help=(
            "Detect column layout using Surya neural text-line detection "
            "(alternative to --detect-columns; produces the same columns_report.csv; "
            "requires surya-ocr)"
        ),
    )
    stages.add_argument(
        "--detect-columns",
        dest="detect_columns",
        action="store_true",
        help="Detect per-image column layout and produce columns_report.csv",
    )
    stages.add_argument(
        "--surya-ocr",
        dest="surya_ocr",
        action="store_true",
        help=(
            "Run Surya OCR on downloaded images, producing line-level bboxes "
            "(*_surya.json) used by --align-ocr. Requires surya-ocr."
        ),
    )
    stages.add_argument(
        "--gemini-ocr",
        dest="gemini_ocr",
        action="store_true",
        help="Run Gemini OCR on downloaded images (see --ocr-model)",
    )
    stages.add_argument(
        "--compare-ocr",
        dest="compare_ocr",
        action="store_true",
        help="Compare multiple Gemini models side-by-side (see --models)",
    )
    stages.add_argument(
        "--align-ocr",
        dest="align_ocr",
        action="store_true",
        help="Align Gemini text to Surya bboxes. See --ocr-model / --models.",
    )
    stages.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Draw alignment bounding boxes on images and save *_viz.jpg (see --ocr-model / --models)",
    )
    stages.add_argument(
        "--detect-spreads",
        dest="detect_spreads",
        action="store_true",
        help="Detect double-page spreads in downloaded images",
    )
    stages.add_argument(
        "--split-spreads",
        dest="split_spreads",
        action="store_true",
        help="Split detected spreads into left/right page files (requires --detect-spreads output)",
    )
    stages.add_argument(
        "--select-pages",
        dest="select_pages",
        action="store_true",
        help=(
            "Generate a browser-based page-selector UI for each volume. "
            "Opens select_pages.html in the browser — click thumbnails to pick "
            "4–8 representative sample pages, then click 'Download selection.txt'. "
            "Place selection.txt in output/{slug}/ before running --generate-prompts. "
            "Use --no-open to generate the HTML without launching a browser."
        ),
    )
    stages.add_argument(
        "--generate-prompts",
        dest="generate_prompts",
        action="store_true",
        help=(
            "Generate volume-specific OCR and NER prompts by having Gemini analyze "
            "sample pages selected by --select-pages. Requires selection.txt in "
            "output/{slug}/ or an explicit --selection path. Prompts are saved to "
            "output/{slug}/ocr_prompt.md and ner_prompt.md and auto-discovered "
            "by --gemini-ocr and --extract-entries."
        ),
    )
    stages.add_argument(
        "--review-alignment",
        dest="review_alignment",
        action="store_true",
        help=(
            "Launch the interactive alignment review UI (Flask web server). "
            "Browse pages by unmatched count, draw bounding boxes over unmatched "
            "regions, re-run Surya OCR on crops, and save accepted matches back "
            "to *_aligned.json. Blocks until you press Ctrl+C."
        ),
    )
    stages.add_argument(
        "--extract-entries",
        dest="extract_entries",
        action="store_true",
        help="Extract structured entries from aligned OCR (see --ocr-model / --models)",
    )
    stages.add_argument(
        "--geocode",
        dest="geocode",
        action="store_true",
        help="Geocode extracted entries to lat/lon and write *_geocoded.csv (see --ocr-model / --models)",
    )
    stages.add_argument(
        "--map",
        dest="map",
        action="store_true",
        help="Generate interactive HTML map from geocoded entries (see --ocr-model / --models)",
    )
    stages.add_argument(
        "--explore",
        dest="explore",
        action="store_true",
        help=(
            "Generate a self-contained interactive HTML explorer from extracted entries. "
            "Auto-introspects the CSV schema — works for any document type. "
            "Produces entries_{model}_explorer.html alongside the CSV."
        ),
    )
    stages.add_argument(
        "--extract",
        dest="extract",
        action="store_true",
        help=(
            "Automated shorthand: --download --gemini-ocr --extract-entries --explore. "
            "Produces a structured entries CSV and self-contained HTML explorer from any supported URL. "
            "Run --select-pages and --generate-prompts once first for a new collection type; "
            "after that, --extract works on any subsequent volume in the same series."
        ),
    )
    stages.add_argument(
        "--guided",
        dest="guided",
        action="store_true",
        help=(
            "Human-in-the-loop shorthand: the --extract pipeline plus page scoping, "
            "alignment, and review — "
            "--download --select-pages --surya-ocr --gemini-ocr --align-ocr --review-alignment "
            "--extract-entries --explore. "
            "Geo stages are not included; add --geocode --map for materials with addresses. "
            "Pauses at --select-pages (page scoping), --review-alignment (unmatched line correction), "
            "and expects --generate-prompts to have been run first for new collection types. "
            "Defaults --batch-size to 4 and --workers to 8 unless already set. "
            "Add --ocr-model to override the Gemini model (each stage uses its own default otherwise). "
            "Combine with --loc-csv / --ia-csv to also export metadata. "
            "For microfilm or bound-volume scans that contain double-page spreads, run "
            "--detect-spreads --split-spreads before --guided so pages are correctly "
            "separated before OCR."
        ),
    )

    # --- Pass-through options ---
    opts = parser.add_argument_group(
        "stage options",
        "Passed through to the relevant script. Omit to use each script's default.",
    )
    opts.add_argument(
        "--ocr-model",
        dest="ocr_model",
        default=None,
        metavar="MODEL",
        help=(
            "Gemini model for OCR, alignment, entry extraction, geocoding, and map stages. "
            "If omitted, each stage uses its own built-in default. "
            "Also accepted as --model for backward compatibility."
        ),
    )
    opts.add_argument(
        "--model",
        dest="ocr_model",
        default=argparse.SUPPRESS,
        metavar="MODEL",
        help=argparse.SUPPRESS,  # hidden alias for --ocr-model
    )
    opts.add_argument(
        "--ocr-prompt",
        dest="ocr_prompt",
        default=None,
        metavar="FILE",
        help=(
            "OCR system prompt file for --gemini-ocr. "
            "If omitted, looks for ocr_prompt.md in the volume output directory, "
            "then falls back to prompts/ocr_prompt.md (generic default). "
            "Use this to reuse the prompt from another volume in the same series."
        ),
    )
    opts.add_argument(
        "--ner-prompt",
        dest="ner_prompt",
        default=None,
        metavar="FILE",
        help=(
            "NER system prompt file for --extract-entries. "
            "If omitted, looks for ner_prompt.md in the volume output directory, "
            "then falls back to prompts/ner_prompt.md (generic default). "
            "Use this to reuse the prompt from another volume in the same series."
        ),
    )
    opts.add_argument(
        "--prompt-model",
        dest="prompt_model",
        default=None,
        metavar="MODEL",
        help=(
            "Gemini model for --generate-prompts (default: gemini-3-flash-preview). "
            "A more capable model is used by default for prompt generation; override here."
        ),
    )
    opts.add_argument(
        "--models",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="Two or more Gemini models for --compare-ocr",
    )
    opts.add_argument(
        "--mode",
        choices=["text-only", "multimodal"],
        default=None,
        help=(
            "Extraction mode for --extract-entries: 'text-only' sends corrected OCR text; "
            "'multimodal' also sends the page image for layout context. "
            "Multimodal improves accuracy on materials with mid-page geographic headings "
            "or ambiguous column layouts (default: text-only)."
        ),
    )
    opts.add_argument(
        "--flex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Gemini Flex inference for --gemini-ocr and --extract-entries: "
            "~50%% cheaper, 1–15 min latency per request. On by default — "
            "pass --no-flex for time-sensitive runs."
        ),
    )
    opts.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel workers for OCR stages",
    )
    opts.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        metavar="N",
        help="Images per Surya inference batch (default: 4; reduce if OOM)",
    )
    opts.add_argument(
        "--width",
        type=int,
        default=None,
        metavar="PX",
        help="Image download width in pixels (for --download)",
    )
    opts.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FRAC",
        help="Gutter detection threshold for --detect-spreads and --detect-columns",
    )
    opts.add_argument(
        "--max-columns",
        type=int,
        default=None,
        dest="max_columns",
        metavar="N",
        help=(
            "Maximum columns to detect for --detect-columns (default: 2). "
            "Increase for genuine 3+ column layouts."
        ),
    )
    opts.add_argument(
        "--force",
        action="store_true",
        help="Force --split-spreads to delete and re-split existing output files",
    )
    opts.add_argument(
        "--retry-merged",
        action="store_true",
        dest="retry_merged",
        help=(
            "Find pages flagged possible_column_merge=true in existing aligned JSON "
            "files, delete their Gemini OCR .txt outputs, then automatically run "
            "--gemini-ocr and --align-ocr to fix them. Combine with --ocr-prompt to "
            "use an updated prompt for the re-OCR pass."
        ),
    )
    opts.add_argument(
        "--min-surya-confidence",
        dest="min_surya_confidence",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "For --align-ocr: skip Surya lines below this detection confidence "
            "(0.0–1.0). Filters ghost detections before NW alignment. "
            "Suggested starting point: 0.35."
        ),
    )
    opts.add_argument(
        "--no-text",
        dest="no_text",
        action="store_true",
        help="For --visualize: draw boxes only, skip Gemini text labels on images.",
    )
    opts.add_argument(
        "--no-open",
        dest="no_open",
        action="store_true",
        help="For --select-pages: generate select_pages.html but do not open it in a browser.",
    )
    opts.add_argument(
        "--selection",
        metavar="PATH",
        default=None,
        help=(
            "For --generate-prompts: path to selection.txt produced by --select-pages. "
            "If omitted, automatically looks for selection.txt in output/{slug}/."
        ),
    )
    opts.add_argument(
        "--sections",
        metavar="PATH",
        default=None,
        help=(
            "Path to sections.txt marking structural section boundaries in a volume "
            "(e.g. alphabetical / street / business). When provided, passed to "
            "--select-pages (seeds selection), --generate-prompts (per-section prompts), "
            "--gemini-ocr (per-page prompt routing), and --extract-entries "
            "(context reset + per-page prompt routing at each boundary)."
        ),
    )
    opts.add_argument(
        "--ocr-only",
        dest="ocr_only",
        action="store_true",
        help="For --generate-prompts: generate only the OCR transcription prompt (skip NER).",
    )
    opts.add_argument(
        "--ner-only",
        dest="ner_only",
        action="store_true",
        help="For --generate-prompts: generate only the NER entry-extraction prompt (skip OCR).",
    )
    opts.add_argument(
        "--expand-dittos",
        dest="expand_dittos",
        action="store_true",
        help=(
            "For --gemini-ocr: expand ditto marks ('' or 〃, often misread as 66) "
            "in place rather than transcribing them literally. Useful for tabular "
            "documents where each row repeats a unit or category from the row above."
        ),
    )
    opts.add_argument(
        "--high-res",
        dest="high_res",
        action="store_true",
        help=(
            "For --gemini-ocr / --compare-ocr: use high media resolution when sending images "
            "to Gemini (higher token cost). Auto-enabled when the OCR prompt mentions "
            "handwriting or manuscripts."
        ),
    )
    opts.add_argument(
        "--skip-empty-rerun",
        dest="skip_empty_rerun",
        action="store_true",
        help="For --compare-ocr: skip re-running Gemini API calls for empty output files",
    )
    opts.add_argument(
        "--slug",
        default=None,
        metavar="SLUG",
        help=(
            "Override the auto-generated slug. Useful when re-running without "
            "an API token or when you want a specific folder name."
        ),
    )
    opts.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "Resolve slugs and create output directories, but print the command "
            "each stage would run instead of executing it."
        ),
    )

    args = parser.parse_args()

    # Expand --extract into its constituent stage flags (automated path)
    if args.extract:
        for flag in ("download", "gemini_ocr", "extract_entries", "explore"):
            setattr(args, flag, True)

    # Expand --guided into its constituent stage flags and defaults: the same
    # arc as --extract plus the human-in-the-loop stages. Geo stages are
    # material-dependent (need address fields) — run them via --geocode --map
    # or `pipeline geo` afterwards.
    if args.guided:
        for flag in ("download", "select_pages", "surya_ocr", "gemini_ocr", "align_ocr",
                     "review_alignment", "extract_entries", "explore"):
            setattr(args, flag, True)
        if args.batch_size is None:
            args.batch_size = 4
        if args.workers is None:
            args.workers = 8

    # Validate: at least one stage must be selected
    enabled = {sd.name for sd in STAGES if getattr(args, sd.name)}

    # If --download is requested without an explicit *-csv stage, auto-detect
    # the right source CSV stage per target (based on URL type) at run time.
    _auto_infer_csv = (
        "download" in enabled
        and not enabled & {"loc_csv", "ia_csv"}
    )

    if not enabled and not getattr(args, "retry_merged", False):
        parser.error(
            "No pipeline stages selected. "
            "Use --loc-csv, --ia-csv, --iiif-csv, --download, --surya-ocr, "
            "--gemini-ocr, --align-ocr, --review-alignment, --extract-entries, "
            "--geocode, --map, etc."
        )

    # Validate: --compare-ocr needs models
    if "compare_ocr" in enabled and len(args.models) < 2:
        parser.error("--compare-ocr requires at least 2 models via --models.")

    # Preflight: stages backed by optional extras fail fast with an install
    # hint before any work starts. Without this, a missing dependency surfaces
    # as a mid-run per-stage warning and downstream stages (e.g. align_ocr
    # after a failed surya_ocr) run against nothing.
    if not args.dry_run:
        missing, install_cmd = preflight_requirements(enabled)
        if missing:
            for sd in missing:
                print(
                    f"Error: {sd.flag} requires the '{sd.requires}' package, "
                    "which is not installed.",
                    file=sys.stderr,
                )
            print(f"  Install with: {install_cmd}", file=sys.stderr)
            sys.exit(1)

    # Warn: --slug with multiple targets is ambiguous
    if args.slug:
        targets_raw = load_targets(args.source)
        if len(targets_raw) > 1:
            parser.error("--slug can only be used with a single target.")

    targets = load_targets(args.source)
    if not targets:
        print("No targets found in source.", file=sys.stderr)
        sys.exit(1)

    stage_labels = " → ".join(sd.flag for sd in STAGES if sd.name in enabled)
    dry_tag = "  *** DRY RUN — no scripts will be executed ***\n" if args.dry_run else ""
    print(
        f"\n{'═' * 62}\n"
        f"{dry_tag}"
        f"  Pipeline: {stage_labels}\n"
        f"  Targets:  {len(targets)}\n"
        f"{'═' * 62}",
        file=sys.stderr,
    )

    outcomes: dict[str, list[str]] = {"ok": [], "partial": [], "failed": []}

    for i, target in enumerate(targets, 1):
        # Existing output directory: slug comes from the path, no URL/UUID needed.
        # Accepts either output/{slug} or output/{slug}/{item_id}.
        if Path(target).is_dir():
            p = Path(target).resolve()
            # Walk up to find the path component right under an "output" dir.
            parts = p.parts
            try:
                out_idx = next(j for j in range(len(parts) - 1, -1, -1) if parts[j] == "output")
                slug = args.slug or parts[out_idx + 1]
            except StopIteration:
                slug = args.slug or p.name
            uuid = None
            label = slug
            kind = "dir"

        # CSV-as-source: slug comes from the filename; no UUID or title fetch needed
        elif target.lower().endswith(".csv"):
            slug = args.slug or Path(target).stem
            uuid = None
            label = slug
            kind = "csv"

        # Local IIIF manifest file (.json): treated like a generic IIIF manifest URL
        elif target.lower().endswith(".json") and Path(target).exists():
            slug = args.slug or Path(target).stem
            uuid = None
            label = slug
            kind = "iiif-manifest"

        # Library of Congress URL: slug derived from URL path
        elif is_loc_url(target):
            if "/resource/" in target:
                print(
                    "  Resolving /resource/ URL to /item/ URL via LoC API…",
                    file=sys.stderr,
                )
                target = _resource_url_to_item_url(target)
            slug = args.slug or loc_slug(target)
            uuid = None
            label = slug
            kind = "loc-item" if ("/item/" in target or "/resource/" in target) else "loc-collection"

        # Internet Archive URL: identifier extracted from /details/{id}
        elif is_ia_url(target):
            ia_id = _extract_ia_identifier(target)
            if not ia_id:
                print(
                    f"\n[{i}/{len(targets)}] Skipping — no IA identifier found: {target}",
                    file=sys.stderr,
                )
                continue
            if args.slug:
                slug = args.slug
                label = ia_id
                kind = "ia"
            else:
                ia_title, kind = _fetch_ia_info(ia_id)
                slug = _make_ia_slug(ia_title, ia_id)
                label = ia_title or ia_id
            uuid = None

        # Generic IIIF manifest URL (not LoC or IA)
        elif _is_generic_iiif_url(target):
            slug = args.slug or _iiif_manifest_item_id(target)
            uuid = None
            label = slug
            kind = "iiif-manifest"

        else:
            print(
                f"\n[{i}/{len(targets)}] Skipping — unrecognised source: {target}\n"
                "  Expected: loc.gov URL, archive.org URL, IIIF manifest URL, "
                "existing output directory, or pre-built CSV path.",
                file=sys.stderr,
            )
            continue

        # Always create the output directory (even in dry-run, so the folder
        # structure is in place for inspection or manual follow-up).
        output_dir = Path("output") / slug
        output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{i}/{len(targets)}] {label}\n"
            f"  uuid: {uuid}  |  slug: {slug}  |  type: {kind}\n"
            f"  csv:  output/{slug}/{slug}.csv\n"
            f"  imgs: output/{slug}/",
            file=sys.stderr,
        )

        stage_outcomes: dict[str, str] = {}

        # Per-target stage set — may include an auto-detected *-csv stage.
        target_enabled = set(enabled)

        # --retry-merged: find aligned JSONs flagged possible_column_merge, delete
        # their .txt files so --gemini-ocr will re-OCR them, then force both
        # --gemini-ocr and --align-ocr to run for this target.
        if getattr(args, "retry_merged", False):
            _ocr_slug = (args.ocr_model or DEFAULT_OCR_MODEL).replace("/", "_")
            _flagged: list[Path] = []
            for _json_path in sorted(output_dir.rglob(f"*_{_ocr_slug}_aligned.json")):
                if uuid and uuid not in _json_path.parts:
                    continue
                try:
                    _data = json.loads(_json_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if _data.get("possible_column_merge"):
                    _txt = _json_path.with_name(
                        _json_path.stem[: -len("_aligned")] + ".txt"
                    )
                    if _txt.exists():
                        _flagged.append(_txt)
            if not _flagged:
                print(
                    "  --retry-merged: no merged-column pages found.", file=sys.stderr
                )
            else:
                print(
                    f"  --retry-merged: {len(_flagged)} page(s) flagged — "
                    "deleting OCR files and queuing re-OCR + re-alignment…",
                    file=sys.stderr,
                )
                for _txt in _flagged:
                    _txt.unlink()
                target_enabled.add("gemini_ocr")
                target_enabled.add("align_ocr")
                args.force = True
        if _auto_infer_csv:
            csv_check = Path("output") / slug / f"{slug}.csv"
            if not csv_check.exists() and not args.dry_run:
                if kind in ("loc-item", "loc-collection"):
                    target_enabled.add("loc_csv")
                    print("  Auto-adding --loc-csv (collection CSV not found)", file=sys.stderr)
                elif kind in ("ia-item", "ia-collection", "ia"):
                    target_enabled.add("ia_csv")
                    print("  Auto-adding --ia-csv (collection CSV not found)", file=sys.stderr)
                elif kind == "iiif-manifest":
                    # CSV is optional for IIIF URLs — download falls back to
                    # --manifest mode automatically; only add --iiif-csv if the
                    # user has explicitly requested other post-download stages
                    # that need the full CSV (e.g. --extract-entries).
                    pass

        # Seed pipeline_state.json with slug + source_url for this target.
        if not args.dry_run:
            write_state(output_dir, {"slug": slug, "source_url": target})

        for sd_stage in STAGES:
            stage, script = sd_stage.name, sd_stage.script
            if stage not in target_enabled:
                continue

            print(f"\n  ── {script}", file=sys.stderr)
            stage_args_raw = build_stage_args(stage, target, slug, args, dry_run=args.dry_run)
            if stage_args_raw is None:
                stage_outcomes[stage] = "skipped"
                continue

            # build_stage_args may return a list of arg-lists (e.g. gemini_ocr
            # with multiple --models); normalise to always be a list of runs.
            if stage_args_raw and isinstance(stage_args_raw[0], list):
                all_runs = stage_args_raw
            else:
                all_runs = [stage_args_raw]

            all_ok = True
            is_interactive = sd_stage.interactive
            for stage_args in all_runs:
                ok = run_stage(script, stage_args, dry_run=args.dry_run, interactive=is_interactive)
                if not ok:
                    all_ok = False
            stage_outcomes[stage] = "ok" if all_ok else "failed"

            if all_ok and not args.dry_run:
                record_stage(output_dir, stage)
                # Record which model was used so downstream scripts don't need --model.
                if stage == "gemini_ocr" and args.ocr_model:
                    write_state(output_dir, {"ocr_model": args.ocr_model})
                if stage == "extract_entries" and args.ocr_model:
                    write_state(output_dir, {"ner_model": args.ocr_model})

        # Summarise this target
        n_ok    = sum(1 for v in stage_outcomes.values() if v == "ok")
        n_fail  = sum(1 for v in stage_outcomes.values() if v == "failed")
        n_skip  = sum(1 for v in stage_outcomes.values() if v == "skipped")
        if n_fail:
            outcomes["failed"].append(slug)
        elif n_ok == 0 and n_skip > 0:
            outcomes["partial"].append(slug)
        else:
            outcomes["ok"].append(slug)

    # Final summary
    total = len(targets)
    print(f"\n{'═' * 62}", file=sys.stderr)
    print(f"  Done. {total} target(s) processed.", file=sys.stderr)
    if outcomes["ok"]:
        print(f"    ✓ OK:       {', '.join(outcomes['ok'])}", file=sys.stderr)
    if outcomes["partial"]:
        print(f"    ~ Partial:  {', '.join(outcomes['partial'])}", file=sys.stderr)
    if outcomes["failed"]:
        print(f"    ✗ Failed:   {', '.join(outcomes['failed'])}", file=sys.stderr)
    print(f"{'═' * 62}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
