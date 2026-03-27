#!/usr/bin/env python3
"""Digital Collections pipeline orchestrator (NYPL, Library of Congress, Internet Archive).

Reads a list of collection URLs/UUIDs from a text file (or a single URL/UUID
passed directly) and runs the requested pipeline stages in order for each.
A human-readable slug is derived automatically and used as the base name for
every output file and directory, so all stages stay in sync:

    output/{slug}/{slug}.csv
    output/{slug}/

Stages run in this fixed order regardless of the order flags appear on the
command line:

  --nypl-csv        sources/nypl_collection_csv.py  → output/{slug}/{slug}.csv  (NYPL only)
  --loc-csv         sources/loc_collection_csv.py   → output/{slug}/{slug}.csv  (LoC only)
  --ia-csv          sources/ia_collection_csv.py    → output/{slug}/{slug}.csv  (Internet Archive only)
  --iiif-csv        sources/iiif_manifest_csv.py    → output/{slug}/{slug}.csv  (any IIIF manifest or collection)
  --download        pipeline/download_images.py       → output/{slug}/
  --detect-spreads  pipeline/detect_spreads.py        (double-page spread detection)
  --split-spreads   pipeline/split_spreads.py         (split spreads into left/right pages)
  --surya-detect    pipeline/surya_detect.py          (Surya neural column detection → columns_report.csv)
  --detect-columns  pipeline/detect_columns.py        (pixel-projection column detection → columns_report.csv)
  --tesseract       old/run_ocr.py                    (Tesseract OCR — legacy; use --surya-ocr instead)
  --surya-ocr       pipeline/run_surya_ocr.py         (Surya OCR → *_surya.json line bboxes + *_surya.txt)
  --gemini-ocr      pipeline/run_gemini_ocr.py        (Gemini OCR)
  --compare-ocr     analysis/compare_ocr.py           (side-by-side model comparison; accepts "surya" token)
  --align-ocr       pipeline/align_ocr.py             (NW alignment of Gemini text to Surya/Tesseract bboxes)
  --visualize       analysis/visualize_alignment.py   (draw alignment boxes on images → *_viz.jpg)
  --review-alignment pipeline/review_alignment.py     (interactive UI to correct unmatched entries → *_aligned.json)
  --extract-entries pipeline/extract_entries.py        (extract structured entries from aligned OCR)
  --geocode         pipeline/geo/geocode_entries.py    (geocode entries to lat/lon)
  --map             pipeline/geo/map_entries.py        (generate interactive HTML map)

  --select-pages        pipeline/select_pages.py  (interactive browser UI for picking sample pages and scoping entry pages; opens HTML in browser)
  --generate-prompts    pipeline/generate_prompt.py  (Gemini generates volume-specific OCR + NER prompts from sample pages)

  --guided              shorthand for --download --select-pages --surya-ocr --gemini-ocr
                          --align-ocr --review-alignment --extract-entries --geocode --map
                          (defaults --batch-size and --workers to 8;
                          run --generate-prompts first for new collection types)

Key model flags:
  --ocr-model MODEL     Gemini model for OCR and downstream stages (each stage uses its own default if omitted)
  --prompt-model MODEL  Gemini model for --generate-prompts (default: gemini-3-flash-preview)

Usage
-----
    # NYPL collection or item:
    python main.py collections.txt --nypl-csv --download --gemini-ocr
    python main.py https://digitalcollections.nypl.org/collections/<uuid> \\
        --nypl-csv --download --gemini-ocr

    # Library of Congress collection or item:
    python main.py https://www.loc.gov/collections/civil-war-maps/ \\
        --loc-csv --download --gemini-ocr
    python main.py https://www.loc.gov/item/01015253/ \\
        --loc-csv --download --tesseract

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
    python main.py collections.txt --nypl-csv --download \\
        --compare-ocr --models gemini-2.0-flash gemini-1.5-flash \\
        --workers 8
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).parent

sys.path.insert(0, str(SCRIPT_DIR))
from utils.iiif_utils import manifest_item_id as _iiif_manifest_item_id
from sources.nypl_utils import fetch_title, make_slug
from sources.loc_utils import _resource_url_to_item_url, loc_slug
from sources.ia_utils import _extract_ia_identifier, _fetch_ia_info, _make_ia_slug

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pipeline definition — stages always execute in this order.
# Each entry is (dest_attr_name, script_filename, human_label).
# ---------------------------------------------------------------------------
PIPELINE: list[tuple[str, str, str]] = [
    ("nypl_csv",        "sources/nypl_collection_csv.py",      "--nypl-csv"),
    ("loc_csv",         "sources/loc_collection_csv.py",       "--loc-csv"),
    ("ia_csv",          "sources/ia_collection_csv.py",        "--ia-csv"),
    ("iiif_csv",        "sources/iiif_manifest_csv.py",        "--iiif-csv"),
    ("download",        "pipeline/download_images.py",         "--download"),
    ("detect_spreads",  "pipeline/detect_spreads.py",          "--detect-spreads"),
    ("split_spreads",   "pipeline/split_spreads.py",           "--split-spreads"),
    ("select_pages",    "pipeline/select_pages.py",     "--select-pages"),
    ("generate_prompts","pipeline/generate_prompt.py",     "--generate-prompts"),
    ("surya_detect",    "pipeline/surya_detect.py",            "--surya-detect"),
    ("detect_columns",  "pipeline/detect_columns.py",          "--detect-columns"),
    ("tesseract",       "old/run_ocr.py",                      "--tesseract"),
    ("surya_ocr",       "pipeline/run_surya_ocr.py",           "--surya-ocr"),
    ("gemini_ocr",      "pipeline/run_gemini_ocr.py",          "--gemini-ocr"),
    ("chandra_ocr",     "pipeline/run_chandra_ocr.py",         "--chandra-ocr"),
    ("compare_ocr",     "analysis/compare_ocr.py",             "--compare-ocr"),
    ("align_ocr",       "pipeline/align_ocr.py",               "--align-ocr"),
    ("visualize",       "analysis/visualize_alignment.py",     "--visualize"),
    ("review_alignment","pipeline/review_alignment.py",        "--review-alignment"),
    ("extract_entries", "pipeline/extract_entries.py",         "--extract-entries"),
    ("geocode",         "pipeline/geo/geocode_entries.py",     "--geocode"),
    ("map",             "pipeline/geo/map_entries.py",         "--map"),
    ("explore",         "pipeline/explore_entries.py",         "--explore"),
]


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


def extract_uuid(text: str) -> str | None:
    m = UUID_RE.search(text)
    return m.group(0).lower() if m else None


def is_item_url(text: str) -> bool:
    return "/items/" in text


def is_loc_url(text: str) -> bool:
    return "loc.gov" in text


def is_ia_url(text: str) -> bool:
    return "archive.org" in text


def _is_generic_iiif_url(text: str) -> bool:
    """True for any HTTP(S) URL that is not a NYPL, LoC, or IA URL.

    Used to detect arbitrary IIIF manifest URLs from external institutions.
    """
    return (
        text.startswith("http")
        and not is_loc_url(text)
        and not is_ia_url(text)
        and "nypl.org" not in text
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
    """
    cmd = [sys.executable, str(SCRIPT_DIR / script)] + stage_args
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
    output_dir = Path("output") / slug

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

    if stage == "nypl_csv":
        if source.lower().endswith(".csv"):
            print(
                "    Skipping: --nypl-csv is not applicable when source is a CSV file.",
                file=sys.stderr,
            )
            return None
        if is_loc_url(source):
            print(
                "    Skipping: --nypl-csv is not applicable for loc.gov URLs. "
                "Use --loc-csv instead.",
                file=sys.stderr,
            )
            return None
        a = [source, "--output", f"{slug}.csv"]
        if parsed.token:
            a += ["--token", parsed.token]
        return a

    if stage == "loc_csv":
        if source.lower().endswith(".csv"):
            print(
                "    Skipping: --loc-csv is not applicable when source is a CSV file.",
                file=sys.stderr,
            )
            return None
        if not is_loc_url(source):
            print(
                "    Skipping: --loc-csv requires a loc.gov URL. "
                "Use --nypl-csv for NYPL sources.",
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
                "Use --nypl-csv or --loc-csv for other sources.",
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
                "(not NYPL, LoC, or IA). Use --nypl-csv, --loc-csv, or --ia-csv instead.",
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
                "Run --iiif-csv, --nypl-csv, or --loc-csv first.",
                file=sys.stderr,
            )
            return None
        a = [str(actual_csv), "--resume"]
        if parsed.width is not None:
            a += ["--width", str(parsed.width)]
        return a

    if stage == "tesseract":
        if not _require_images():
            return None
        a = [str(output_dir)]
        if parsed.workers is not None:
            a += ["--workers", str(parsed.workers)]
        if getattr(parsed, "psm", None) is not None:
            a += ["--psm", str(parsed.psm)]
        if getattr(parsed, "oem", None) is not None:
            a += ["--oem", str(parsed.oem)]
        if getattr(parsed, "dpi", None) is not None:
            a += ["--dpi", str(parsed.dpi)]
        if getattr(parsed, "use_dict", False):
            a += ["--dict"]
        return a

    if stage == "surya_ocr":
        if not _require_images():
            return None
        a = [str(output_dir)]
        if getattr(parsed, "batch_size", None) is not None:
            a += ["--batch-size", str(parsed.batch_size)]
        return a

    if stage == "gemini_ocr":
        if not _require_images():
            return None
        model_list = parsed.models if parsed.models else ([parsed.ocr_model] if parsed.ocr_model else [None])
        runs = []
        for m in model_list:
            a = [str(output_dir)]
            if m:
                a += ["--model", m]
            if parsed.workers is not None:
                a += ["--workers", str(parsed.workers)]
            if getattr(parsed, "expand_dittos", False):
                a += ["--expand-dittos"]
            if getattr(parsed, "high_res", False):
                a += ["--high-res"]
            if getattr(parsed, "ocr_prompt", None):
                a += ["--prompt-file", parsed.ocr_prompt]
            runs.append(a)
        return runs  # list[list[str]] — one run per model

    if stage == "chandra_ocr":
        if not _require_images():
            return None
        a = [str(output_dir)]
        if getattr(parsed, "chandra_method", None):
            a += ["--method", parsed.chandra_method]
        if getattr(parsed, "batch_size", None) is not None:
            a += ["--batch-size", str(parsed.batch_size)]
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

    if stage == "align_ocr":
        if not _require_images():
            return None
        model_list = parsed.models if parsed.models else ([parsed.ocr_model] if parsed.ocr_model else [None])
        runs = []
        for m in model_list:
            a = [str(output_dir)]
            if m:
                a += ["--model", m]
            if parsed.workers is not None:
                a += ["--workers", str(parsed.workers)]
            if getattr(parsed, "force", False):
                a += ["--force"]
            if getattr(parsed, "min_surya_confidence", None) is not None:
                a += ["--min-surya-confidence", str(parsed.min_surya_confidence)]
            runs.append(a)
        return runs  # list[list[str]] — one run per model

    if stage == "visualize":
        if not _require_images():
            return None
        model_list = parsed.models if parsed.models else ([parsed.ocr_model] if parsed.ocr_model else [None])
        runs = []
        for m in model_list:
            a = [str(output_dir)]
            if m:
                a += ["--model", m]
            if getattr(parsed, "no_text", False):
                a += ["--no-text"]
            if getattr(parsed, "force", False):
                a += ["--force"]
            runs.append(a)
        return runs  # list[list[str]] — one run per model

    if stage == "surya_detect":
        if not _require_images():
            return None
        a = [str(output_dir)]
        if getattr(parsed, "force", False):
            a += ["--force"]
        return a

    if stage == "detect_columns":
        if not _require_images():
            return None
        a = [str(output_dir)]
        if parsed.workers is not None:
            a += ["--workers", str(parsed.workers)]
        if parsed.threshold is not None:
            a += ["--threshold", str(parsed.threshold)]
        if getattr(parsed, "max_columns", None) is not None:
            a += ["--max-columns", str(parsed.max_columns)]
        if getattr(parsed, "force", False):
            a += ["--force"]
        return a

    if stage == "detect_spreads":
        if not _require_images():
            return None
        a = [str(output_dir)]
        # Pass --csv for microform prior: in dry-run, use it if --nypl-csv is
        # also selected (it would be created); in live runs, only if it exists.
        if dry_run:
            if getattr(parsed, "nypl_csv", False):
                a += ["--csv", str(csv_path)]
        elif csv_path.exists():
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
        return a

    if stage == "generate_prompts":
        if not _require_images():
            return None
        selection = getattr(parsed, "selection", None)
        def _make_prompt_args(item_dir: Path, sel_path: Path) -> list[str]:
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

    if stage == "review_alignment":
        if not _require_images():
            return None
        a = [str(output_dir)]
        m = (parsed.models[0] if parsed.models else None) or parsed.ocr_model
        if m:
            a += ["--model", m]
        return a

    if stage == "extract_entries":
        if not _require_images():
            return None
        # Pass --aligned-model (OCR file slug) separately from --model (NER model).
        # Omitting --model lets extract_entries use its own NER DEFAULT_MODEL.
        ocr_models = parsed.models if parsed.models else ([parsed.ocr_model] if parsed.ocr_model else [None])
        runs = []
        for ocr_m in ocr_models:
            a = [str(output_dir)]
            if ocr_m:
                a += ["--aligned-model", ocr_m]
            if getattr(parsed, "ner_prompt", None):
                a += ["--prompt", parsed.ner_prompt]
            if getattr(parsed, "force", False):
                a += ["--force"]
            runs.append(a)
        return runs

    if stage == "geocode":
        if not _require_images():
            return None
        model_list = parsed.models if parsed.models else ([parsed.ocr_model] if parsed.ocr_model else [None])
        runs = []
        for m in model_list:
            a = [str(output_dir)]
            if m:
                a += ["--model", m]
            runs.append(a)
        return runs

    if stage == "map":
        if not _require_images():
            return None
        model_list = parsed.models if parsed.models else ([parsed.ocr_model] if parsed.ocr_model else [None])
        runs = []
        for m in model_list:
            a = [str(output_dir)]
            if m:
                a += ["--model", m]
            runs.append(a)
        return runs

    if stage == "explore":
        if not _require_images():
            return None
        model_list = parsed.models if parsed.models else ([parsed.ocr_model] if parsed.ocr_model else [None])
        runs = []
        for m in model_list:
            a = [str(output_dir)]
            if m:
                a += ["--model", m]
            runs.append(a)
        return runs

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run digital collections pipeline stages (NYPL and Library of Congress).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help=(
            "A text file of URLs/UUIDs (one per line), "
            "a single NYPL or Library of Congress URL or UUID, "
            "or a pre-built collection CSV file."
        ),
    )

    # --- Stage flags ---
    stages = parser.add_argument_group(
        "pipeline stages",
        "Select one or more stages to run. They always execute in the order listed.",
    )
    stages.add_argument(
        "--nypl-csv",
        dest="nypl_csv",
        action="store_true",
        help="Export NYPL item metadata to output/{slug}/{slug}.csv (NYPL URLs only)",
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
            "Use for institutions not supported by --nypl-csv, --loc-csv, or --ia-csv."
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
            "(alternative to --detect-columns; produces the same columns_report.csv "
            "so --tesseract works unchanged; requires surya-ocr)"
        ),
    )
    stages.add_argument(
        "--detect-columns",
        dest="detect_columns",
        action="store_true",
        help="Detect per-image column layout and produce columns_report.csv (used by --tesseract)",
    )
    stages.add_argument(
        "--tesseract",
        dest="tesseract",
        action="store_true",
        help="Run Tesseract OCR on downloaded images (legacy; prefer --surya-ocr)",
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
        "--chandra-ocr",
        dest="chandra_ocr",
        action="store_true",
        help=(
            "Run Chandra OCR on downloaded images (local 5B model, no API key needed). "
            "Requires: pip install chandra-ocr[hf]. Use --chandra-method to select backend."
        ),
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
        help=(
            "Align Gemini text to OCR bboxes (Surya line-level preferred; "
            "Tesseract word-level as legacy fallback). See --ocr-model / --models."
        ),
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
            "Human-in-the-loop shorthand for the full pipeline: "
            "--download --select-pages --surya-ocr --gemini-ocr --align-ocr --review-alignment "
            "--extract-entries --geocode --map. "
            "Pauses at --select-pages (page scoping), --review-alignment (unmatched line correction), "
            "and expects --generate-prompts to have been run first for new collection types. "
            "Defaults --batch-size to 4 and --workers to 8 unless already set. "
            "Add --ocr-model to override the Gemini model (each stage uses its own default otherwise). "
            "Combine with --nypl-csv / --loc-csv / --ia-csv to also export metadata. "
            "For microfilm or bound-volume scans that contain double-page spreads, run "
            "--detect-spreads --split-spreads before --guided so pages are correctly "
            "separated before OCR."
        ),
    )

    # --- Authentication ---
    auth = parser.add_argument_group("authentication")
    auth.add_argument(
        "--token",
        default=os.environ.get("NYPL_API_TOKEN", ""),
        help="NYPL API token (or set NYPL_API_TOKEN env var)",
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
        help="Images per Surya/Chandra inference batch (default: 4 for Surya, 1 for Chandra; reduce if OOM)",
    )
    opts.add_argument(
        "--chandra-method",
        dest="chandra_method",
        choices=["hf", "vllm"],
        default="hf",
        metavar="METHOD",
        help="Chandra inference backend for --chandra-ocr: 'hf' (HuggingFace, default) or 'vllm'",
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
        "--psm",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Tesseract page segmentation mode for --tesseract "
            "(default: Tesseract built-in, 3 = auto). "
            "Recommended for multi-column pages: 1 (auto with OSD)."
        ),
    )
    opts.add_argument(
        "--oem",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Tesseract OCR engine mode for --tesseract "
            "(0=legacy, 1=LSTM, 2=legacy+LSTM). "
            "Try 2 for degraded historical scans."
        ),
    )
    opts.add_argument(
        "--dpi",
        type=int,
        default=None,
        metavar="N",
        help="Image DPI hint for --tesseract (300 or 400 typical for archival scans).",
    )
    opts.add_argument(
        "--dict",
        dest="use_dict",
        action="store_true",
        help=(
            "Re-enable Tesseract dictionary correction for --tesseract "
            "(disabled by default to preserve proper nouns and addresses "
            "for NW alignment with Gemini text)."
        ),
    )
    opts.add_argument(
        "--force",
        action="store_true",
        help="Force --split-spreads to delete and re-split existing output files",
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

    # Expand --guided into its constituent stage flags and defaults (human-in-the-loop path)
    if args.guided:
        for flag in ("download", "select_pages", "surya_ocr", "gemini_ocr", "align_ocr",
                     "review_alignment", "extract_entries", "geocode", "map"):
            setattr(args, flag, True)
        if args.batch_size is None:
            args.batch_size = 4
        if args.workers is None:
            args.workers = 8

    # Validate: at least one stage must be selected
    enabled = {stage for stage, _, _ in PIPELINE if getattr(args, stage)}

    # If --download is requested without an explicit *-csv stage, auto-detect
    # the right source CSV stage per target (based on URL type) at run time.
    _auto_infer_csv = (
        "download" in enabled
        and not enabled & {"nypl_csv", "loc_csv", "ia_csv"}
    )

    if not enabled:
        parser.error(
            "No pipeline stages selected. "
            "Use --nypl-csv, --loc-csv, --ia-csv, --iiif-csv, --download, --surya-ocr, "
            "--gemini-ocr, --align-ocr, --review-alignment, --extract-entries, --geocode, "
            "--map, etc."
        )

    # Validate: stages that need a token (not enforced in dry-run)
    if "nypl_csv" in enabled and not args.token and not args.dry_run:
        parser.error(
            "--nypl-csv requires an API token. "
            "Use --token or set NYPL_API_TOKEN."
        )

    # Validate: --compare-ocr needs models
    if "compare_ocr" in enabled and len(args.models) < 2:
        parser.error("--compare-ocr requires at least 2 models via --models.")

    # Warn: --slug with multiple targets is ambiguous
    if args.slug:
        targets_raw = load_targets(args.source)
        if len(targets_raw) > 1:
            parser.error("--slug can only be used with a single target.")

    targets = load_targets(args.source)
    if not targets:
        print("No targets found in source.", file=sys.stderr)
        sys.exit(1)

    # Session for title fetching (only built when a token is available)
    session: requests.Session | None = None
    if args.token:
        session = requests.Session()
        session.headers["Authorization"] = f'Token token="{args.token}"'

    stage_labels = " → ".join(label for stage, _, label in PIPELINE if stage in enabled)
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
                    f"  Resolving /resource/ URL to /item/ URL via LoC API…",
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

        # Generic IIIF manifest URL (not NYPL, LoC, or IA)
        elif _is_generic_iiif_url(target):
            slug = args.slug or _iiif_manifest_item_id(target)
            uuid = None
            label = slug
            kind = "iiif-manifest"

        else:
            uuid = extract_uuid(target)
            if not uuid:
                print(
                    f"\n[{i}/{len(targets)}] Skipping — no UUID found: {target}",
                    file=sys.stderr,
                )
                continue

            item = is_item_url(target)

            # Derive slug
            if args.slug:
                slug = args.slug
                title = ""
            elif session:
                title = fetch_title(session, uuid, item)
                time.sleep(0.15)  # polite pause between API calls
                slug = make_slug(title, uuid)
            else:
                title = ""
                slug = make_slug("", uuid)

            label = title or uuid
            kind = "item" if item else "collection"

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
        if _auto_infer_csv:
            csv_check = Path("output") / slug / f"{slug}.csv"
            if not csv_check.exists() and not args.dry_run:
                if kind in ("loc-item", "loc-collection"):
                    target_enabled.add("loc_csv")
                    print("  Auto-adding --loc-csv (collection CSV not found)", file=sys.stderr)
                elif kind in ("ia-item", "ia-collection", "ia"):
                    target_enabled.add("ia_csv")
                    print("  Auto-adding --ia-csv (collection CSV not found)", file=sys.stderr)
                elif kind in ("item", "collection"):
                    target_enabled.add("nypl_csv")
                    print("  Auto-adding --nypl-csv (collection CSV not found)", file=sys.stderr)
                elif kind == "iiif-manifest":
                    # CSV is optional for IIIF URLs — download falls back to
                    # --manifest mode automatically; only add --iiif-csv if the
                    # user has explicitly requested other post-download stages
                    # that need the full CSV (e.g. --extract-entries).
                    pass

        for stage, script, _ in PIPELINE:
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
            is_interactive = stage in ("select_pages", "review_alignment")
            for stage_args in all_runs:
                ok = run_stage(script, stage_args, dry_run=args.dry_run, interactive=is_interactive)
                if not ok:
                    all_ok = False
            stage_outcomes[stage] = "ok" if all_ok else "failed"

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
