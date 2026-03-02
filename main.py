#!/usr/bin/env python3
"""Digital Collections pipeline orchestrator (NYPL, Library of Congress, Internet Archive).

Reads a list of collection URLs/UUIDs from a text file (or a single URL/UUID
passed directly) and runs the requested pipeline stages in order for each.
A human-readable slug is derived automatically and used as the base name for
every output file and directory, so all stages stay in sync:

    collection_csv/{slug}.csv
    images/{slug}/

Stages run in this fixed order regardless of the order flags appear on the
command line:

  --nypl-csv        sources/nypl_collection_csv.py  → collection_csv/{slug}.csv  (NYPL only)
  --loc-csv         sources/loc_collection_csv.py   → collection_csv/{slug}.csv  (LoC only)
  --ia-csv          sources/ia_collection_csv.py    → collection_csv/{slug}.csv  (Internet Archive only)
  --download        pipeline/download_images.py       → images/{slug}/
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
  --geocode         pipeline/geocode_entries.py        (geocode entries to lat/lon)
  --map             pipeline/map_entries.py            (generate interactive HTML map)

  --full-run            shorthand for --download --surya-ocr --gemini-ocr --align-ocr
                          --review-alignment --extract-entries --geocode --map
                          (defaults --batch-size and --workers to 8)

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

    # Pre-built CSV (any source):
    python main.py collection_csv/my_items.csv --download --gemini-ocr

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
API_BASE = "https://api.repo.nypl.org/api/v2"

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
    ("download",        "pipeline/download_images.py",         "--download"),
    ("detect_spreads",  "pipeline/detect_spreads.py",          "--detect-spreads"),
    ("split_spreads",   "pipeline/split_spreads.py",           "--split-spreads"),
    ("surya_detect",    "pipeline/surya_detect.py",            "--surya-detect"),
    ("detect_columns",  "pipeline/detect_columns.py",          "--detect-columns"),
    ("tesseract",       "old/run_ocr.py",                      "--tesseract"),
    ("surya_ocr",       "pipeline/run_surya_ocr.py",           "--surya-ocr"),
    ("gemini_ocr",      "pipeline/run_gemini_ocr.py",          "--gemini-ocr"),
    ("compare_ocr",     "analysis/compare_ocr.py",             "--compare-ocr"),
    ("align_ocr",       "pipeline/align_ocr.py",               "--align-ocr"),
    ("visualize",       "analysis/visualize_alignment.py",     "--visualize"),
    ("review_alignment","pipeline/review_alignment.py",        "--review-alignment"),
    ("extract_entries", "pipeline/extract_entries.py",         "--extract-entries"),
    ("geocode",         "pipeline/geocode_entries.py",         "--geocode"),
    ("map",             "pipeline/map_entries.py",             "--map"),
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
        if p.suffix.lower() == ".csv":
            return [source]  # pre-built collection CSV — use directly
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


def _fetch_loc_title(item_id: str) -> str:
    """Best-effort fetch of a LoC item title from the public JSON API."""
    try:
        resp = requests.get(
            f"https://www.loc.gov/item/{item_id}/?fo=json",
            timeout=15,
        )
        resp.raise_for_status()
        v = resp.json().get("item", {}).get("title", "")
        if isinstance(v, list):
            return str(v[0]).strip() if v else ""
        return str(v).strip() if v else ""
    except Exception:  # noqa: BLE001
        return ""


def _make_loc_slug(title: str, item_id: str) -> str:
    """Build a slug from a LoC title and item ID.

    Mirrors _make_slug() in loc_collection_csv.py so both scripts
    produce the same slug for the same item.

    ('The Brooklyn city directory', '01015253')
        → 'the_brooklyn_city_directory_01015253'
    """
    id_suffix = item_id[:12]
    if title:
        sanitized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
        if sanitized:
            return f"{sanitized}_{id_suffix}"
    return id_suffix


def loc_slug(url: str) -> str:
    """Derive a filesystem-safe slug from a LoC URL.

    For item URLs, fetches the item title from the LoC JSON API so the
    slug matches what loc_collection_csv.py produces by default:
        /item/01015253/              → 'the_brooklyn_city_directory_01015253'
        /collections/civil-war-maps/ → 'civil-war-maps'
    """
    m = re.search(r"/collections/([^/?#]+)", url)
    if m:
        return m.group(1).rstrip("/")
    m = re.search(r"/item/([^/?#]+)", url)
    if m:
        item_id = m.group(1).rstrip("/")
        title = _fetch_loc_title(item_id)
        return _make_loc_slug(title, item_id)
    return re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")[:40]


_IA_IDENTIFIER_RE = re.compile(r"archive\.org/details/([^/?#]+)")
_IA_METADATA_API  = "https://archive.org/metadata/{identifier}"


def _extract_ia_identifier(url: str) -> str | None:
    m = _IA_IDENTIFIER_RE.search(url)
    return m.group(1).rstrip("/") if m else None


def _fetch_ia_info(identifier: str) -> tuple[str, str]:
    """Return (title, kind) for an IA identifier.

    kind is "ia-collection" or "ia-item".  Returns ("", "ia-item") on failure.
    """
    try:
        resp = requests.get(
            _IA_METADATA_API.format(identifier=identifier),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("metadata", {})
        title = meta.get("title", "")
        if isinstance(title, list):
            title = title[0] if title else ""
        mediatype = meta.get("mediatype", "")
        kind = "ia-collection" if mediatype == "collection" else "ia-item"
        return str(title).strip(), kind
    except Exception:  # noqa: BLE001
        return "", "ia-item"


def _make_ia_slug(title: str, identifier: str) -> str:
    """Build a filesystem-safe slug from an IA title and identifier."""
    id_part = identifier[:20]
    if title:
        sanitized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
        if sanitized:
            return f"{sanitized}_{id_part}"
    return id_part


def _as_list(obj) -> list:
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]


def _str_val(obj) -> str:
    """Unwrap a {'$': 'value'} node or coerce to str."""
    if isinstance(obj, dict):
        return str(obj.get("$", ""))
    return str(obj) if obj else ""


def fetch_title(session: requests.Session, uuid: str, item: bool) -> str:
    """
    Best-effort: fetch a human-readable title for a UUID from the NYPL API.
    Returns an empty string on any failure — callers fall back to UUID-only slugs.
    """
    try:
        if item:
            resp = session.get(
                f"{API_BASE}/items/item_details/{uuid}",
                params={"page": 1, "per_page": 1},
                timeout=15,
            )
            resp.raise_for_status()
            mods = (
                resp.json().get("nyplAPI", {}).get("response", {}).get("mods") or {}
            )
            for ti in _as_list(mods.get("titleInfo")):
                t = _str_val(ti.get("title") if isinstance(ti, dict) else None)
                if t:
                    return t
        else:
            resp = session.get(
                f"{API_BASE}/collections/{uuid}",
                params={"page": 1, "per_page": 1},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("nyplAPI", {}).get("response", {})
            # Try top-level keys first
            candidates = [
                data.get("title"),
                data.get("heading"),
                (data.get("collection") or {}).get("title"),
            ]
            for candidate in candidates:
                t = _str_val(candidate)
                if t:
                    return t
            # Fall back to mods structure (some collection responses embed mods)
            mods = data.get("mods") or {}
            for ti in _as_list(mods.get("titleInfo")):
                t = _str_val(ti.get("title") if isinstance(ti, dict) else None)
                if t:
                    return t
            # Last resort: item_details endpoint — always has full mods with title
            resp2 = session.get(
                f"{API_BASE}/items/item_details/{uuid}",
                params={"page": 1, "per_page": 1},
                timeout=15,
            )
            resp2.raise_for_status()
            mods2 = (
                resp2.json().get("nyplAPI", {}).get("response", {}).get("mods") or {}
            )
            for ti in _as_list(mods2.get("titleInfo")):
                t = _str_val(ti.get("title") if isinstance(ti, dict) else None)
                if t:
                    return t
    except Exception:  # noqa: BLE001
        pass
    return ""


def make_slug(title: str, uuid: str) -> str:
    """
    Build a filesystem-safe slug: {title_words}_{uuid8}.
    Falls back to just {uuid8} if no usable title is available.
    """
    uuid8 = uuid.replace("-", "")[:8]
    if title:
        sanitized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
        if sanitized:
            return f"{sanitized}_{uuid8}"
    return uuid8


def run_stage(script: str, stage_args: list[str], dry_run: bool = False) -> bool:
    """
    Run a pipeline stage as a subprocess, streaming output directly to the
    terminal.  Returns True if the script exited with code 0.
    In dry-run mode, prints the command that would be run without executing it.
    """
    cmd = [sys.executable, str(SCRIPT_DIR / script)] + stage_args
    if dry_run:
        print(f"    [dry run] $ {' '.join(str(a) for a in cmd)}", file=sys.stderr)
        return True
    print(f"    $ {' '.join(str(a) for a in cmd)}", file=sys.stderr)
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
    csv_path = Path("collection_csv") / f"{slug}.csv"
    images_dir = Path("images") / slug

    def _require_images() -> bool:
        if dry_run:
            return True
        if not images_dir.exists():
            print(
                f"    Skipping: images directory not found ({images_dir}). "
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
        return [source, "--output", f"{slug}.csv"]

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
        return [source, "--output", f"{slug}.csv"]

    if stage == "download":
        actual_csv = Path(source) if source.lower().endswith(".csv") else csv_path
        if not dry_run and not actual_csv.exists():
            print(
                f"    Skipping: collection CSV not found ({actual_csv}). "
                "Run --nypl-csv or --loc-csv first.",
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
        a = [str(images_dir)]
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
        a = [str(images_dir)]
        if getattr(parsed, "batch_size", None) is not None:
            a += ["--batch-size", str(parsed.batch_size)]
        return a

    if stage == "gemini_ocr":
        if not _require_images():
            return None
        models = parsed.models if parsed.models else [parsed.model]
        runs = []
        for m in models:
            a = [str(images_dir), "--model", m]
            if parsed.workers is not None:
                a += ["--workers", str(parsed.workers)]
            runs.append(a)
        return runs  # list[list[str]] — one run per model

    if stage == "compare_ocr":
        if not _require_images():
            return None
        a = [str(images_dir), "--models"] + list(parsed.models)
        if parsed.workers is not None:
            a += ["--workers", str(parsed.workers)]
        if getattr(parsed, "skip_empty_rerun", False):
            a += ["--skip-empty-rerun"]
        return a

    if stage == "align_ocr":
        if not _require_images():
            return None
        models = parsed.models if parsed.models else (
            [parsed.model] if getattr(parsed, "model", None) else []
        )
        if not models:
            print(
                "    Skipping: --align-ocr requires --model or --models.",
                file=sys.stderr,
            )
            return None
        runs = []
        for m in models:
            a = [str(images_dir), "--model", m]
            if parsed.workers is not None:
                a += ["--workers", str(parsed.workers)]
            if getattr(parsed, "force", False):
                a += ["--force"]
            runs.append(a)
        return runs  # list[list[str]] — one run per model

    if stage == "visualize":
        if not _require_images():
            return None
        models = parsed.models if parsed.models else (
            [parsed.model] if getattr(parsed, "model", None) else []
        )
        if not models:
            print(
                "    Skipping: --visualize requires --model or --models.",
                file=sys.stderr,
            )
            return None
        runs = []
        for m in models:
            a = [str(images_dir), "--model", m]
            if getattr(parsed, "no_text", False):
                a += ["--no-text"]
            if getattr(parsed, "force", False):
                a += ["--force"]
            runs.append(a)
        return runs  # list[list[str]] — one run per model

    if stage == "surya_detect":
        if not _require_images():
            return None
        a = [str(images_dir)]
        if getattr(parsed, "force", False):
            a += ["--force"]
        return a

    if stage == "detect_columns":
        if not _require_images():
            return None
        a = [str(images_dir)]
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
        a = [str(images_dir)]
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
        spreads_report = images_dir / "spreads_report.csv"
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

    if stage == "review_alignment":
        if not _require_images():
            return None
        m = (parsed.models[0] if parsed.models else None) or getattr(parsed, "model", "gemini-2.0-flash")
        a = [str(images_dir), "--model", m]
        return a

    if stage == "extract_entries":
        if not _require_images():
            return None
        models = parsed.models if parsed.models else (
            [parsed.model] if getattr(parsed, "model", None) else []
        )
        if not models:
            print(
                "    Skipping: --extract-entries requires --model or --models.",
                file=sys.stderr,
            )
            return None
        runs = []
        for m in models:
            a = [str(images_dir), "--model", m]
            if getattr(parsed, "force", False):
                a += ["--force"]
            runs.append(a)
        return runs

    if stage == "geocode":
        if not _require_images():
            return None
        models = parsed.models if parsed.models else (
            [parsed.model] if getattr(parsed, "model", None) else []
        )
        if not models:
            print(
                "    Skipping: --geocode requires --model or --models.",
                file=sys.stderr,
            )
            return None
        runs = []
        for m in models:
            runs.append([str(images_dir), "--model", m])
        return runs

    if stage == "map":
        if not _require_images():
            return None
        models = parsed.models if parsed.models else (
            [parsed.model] if getattr(parsed, "model", None) else []
        )
        if not models:
            print(
                "    Skipping: --map requires --model or --models.",
                file=sys.stderr,
            )
            return None
        runs = []
        for m in models:
            runs.append([str(images_dir), "--model", m])
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
        help="Export NYPL item metadata to collection_csv/{slug}.csv (NYPL URLs only)",
    )
    stages.add_argument(
        "--loc-csv",
        dest="loc_csv",
        action="store_true",
        help="Export Library of Congress item metadata to collection_csv/{slug}.csv (loc.gov URLs only)",
    )
    stages.add_argument(
        "--ia-csv",
        dest="ia_csv",
        action="store_true",
        help="Export Internet Archive item metadata to collection_csv/{slug}.csv (archive.org URLs only)",
    )
    stages.add_argument(
        "--download",
        dest="download",
        action="store_true",
        help="Download IIIF images to images/{slug}/",
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
        help="Run Gemini OCR on downloaded images (see --model)",
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
            "Tesseract word-level as legacy fallback). See --model / --models."
        ),
    )
    stages.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Draw alignment bounding boxes on images and save *_viz.jpg (see --model / --models)",
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
        help="Extract structured entries from aligned OCR (see --model / --models)",
    )
    stages.add_argument(
        "--geocode",
        dest="geocode",
        action="store_true",
        help="Geocode extracted entries to lat/lon and write *_geocoded.csv (see --model / --models)",
    )
    stages.add_argument(
        "--map",
        dest="map",
        action="store_true",
        help="Generate interactive HTML map from geocoded entries (see --model / --models)",
    )
    stages.add_argument(
        "--full-run",
        dest="full_run",
        action="store_true",
        help=(
            "Shorthand for the standard end-to-end pipeline: "
            "--download --surya-ocr --gemini-ocr --align-ocr --review-alignment "
            "--extract-entries --geocode --map. "
            "Defaults --batch-size to 8 and --workers to 8 unless already set. "
            "Add --model / --models to specify the Gemini model (default: gemini-2.0-flash). "
            "Combine with --nypl-csv / --loc-csv / --ia-csv to also export metadata."
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
        "--model",
        default="gemini-2.0-flash",
        metavar="MODEL",
        help="Gemini model for --gemini-ocr (default: gemini-2.0-flash)",
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
        help="Images per Surya inference batch for --surya-ocr (default: 4; reduce if OOM)",
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
        "--no-text",
        dest="no_text",
        action="store_true",
        help="For --visualize: draw boxes only, skip Gemini text labels on images.",
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

    # Expand --full-run into its constituent stage flags and defaults
    if args.full_run:
        for flag in ("download", "surya_ocr", "gemini_ocr", "align_ocr",
                     "review_alignment", "extract_entries", "geocode", "map"):
            setattr(args, flag, True)
        if args.batch_size is None:
            args.batch_size = 8
        if args.workers is None:
            args.workers = 8

    # Validate: at least one stage must be selected
    enabled = {stage for stage, _, _ in PIPELINE if getattr(args, stage)}
    if not enabled:
        parser.error(
            "No pipeline stages selected. "
            "Use --nypl-csv, --loc-csv, --ia-csv, --download, --surya-ocr, --gemini-ocr, "
            "--align-ocr, --review-alignment, --extract-entries, --geocode, --map, etc."
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
        # CSV-as-source: slug comes from the filename; no UUID or title fetch needed
        if target.lower().endswith(".csv"):
            slug = args.slug or Path(target).stem
            uuid = None
            label = slug
            kind = "csv"

        # Library of Congress URL: slug derived from URL path
        elif is_loc_url(target):
            slug = args.slug or loc_slug(target)
            uuid = None
            label = slug
            kind = "loc-item" if "/item/" in target else "loc-collection"

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

        # Always create the output directories (even in dry-run, so the folder
        # structure is in place for inspection or manual follow-up).
        csv_dir = Path("collection_csv")
        images_dir = Path("images") / slug
        csv_dir.mkdir(exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{i}/{len(targets)}] {label}\n"
            f"  uuid: {uuid}  |  slug: {slug}  |  type: {kind}\n"
            f"  csv:  collection_csv/{slug}.csv\n"
            f"  imgs: images/{slug}/",
            file=sys.stderr,
        )

        stage_outcomes: dict[str, str] = {}

        for stage, script, _ in PIPELINE:
            if stage not in enabled:
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
            for stage_args in all_runs:
                ok = run_stage(script, stage_args, dry_run=args.dry_run)
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
