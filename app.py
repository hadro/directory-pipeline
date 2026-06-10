#!/usr/bin/env python3
"""
app.py — directory-pipeline web dashboard.

A browser-based "air traffic control" interface for managing collection items
through the pipeline stages.

Usage:
    python app.py              # opens at http://127.0.0.1:5001
    python app.py --port 8080
"""

import argparse
import json
import os
import queue
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import webbrowser
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "pipeline.db"
OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_PORT = 5001
from utils.models import DEFAULT_OCR_MODEL as DEFAULT_MODEL

# Matches the first segment of a UUID (8 hex chars)
_UUID_RE = re.compile(r"([0-9a-f]{8})-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
# Matches any http://127.0.0.1:PORT/... URL in subprocess output
_SERVER_URL_RE = re.compile(r"http://127\.0\.0\.1:\d+\S*")
# Matches output/slug in subprocess output (slug has no slashes)
_OUTPUT_SLUG_RE = re.compile(r"\boutput/([^/\s]+)")

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    # (name, label, main.py flag, group, interactive, standalone script path, requires)
    # 'requires': optional Python package — if missing, stage button is disabled in the UI.
    {"name": "download",           "label": "Download",           "flag": "--download",          "group": "ingest",    "interactive": False, "script": None},
    {"name": "select_pages",       "label": "Select pages",       "flag": "--select-pages",      "group": "calibrate", "interactive": True,  "script": None},
    {"name": "generate_prompts",   "label": "Generate prompts",   "flag": "--generate-prompts",  "group": "calibrate", "interactive": False, "script": None},
    {"name": "surya_ocr",          "label": "Surya OCR",          "flag": "--surya-ocr",         "group": "ocr",       "interactive": False, "script": None,
     "requires": "surya", "install_hint": "uv sync --extra gpu"},
    {"name": "gemini_ocr",         "label": "Gemini OCR",         "flag": "--gemini-ocr",        "group": "ocr",       "interactive": False, "script": None},
    {"name": "align_ocr",          "label": "Align OCR",          "flag": "--align-ocr",         "group": "ocr",       "interactive": False, "script": None},
    {"name": "review_alignment",   "label": "Review alignment",   "flag": "--review-alignment",  "group": "review",    "interactive": True,  "script": None},
    {"name": "extract_entries",    "label": "Extract entries",    "flag": "--extract-entries",   "group": "extract",   "interactive": False, "script": None},
    {"name": "explore",            "label": "Explore",            "flag": "--explore",           "group": "extract",   "interactive": False, "script": None},
    {"name": "geocode",            "label": "Geocode",            "flag": "--geocode",           "group": "extract",   "interactive": False, "script": None,
     "requires": "geopy", "install_hint": "uv sync --extra geo"},
    {"name": "map",                "label": "Map",                "flag": "--map",               "group": "extract",   "interactive": False, "script": None},
    {"name": "postprocess",        "label": "Postprocess",        "flag": None,                  "group": "extract",   "interactive": False, "collection_script": "pipeline/postprocess.py"},
    {"name": "export_annotations", "label": "Export annotations", "flag": None,                  "group": "iiif",      "interactive": False, "script": "pipeline/iiif/export_annotations.py"},
    {"name": "export_entry_boxes", "label": "Export entry boxes", "flag": None,                  "group": "iiif",      "interactive": False, "script": "pipeline/iiif/export_entry_boxes.py"},
    {"name": "build_ranges",       "label": "Build ranges",       "flag": None,                  "group": "iiif",      "interactive": False, "script": "pipeline/iiif/build_ranges.py"},
]

STAGE_BY_NAME = {s["name"]: s for s in STAGES}


def _check_available_packages() -> dict[str, bool]:
    """Check which optional packages are importable. Runs once at startup."""
    import importlib
    packages = {s["requires"] for s in STAGES if s.get("requires")}
    return {pkg: importlib.util.find_spec(pkg) is not None for pkg in packages}


# Checked once at import time so the result is stable across requests.
_AVAILABLE_PACKAGES: dict[str, bool] = _check_available_packages()


def _stage_available(stage_name: str) -> tuple[bool, str]:
    """Return (available, install_hint). Unavailable = required package missing."""
    s = STAGE_BY_NAME.get(stage_name, {})
    pkg = s.get("requires")
    if pkg and not _AVAILABLE_PACKAGES.get(pkg, True):
        return False, s.get("install_hint", f"pip install {pkg}")
    return True, ""

# ---------------------------------------------------------------------------
# Per-stage argument definitions
# ---------------------------------------------------------------------------
# Each entry maps stage name → list of arg descriptors.
# descriptor keys: name, flag, type ("str"|"int"|"float"|"bool"), label, default, placeholder

STAGE_ARG_DEFS: dict[str, list[dict]] = {
    "download": [
        {"name": "width", "flag": "--width", "type": "int",
         "label": "Width", "default": "", "placeholder": "e.g. 1200",
         "hint": "Download width in pixels. Capped at native resolution to avoid upscaling. Omit for full resolution."},
    ],
    "surya_ocr": [
        {"name": "batch_size", "flag": "--batch-size", "type": "int",
         "label": "Batch size", "default": "", "placeholder": "e.g. 4",
         "hint": "Images processed per batch. Reduce if running out of GPU/Apple Silicon memory (default: 4)."},
    ],
    "gemini_ocr": [
        {"name": "model",         "flag": "--model",          "type": "str",  "label": "Model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "Gemini model for OCR text extraction. Output is cached per model — changing the model only reruns uncached pages."},
        {"name": "workers",       "flag": "--workers",         "type": "int",  "label": "Workers",
         "default": "", "placeholder": "e.g. 4",
         "hint": "Parallel API calls. Higher values increase throughput but may hit rate limits."},
        {"name": "expand_dittos", "flag": "--expand-dittos",   "type": "bool", "label": "Expand dittos",
         "default": False,
         "hint": "Expand ditto marks (″) into the repeated text from the previous entry."},
        {"name": "flex",          "flag": "--flex",            "type": "bool", "label": "Flex inference",
         "default": True,
         "hint": "Use Gemini Flex inference (~50% cheaper, 1–15 min latency per request). Recommended for large batches where real-time throughput isn't needed. Uncheck for time-sensitive runs."},
        {"name": "high_res",      "flag": "--high-res",        "type": "bool", "label": "High res",
         "default": False,
         "hint": "Send images at higher resolution to the API. Slower and costlier; useful for small or dense text, and recommended for handwritten text (HTR)."},
    ],
    "align_ocr": [
        {"name": "model",                "flag": "--model",                 "type": "str",   "label": "OCR model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "Model used for Gemini OCR. Auto-detected from pipeline_state.json if left at the default — only override if you ran OCR with a different model."},
        {"name": "workers",              "flag": "--workers",               "type": "int",   "label": "Workers",
         "default": "", "placeholder": "e.g. 4",
         "hint": "Parallel alignment workers."},
        {"name": "min_surya_confidence", "flag": "--min-surya-confidence",  "type": "float", "label": "Min Surya confidence",
         "default": "", "placeholder": "e.g. 0.85", "min": 0, "max": 1,
         "hint": "Minimum Surya line confidence (0–1) to include in alignment. Lines below this threshold are excluded."},
        {"name": "force",                "flag": "--force",                 "type": "bool",  "label": "Force re-run",
         "default": False,
         "hint": "Re-run alignment even for pages that already have *_aligned.json output."},
    ],
    "generate_prompts": [
        {"name": "model",    "flag": "--model",    "type": "str",  "label": "Model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "Gemini model used to analyze sample pages and write the prompts."},
        {"name": "ocr_only", "flag": "--ocr-only", "type": "bool", "label": "OCR only",  "default": False,
         "hint": "Generate only ocr_prompt.md; skip the NER prompt."},
        {"name": "ner_only", "flag": "--ner-only", "type": "bool", "label": "NER only",  "default": False,
         "hint": "Generate only ner_prompt.md; skip the OCR prompt."},
    ],
    "review_alignment": [
        {"name": "model", "flag": "--model", "type": "str", "label": "OCR model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "Must match the model slug in *_aligned.json filenames. Opens a local Flask UI at localhost:5000."},
    ],
    "extract_entries": [
        {"name": "aligned_model", "flag": "--ocr-model", "type": "str", "label": "Aligned model (OCR)",
         "default": "", "placeholder": "e.g. gemini-3.1-flash-lite",
         "hint": "OCR model whose *_aligned.json files to use for bounding boxes. Auto-detected from pipeline_state.json if omitted. Leave blank for text-only (no #xywh= in output)."},
        {"name": "ner_prompt",    "flag": "--ner-prompt",     "type": "str", "label": "NER prompt path",
         "default": "", "placeholder": "e.g. output/slug/ner_prompt.md",
         "hint": "Path to a custom NER prompt. Auto-discovered from output/{slug}/ner_prompt.md if omitted. Reuse across volumes in the same series."},
        {"name": "force",         "flag": "--force",          "type": "bool", "label": "Force re-run",
         "default": False,
         "hint": "Overwrite cached *_entries.json files. Required if a prior run produced text-only entries and you now want aligned mode."},
    ],
    "geocode": [
        {"name": "model", "flag": "--model", "type": "str", "label": "Model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "Model slug identifying which entries CSV to geocode (entries_{model}.csv). Uses Google Maps API if GOOGLE_MAPS_API_KEY is set, otherwise Nominatim city-level fallback."},
    ],
    "map": [
        {"name": "model", "flag": "--model", "type": "str", "label": "Model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "Model slug identifying which geocoded CSV to map (entries_{model}_geocoded.csv). Requires geocoding to have run first."},
    ],
    "explore": [
        {"name": "model", "flag": "--model", "type": "str", "label": "Model",
         "default": "", "placeholder": "auto-detect",
         "hint": "Model slug identifying which entries CSV to explore. Leave blank to auto-detect the available CSV. Produces a self-contained interactive HTML explorer with search and filtering."},
    ],
    "postprocess": [
        {"name": "no_combine", "flag": "--no-combine", "type": "bool", "label": "No combine",
         "default": False,
         "hint": "Skip the combine step. Use this for single-volume directories that don't need merging."},
    ],
    "export_annotations": [
        {"name": "model", "flag": "--model", "type": "str",  "label": "Model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "OCR model slug whose aligned JSON files to convert to W3C Annotation Pages (line-level transcription + entry-level structured data)."},
        {"name": "force", "flag": "--force", "type": "bool", "label": "Force", "default": False,
         "hint": "Re-export even if annotation files already exist."},
    ],
    "export_entry_boxes": [
        {"name": "model", "flag": "--model", "type": "str",  "label": "Model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "OCR model slug whose entries JSON files to convert to colored bounding box annotations for IIIF viewers."},
        {"name": "force", "flag": "--force", "type": "bool", "label": "Force", "default": False,
         "hint": "Re-export even if box annotation files already exist."},
    ],
    "build_ranges": [
        {"name": "model", "flag": "--model", "type": "str", "label": "Model",
         "default": DEFAULT_MODEL, "placeholder": "",
         "hint": "Model slug identifying which geocoded CSV to build ranges from. Groups entries by State → City → Category for IIIF table of contents."},
    ],
}

# Brief per-stage descriptions shown at the top of each arg panel.
STAGE_HINTS: dict[str, str] = {
    "download":          "Fetches IIIF images to output/{slug}/. Also writes manifest.json, needed for image thumbnails.",
    "select_pages":      "Browser UI with two tabs: Sample (pick 4–10 pages for prompt calibration → selection.txt) and Scope (exclude frontmatter/back-matter → included_pages.txt). Run once per collection type.",
    "generate_prompts":  "Sends selected sample pages to Gemini and generates volume-specific ocr_prompt.md and ner_prompt.md. Run once per collection type; reuse prompts for subsequent volumes.",
    "surya_ocr":         "Runs Surya's neural line-detection model to produce bounding boxes (*_surya.json). Requires GPU or Apple Silicon. Needed before --align-ocr.",
    "gemini_ocr":        "Sends each image to Gemini for text extraction (*_{model}.txt). Handles rate limits with exponential backoff. Output is cached — safe to re-run.",
    "align_ocr":         "Aligns Gemini text to Surya bounding boxes using Needleman-Wunsch (*_{model}_aligned.json). Produces the #xywh= fragment coordinates used in the entries CSV.",
    "review_alignment":  "Interactive Flask UI for manually correcting pages where automatic alignment left unmatched entries. Draw boxes, run Surya on crops, accept proposed matches.",
    "extract_entries":   "Calls Gemini NER on each aligned page to produce structured entries (entries_{model}.csv). Schema is defined by the NER prompt — no code changes needed for new collection types.",
    "explore":           "Generates a self-contained interactive HTML explorer from the entries CSV. Auto-introspects the schema — search, filter, and browse entries without any additional setup.",
    "geocode":           "Resolves entries to lat/lon coordinates. Uses Google Maps API for address-level geocoding (requires GOOGLE_MAPS_API_KEY) and Nominatim as a city-level fallback.",
    "map":               "Generates a self-contained Leaflet HTML map from the geocoded CSV with clustered markers, search, and IIIF thumbnail popups.",
    "postprocess":        "Runs the full post-extraction sequence: fix_entries (normalize, deduplicate) → combine_volumes (merge per-volume CSVs) → explore_entries (build interactive HTML explorer). Model is read from pipeline_state.json automatically.",
    "export_annotations": "Converts aligned JSON to W3C Annotation Pages (JSON-LD) for IIIF viewers like Mirador and Universal Viewer.",
    "export_entry_boxes": "Converts entries JSON to colored bounding box annotations (W3C Annotation Pages). Color-coded by establishment category.",
    "build_ranges":      "Builds a IIIF Presentation v3 Range hierarchy (table of contents) from the geocoded CSV, grouping entries by State → City → Category.",
}

GROUPS = [
    ("ingest",    "Ingest"),
    ("calibrate", "Calibrate"),
    ("ocr",       "OCR"),
    ("review",    "Review"),
    ("extract",   "Extract"),
    ("iiif",      "IIIF"),
]

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def _init_db() -> None:
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id         INTEGER PRIMARY KEY,
                url        TEXT NOT NULL,
                slug       TEXT,
                output_dir TEXT,
                model      TEXT NOT NULL DEFAULT 'gemini-3.1-flash-lite',
                added_at   TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS runs (
                id           INTEGER PRIMARY KEY,
                item_id      INTEGER REFERENCES items(id),
                stage        TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'running',
                forced       INTEGER DEFAULT 0,
                started_at   TEXT DEFAULT (datetime('now')),
                finished_at  TEXT,
                log          TEXT DEFAULT '',
                server_url   TEXT
            );
            -- Mark any runs left 'running' from a prior server process as interrupted.
            UPDATE runs SET status='interrupted', finished_at=datetime('now')
            WHERE status='running';
        """)


@contextmanager
def _db():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _get_item(item_id: int) -> dict | None:
    with _db() as conn:
        row = conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        return dict(row) if row else None

# ---------------------------------------------------------------------------
# Output dir detection
# ---------------------------------------------------------------------------

def _detect_output_dir(url: str) -> Path | None:
    """Find the output dir for a URL by matching the UUID8 in output/ dir names."""
    m = _UUID_RE.search(url)
    if m and OUTPUT_ROOT.exists():
        uuid8 = m.group(1).lower()
        for d in OUTPUT_ROOT.iterdir():
            if d.is_dir() and uuid8 in d.name.lower():
                return d
    return None


def _resolve_output_dir(item: dict) -> Path | None:
    """Return the output_dir Path for an item, or None if not yet known."""
    od = item.get("output_dir")
    if od:
        p = SCRIPT_DIR / od
        return p if p.exists() else None
    od = _detect_output_dir(item.get("url", ""))
    if od:
        # Persist it
        rel = str(od.relative_to(SCRIPT_DIR))
        with _db() as conn:
            conn.execute("UPDATE items SET output_dir=? WHERE id=? AND output_dir IS NULL",
                         (rel, item["id"]))
        return od
    return None


def _find_uuid_dirs(output_dir: Path) -> list[Path]:
    """Return immediate subdirectories of output_dir (UUID item dirs)."""
    if not output_dir or not output_dir.exists():
        return []
    return sorted(d for d in output_dir.iterdir() if d.is_dir())

# ---------------------------------------------------------------------------
# Stage completion detection
# ---------------------------------------------------------------------------

def _check_stage_done(item: dict) -> dict[str, bool]:
    """Return {stage_name: done_bool} for all stages of an item."""
    od = _resolve_output_dir(item)
    # Use same model resolution as _build_cmd so completion checks stay in sync.
    model = (
        _read_pipeline_model(od)
        or item.get("model")
        or DEFAULT_MODEL
    )
    item_id = item["id"]

    results: dict[str, bool] = {}
    for s in STAGES:
        name = s["name"]
        results[name] = _stage_done(name, od, model, item_id)
    return results


def _stage_done(stage: str, od: Path | None, model: str, item_id: int) -> bool:
    if not od or not od.exists():
        # Before the output dir exists, nothing can be done.
        if stage == "review_alignment":
            # Fall back to DB check even with no output dir.
            with _db() as conn:
                row = conn.execute(
                    "SELECT id FROM runs WHERE item_id=? AND stage=? AND status='done'",
                    (item_id, stage)
                ).fetchone()
            return row is not None
        return False

    checks = {
        "download": lambda: any(
            d.is_dir() and (list(d.glob("*.jpg")) or list(d.glob("*.jpeg"))
                            or list(d.glob("*.png")) or list(d.glob("*.tif")))
            for d in od.iterdir() if d.is_dir()
        ),
        # selection.txt may land in the top-level dir or an item subdir.
        "select_pages":      lambda: bool(list(od.glob("**/selection.txt"))),
        "generate_prompts":  lambda: (od / "ner_prompt.md").exists() or (od / "ocr_prompt.md").exists(),
        "surya_ocr":         lambda: bool(list(od.glob("**/*_surya.json"))),
        "gemini_ocr":        lambda: bool(list(od.glob(f"**/*_{model}.txt"))),
        "align_ocr":         lambda: bool(list(od.glob(f"**/*_{model}_aligned.json"))),
        # review_alignment: any aligned JSON with a manually-confirmed line is sufficient.
        "review_alignment":  lambda: bool([
            f for f in od.glob(f"**/*_{model}_aligned.json")
            if any(ln.get("confidence") == "manual"
                   for ln in (json.loads(f.read_text(encoding="utf-8")).get("lines") or []))
        ]) or _db_stage_done(stage, item_id),
        "extract_entries":   lambda: bool(list(od.glob("**/entries_*.csv"))),
        "explore":           lambda: bool(list(od.glob("**/entries_*_explorer.html"))),
        "postprocess":       lambda: (od / "combined.csv").exists() or bool(list(od.glob("**/entries_*_fixed.csv"))),
        "geocode":           lambda: bool(list(od.glob("**/entries_*_geocoded.csv"))),
        "map":               lambda: bool(list(od.glob("**/entries_*.html"))),
        "export_annotations": lambda: bool([
            f for f in od.glob("**/*_annotations.json") if "_box_" not in f.name
        ]),
        "export_entry_boxes": lambda: bool(list(od.glob("**/*_box_annotations.json"))),
        "build_ranges":       lambda: bool(list(od.glob("**/ranges_*.json"))),
    }
    fn = checks.get(stage)
    try:
        result = fn() if fn else False
    except Exception:
        result = False

    # Final fallback: check pipeline_state.json for stages run via the CLI.
    if not result:
        result = _pipeline_state_stage_done(od, stage)

    return result


def _db_stage_done(stage: str, item_id: int) -> bool:
    """Check the runs DB for a 'done' record (fallback for stages with no file artifact)."""
    with _db() as conn:
        row = conn.execute(
            "SELECT id FROM runs WHERE item_id=? AND stage=? AND status='done'",
            (item_id, stage)
        ).fetchone()
    return row is not None


def _pipeline_state_stage_done(od: Path | None, stage: str) -> bool:
    """Check pipeline_state.json for stages completed via the CLI outside the dashboard."""
    if not od:
        return False
    state_file = od / "pipeline_state.json"
    if not state_file.exists():
        return False
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        return stage in (state.get("stages_completed") or [])
    except Exception:
        return False


def _item_status(item: dict) -> dict[str, str]:
    """Return {stage_name: 'pending'|'running'|'done'|'failed'} for one item."""
    with _db() as conn:
        active_rows = conn.execute(
            "SELECT id, stage FROM runs WHERE item_id=? AND status='running'",
            (item["id"],)
        ).fetchall()

        # Cross-check with live _RUNS — clean up stale 'running' rows whose
        # process has already exited (e.g. killed but children kept pipe open).
        with _RUNS_LOCK:
            live_ids = set(_RUNS.keys())
        stale = [r["id"] for r in active_rows if r["id"] not in live_ids]
        if stale:
            conn.execute(
                f"UPDATE runs SET status='interrupted', finished_at=datetime('now')"
                f" WHERE id IN ({','.join('?'*len(stale))})",
                stale,
            )
        active_stages = {r["stage"] for r in active_rows if r["id"] in live_ids}

        # Latest run per stage
        failed_rows = conn.execute(
            """SELECT stage FROM runs
               WHERE item_id=?
                 AND status='failed'
                 AND id = (SELECT MAX(id) FROM runs r2
                           WHERE r2.item_id = runs.item_id
                             AND r2.stage   = runs.stage)""",
            (item["id"],)
        ).fetchall()
        failed_stages = {r["stage"] for r in failed_rows}

    done_map = _check_stage_done(item)
    statuses: dict[str, str] = {}
    for s in STAGES:
        name = s["name"]
        if name in active_stages:
            statuses[name] = "running"
        elif done_map.get(name):
            statuses[name] = "done"
        elif name in failed_stages:
            statuses[name] = "failed"
        else:
            statuses[name] = "pending"
    return statuses

# ---------------------------------------------------------------------------
# Subprocess / run management
# ---------------------------------------------------------------------------

_RUNS: dict[int, dict] = {}       # run_id -> state dict
_SUBS: dict[int, list] = {}       # run_id -> list[Queue]
_RUNS_LOCK = threading.Lock()

# Interactive stages: Ctrl-C (SIGINT, exit 130) means the user is finished,
# not that something went wrong.
_INTERACTIVE_STAGES = {"select_pages", "review_alignment"}


def _reader_thread(run_id: int, item_id: int, stage_name: str) -> None:
    """Read subprocess stdout/stderr, notify SSE subscribers, update DB on finish."""
    run = _RUNS[run_id]
    proc = run["process"]

    for raw in proc.stdout:
        line = raw.rstrip("\n")
        with run["lock"]:
            run["log_lines"].append(line)

        for q in list(_SUBS.get(run_id, [])):
            try:
                q.put_nowait(("line", line))
            except Exception:
                pass

        # Detect interactive server URL
        m = _SERVER_URL_RE.search(line)
        if m:
            url = m.group(0)
            with run["lock"]:
                run["server_url"] = url
            for q in list(_SUBS.get(run_id, [])):
                try:
                    q.put_nowait(("server_url", url))
                except Exception:
                    pass

    proc.wait()
    exit_code = proc.returncode
    with run["lock"]:
        run["finished"] = True
        run["exit_code"] = exit_code

    # For interactive stages, Ctrl-C (exit 130) means the user finished normally.
    sigint_exit = exit_code in (130, -2)
    status = "done" if exit_code == 0 or (sigint_exit and stage_name in _INTERACTIVE_STAGES) else "failed"
    log_text = "\n".join(run["log_lines"])

    # Try to detect output_dir from log output
    output_dir_found = None
    for line in run["log_lines"]:
        m2 = _OUTPUT_SLUG_RE.search(line)
        if m2:
            slug = m2.group(1).rstrip("/")
            candidate = SCRIPT_DIR / "output" / slug
            if candidate.exists() and candidate.is_dir():
                output_dir_found = f"output/{slug}"
                break

    with _db() as conn:
        conn.execute(
            "UPDATE runs SET status=?, finished_at=datetime('now'), log=? WHERE id=?",
            (status, log_text, run_id)
        )
        if output_dir_found:
            conn.execute(
                "UPDATE items SET output_dir=? WHERE id=? AND output_dir IS NULL",
                (output_dir_found, item_id)
            )

    # Signal SSE consumers
    for q in list(_SUBS.get(run_id, [])):
        try:
            q.put_nowait(("done", status))
        except Exception:
            pass


def _start_run(item_id: int, stage_name: str, cmd: list[str], forced: bool = False) -> int:
    """Launch a stage subprocess. Returns the run_id."""
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO runs (item_id, stage, status, forced) VALUES (?, ?, 'running', ?)",
            (item_id, stage_name, 1 if forced else 0)
        )
        run_id = cur.lastrowid

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(SCRIPT_DIR),
        start_new_session=True,
    )

    run_state = {
        "process":   proc,
        "log_lines": [],
        "lock":      threading.Lock(),
        "finished":  False,
        "exit_code": None,
        "server_url": None,
    }
    with _RUNS_LOCK:
        _RUNS[run_id] = run_state

    t = threading.Thread(target=_reader_thread, args=(run_id, item_id, stage_name), daemon=True)
    t.start()
    return run_id


def _read_pipeline_model(od: Path | None) -> str | None:
    """Read the OCR/NER model from pipeline_state.json if present."""
    if not od:
        return None
    state_file = od / "pipeline_state.json"
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        return state.get("ner_model") or state.get("ocr_model")
    except Exception:
        return None


def _build_cmd(
    item: dict,
    stage: str,
    force: bool = False,
    workers: int | None = None,
    overrides: dict | None = None,
) -> list[str]:
    """Build the subprocess command for a pipeline stage."""
    overrides = overrides or {}
    s = STAGE_BY_NAME[stage]
    od = _resolve_output_dir(item)
    # Model resolution: explicit override > pipeline_state.json > DB item model > default
    model = (
        overrides.get("model")
        or _read_pipeline_model(od)
        or item.get("model")
        or DEFAULT_MODEL
    )
    effective_force = force or bool(overrides.get("force"))
    effective_workers = overrides.get("workers") or workers

    if s.get("collection_script"):
        # Collection-level script — pass the output dir directly (not a UUID subdir)
        script_path = SCRIPT_DIR / s["collection_script"]
        if not od:
            raise ValueError("Output directory not found. Run --download first.")
        cmd = [sys.executable, str(script_path), str(od)]
        if effective_force:
            cmd.append("--force")
    elif s["flag"] is not None:
        # Route through main.py
        # For 'explore', omit --model unless the user explicitly set it so that
        # explore_entries.py can auto-detect whichever entries_*.csv is present.
        cmd = [sys.executable, str(SCRIPT_DIR / "main.py"), item["url"], s["flag"]]
        if stage != "explore" or overrides.get("model"):
            cmd += ["--model", model]
        if effective_force:
            cmd.append("--force")
        if effective_workers:
            cmd += ["--workers", str(effective_workers)]
    else:
        # IIIF standalone script — needs a UUID subdir
        script_path = SCRIPT_DIR / s["script"]
        if not od:
            raise ValueError("Output directory not found. Run --download first.")
        uuid_dirs = _find_uuid_dirs(od)
        if not uuid_dirs:
            raise ValueError(f"No item subdirectory found in {od}. Run --download first.")
        uuid_dir = uuid_dirs[0]
        cmd = [sys.executable, str(script_path), str(uuid_dir), "--model", model]
        if effective_force:
            cmd.append("--force")

    # Append remaining stage-specific overrides, falling back to arg_def defaults.
    for arg_def in STAGE_ARG_DEFS.get(stage, []):
        name = arg_def["name"]
        if name in ("model", "force", "workers"):
            continue  # already handled above
        val = overrides.get(name)
        if val is None:
            val = arg_def.get("default")  # use UI default when user hasn't overridden
        if val is None or val == "" or val is False:
            continue
        flag = arg_def["flag"]
        if arg_def["type"] == "bool":
            cmd.append(flag)
        else:
            cmd += [flag, str(val)]

    return cmd

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    stages_for_js = []
    for s in STAGES:
        available, install_hint = _stage_available(s["name"])
        stages_for_js.append({
            "name":         s["name"],
            "label":        s["label"],
            "group":        s["group"],
            "interactive":  s["interactive"],
            "flag":         s["flag"],
            "available":    available,
            "install_hint": install_hint,
        })
    return render_template_string(
        _HTML,
        stages_json=json.dumps(stages_for_js),
        stage_args_json=json.dumps(STAGE_ARG_DEFS),
        stage_hints_json=json.dumps(STAGE_HINTS),
    )


@app.route("/api/items", methods=["GET"])
def list_items():
    with _db() as conn:
        rows = conn.execute("SELECT * FROM items ORDER BY added_at DESC").fetchall()

    result = []
    for row in rows:
        item = dict(row)
        if not item["output_dir"]:
            od = _detect_output_dir(item["url"])
            if od:
                rel = str(od.relative_to(SCRIPT_DIR))
                item["output_dir"] = rel
                with _db() as conn:
                    conn.execute("UPDATE items SET output_dir=? WHERE id=? AND output_dir IS NULL",
                                 (rel, item["id"]))

        with _db() as conn:
            active = conn.execute(
                "SELECT id, stage FROM runs WHERE item_id=? AND status='running'"
                " ORDER BY id DESC LIMIT 1",
                (item["id"],)
            ).fetchone()

        item["stages"] = _item_status(item)
        item["active_run"] = dict(active) if active else None
        result.append(item)

    return jsonify(result)


@app.route("/api/items", methods=["POST"])
def add_item():
    data = request.json or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url required"}), 400
    model = (data.get("model") or DEFAULT_MODEL).strip()

    od = _detect_output_dir(url)
    output_dir = str(od.relative_to(SCRIPT_DIR)) if od else None

    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO items (url, model, output_dir) VALUES (?, ?, ?)",
            (url, model, output_dir)
        )
        item_id = cur.lastrowid
        row = conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        item = dict(row)

    item["stages"] = _item_status(item)
    item["active_run"] = None
    return jsonify(item), 201


@app.route("/api/items/<int:item_id>", methods=["PATCH"])
def update_item(item_id):
    data = request.json or {}
    item = _get_item(item_id)
    if not item:
        return jsonify({"error": "not found"}), 404

    allowed = {"model"}
    updates = {k: v for k, v in data.items() if k in allowed and v}
    if updates:
        set_clause = ", ".join(f"{k}=?" for k in updates)
        with _db() as conn:
            conn.execute(f"UPDATE items SET {set_clause} WHERE id=?",
                         (*updates.values(), item_id))
        item = _get_item(item_id)

    item["stages"] = _item_status(item)
    return jsonify(item)


@app.route("/api/items/<int:item_id>", methods=["DELETE"])
def remove_item(item_id):
    with _db() as conn:
        conn.execute("DELETE FROM runs WHERE item_id=?", (item_id,))
        conn.execute("DELETE FROM items WHERE id=?", (item_id,))
    return jsonify({"ok": True})


@app.route("/api/items/<int:item_id>/status", methods=["GET"])
def item_status(item_id):
    item = _get_item(item_id)
    if not item:
        return jsonify({"error": "not found"}), 404

    # Refresh output_dir if needed
    if not item["output_dir"]:
        od = _detect_output_dir(item["url"])
        if od:
            rel = str(od.relative_to(SCRIPT_DIR))
            item["output_dir"] = rel
            with _db() as conn:
                conn.execute("UPDATE items SET output_dir=? WHERE id=? AND output_dir IS NULL",
                             (rel, item["id"]))

    with _db() as conn:
        active = conn.execute(
            "SELECT id, stage FROM runs WHERE item_id=? AND status='running'"
            " ORDER BY id DESC LIMIT 1",
            (item["id"],)
        ).fetchone()

    return jsonify({
        "stages":     _item_status(item),
        "active_run": dict(active) if active else None,
        "output_dir": item["output_dir"],
        "slug":       item["slug"],
    })


@app.route("/api/items/<int:item_id>/run/<stage>", methods=["POST"])
def run_stage_endpoint(item_id, stage):
    item = _get_item(item_id)
    if not item:
        return jsonify({"error": "item not found"}), 404
    if stage not in STAGE_BY_NAME:
        return jsonify({"error": "unknown stage"}), 400

    # Reject immediately if a required package isn't installed.
    available, install_hint = _stage_available(stage)
    if not available:
        pkg = STAGE_BY_NAME[stage].get("requires", "")
        return jsonify({
            "error": f"'{pkg}' is not installed. Run: {install_hint}",
            "unavailable": True,
            "install_hint": install_hint,
        }), 422

    # Prevent duplicate concurrent runs
    with _db() as conn:
        running = conn.execute(
            "SELECT id FROM runs WHERE item_id=? AND stage=? AND status='running'",
            (item_id, stage)
        ).fetchone()
    if running:
        return jsonify({"error": "already running", "run_id": running["id"]}), 409

    data = request.json or {}
    force = bool(data.get("force", False))
    workers = data.get("workers")
    overrides = data.get("overrides") or {}

    # Guard: refuse to re-run a completed stage unless force=True.
    # This catches the case where the UI's model mismatch caused a "pending"
    # status but the files actually exist on disk.
    if not force:
        done_map = _check_stage_done(item)
        if done_map.get(stage):
            return jsonify({
                "error": "already done — use force to re-run",
                "status": "done",
            }), 409

    try:
        cmd = _build_cmd(item, stage, force=force, workers=workers, overrides=overrides)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    run_id = _start_run(item_id, stage, cmd, forced=force)
    return jsonify({"run_id": run_id})


@app.route("/api/runs/<int:run_id>/cancel", methods=["POST"])
def cancel_run(run_id):
    run = _RUNS.get(run_id)
    if not run:
        return jsonify({"error": "run not found or already finished"}), 404
    proc = run["process"]
    try:
        # Kill the entire process group so Flask dev-server child processes
        # (and any other children) are also terminated, closing the stdout
        # pipe and allowing _reader_thread to detect EOF cleanly.
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    return jsonify({"ok": True})


@app.route("/api/runs/<int:run_id>/log")
def stream_log(run_id):
    def generate():
        run = _RUNS.get(run_id)

        if run is None:
            # Historical run — load from DB
            with _db() as conn:
                row = conn.execute(
                    "SELECT log, status, server_url FROM runs WHERE id=?", (run_id,)
                ).fetchone()
            if not row:
                yield "event: error\ndata: run not found\n\n"
                return
            for line in (row["log"] or "").split("\n"):
                yield f"data: {line}\n\n"
            if row["server_url"]:
                yield f"event: server_url\ndata: {row['server_url']}\n\n"
            yield f"event: done\ndata: {row['status']}\n\n"
            return

        # Live run — send buffered lines then subscribe for new ones
        q: queue.Queue = queue.Queue()
        _SUBS.setdefault(run_id, []).append(q)

        try:
            with run["lock"]:
                buffered = list(run["log_lines"])
                finished = run["finished"]
                exit_code = run.get("exit_code")
                server_url = run.get("server_url")

            for line in buffered:
                yield f"data: {line}\n\n"

            if server_url:
                yield f"event: server_url\ndata: {server_url}\n\n"

            if finished:
                status = "done" if exit_code == 0 else "failed"
                yield f"event: done\ndata: {status}\n\n"
                return

            # Stream live
            while True:
                try:
                    event_type, payload = q.get(timeout=25)
                    if event_type == "line":
                        yield f"data: {payload}\n\n"
                    elif event_type == "server_url":
                        yield f"event: server_url\ndata: {payload}\n\n"
                    elif event_type == "done":
                        yield f"event: done\ndata: {payload}\n\n"
                        return
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            subs = _SUBS.get(run_id, [])
            if q in subs:
                subs.remove(q)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ---------------------------------------------------------------------------
# HTML / JS dashboard
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>directory-pipeline</title>
<style>
:root {
  --bg:     #1a1a1a;
  --bg2:    #232323;
  --bg3:    #2c2c2c;
  --border: #383838;
  --text:   #ddd;
  --muted:  #777;
  --green:  #5a9e5a;
  --amber:  #c98a2a;
  --red:    #b85050;
  --blue:   #4a8ec2;
  --font:   'Menlo', 'Consolas', 'Monaco', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; }

/* Header */
header {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px;
  position: sticky; top: 0; background: var(--bg); z-index: 10;
}
header h1 { font-size: 13px; font-weight: normal; color: var(--muted); flex-shrink: 0; }
#add-form { display: flex; gap: 6px; align-items: center; flex: 1; }
#add-form input {
  background: var(--bg3); border: 1px solid var(--border); color: var(--text);
  font-family: var(--font); font-size: 12px; padding: 4px 8px; border-radius: 3px;
}
#add-form input:focus { outline: none; border-color: var(--blue); }
#url-input  { flex: 1; min-width: 200px; }
#model-input { width: 165px; }
#add-form button {
  background: var(--bg3); border: 1px solid var(--border); color: var(--text);
  font-family: var(--font); font-size: 12px; padding: 4px 10px; border-radius: 3px;
  cursor: pointer; white-space: nowrap;
}
#add-form button:hover { border-color: #666; }
#add-error { color: var(--red); font-size: 11px; }

/* Items container */
#items { padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; }
.empty-state { color: var(--muted); padding: 48px; text-align: center; }

/* Item card */
.item-card {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 4px;
  overflow: hidden;
}

/* Card header */
.item-header {
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: baseline; gap: 8px;
}
.item-slug   { font-weight: bold; flex-shrink: 0; }
.item-divider { color: var(--muted); flex-shrink: 0; }
.item-url    { color: var(--muted); font-size: 11px; flex: 1; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }
.item-model  { flex-shrink: 0; display: flex; align-items: center; gap: 4px; }
.item-model span { color: var(--muted); font-size: 10px; }
.item-model input {
  background: transparent; border: none; border-bottom: 1px dashed var(--border);
  color: #999; font-family: var(--font); font-size: 11px; width: 155px;
  padding: 1px 2px;
}
.item-model input:focus { outline: none; border-bottom-color: var(--blue); color: var(--text); }
.btn-remove {
  background: none; border: none; color: var(--muted); cursor: pointer;
  font-size: 13px; padding: 0 2px; flex-shrink: 0;
}
.btn-remove:hover { color: var(--red); }

/* Stage grid */
.item-stages { padding: 7px 10px; display: flex; flex-direction: column; gap: 5px; }
.stage-row   { display: flex; align-items: flex-start; gap: 8px; }
.group-label {
  color: var(--muted); font-size: 10px; letter-spacing: 0.07em; text-transform: uppercase;
  width: 62px; flex-shrink: 0; text-align: right; padding-top: 3px;
}
.stage-btns  { display: flex; flex-direction: column; gap: 4px; align-items: flex-start; flex: 1; }

/* Stage buttons */
.stage-btn {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 3px; border: 1px solid var(--border);
  font-family: var(--font); font-size: 11px; cursor: pointer;
  background: var(--bg3); color: var(--muted); white-space: nowrap;
  transition: border-color 0.1s, color 0.1s;
}
.stage-btn:hover:not(:disabled) { border-color: #666; color: var(--text); }
.stage-btn:disabled { cursor: default; opacity: 0.6; }
.stage-btn.done   { border-color: var(--green); color: var(--green); }
.stage-btn.running { border-color: var(--amber); color: var(--amber); }
.stage-btn.failed  { border-color: var(--red);   color: var(--red); }

/* Arg panel */
.arg-panel {
  display: none;
  margin: 3px 0 2px 0;
  background: var(--bg); border: 1px solid var(--border); border-radius: 3px;
  padding: 8px 10px; width: 100%;
}
.arg-panel.open { display: block; }
.arg-row {
  display: flex; align-items: center; gap: 6px;
  margin-bottom: 5px; font-size: 11px;
}
.arg-row label { color: var(--muted); width: 140px; flex-shrink: 0; text-align: right; }
.arg-row input[type=text],
.arg-row input[type=number] {
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--text); font-family: var(--font); font-size: 11px;
  padding: 2px 6px; border-radius: 3px; width: 220px;
}
.arg-row input[type=text]:focus,
.arg-row input[type=number]:focus { outline: none; border-color: var(--blue); }
.arg-row input[type=checkbox] { margin: 0; cursor: pointer; }
.stage-hint {
  font-size: 10px; color: var(--muted); margin-bottom: 8px; line-height: 1.5;
  padding-bottom: 7px; border-bottom: 1px solid var(--border);
}
.arg-hint {
  font-size: 10px; color: #555; line-height: 1.4; margin-left: 2px;
  flex: 1;
}
.cmd-preview {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 3px;
  padding: 5px 8px; font-size: 10px; color: #666;
  white-space: pre-wrap; word-break: break-all;
  margin: 6px 0; line-height: 1.5;
}
.panel-actions { display: flex; gap: 6px; margin-top: 6px; align-items: center; }
.btn-run {
  background: var(--bg3); border: 1px solid var(--border); color: var(--text);
  font-family: var(--font); font-size: 11px; padding: 3px 12px;
  border-radius: 3px; cursor: pointer;
}
.btn-run:hover { border-color: var(--green); color: var(--green); }
.stage-btn.active { border-color: var(--blue); color: var(--blue); }
.btn-cancel {
  background: none; border: none; color: var(--muted);
  font-family: var(--font); font-size: 11px; cursor: pointer; padding: 0;
}
.btn-cancel:hover { color: var(--text); }

/* Log panel */
.item-log { border-top: 1px solid var(--border); }
.log-header {
  padding: 5px 10px; display: flex; align-items: center; gap: 6px;
  font-size: 11px; color: var(--muted); cursor: pointer; user-select: none;
}
.log-header:hover { color: var(--text); }
.log-toggle  { font-size: 10px; }
.log-label   { flex: 1; }
.log-status  { font-weight: bold; }
.log-stop-btn {
  background: none; border: 1px solid var(--border); color: var(--muted);
  font-family: var(--font); font-size: 10px; padding: 1px 6px;
  border-radius: 3px; cursor: pointer; flex-shrink: 0;
}
.log-stop-btn:hover { border-color: #c44; color: #c44; }
.log-stop-btn:disabled { opacity: 0.5; cursor: default; }
.log-body    { display: none; padding: 0 10px 8px; }
.log-body.open { display: block; }
.log-pre {
  background: var(--bg); border: 1px solid var(--border); border-radius: 3px;
  padding: 6px 8px; font-size: 11px; max-height: 260px; overflow-y: auto;
  white-space: pre-wrap; word-break: break-all; line-height: 1.5; color: #bbb;
}
.server-url-btn {
  display: inline-block; margin-top: 6px;
  background: var(--blue); border: none; color: #fff;
  font-family: var(--font); font-size: 11px;
  padding: 3px 10px; border-radius: 3px; cursor: pointer;
}
.server-url-btn:hover { opacity: 0.85; }
</style>
</head>
<body>

<header>
  <h1>directory-pipeline</h1>
  <form id="add-form" onsubmit="addItem(event)">
    <input id="url-input"   type="text"
           placeholder="IIIF manifest URL or collection URL"
           autocomplete="off" spellcheck="false">
    <input id="model-input" type="text" value="gemini-3.1-flash-lite" placeholder="model">
    <button type="submit">+ Add item</button>
    <span id="add-error"></span>
  </form>
</header>

<div id="items"></div>

<script>
// ---- Constants (injected by Flask) ----------------------------------------
const STAGES = {{ stages_json | safe }};
const STAGE_ARG_DEFS = {{ stage_args_json | safe }};
const STAGE_HINTS = {{ stage_hints_json | safe }};
const GROUPS = [
  ['ingest',    'Ingest'],
  ['calibrate', 'Calibrate'],
  ['ocr',       'OCR'],
  ['review',    'Review'],
  ['extract',   'Extract'],
  ['iiif',      'IIIF'],
];

// ---- State ----------------------------------------------------------------
let _items   = [];     // [{id, url, slug, model, output_dir, stages, active_run}]
let _polls   = {};     // itemId -> setInterval handle
let _sources = {};     // itemId -> EventSource (for cleanup)

// ---- Helpers --------------------------------------------------------------
function statusIcon(s) {
  return {done:'✓', running:'▶', failed:'✗', pending:'○'}[s] || '○';
}

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

// ---- Rendering ------------------------------------------------------------
function render() {
  const container = document.getElementById('items');

  if (!_items.length) {
    container.innerHTML =
      '<div class="empty-state">No items yet. Paste a URL above.</div>';
    return;
  }

  // Clear empty-state placeholder if present
  const emptyState = container.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  // Remove cards for deleted items
  const ids = new Set(_items.map(i => i.id));
  container.querySelectorAll('.item-card').forEach(c => {
    if (!ids.has(+c.dataset.id)) c.remove();
  });

  // Insert or replace cards
  _items.forEach((item, idx) => {
    const existing = container.querySelector(`.item-card[data-id="${item.id}"]`);

    // Save open arg-panel state (and focused field) before rebuilding
    const panelState = {};
    let focusedArgName = null, focusedStage = null, focusedSelStart = null, focusedSelEnd = null;
    if (existing) {
      existing.querySelectorAll('.arg-panel.open').forEach(p => {
        if (!p.dataset.stage) return;
        const vals = {};
        p.querySelectorAll('[data-arg-name]').forEach(inp => {
          vals[inp.dataset.argName] = inp.type === 'checkbox' ? inp.checked : inp.value;
          if (document.activeElement === inp) {
            focusedArgName = inp.dataset.argName;
            focusedStage   = p.dataset.stage;
            focusedSelStart = inp.selectionStart;
            focusedSelEnd   = inp.selectionEnd;
          }
        });
        panelState[p.dataset.stage] = vals;
      });
    }

    const newCard = buildCard(item);

    // Restore arg-panel open state into freshly built card
    if (Object.keys(panelState).length) {
      newCard.querySelectorAll('.arg-panel').forEach(p => {
        const st = p.dataset.stage;
        if (!st || !panelState[st]) return;
        const vals = panelState[st];
        p.querySelectorAll('[data-arg-name]').forEach(inp => {
          const v = vals[inp.dataset.argName];
          if (v !== undefined) {
            if (inp.type === 'checkbox') inp.checked = v;
            else inp.value = v;
          }
        });
        p.classList.add('open');
        const stageBtn = newCard.querySelector(`.stage-btn[data-stage="${st}"]`);
        if (stageBtn) stageBtn.classList.add('active');
        // Refresh the command preview with restored values
        p.querySelectorAll('[data-arg-name]').forEach(inp => inp.dispatchEvent(new Event('input')));
      });
    }

    if (existing) {
      // Preserve log panel across re-renders so output stays scrollable
      const existingLog = existing.querySelector(`#log-${item.id}`);
      const logPre = existingLog ? existingLog.querySelector(`#log-pre-${item.id}`) : null;
      const savedScroll = logPre ? logPre.scrollTop : 0;
      if (existingLog && existingLog.style.display !== 'none') {
        const newLog = newCard.querySelector(`#log-${item.id}`);
        if (newLog) newLog.replaceWith(existingLog);
      }
      existing.replaceWith(newCard);
      // Restore scroll position — DOM move resets it to 0
      if (logPre && savedScroll > 0) logPre.scrollTop = savedScroll;
    } else {
      const ref = container.children[idx];
      if (ref) container.insertBefore(newCard, ref);
      else container.appendChild(newCard);
    }

    // Restore focus to whichever arg input was active before the rebuild
    if (focusedArgName && focusedStage) {
      const panel = newCard.querySelector(`.arg-panel[data-stage="${focusedStage}"]`);
      if (panel) {
        const inp = panel.querySelector(`[data-arg-name="${focusedArgName}"]`);
        if (inp) {
          inp.focus();
          if (focusedSelStart !== null && inp.setSelectionRange) {
            inp.setSelectionRange(focusedSelStart, focusedSelEnd);
          }
        }
      }
    }
  });
}

function buildCard(item) {
  const card = el('div', 'item-card');
  card.dataset.id = item.id;

  // --- Header ---
  const hdr = el('div', 'item-header');

  const slugEl = el('span', 'item-slug');
  slugEl.textContent = item.slug || '(pending)';
  hdr.appendChild(slugEl);

  const div = el('span', 'item-divider');
  div.textContent = '—';
  hdr.appendChild(div);

  const urlEl = el('span', 'item-url');
  urlEl.textContent = item.url;
  urlEl.title = item.url;
  hdr.appendChild(urlEl);

  if (item.output_dir) {
    const dirEl = el('span');
    dirEl.style.cssText = 'color:#555;font-size:10px;flex-shrink:0;white-space:nowrap;';
    dirEl.textContent = item.output_dir;
    hdr.appendChild(dirEl);
  }

  const modelWrap = el('div', 'item-model');
  const modelLbl = el('span');
  modelLbl.textContent = 'model:';
  modelWrap.appendChild(modelLbl);
  const modelInput = el('input');
  modelInput.type = 'text';
  modelInput.value = item.model;
  modelInput.title = 'OCR model — press Enter to save';
  modelInput.addEventListener('change', () => patchModel(item.id, modelInput.value.trim()));
  modelWrap.appendChild(modelInput);
  hdr.appendChild(modelWrap);

  const rmBtn = el('button', 'btn-remove');
  rmBtn.textContent = '✕';
  rmBtn.title = 'Remove from tracking (output files kept)';
  rmBtn.addEventListener('click', () => removeItem(item.id));
  hdr.appendChild(rmBtn);

  card.appendChild(hdr);

  // --- Stage grid ---
  const stagesDiv = el('div', 'item-stages');
  GROUPS.forEach(([group, label]) => {
    const groupStages = STAGES.filter(s => s.group === group);
    if (!groupStages.length) return;

    const row = el('div', 'stage-row');
    const lbl = el('span', 'group-label');
    lbl.textContent = label;
    row.appendChild(lbl);

    const btns = el('div', 'stage-btns');
    groupStages.forEach(s => {
      const status = (item.stages || {})[s.name] || 'pending';
      btns.appendChild(buildStageControl(item, s, status));
    });
    row.appendChild(btns);
    stagesDiv.appendChild(row);
  });
  card.appendChild(stagesDiv);

  // --- Log panel (hidden until a run starts) ---
  const logDiv = el('div', 'item-log');
  logDiv.id = `log-${item.id}`;
  logDiv.style.display = 'none';
  card.appendChild(logDiv);

  return card;
}

function buildStageControl(item, stage, status) {
  const wrap = el('div');

  // Stage toggle button
  const btn = el('button', `stage-btn ${status}`);
  btn.dataset.stage = stage.name;
  const icon = el('span'); icon.textContent = statusIcon(status);
  btn.appendChild(icon);
  const lbl = el('span');
  lbl.textContent = stage.label + (stage.interactive ? ' ↗' : '');
  btn.appendChild(lbl);

  if (status === 'running') {
    btn.disabled = true;
    wrap.appendChild(btn);
    return wrap;
  }

  // Unavailable: required package not installed.
  if (stage.available === false) {
    btn.disabled = true;
    btn.title = `Not available — install with: ${stage.install_hint}`;
    const hint = el('span');
    hint.style.cssText = 'font-size:11px;color:var(--muted);margin-left:6px;';
    hint.textContent = stage.install_hint;
    wrap.appendChild(btn);
    wrap.appendChild(hint);
    return wrap;
  }

  // Arg panel
  const panel = el('div', 'arg-panel');
  panel.dataset.stage = stage.name;
  const argDefs = STAGE_ARG_DEFS[stage.name] || [];
  const fields = {};

  // Stage-level description
  const stageHint = STAGE_HINTS[stage.name];
  if (stageHint) {
    const hintEl = el('div', 'stage-hint');
    hintEl.textContent = stageHint;
    panel.appendChild(hintEl);
  }

  argDefs.forEach(def => {
    const row = el('div', 'arg-row');
    const labelEl = el('label'); labelEl.textContent = def.label + ':';
    row.appendChild(labelEl);
    let input;
    if (def.type === 'bool') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = def.default || false;
    } else {
      input = document.createElement('input');
      input.type = (def.type === 'int' || def.type === 'float') ? 'number' : 'text';
      input.value = def.default || '';
      if (def.placeholder) input.placeholder = def.placeholder;
      if (def.type === 'float') input.step = 'any';
      if (def.min !== undefined) input.min = def.min;
      if (def.max !== undefined) input.max = def.max;
    }
    input.dataset.argName = def.name;
    input.addEventListener('input', () => updateCmdPreview(item, stage, fields, previewEl));
    input.addEventListener('change', () => updateCmdPreview(item, stage, fields, previewEl));
    row.appendChild(input);
    if (def.hint) {
      const hintEl = el('span', 'arg-hint');
      hintEl.textContent = def.hint;
      row.appendChild(hintEl);
    }
    panel.appendChild(row);
    fields[def.name] = { input, def };
  });

  const previewEl = el('div', 'cmd-preview');
  panel.appendChild(previewEl);

  const actions = el('div', 'panel-actions');
  const runBtn = el('button', 'btn-run');
  runBtn.textContent = status === 'done' ? 'Force-run' : 'Run';
  runBtn.addEventListener('click', () => {
    const overrides = gatherOverrides(fields);
    const force = status === 'done' || bool(overrides.force);
    panel.classList.remove('open');
    btn.classList.remove('active');
    runStage(item.id, stage.name, force, overrides);
  });
  const cancelBtn = el('button', 'btn-cancel');
  cancelBtn.textContent = 'cancel';
  cancelBtn.addEventListener('click', () => {
    panel.classList.remove('open');
    btn.classList.remove('active');
  });
  actions.appendChild(runBtn);
  actions.appendChild(cancelBtn);
  panel.appendChild(actions);

  btn.addEventListener('click', () => {
    const isOpen = panel.classList.contains('open');
    if (isOpen) {
      panel.classList.remove('open');
      btn.classList.remove('active');
    } else {
      panel.classList.add('open');
      btn.classList.add('active');
      updateCmdPreview(item, stage, fields, previewEl);
    }
  });

  wrap.appendChild(btn);
  wrap.appendChild(panel);
  return wrap;
}

function gatherOverrides(fields) {
  const overrides = {};
  Object.entries(fields).forEach(([name, {input, def}]) => {
    if (def.type === 'bool') {
      if (input.checked) overrides[name] = true;
    } else if (input.value.trim()) {
      overrides[name] = input.value.trim();
    }
  });
  return overrides;
}

function bool(v) { return v === true || v === 'true'; }

function buildCmdPreview(item, stage, overrides) {
  const model = overrides.model || item.model || 'gemini-3.1-flash-lite';
  const parts = ['python main.py', item.url || '<url>'];
  if (stage.flag) {
    parts.push(stage.flag);
    parts.push('--model', model);
  } else {
    // IIIF standalone script
    parts[0] = 'python';
    parts.splice(1, 1, `pipeline/iiif/${stage.name}.py`, '<output_dir>');
    if (overrides.model) parts.push('--model', overrides.model);
  }
  const argDefs = STAGE_ARG_DEFS[stage.name] || [];
  argDefs.forEach(def => {
    if (def.name === 'model') return;
    const val = overrides[def.name];
    if (!val && val !== true) return;
    if (def.type === 'bool') { parts.push(def.flag); }
    else { parts.push(def.flag, String(val)); }
  });
  return parts.join(' ');
}

function updateCmdPreview(item, stage, fields, previewEl) {
  const overrides = gatherOverrides(fields);
  previewEl.textContent = buildCmdPreview(item, stage, overrides);
}

// ---- Log panel management -------------------------------------------------
function initLogPanel(itemId, stageLabel, runId) {
  const logDiv = document.getElementById(`log-${itemId}`);
  if (!logDiv) return;
  logDiv.style.display = 'block';
  logDiv.innerHTML = '';
  logDiv.dataset.runId = runId || '';

  const hdr = el('div', 'log-header');
  hdr.addEventListener('click', (e) => {
    if (e.target.closest('.log-stop-btn')) return;
    const body = logDiv.querySelector('.log-body');
    if (body) body.classList.toggle('open');
  });

  const toggle = el('span', 'log-toggle'); toggle.textContent = '▾';
  hdr.appendChild(toggle);
  const lbl = el('span', 'log-label');
  lbl.innerHTML = `${stageLabel} &mdash; <span class="log-status">starting…</span>`;
  hdr.appendChild(lbl);

  if (runId) {
    const stopBtn = document.createElement('button');
    stopBtn.className = 'log-stop-btn';
    stopBtn.textContent = '■ Stop';
    stopBtn.title = 'Terminate the running process';
    stopBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      stopBtn.disabled = true;
      stopBtn.textContent = 'stopping…';
      const res = await fetch(`/api/runs/${runId}/cancel`, {method: 'POST'});
      if (res.ok) {
        appendLog(itemId, '\n[stopped by user]');
        stopBtn.textContent = 'stopped';
      } else {
        stopBtn.disabled = false;
        stopBtn.textContent = '■ Stop';
      }
    });
    hdr.appendChild(stopBtn);
  }

  logDiv.appendChild(hdr);

  const body = el('div', 'log-body open');
  const pre = el('pre', 'log-pre');
  pre.id = `log-pre-${itemId}`;
  body.appendChild(pre);

  const serverWrap = el('div');
  serverWrap.id = `server-btn-${itemId}`;
  body.appendChild(serverWrap);

  logDiv.appendChild(body);
}

function appendLog(itemId, text) {
  const pre = document.getElementById(`log-pre-${itemId}`);
  if (!pre) return;
  pre.textContent += text + '\n';
  // auto-scroll if near bottom
  if (pre.scrollHeight - pre.scrollTop < pre.clientHeight + 40)
    pre.scrollTop = pre.scrollHeight;
}

function setLogStatus(itemId, text) {
  const el = document.querySelector(`#log-${itemId} .log-status`);
  if (el) el.textContent = text;
}

function showServerUrl(itemId, url) {
  const wrap = document.getElementById(`server-btn-${itemId}`);
  if (!wrap || wrap.querySelector('.server-url-btn')) return;
  const btn = document.createElement('button');
  btn.className = 'server-url-btn';
  btn.textContent = `Open  ↗  ${url}`;
  btn.addEventListener('click', () => window.open(url, '_blank'));
  wrap.appendChild(btn);
}

// ---- API ------------------------------------------------------------------
async function loadItems() {
  const res = await fetch('/api/items');
  _items = await res.json();
  render();
  // Reconnect to any runs that were active before the page was refreshed
  _items.forEach(item => {
    if (item.active_run && item.active_run.id && !_sources[item.id]) {
      const stage = STAGES.find(s => s.name === item.active_run.stage);
      const label = stage ? stage.label : item.active_run.stage;
      initLogPanel(item.id, label, item.active_run.id);
      openLogStream(item.id, item.active_run.id);
      startPolling(item.id);
    }
  });
}

async function addItem(e) {
  e.preventDefault();
  const url   = document.getElementById('url-input').value.trim();
  const model = document.getElementById('model-input').value.trim();
  const errEl = document.getElementById('add-error');
  errEl.textContent = '';
  if (!url) { errEl.textContent = 'URL required'; return; }

  const res = await fetch('/api/items', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url, model}),
  });
  const data = await res.json();
  if (!res.ok) { errEl.textContent = data.error || 'Error'; return; }

  _items.unshift(data);
  render();
  document.getElementById('url-input').value = '';
}

async function removeItem(itemId) {
  if (!confirm('Remove item from tracking?\n(Output files will be kept.)')) return;
  await fetch(`/api/items/${itemId}`, {method: 'DELETE'});
  _items = _items.filter(i => i.id !== itemId);
  stopPolling(itemId);
  if (_sources[itemId]) { _sources[itemId].close(); delete _sources[itemId]; }
  render();
}

async function patchModel(itemId, model) {
  if (!model) return;
  await fetch(`/api/items/${itemId}`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model}),
  });
}

async function runStage(itemId, stageName, force, overrides = {}) {
  const res = await fetch(`/api/items/${itemId}/run/${stageName}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({force, overrides}),
  });
  const data = await res.json();
  if (!res.ok) {
    if (data.status === 'done') {
      // Stale UI — stage was already done but showed as pending (e.g. model slug changed).
      // Refresh status silently so the button updates to "Force-run".
      await refreshItem(itemId);
      return;
    }
    if (data.unavailable) {
      alert(`Cannot run ${stageName}: ${data.error}`);
      return;
    }
    alert(data.error || 'Failed to start stage');
    return;
  }

  const runId = data.run_id;
  const stage = STAGES.find(s => s.name === stageName);
  const label = stage ? stage.label : stageName;

  // Optimistically mark as running
  const idx = _items.findIndex(i => i.id === itemId);
  if (idx >= 0) {
    _items[idx] = {
      ..._items[idx],
      stages: {...(_items[idx].stages || {}), [stageName]: 'running'},
      active_run: {id: runId, stage: stageName},
    };
  }
  render();

  initLogPanel(itemId, label, runId);
  openLogStream(itemId, runId);
  startPolling(itemId);
}

function openLogStream(itemId, runId) {
  if (_sources[itemId]) _sources[itemId].close();

  const es = new EventSource(`/api/runs/${runId}/log`);
  _sources[itemId] = es;

  es.onmessage = (e) => { appendLog(itemId, e.data); };

  es.addEventListener('server_url', (e) => {
    showServerUrl(itemId, e.data);
    window.open(e.data, '_blank');
  });

  es.addEventListener('done', (e) => {
    const ok = e.data === 'done';
    setLogStatus(itemId, ok ? 'done ✓' : 'failed ✗');
    es.close();
    delete _sources[itemId];
    stopPolling(itemId);
    refreshItem(itemId);
  });

  es.onerror = () => {
    es.close();
    delete _sources[itemId];
    stopPolling(itemId);
    refreshItem(itemId);
  };
}

async function refreshItem(itemId) {
  const res = await fetch(`/api/items/${itemId}/status`);
  if (!res.ok) return;
  const data = await res.json();
  const idx = _items.findIndex(i => i.id === itemId);
  if (idx >= 0) {
    _items[idx] = {..._items[idx], ...data};
    render();
  }
}

function startPolling(itemId) {
  if (_polls[itemId]) return;
  _polls[itemId] = setInterval(() => refreshItem(itemId), 3000);
}

function stopPolling(itemId) {
  if (_polls[itemId]) { clearInterval(_polls[itemId]); delete _polls[itemId]; }
}

// ---- Init -----------------------------------------------------------------
loadItems();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="directory-pipeline web dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open the browser automatically")
    args = parser.parse_args()

    _init_db()

    url = f"http://127.0.0.1:{args.port}"
    print(f"directory-pipeline dashboard — {url}", flush=True)

    if not args.no_open:
        import threading as _t
        _t.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
