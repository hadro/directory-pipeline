#!/usr/bin/env python3
"""Read and write pipeline_state.json in each output slug directory.

pipeline_state.json records which model was used for each stage, so downstream
scripts (fix_entries.py, align_ocr.py, extract_entries.py) can discover the
right model without a --model flag.

Schema
------
{
    "source_url": "https://...",
    "slug": "ldpd_11290437_000",
    "ocr_model": "gemini-3.1-flash-lite",
    "ner_model": "gemini-3.1-flash-lite",
    "stages_completed": ["download", "surya_ocr", "gemini_ocr", "align_ocr"],
    "last_run": "2026-05-31T14:22:00Z"
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path

_FILENAME = "pipeline_state.json"


def _state_path(output_dir: Path) -> Path:
    return output_dir / _FILENAME


def read_state(output_dir: Path) -> dict:
    p = _state_path(output_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def write_state(output_dir: Path, updates: dict) -> None:
    p = _state_path(output_dir)
    state = read_state(output_dir)
    state.update(updates)
    state["last_run"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_stage(output_dir: Path, stage: str) -> None:
    state = read_state(output_dir)
    completed = state.get("stages_completed", [])
    if stage not in completed:
        completed.append(stage)
    write_state(output_dir, {"stages_completed": completed})


def get_ocr_model(output_dir: Path) -> str | None:
    return read_state(output_dir).get("ocr_model")


def get_ner_model(output_dir: Path) -> str | None:
    return read_state(output_dir).get("ner_model")
