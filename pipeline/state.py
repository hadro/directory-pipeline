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
    "last_run": "2026-05-31T14:22:00Z",
    "last_run_args": "main.py output/vol1 --gemini-ocr --flex"
}

``last_run_args`` records the command line behind the most recent write, so a
run's flags (``--flex``, ``--mode multimodal``, a ``--prompt-file`` override)
stay recoverable afterwards — the rest of the schema captures *what* a stage
used, not *how* it was invoked.  It is captured automatically; pass
``args=None`` to suppress it.
"""

import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

_FILENAME = "pipeline_state.json"

# Sentinel: lets `args=None` mean "record nothing" while an omitted argument
# still auto-captures the invocation.
_UNSET = object()


def _invocation() -> str:
    """The current command line as a copy-pasteable string.

    The interpreter path is reduced to the script's basename so the value stays
    stable across machines and virtualenvs.
    """
    argv = sys.argv or []
    if not argv:
        return ""
    try:
        return shlex.join([Path(argv[0]).name, *argv[1:]])
    except Exception:
        return ""


def _state_path(output_dir: Path) -> Path:
    return output_dir / _FILENAME


def find_state_dir(start: Path, max_up: int = 2) -> Path:
    """Return the directory whose ``pipeline_state.json`` governs *start*.

    main.py always writes state at the slug directory, but the leaf scripts are
    handed either that directory or an item directory inside it.  Walking up a
    bounded number of levels to an existing state file means a direct leaf
    invocation updates the same file the orchestrator does instead of scattering
    a second one alongside the images.  Falls back to *start* when none exists.
    """
    cur = Path(start).resolve()
    for _ in range(max_up + 1):
        if (cur / _FILENAME).exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path(start)


def read_state(output_dir: Path) -> dict:
    p = _state_path(output_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def write_state(output_dir: Path, updates: dict, args=_UNSET) -> None:
    """Merge *updates* into the state file and stamp ``last_run``.

    *args* records the invocation in ``last_run_args``: omit it to capture the
    current command line, pass a string or argv list to record something
    specific, or pass ``None`` to leave the field untouched.
    """
    p = _state_path(output_dir)
    state = read_state(output_dir)
    state.update(updates)
    state["last_run"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args is _UNSET:
        args = _invocation()
    if args:
        state["last_run_args"] = args if isinstance(args, str) else shlex.join(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_stage(output_dir: Path, stage: str, args=_UNSET, *, updates: dict | None = None) -> None:
    """Append *stage* to ``stages_completed``, merging *updates* in the same write.

    Pass fields the stage owns (``ocr_model``, ``ner_model``, …) via *updates*
    rather than following this with a separate ``write_state`` call — the two
    belong to one logical update, and merging them halves the read-merge-write
    cycles on the state file.
    """
    state = read_state(output_dir)
    completed = state.get("stages_completed", [])
    if stage not in completed:
        completed.append(stage)
    write_state(output_dir, {**(updates or {}), "stages_completed": completed}, args=args)


def get_last_run_args(output_dir: Path) -> str | None:
    return read_state(output_dir).get("last_run_args")


def get_ocr_model(output_dir: Path) -> str | None:
    return read_state(output_dir).get("ocr_model")


def get_ner_model(output_dir: Path) -> str | None:
    return read_state(output_dir).get("ner_model")
