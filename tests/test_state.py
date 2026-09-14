"""pipeline/state.py — pipeline_state.json round-trips and model lookup."""

import json
import sys

from pipeline.state import (
    find_state_dir,
    get_last_run_args,
    get_ner_model,
    get_ocr_model,
    read_state,
    record_stage,
    write_state,
)


def test_read_missing_returns_empty(tmp_path):
    assert read_state(tmp_path) == {}


def test_write_then_read_round_trip(tmp_path):
    write_state(tmp_path, {"slug": "vol1", "ocr_model": "gemini-x"})
    state = read_state(tmp_path)
    assert state["slug"] == "vol1"
    assert state["ocr_model"] == "gemini-x"
    assert "last_run" in state


def test_write_merges_rather_than_replaces(tmp_path):
    write_state(tmp_path, {"slug": "vol1"})
    write_state(tmp_path, {"ocr_model": "gemini-x"})
    state = read_state(tmp_path)
    assert state["slug"] == "vol1"
    assert state["ocr_model"] == "gemini-x"


def test_write_creates_directory(tmp_path):
    target = tmp_path / "new" / "deep"
    write_state(target, {"slug": "s"})
    assert read_state(target)["slug"] == "s"


def test_record_stage_appends_and_dedups(tmp_path):
    record_stage(tmp_path, "download")
    record_stage(tmp_path, "gemini_ocr")
    record_stage(tmp_path, "download")
    assert read_state(tmp_path)["stages_completed"] == ["download", "gemini_ocr"]


def test_record_stage_merges_updates_in_one_write(tmp_path):
    record_stage(tmp_path, "gemini_ocr", updates={"ocr_model": "gemini-o"})
    state = read_state(tmp_path)
    assert state["stages_completed"] == ["gemini_ocr"]
    assert state["ocr_model"] == "gemini-o"


def test_record_stage_updates_cannot_clobber_stage_list(tmp_path):
    record_stage(tmp_path, "download")
    record_stage(tmp_path, "gemini_ocr", updates={"stages_completed": ["bogus"]})
    assert read_state(tmp_path)["stages_completed"] == ["download", "gemini_ocr"]


def test_model_getters(tmp_path):
    assert get_ocr_model(tmp_path) is None
    assert get_ner_model(tmp_path) is None
    write_state(tmp_path, {"ocr_model": "gemini-o", "ner_model": "gemini-n"})
    assert get_ocr_model(tmp_path) == "gemini-o"
    assert get_ner_model(tmp_path) == "gemini-n"


def test_corrupted_state_file_reads_as_empty(tmp_path):
    (tmp_path / "pipeline_state.json").write_text("{not json", encoding="utf-8")
    assert read_state(tmp_path) == {}


def test_state_file_is_valid_json_on_disk(tmp_path):
    write_state(tmp_path, {"slug": "s"})
    raw = (tmp_path / "pipeline_state.json").read_text(encoding="utf-8")
    assert json.loads(raw)["slug"] == "s"


def test_last_run_args_captured_automatically(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["/long/path/to/main.py", "output/vol1", "--flex"])
    write_state(tmp_path, {"slug": "s"})
    # argv[0] is reduced to a basename so the value is stable across machines
    assert get_last_run_args(tmp_path) == "main.py output/vol1 --flex"


def test_last_run_args_survives_empty_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [])
    write_state(tmp_path, {"slug": "s"})
    assert "last_run_args" not in read_state(tmp_path)


def test_last_run_args_explicit_string(tmp_path):
    write_state(tmp_path, {"slug": "s"}, args="main.py out/vol --gemini-ocr --flex")
    assert get_last_run_args(tmp_path) == "main.py out/vol --gemini-ocr --flex"


def test_last_run_args_accepts_argv_list(tmp_path):
    write_state(tmp_path, {"slug": "s"}, args=["main.py", "out/a b", "--flex"])
    assert get_last_run_args(tmp_path) == "main.py 'out/a b' --flex"


def test_last_run_args_none_leaves_field_absent(tmp_path):
    write_state(tmp_path, {"slug": "s"}, args=None)
    assert "last_run_args" not in read_state(tmp_path)


def test_last_run_args_none_preserves_existing_value(tmp_path):
    write_state(tmp_path, {"slug": "s"}, args="first.py --a")
    write_state(tmp_path, {"ocr_model": "m"}, args=None)
    assert get_last_run_args(tmp_path) == "first.py --a"


def test_record_stage_passes_args_through(tmp_path):
    record_stage(tmp_path, "gemini_ocr", args="run_gemini_ocr.py out/vol --flex")
    state = read_state(tmp_path)
    assert state["stages_completed"] == ["gemini_ocr"]
    assert state["last_run_args"] == "run_gemini_ocr.py out/vol --flex"


def test_find_state_dir_prefers_existing_parent_state(tmp_path):
    """A leaf script handed an item dir must update the slug-level state file
    rather than scattering a second one next to the images."""
    slug = tmp_path / "slug"
    item = slug / "item"
    item.mkdir(parents=True)
    write_state(slug, {"slug": "demo"})
    assert find_state_dir(item) == slug.resolve()


def test_find_state_dir_returns_start_when_no_state_anywhere(tmp_path):
    item = tmp_path / "slug" / "item"
    item.mkdir(parents=True)
    assert find_state_dir(item) == item


def test_find_state_dir_prefers_own_state_over_parent(tmp_path):
    slug = tmp_path / "slug"
    item = slug / "item"
    item.mkdir(parents=True)
    write_state(slug, {"slug": "parent"})
    write_state(item, {"slug": "own"})
    assert find_state_dir(item) == item.resolve()


def test_find_state_dir_respects_max_up(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    write_state(tmp_path, {"slug": "far"})
    # too many levels up to reach it — falls back to the start dir
    assert find_state_dir(deep, max_up=2) == deep
