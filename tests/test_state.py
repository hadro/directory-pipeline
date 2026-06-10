"""pipeline/state.py — pipeline_state.json round-trips and model lookup."""

import json

from pipeline.state import (
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
