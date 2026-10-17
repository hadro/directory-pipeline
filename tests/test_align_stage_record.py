"""align_ocr records its stage in pipeline_state.json only when nothing failed.

main.py trusts the state file, so a run that aligned some pages and failed on
others must not look finished. The OCR stage already gates the same way.
"""

import json
import sys

import pipeline.align_ocr as A

MODEL = "gemini-3.1-flash-lite"


def _volume(tmp_path):
    for stem in ("0001_a", "0002_b"):
        (tmp_path / f"{stem}.jpg").write_bytes(b"")
        (tmp_path / f"{stem}_{MODEL}.txt").write_text("x\n", encoding="utf-8")
        (tmp_path / f"{stem}_surya.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _run(tmp_path, monkeypatch, fail_stem):
    def fake_align(image_path, *args, **kwargs):
        return ("failed" if image_path.stem == fail_stem else "ok"), False

    monkeypatch.setattr(A, "align_image", fake_align)
    monkeypatch.setattr(sys, "argv", ["align_ocr.py", str(tmp_path), "--model", MODEL, "--quiet"])
    A.main()
    state = tmp_path / "pipeline_state.json"
    if not state.exists():
        return []
    return json.loads(state.read_text(encoding="utf-8")).get("stages_completed", [])


def test_a_partial_failure_leaves_the_stage_unrecorded(tmp_path, monkeypatch):
    assert "align_ocr" not in _run(_volume(tmp_path), monkeypatch, fail_stem="0002_b")


def test_a_clean_run_records_the_stage(tmp_path, monkeypatch):
    assert "align_ocr" in _run(_volume(tmp_path), monkeypatch, fail_stem=None)
