"""utils/models.py — model slugs and OCR-model discovery from filenames."""

from pathlib import Path

from utils.models import (
    DEFAULT_NER_MODEL,
    DEFAULT_OCR_MODEL,
    discover_ocr_slug,
    model_slug,
)


def test_model_slug_passthrough():
    assert model_slug("gemini-2.0-flash") == "gemini-2.0-flash"


def test_model_slug_replaces_slashes():
    assert model_slug("models/gemini-2.0-flash") == "models_gemini-2.0-flash"


def test_defaults_are_nonempty_strings():
    assert DEFAULT_OCR_MODEL and isinstance(DEFAULT_OCR_MODEL, str)
    assert DEFAULT_NER_MODEL and isinstance(DEFAULT_NER_MODEL, str)


def _touch(d: Path, *names: str) -> None:
    for n in names:
        (d / n).write_text("")


def test_discover_missing_dir_returns_none(tmp_path):
    assert discover_ocr_slug(tmp_path / "nope") is None


def test_discover_empty_dir_returns_none(tmp_path):
    assert discover_ocr_slug(tmp_path) is None


def test_discover_from_aligned_json(tmp_path):
    _touch(tmp_path, "0001_5142941_gemini-3.1-flash-lite_aligned.json")
    assert discover_ocr_slug(tmp_path) == "gemini-3.1-flash-lite"


def test_discover_handles_nonnumeric_image_ids(tmp_path):
    # IA-style stems contain URL-encoded slashes and dots — the regex must
    # anchor on the final "_{slug}_aligned.json", not on a numeric page stem.
    _touch(tmp_path, "0077_1897BPL%2F1897BPL_jp2.zip%2F1897BPL_0076.jp2_gemini-3.1-flash-lite-preview_aligned.json")
    assert discover_ocr_slug(tmp_path) == "gemini-3.1-flash-lite-preview"


def test_discover_aligned_is_model_prefix_agnostic(tmp_path):
    _touch(tmp_path, "0001_id_chandra-v1_aligned.json")
    assert discover_ocr_slug(tmp_path) == "chandra-v1"


def test_discover_txt_requires_gemini_prefix(tmp_path):
    # Bare .txt names are ambiguous: surya output and scoping files must not
    # be mistaken for model slugs.
    _touch(tmp_path, "0001_id_surya.txt", "included_pages.txt", "0001_id_chandra-v1.txt")
    assert discover_ocr_slug(tmp_path) is None


def test_discover_from_gemini_txt(tmp_path):
    _touch(tmp_path, "0001_id_surya.txt", "0001_id_gemini-2.0-flash.txt")
    assert discover_ocr_slug(tmp_path) == "gemini-2.0-flash"


def test_discover_aligned_wins_over_txt(tmp_path):
    _touch(
        tmp_path,
        "0001_id_gemini-old_aligned.json",
        "0001_id_gemini-new.txt",
        "0002_id_gemini-new.txt",
    )
    assert discover_ocr_slug(tmp_path) == "gemini-old"


def test_discover_most_common_slug_wins(tmp_path):
    _touch(
        tmp_path,
        "0001_id_gemini-a_aligned.json",
        "0002_id_gemini-b_aligned.json",
        "0003_id_gemini-b_aligned.json",
    )
    assert discover_ocr_slug(tmp_path) == "gemini-b"


def test_discover_scans_immediate_subdirs(tmp_path):
    item = tmp_path / "item1"
    item.mkdir()
    _touch(item, "0001_id_gemini-sub_aligned.json")
    assert discover_ocr_slug(tmp_path) == "gemini-sub"
