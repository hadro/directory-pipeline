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


# --- require_txt: the aligner's input is the OCR text, not the aligned JSON ---
#
# Without this, a directory holding alignments from a model whose .txt files are
# absent resolves to that model, and align_ocr skips every page as "missing" —
# silently under --quiet. It cost a full re-align of 1,448 files, including the
# volume holding most of this project's hand-corrected alignments.

def _mk(d, name):
    (d / name).write_text("x", encoding="utf-8")


def test_require_txt_skips_a_model_with_no_ocr_text(tmp_path):
    """The exact production case: old alignments outnumber the usable model."""
    for i in range(3):
        _mk(tmp_path, f"000{i}_id_gemini-3-flash-preview_aligned.json")
        _mk(tmp_path, f"000{i}_id_gemini-2.0-flash_aligned.json")
        _mk(tmp_path, f"000{i}_id_gemini-2.0-flash.txt")

    assert discover_ocr_slug(tmp_path, require_txt=True) == "gemini-2.0-flash"


def test_default_behaviour_is_unchanged(tmp_path):
    """Callers that read aligned JSON (export_alto) must keep the old result."""
    for i in range(3):
        _mk(tmp_path, f"000{i}_id_gemini-3-flash-preview_aligned.json")
        _mk(tmp_path, f"000{i}_id_gemini-2.0-flash.txt")

    assert discover_ocr_slug(tmp_path) == "gemini-3-flash-preview"


def test_require_txt_still_prefers_aligned_among_usable_models(tmp_path):
    """It narrows the candidates; it does not invert the aligned-first order."""
    for i in range(3):
        _mk(tmp_path, f"000{i}_id_gemini-a.txt")
        _mk(tmp_path, f"000{i}_id_gemini-b.txt")
        _mk(tmp_path, f"000{i}_id_gemini-b_aligned.json")

    assert discover_ocr_slug(tmp_path, require_txt=True) == "gemini-b"


def test_require_txt_returns_none_when_no_model_has_text(tmp_path):
    for i in range(3):
        _mk(tmp_path, f"000{i}_id_gemini-3-flash-preview_aligned.json")

    assert discover_ocr_slug(tmp_path, require_txt=True) is None


def test_require_txt_searches_subdirectories(tmp_path):
    sub = tmp_path / "item"
    sub.mkdir()
    for i in range(2):
        _mk(sub, f"000{i}_id_gemini-3-flash-preview_aligned.json")
        _mk(sub, f"000{i}_id_gemini-2.0-flash.txt")

    assert discover_ocr_slug(tmp_path, require_txt=True) == "gemini-2.0-flash"


def test_discover_stops_one_level_down_by_default(tmp_path):
    deep = tmp_path / "collection" / "volume" / "item"
    deep.mkdir(parents=True)
    _mk(deep, "0001_id_gemini-2.0-flash_aligned.json")

    assert discover_ocr_slug(tmp_path) is None


def test_recursive_discovery_reaches_a_collection_root(tmp_path):
    """The repair tools are pointed at roots like green_books_and_related/, where
    aligned files sit three levels down and the old era is still the majority."""
    old = tmp_path / "the_green_book" / "item_a"
    new = tmp_path / "travelguide" / "item_b"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    for i in range(3):
        _mk(old, f"000{i}_id_gemini-2.0-flash_aligned.json")
    _mk(new, "0001_id_gemini-3.1-flash-lite_aligned.json")

    assert discover_ocr_slug(tmp_path, recursive=True) == "gemini-2.0-flash"
