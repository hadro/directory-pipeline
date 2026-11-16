"""pipeline/extract_entries.py _detect_aligned_slug — choosing between models.

Where a directory holds alignments from several models, this used to return the
first hit from a sorted glob — a plain alphabetical sort over filenames, so
'gemini-2.0-flash' beat 'gemini-3-flash-preview' on every page that had both.
The worse alignment was picked deterministically rather than occasionally, and
because nothing records which alignment produced a given canvas_fragment, the
wrong choice is invisible downstream.
"""

import json

from pipeline.extract_entries import _detect_aligned_slug


def _page(item_dir, stem, slug, n_lines, n_unmatched=0):
    (item_dir / f"{stem}_{slug}_aligned.json").write_text(
        json.dumps({
            "lines": [{"gemini_text": f"line {i}"} for i in range(n_lines)],
            "unmatched_gemini": [f"missed {i}" for i in range(n_unmatched)],
        }),
        encoding="utf-8",
    )


def test_single_model_is_returned(tmp_path):
    _page(tmp_path, "0001_id", "gemini-2.0-flash", 10)
    assert _detect_aligned_slug(tmp_path) == "gemini-2.0-flash"


def test_better_alignment_wins_over_alphabetically_earlier_one(tmp_path):
    """The regression: a healthier alignment that sorts later must still win."""
    _page(tmp_path, "0001_id", "gemini-2.0-flash", 10, n_unmatched=40)
    _page(tmp_path, "0001_id", "gemini-3-flash-preview", 40, n_unmatched=2)

    assert _detect_aligned_slug(tmp_path) == "gemini-3-flash-preview"


def test_line_count_alone_does_not_decide(tmp_path):
    """More lines but badly boxed loses to fewer lines cleanly boxed.

    Modelled on the real case: 3,967 lines at 53% boxed vs 2,889 at 87%.
    """
    _page(tmp_path, "0001_id", "gemini-2.0-flash", 3967, n_unmatched=3439)
    _page(tmp_path, "0001_id", "gemini-3-flash-preview", 2889, n_unmatched=418)

    assert _detect_aligned_slug(tmp_path) == "gemini-3-flash-preview"


def test_boxed_fraction_alone_does_not_decide(tmp_path):
    """A model that emitted almost no text must not win on a perfect ratio."""
    _page(tmp_path, "0001_id", "gemini-2.0-flash", 500, n_unmatched=50)
    _page(tmp_path, "0001_id", "gemini-3-flash-preview", 3, n_unmatched=0)

    assert _detect_aligned_slug(tmp_path) == "gemini-2.0-flash"


def test_selection_is_stable_regardless_of_filesystem_order(tmp_path):
    """Equal-quality models resolve deterministically, not by glob order."""
    _page(tmp_path, "0001_id", "gemini-2.0-flash", 20, n_unmatched=5)
    _page(tmp_path, "0001_id", "gemini-3-flash-preview", 20, n_unmatched=5)

    assert _detect_aligned_slug(tmp_path) == _detect_aligned_slug(tmp_path)


def test_falls_back_to_ocr_text_when_no_alignment_exists(tmp_path):
    (tmp_path / "0001_id_gemini-2.0-flash.txt").write_text("some text", encoding="utf-8")
    assert _detect_aligned_slug(tmp_path) == "gemini-2.0-flash"


def test_empty_directory_returns_none(tmp_path):
    assert _detect_aligned_slug(tmp_path) is None


def test_unreadable_alignment_does_not_crash_selection(tmp_path):
    (tmp_path / "0001_id_gemini-2.0-flash_aligned.json").write_text("{ bad", encoding="utf-8")
    _page(tmp_path, "0002_id", "gemini-3-flash-preview", 30)

    assert _detect_aligned_slug(tmp_path) == "gemini-3-flash-preview"
