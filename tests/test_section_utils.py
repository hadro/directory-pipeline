"""utils/section_utils.py — sections.txt parsing and per-section prompt routing.

These are the consumer half of section handling: five call sites across
extract_entries, generate_prompt, run_gemini_ocr and select_pages depend on
them to route the right prompt at the right page and to reset carried heading
context at a section boundary.
"""

import pytest

from utils.section_utils import (
    is_section_boundary,
    load_sections,
    prompt_for_page,
    section_for_page,
)

FILES = [f"{i:04d}_img.jpg" for i in range(10)]


def _sections_file(tmp_path, body):
    p = tmp_path / "sections.txt"
    p.write_text(body, encoding="utf-8")
    return p


# ── load_sections ─────────────────────────────────────────────────────────

def test_parses_labels_and_derives_page_ranges(tmp_path):
    p = _sections_file(tmp_path, "0000_img.jpg alphabetical\n0004_img.jpg street\n")
    secs = load_sections(p, FILES)
    assert [s["label"] for s in secs] == ["alphabetical", "street"]
    assert secs[0]["start_idx"] == 0 and secs[0]["end_idx"] == 3
    # the final section runs to the end of the volume
    assert secs[1]["start_idx"] == 4 and secs[1]["end_idx"] == 9
    assert secs[0]["page_indices"] == [0, 1, 2, 3]


def test_comments_and_blank_lines_ignored(tmp_path):
    p = _sections_file(tmp_path, "# leading comment\n\n0002_img.jpg street\n\n# trailing\n")
    assert [s["label"] for s in load_sections(p, FILES)] == ["street"]


def test_entries_are_sorted_by_position_not_file_order(tmp_path):
    p = _sections_file(tmp_path, "0006_img.jpg business\n0001_img.jpg alphabetical\n")
    secs = load_sections(p, FILES)
    assert [s["label"] for s in secs] == ["alphabetical", "business"]
    assert secs[0]["end_idx"] == 5


def test_label_may_contain_spaces(tmp_path):
    p = _sections_file(tmp_path, "0000_img.jpg street and avenue\n")
    assert load_sections(p, FILES)[0]["label"] == "street and avenue"


def test_line_without_a_label_is_skipped(tmp_path):
    p = _sections_file(tmp_path, "0000_img.jpg\n0003_img.jpg street\n")
    assert [s["label"] for s in load_sections(p, FILES)] == ["street"]


def test_empty_file_yields_no_sections(tmp_path):
    assert load_sections(_sections_file(tmp_path, "# only a comment\n"), FILES) == []


def test_matches_by_basename_when_paths_differ(tmp_path):
    p = _sections_file(tmp_path, "some/other/dir/0003_img.jpg street\n")
    assert load_sections(p, FILES)[0]["start_idx"] == 3


def test_unknown_filename_raises_rather_than_silently_misrouting(tmp_path):
    p = _sections_file(tmp_path, "9999_missing.jpg street\n")
    with pytest.raises(ValueError, match="not found in the image list"):
        load_sections(p, FILES)


# ── section_for_page ──────────────────────────────────────────────────────

def test_page_resolves_to_its_containing_section(tmp_path):
    secs = load_sections(_sections_file(
        tmp_path, "0000_img.jpg alphabetical\n0005_img.jpg street\n"), FILES)
    assert section_for_page("0004_img.jpg", secs, FILES) == "alphabetical"
    assert section_for_page("0005_img.jpg", secs, FILES) == "street"
    assert section_for_page("0009_img.jpg", secs, FILES) == "street"


def test_page_before_the_first_section_has_no_label(tmp_path):
    """Front matter ahead of the first marked section is unsectioned."""
    secs = load_sections(_sections_file(tmp_path, "0005_img.jpg street\n"), FILES)
    assert section_for_page("0001_img.jpg", secs, FILES) is None


def test_no_sections_means_no_label():
    assert section_for_page("0001_img.jpg", [], FILES) is None


def test_unknown_page_has_no_label(tmp_path):
    secs = load_sections(_sections_file(tmp_path, "0000_img.jpg street\n"), FILES)
    assert section_for_page("nope.jpg", secs, FILES) is None


# ── prompt_for_page ───────────────────────────────────────────────────────

def test_section_specific_prompt_wins_when_present(tmp_path):
    secs = load_sections(_sections_file(tmp_path, "0000_img.jpg street\n"), FILES)
    (tmp_path / "ner_prompt.md").write_text("generic", encoding="utf-8")
    (tmp_path / "ner_prompt_street.md").write_text("street", encoding="utf-8")
    got = prompt_for_page("0002_img.jpg", secs, FILES, tmp_path, "ner_prompt")
    assert got.name == "ner_prompt_street.md"


def test_falls_back_to_the_generic_prompt(tmp_path):
    secs = load_sections(_sections_file(tmp_path, "0000_img.jpg street\n"), FILES)
    (tmp_path / "ner_prompt.md").write_text("generic", encoding="utf-8")
    got = prompt_for_page("0002_img.jpg", secs, FILES, tmp_path, "ner_prompt")
    assert got.name == "ner_prompt.md"


def test_unsectioned_page_uses_the_generic_prompt(tmp_path):
    secs = load_sections(_sections_file(tmp_path, "0005_img.jpg street\n"), FILES)
    (tmp_path / "ocr_prompt.md").write_text("generic", encoding="utf-8")
    (tmp_path / "ocr_prompt_street.md").write_text("street", encoding="utf-8")
    got = prompt_for_page("0001_img.jpg", secs, FILES, tmp_path, "ocr_prompt")
    assert got.name == "ocr_prompt.md"


def test_returns_the_generic_path_even_when_nothing_exists(tmp_path):
    """Callers check .exists() themselves and fall back to their own default."""
    got = prompt_for_page("0001_img.jpg", [], FILES, tmp_path, "ner_prompt")
    assert got == tmp_path / "ner_prompt.md"
    assert not got.exists()


# ── is_section_boundary ───────────────────────────────────────────────────

def test_first_page_of_a_later_section_is_a_boundary(tmp_path):
    secs = load_sections(_sections_file(
        tmp_path, "0000_img.jpg alphabetical\n0004_img.jpg street\n"), FILES)
    assert is_section_boundary("0004_img.jpg", secs, FILES) is True


def test_first_section_start_is_not_a_boundary(tmp_path):
    """Nothing precedes it, so there is no carried context to reset."""
    secs = load_sections(_sections_file(
        tmp_path, "0000_img.jpg alphabetical\n0004_img.jpg street\n"), FILES)
    assert is_section_boundary("0000_img.jpg", secs, FILES) is False


def test_interior_pages_are_not_boundaries(tmp_path):
    secs = load_sections(_sections_file(
        tmp_path, "0000_img.jpg alphabetical\n0004_img.jpg street\n"), FILES)
    assert is_section_boundary("0005_img.jpg", secs, FILES) is False


def test_no_sections_means_no_boundaries():
    assert is_section_boundary("0004_img.jpg", [], FILES) is False
