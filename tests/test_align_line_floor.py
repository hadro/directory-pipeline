"""filter_short_lines: short lines stay only when they are wide.

The cases are Tulsa 1921's: a page of ~19 px lines about 350 px wide, with
header pieces and wrapped tails Surya boxes at 10-14 px.
"""
from pipeline.align_ocr import filter_short_lines


def ln(x0, y0, w, h, text=""):
    return {"bbox": [x0, y0, x0 + w, y0 + h], "text": text}


BODY = [ln(50, 100 + 20 * i, 350, 19, f"body {i}") for i in range(20)]


def kept(lines):
    return {l["text"] for l in filter_short_lines(lines)}


def test_narrow_fragments_under_the_floor_go():
    frags = [ln(795, 169, 31, 13, "811"), ln(698, 168, 48, 14, "PHY"), ln(56, 410, 34, 12, "av.")]
    assert kept(BODY + frags) == {l["text"] for l in BODY}


def test_a_wide_short_line_stays():
    """A line without tall letters, or small type, is under 15 px but full width."""
    short = ln(50, 600, 300, 13, "mmm ewer sun")
    assert "mmm ewer sun" in kept(BODY + [short])


def test_dot_leaders_still_go_however_wide():
    leader = ln(50, 600, 700, 5, ". . . . . .")
    assert ". . . . . ." not in kept(BODY + [leader])


def test_tall_lines_are_never_dropped_for_width():
    stamp = ln(900, 50, 20, 40, "3")
    assert "3" in kept(BODY + [stamp])


def test_explicit_min_height_filters_on_height_alone():
    lines = BODY + [ln(795, 169, 31, 13, "811")]
    assert len(filter_short_lines(lines, min_height=13)) == len(lines)
    assert len(filter_short_lines(lines, min_height=19)) == len(BODY)


def test_small_type_page_keeps_its_lines():
    """polk1925 through Indra: the whole page is 13 px type."""
    page = [ln(50, 100 + 15 * i, 300, 13, f"l{i}") for i in range(30)]
    assert len(filter_short_lines(page)) == 30
