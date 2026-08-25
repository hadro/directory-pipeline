"""utils/column_utils.py — shared gutter detection and column thresholds."""

import pytest

from utils.column_utils import (
    MIN_COLUMN_COVERAGE,
    MIN_COLUMN_FRAC,
    column_breaks,
)

W = 2048


def _cols(n, rows=30, starts=(130, 760, 1380), width=430):
    """Synthetic n-column page of line bboxes."""
    return [
        {"bbox": [starts[c], 200 + r * 40, starts[c] + width, 230 + r * 40],
         "text": f"col{c}row{r}"}
        for c in range(n) for r in range(rows)
    ]


# ── column_breaks ─────────────────────────────────────────────────────────

def test_single_column_has_no_interior_gutter():
    assert column_breaks(_cols(1), W) == []


def test_two_columns_yield_one_gutter():
    breaks = column_breaks(_cols(2, starts=(130, 1100)), W)
    assert len(breaks) == 1
    assert 560 < breaks[0] < 1100


def test_three_columns_yield_two_gutters_left_to_right():
    breaks = column_breaks(_cols(3), W)
    assert len(breaks) == 2
    assert breaks == sorted(breaks)


def test_sparse_page_is_not_judged():
    """Under four lines covering the peak bin, there is nothing to conclude."""
    assert column_breaks(_cols(2, rows=1), W) == []


def test_margins_are_not_reported_as_gutters():
    """The blank page edges are valleys too, but they are not between columns."""
    breaks = column_breaks(_cols(2, starts=(400, 1200), width=300), W)
    assert all(100 < b < W - 100 for b in breaks)


def test_wrapped_lines_crossing_the_gutter_do_not_hide_it():
    """Coverage valleys survive sparse gutter-crossing lines; a consecutive-x1
    gap rule would not."""
    lines = _cols(2, starts=(130, 1100))
    for i in range(3):
        y = 300 + i * 200
        lines.append({"bbox": [130, y, 1530, y + 30], "text": "wrapped"})
    assert len(column_breaks(lines, W)) == 1


def test_empty_input_is_safe():
    assert column_breaks([], W) == []


@pytest.mark.parametrize("page_width", [0, -1])
def test_non_positive_page_width_does_not_raise(page_width):
    # bin_w floors at 16, so this degenerates rather than dividing by zero
    assert column_breaks(_cols(2), page_width) == []


# ── shared thresholds ─────────────────────────────────────────────────────

def test_thresholds_are_sane_fractions():
    assert 0 < MIN_COLUMN_FRAC < 1
    assert 0 < MIN_COLUMN_COVERAGE <= 1


def test_align_and_detect_read_the_same_thresholds():
    """The whole point of this module: the aligner and the column report must
    not be able to disagree about what counts as a column."""
    from pipeline import align_ocr, surya_detect
    assert align_ocr.MIN_COLUMN_FRAC is MIN_COLUMN_FRAC
    assert align_ocr.MIN_COLUMN_COVERAGE is MIN_COLUMN_COVERAGE
    assert surya_detect.MIN_COLUMN_FRAC is MIN_COLUMN_FRAC
    assert surya_detect.MIN_COLUMN_COVERAGE is MIN_COLUMN_COVERAGE


def test_both_stages_use_the_same_gutter_detector():
    from pipeline import align_ocr, surya_detect
    assert align_ocr.column_breaks is column_breaks
    assert surya_detect.column_breaks is column_breaks
