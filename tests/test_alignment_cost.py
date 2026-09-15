"""utils/column_utils.py + pipeline/align_ocr.py — the alignment selection criterion.

`_alignment_cost` decides between the per-column and whole-page candidates at
the end of `build_aligned_lines`. It was arrived at by measurement (see
docs/evidence/2026-09-04-column-selection-criterion/), and the measurement that
mattered was that the obvious rule — pick whichever boxes more text — chose the
geometrically worse candidate on a sixth of multi-column pages. These tests pin
the properties that conclusion rests on, so a future simplification back toward
a coverage-only rule fails here rather than silently in the corpus.
"""
import pytest

from pipeline.align_ocr import _alignment_cost
from utils.column_utils import assign_columns, bbox_centre, inversion_rate


def _line(x, y, text="t"):
    """An aligned line whose box is a 100x20 cell at (x, y)."""
    return {"bbox": [x, y, x + 100, y + 20], "gemini_text": text}


def _column_major(n_cols=2, rows=10, x_step=500):
    """Lines already in correct column-major reading order."""
    return [_line(c * x_step, r * 30)
            for c in range(n_cols) for r in range(rows)]


# ── bbox_centre ───────────────────────────────────────────────────────────

def test_bbox_centre_is_the_midpoint():
    assert bbox_centre([0, 0, 100, 20]) == (50.0, 10.0)


# ── assign_columns ────────────────────────────────────────────────────────

def test_assign_columns_splits_on_the_wide_x_gap():
    pts = [(100, y) for y in range(10)] + [(900, y) for y in range(10)]
    cols = assign_columns(pts)
    assert cols[:10] == [0] * 10
    assert cols[10:] == [1] * 10


def test_assign_columns_keeps_one_column_when_no_gap_is_wide_enough():
    """Evenly spread x-centres are one column, not many narrow ones."""
    pts = [(x, 0) for x in range(0, 1000, 50)]
    assert set(assign_columns(pts)) == {0}


def test_assign_columns_caps_at_n_cols_max():
    """Five visually distinct columns collapse to four at the default cap.

    This is the documented ceiling, and the reason the criterion is weaker on
    dense city-directory pages than on travel guides. Pinned so that raising
    the cap is a deliberate edit with a test to update, not a silent drift.
    """
    pts = [(c * 400, r) for c in range(5) for r in range(5)]
    assert len(set(assign_columns(pts))) == 4
    assert len(set(assign_columns(pts, n_cols_max=5))) == 5


# ── inversion_rate ────────────────────────────────────────────────────────

def test_perfect_column_major_order_has_no_inversions():
    assert inversion_rate([bbox_centre(ln["bbox"]) for ln in _column_major()]) == 0.0


def test_fully_reversed_order_is_near_one():
    pts = [bbox_centre(ln["bbox"]) for ln in _column_major()][::-1]
    assert inversion_rate(pts) > 0.9


def test_reading_across_the_gutter_scores_worse_than_column_major():
    """The failure per-column alignment exists to prevent, in miniature.

    Whole-page NW on a two-column page pairs left-row-0 with right-row-0, so
    the aligned list alternates between columns.

    Perfect interleaving of two n-row columns puts every later left-column line
    behind its right-column partner: 45 inversions of 190 pairs at n=10, a rate
    of 0.237. That is the separation the criterion runs on — not a large
    absolute number, which is why the whole-page candidate has to be *much*
    more disordered before its better coverage is overridden.
    """
    rows = 10
    interleaved = [_line(c * 500, r * 30) for r in range(rows) for c in range(2)]
    ordered = _column_major(rows=rows)
    inv_bad = inversion_rate([bbox_centre(ln["bbox"]) for ln in interleaved])
    inv_good = inversion_rate([bbox_centre(ln["bbox"]) for ln in ordered])
    assert inv_good == 0.0
    assert inv_bad == pytest.approx(45 / 190, abs=1e-6)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_inversion_rate_is_zero_below_three_points(n):
    """Too few boxes to say anything about order — must not divide by zero."""
    assert inversion_rate([(0.0, float(i)) for i in range(n)]) == 0.0


# ── _alignment_cost ───────────────────────────────────────────────────────

def test_cost_is_unmatched_count_when_order_is_perfect():
    lines = _column_major()
    assert _alignment_cost(lines, ["a", "b", "c"]) == pytest.approx(3.0)


def test_disorder_is_penalised_even_with_nothing_unmatched():
    interleaved = [_line(c * 500, r * 30) for r in range(10) for c in range(2)]
    assert _alignment_cost(interleaved, []) > 0.0


def test_a_more_disordered_candidate_loses_at_equal_coverage():
    """The whole point: coverage ties are broken by geometry, not arbitrarily."""
    ordered = _column_major()
    interleaved = [_line(c * 500, r * 30) for r in range(10) for c in range(2)]
    assert _alignment_cost(ordered, []) < _alignment_cost(interleaved, [])


def test_better_coverage_does_not_win_when_it_is_far_more_disordered():
    """The measured regression, as a unit test.

    A whole-page alignment that boxes *every* line but reads across the gutter
    must lose to a column alignment that leaves a few lines unboxed. A
    coverage-only rule picks the first; on 250 Green Book pages that rule chose
    the geometrically worse candidate 34 times out of 42.
    """
    whole = [_line(c * 500, r * 30) for r in range(20) for c in range(2)]
    column = _column_major(rows=18)              # 4 fewer lines placed
    assert _alignment_cost(column, ["u1", "u2", "u3", "u4"]) < _alignment_cost(whole, [])


def test_disorder_scales_with_page_size():
    """A bare rate would let a short page's disorder count as much as a long
    page's. Scaling by line count keeps the two terms commensurable."""
    short = [_line(c * 500, r * 30) for r in range(5) for c in range(2)]
    long_ = [_line(c * 500, r * 30) for r in range(50) for c in range(2)]
    assert _alignment_cost(long_, []) > _alignment_cost(short, [])


def test_lines_without_a_box_do_not_break_the_cost():
    lines = _column_major(rows=5) + [{"gemini_text": "unplaced"}]
    assert _alignment_cost(lines, []) >= 0.0


def test_empty_candidate_costs_only_its_unmatched_text():
    assert _alignment_cost([], ["a", "b"]) == pytest.approx(2.0)
