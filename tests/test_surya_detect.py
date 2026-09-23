"""pipeline/surya_detect.py — column-layout reporting.

The report exists to describe the layout the *aligner* will see, so these
guard both the reporting itself and its agreement with align_ocr.
"""

from pipeline.align_ocr import plan_columns
from pipeline.surya_detect import _analyze_bboxes

W = 2048


def _boxes(n_cols, rows=30, starts=(130, 760, 1380), width=430):
    return [
        [starts[c], 200 + r * 40, starts[c] + width, 230 + r * 40]
        for c in range(n_cols) for r in range(rows)
    ]


def test_no_bboxes_reports_low_confidence():
    out = _analyze_bboxes([], W)
    assert out["num_columns"] == 1
    assert out["confidence"] == "low"


def test_too_few_bboxes_to_judge_reports_single_column():
    assert _analyze_bboxes(_boxes(1, rows=3), W)["num_columns"] == 1


def test_single_column_page():
    out = _analyze_bboxes(_boxes(1), W)
    assert out["num_columns"] == 1
    assert out["gutter_x_positions"] == ""


def test_two_column_page():
    out = _analyze_bboxes(_boxes(2, starts=(130, 1100)), W)
    assert out["num_columns"] == 2
    assert len(out["gutter_x_positions"].split(";")) == 1


def test_three_column_page():
    out = _analyze_bboxes(_boxes(3), W)
    assert out["num_columns"] == 3
    assert len(out["gutter_x_positions"].split(";")) == 2


def test_stacked_blocks_are_not_columns():
    """Two blocks at different x positions that never share a row are stacked
    content, not columns — vertical overlap is what distinguishes them."""
    top    = [[130, 200 + r * 40, 560, 230 + r * 40] for r in range(15)]
    bottom = [[1100, 1400 + r * 40, 1530, 1430 + r * 40] for r in range(15)]
    assert _analyze_bboxes(top + bottom, W)["num_columns"] == 1


def test_report_agrees_with_the_aligner_on_column_count():
    """The invariant utils/column_utils exists to hold: the reported column
    count is the one plan_columns will actually align against."""
    for n in (2, 3):
        boxes = _boxes(n) if n == 3 else _boxes(2, starts=(130, 1100))
        reported = _analyze_bboxes(boxes, W)["num_columns"]
        lines = [{"bbox": list(b), "text": f"line{i}"} for i, b in enumerate(boxes)]
        plan = plan_columns(lines, W)
        assert plan is not None, f"aligner found no columns for a {n}-column page"
        assert len(plan[1]) == reported == n
