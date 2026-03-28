"""Tests for the _canvas_fragment helper in pipeline/align_ocr.py.

_canvas_fragment builds IIIF xywh fragment strings and optionally scales
pixel-space coordinates to canvas space when the two differ.
"""

import sys
from pathlib import Path

import pytest

# align_ocr.py lives in pipeline/ and uses a sys.path.insert at the top of the
# module to locate utils/.  We replicate that so the import succeeds without
# installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.align_ocr import _canvas_fragment, _union_bbox, _y_center


CANVAS = "https://example.org/canvas/0"


# ---------------------------------------------------------------------------
# _canvas_fragment — basic fragment construction
# ---------------------------------------------------------------------------

class TestCanvasFragmentBasic:
    def test_format(self):
        fragment = _canvas_fragment(CANVAS, [10, 20, 110, 70], 1000, 2000, 1000, 2000)
        assert fragment == f"{CANVAS}#xywh=10,20,100,50"

    def test_width_and_height_derived_from_bbox(self):
        fragment = _canvas_fragment(CANVAS, [0, 0, 200, 300], 2000, 3000, 2000, 3000)
        assert "#xywh=0,0,200,300" in fragment

    def test_canvas_uri_preserved(self):
        fragment = _canvas_fragment(CANVAS, [5, 5, 15, 15], 100, 100, 100, 100)
        assert fragment.startswith(CANVAS)

    def test_hash_separator(self):
        fragment = _canvas_fragment(CANVAS, [0, 0, 50, 50], 500, 500, 500, 500)
        assert "#xywh=" in fragment

    def test_minimum_dimension_is_one(self):
        # x1 == x2 and y1 == y2 → w and h must be at least 1
        fragment = _canvas_fragment(CANVAS, [10, 20, 10, 20], 100, 100, 100, 100)
        assert "#xywh=10,20,1,1" in fragment


# ---------------------------------------------------------------------------
# _canvas_fragment — coordinate scaling
# ---------------------------------------------------------------------------

class TestCanvasFragmentScaling:
    def test_no_scaling_when_dimensions_match(self):
        # img and canvas dimensions identical → coords unchanged
        fragment = _canvas_fragment(CANVAS, [100, 200, 300, 400], 1000, 2000, 1000, 2000)
        assert "#xywh=100,200,200,200" in fragment

    def test_scales_up_uniformly(self):
        # Canvas is 2× the image in both axes
        fragment = _canvas_fragment(CANVAS, [50, 100, 150, 200], 500, 1000, 1000, 2000)
        # x1=100, y1=200, x2=300, y2=400 → w=200, h=200
        assert "#xywh=100,200,200,200" in fragment

    def test_scales_down_uniformly(self):
        # Canvas is half the image in both axes
        fragment = _canvas_fragment(CANVAS, [200, 400, 600, 800], 2000, 4000, 1000, 2000)
        # x1=100, y1=200, x2=300, y2=400 → w=200, h=200
        assert "#xywh=100,200,200,200" in fragment

    def test_non_uniform_scaling_x_axis(self):
        # Canvas width = 2× img width, height unchanged
        fragment = _canvas_fragment(CANVAS, [100, 50, 200, 100], 500, 500, 1000, 500)
        # sx=2, sy=1 → x1=200, y1=50, x2=400, y2=100 → w=200, h=50
        assert "#xywh=200,50,200,50" in fragment

    def test_no_scaling_when_canvas_width_zero(self):
        # canvas_w=0 means unknown — skip scaling
        fragment = _canvas_fragment(CANVAS, [10, 20, 110, 70], 1000, 2000, 0, 0)
        assert "#xywh=10,20,100,50" in fragment

    def test_split_page_right_offset(self):
        # Simulate a right-page split: image coords already offset by x_offset=1000
        # Canvas and image dimensions are the same (no scaling), so coords pass through
        fragment = _canvas_fragment(CANVAS, [1010, 50, 1200, 100], 2000, 2000, 2000, 2000)
        assert "#xywh=1010,50,190,50" in fragment


# ---------------------------------------------------------------------------
# _union_bbox
# ---------------------------------------------------------------------------

class TestUnionBbox:
    def test_single_bbox_returns_itself(self):
        assert _union_bbox([[10, 20, 30, 40]]) == [10, 20, 30, 40]

    def test_two_non_overlapping(self):
        assert _union_bbox([[0, 0, 10, 10], [20, 20, 30, 30]]) == [0, 0, 30, 30]

    def test_two_overlapping(self):
        assert _union_bbox([[0, 0, 20, 20], [10, 10, 30, 30]]) == [0, 0, 30, 30]

    def test_contained_bbox(self):
        assert _union_bbox([[0, 0, 100, 100], [10, 10, 20, 20]]) == [0, 0, 100, 100]

    def test_many_bboxes(self):
        bboxes = [[i, i, i + 5, i + 5] for i in range(10)]
        result = _union_bbox(bboxes)
        assert result == [0, 0, 14, 14]


# ---------------------------------------------------------------------------
# _y_center
# ---------------------------------------------------------------------------

class TestYCenter:
    def test_simple(self):
        assert _y_center([0, 10, 100, 30]) == 20.0

    def test_zero_height(self):
        assert _y_center([0, 5, 100, 5]) == 5.0

    def test_float_result(self):
        assert _y_center([0, 0, 100, 3]) == 1.5
