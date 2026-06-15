"""Tests for the pure helpers in pipeline/run_linecrop_ocr.py.

The transcription path is API-driven (not unit-tested), but the crop geometry
and the genuine-merge page selection are deterministic and worth guarding.
"""

import json

from PIL import Image

from pipeline.run_linecrop_ocr import _crop_to_jpeg, _page_genuine_merge_count


# ---------------------------------------------------------------------------
# _crop_to_jpeg — padding, clamping, degenerate rejection
# ---------------------------------------------------------------------------

class TestCropToJpeg:
    def _page(self):
        return Image.new("RGB", (100, 50), "white")

    def test_valid_crop_returns_jpeg_bytes(self):
        out = _crop_to_jpeg(self._page(), [10, 10, 90, 40])
        assert out is not None
        assert out[:2] == b"\xff\xd8"  # JPEG SOI marker

    def test_degenerate_crop_returns_none(self):
        # Zero-size bbox at the image corner stays below the min-crop floor.
        assert _crop_to_jpeg(self._page(), [0, 0, 0, 0]) is None

    def test_out_of_bounds_bbox_is_clamped(self):
        # bbox extends past the image; clamping must keep it valid, not crash.
        out = _crop_to_jpeg(self._page(), [90, 40, 200, 100])
        assert out is not None and out[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# _page_genuine_merge_count — default selection signal
# ---------------------------------------------------------------------------

class TestGenuineMergeCount:
    def _write(self, tmp_path, texts):
        p = tmp_path / "0001_x_gemini_aligned.json"
        p.write_text(json.dumps({"lines": [{"gemini_text": t} for t in texts]}))
        return p

    def test_counts_two_address_lines(self, tmp_path):
        texts = [
            "Royal—1073 Fulton St. Carver—980 Prospect Ave.",   # merged
            "Parkside 759 Gates Ave. Crotona—695 East 170th St.",  # merged
            "Comfort—201 Bay Ave. Virginia Inn—1505 S. New Road",  # merged
            "Carver—980 Prospect Ave.",   # clean
            "Crotona—695 East 170th St.",  # clean
            "CLUBS AND TAVERNS",           # heading
            "Disler's—759 Gates Ave.",     # clean
            "George's—328 Tompkins Ave.",  # clean
        ]
        assert _page_genuine_merge_count(self._write(tmp_path, texts)) == 3

    def test_missing_file_is_zero(self, tmp_path):
        assert _page_genuine_merge_count(tmp_path / "nope.json") == 0

    def test_short_page_is_zero(self, tmp_path):
        # Fewer than 8 lines → not enough to judge; returns 0.
        assert _page_genuine_merge_count(self._write(tmp_path, ["a—1 X St. b—2 Y St."])) == 0
