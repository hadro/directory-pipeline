"""Tests for pipeline/export_alto.py — aligned JSON → ALTO v3 serialization.

Covers coordinate source selection (canvas_fragment vs. bbox fallback), page
dimension resolution, proportional word-box splitting, the --line-strings mode,
and overall ALTO v3 structure.
"""

import xml.etree.ElementTree as ET

from pipeline.export_alto import (
    ALTO_NS,
    _line_xywh,
    _page_dimensions,
    _split_words,
    build_alto,
)


def _q(tag: str) -> str:
    return f"{{{ALTO_NS}}}{tag}"


def _aligned(lines, **top):
    base = {"image": "0001_x.jpg", "model": "gemini-x",
            "canvas_width": 1000, "canvas_height": 2000}
    base.update(top)
    base["lines"] = lines
    return base


# ---------------------------------------------------------------------------
# _line_xywh — coordinate source
# ---------------------------------------------------------------------------

class TestLineXYWH:
    def test_prefers_canvas_fragment(self):
        line = {"canvas_fragment": "https://h/c#xywh=10,20,100,30",
                "bbox": [1, 2, 3, 4]}
        assert _line_xywh(line) == (10, 20, 100, 30)

    def test_falls_back_to_bbox(self):
        line = {"canvas_fragment": "", "bbox": [10, 20, 110, 50]}
        assert _line_xywh(line) == (10, 20, 100, 30)

    def test_none_when_no_coords(self):
        assert _line_xywh({"canvas_fragment": "", "bbox": []}) is None


# ---------------------------------------------------------------------------
# _page_dimensions
# ---------------------------------------------------------------------------

class TestPageDimensions:
    def test_uses_canvas_size_when_present(self):
        assert _page_dimensions({"canvas_width": 1000, "canvas_height": 2000}, []) == (1000, 2000)

    def test_falls_back_to_max_extent(self):
        lines = [{"canvas_fragment": "", "bbox": [0, 0, 100, 50]},
                 {"canvas_fragment": "", "bbox": [0, 0, 60, 300]}]
        assert _page_dimensions({}, lines) == (100, 300)


# ---------------------------------------------------------------------------
# _split_words — proportional character-width allocation
# ---------------------------------------------------------------------------

class TestSplitWords:
    def test_word_count_matches(self):
        words = _split_words(0, 0, 100, 10, "alpha beta gamma")
        assert [w[0] for w in words] == ["alpha", "beta", "gamma"]

    def test_left_to_right_monotonic(self):
        words = _split_words(0, 0, 300, 10, "one two three four")
        xs = [wx for _, wx, _, _, _ in words]
        assert xs == sorted(xs)
        assert xs[0] == 0  # first word starts at the line's x

    def test_inherits_line_y_and_height(self):
        words = _split_words(5, 42, 100, 17, "a b c")
        assert all(wy == 42 and wh == 17 for _, _, wy, _, wh in words)

    def test_min_width_one(self):
        # An empty "word" can't happen via split(), but widths never go below 1.
        words = _split_words(0, 0, 1, 10, "aaaaaaaaaa bbbbbbbbbb")
        assert all(ww >= 1 for _, _, _, ww, _ in words)

    def test_empty_text_yields_nothing(self):
        assert _split_words(0, 0, 100, 10, "   ") == []


# ---------------------------------------------------------------------------
# build_alto — overall structure
# ---------------------------------------------------------------------------

class TestBuildAlto:
    def test_root_and_page_dims(self):
        data = _aligned([{"canvas_fragment": "u#xywh=10,20,100,30", "gemini_text": "Hello world"}])
        root = build_alto(data).getroot()
        assert root.tag == _q("alto")
        page = root.find(f".//{_q('Page')}")
        assert page.get("WIDTH") == "1000" and page.get("HEIGHT") == "2000"
        assert root.find(f".//{_q('MeasurementUnit')}").text == "pixel"
        assert root.find(f".//{_q('fileName')}").text == "0001_x.jpg"

    def test_multiword_line_splits_into_strings_with_sp(self):
        data = _aligned([{"canvas_fragment": "u#xywh=0,0,200,20", "gemini_text": "Foo Bar Baz"}])
        root = build_alto(data).getroot()
        tl = root.find(f".//{_q('TextLine')}")
        strings = tl.findall(_q("String"))
        sps = tl.findall(_q("SP"))
        assert [s.get("CONTENT") for s in strings] == ["Foo", "Bar", "Baz"]
        assert len(sps) == 2  # one between each adjacent pair
        assert tl.get("HPOS") == "0" and tl.get("WIDTH") == "200"

    def test_single_word_line_is_one_string(self):
        data = _aligned([{"canvas_fragment": "u#xywh=0,0,50,20", "gemini_text": "Solo"}])
        root = build_alto(data).getroot()
        strings = root.findall(f".//{_q('String')}")
        assert len(strings) == 1 and strings[0].get("CONTENT") == "Solo"

    def test_line_strings_mode_one_string_per_line(self):
        data = _aligned([{"canvas_fragment": "u#xywh=0,0,200,20", "gemini_text": "Two Words"}])
        root = build_alto(data, line_strings=True).getroot()
        strings = root.findall(f".//{_q('String')}")
        assert len(strings) == 1 and strings[0].get("CONTENT") == "Two Words"

    def test_lines_without_coords_are_dropped(self):
        data = _aligned([
            {"canvas_fragment": "u#xywh=0,0,100,20", "gemini_text": "kept"},
            {"canvas_fragment": "", "bbox": [], "gemini_text": "dropped"},
        ])
        root = build_alto(data).getroot()
        assert len(root.findall(f".//{_q('TextLine')}")) == 1

    def test_output_is_serializable_and_wellformed(self):
        data = _aligned([{"canvas_fragment": "u#xywh=1,2,3,4", "gemini_text": "x y"}])
        xml = ET.tostring(build_alto(data).getroot(), encoding="unicode")
        # round-trips through the parser without error
        ET.fromstring(xml)
        assert ALTO_NS in xml
