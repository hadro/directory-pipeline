"""pipeline/align_ocr.py parse_surya — Surya JSON parsing and confidence handling."""

import json

from pipeline.align_ocr import parse_surya


def _write(tmp_path, payload):
    p = tmp_path / "0001_id_surya.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_page_bbox_from_image_dims(tmp_path):
    p = _write(tmp_path, {"image_width": 100, "image_height": 200, "lines": []})
    page_bbox, lines, median = parse_surya(p)
    assert page_bbox == (0, 0, 100, 200)
    assert lines == []
    assert median is None


def test_low_confidence_lines_filtered_but_count_toward_median(tmp_path):
    p = _write(tmp_path, {
        "image_width": 10, "image_height": 10,
        "lines": [
            {"bbox": [0, 0, 5, 1], "text": "keep", "confidence": 0.9},
            {"bbox": [0, 2, 5, 3], "text": "drop", "confidence": 0.1},
            {"bbox": [0, 4, 5, 5], "text": "keep2", "confidence": 0.8},
        ],
    })
    _, lines, median = parse_surya(p, min_confidence=0.35)
    assert [ln["text"] for ln in lines] == ["keep", "keep2"]
    assert median == 0.8  # median of ALL three (0.1, 0.8, 0.9), pre-filter


def test_median_even_count_averages_middle_two(tmp_path):
    p = _write(tmp_path, {
        "image_width": 10, "image_height": 10,
        "lines": [
            {"bbox": [0, 0, 1, 1], "text": "a", "confidence": 0.4},
            {"bbox": [0, 0, 1, 1], "text": "b", "confidence": 0.6},
        ],
    })
    _, _, median = parse_surya(p)
    assert median == 0.5


def test_lines_without_bbox_are_dropped(tmp_path):
    p = _write(tmp_path, {
        "image_width": 10, "image_height": 10,
        "lines": [{"text": "no bbox", "confidence": 0.9}],
    })
    _, lines, median = parse_surya(p)
    assert lines == []
    assert median is None  # bbox-less lines don't count toward the median either


def test_missing_confidence_defaults_to_kept(tmp_path):
    p = _write(tmp_path, {
        "image_width": 10, "image_height": 10,
        "lines": [{"bbox": [0, 0, 1, 1], "text": "x"}],
    })
    _, lines, median = parse_surya(p, min_confidence=0.35)
    assert len(lines) == 1
    assert lines[0]["surya_confidence"] == 1.0
    assert median is None  # no explicit confidences on the page
