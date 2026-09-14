"""pipeline/align_ocr.py _preserve_manual_lines — hand-made alignments survive --force.

Manual lines are the only artifact in the pipeline that cannot be regenerated
from the page image. ``align_image`` rebuilds a page from scratch and overwrites
the existing JSON, so without preservation a single ``--force`` re-align erases
every correction made through ``--review-alignment``.

The bug this guards against is invisible in normal operation: the non-force path
returns "skipped" before reaching the write, so nothing exercises the destructive
branch until someone force-aligns hand-corrected data once.
"""

import json

from pipeline.align_ocr import _preserve_manual_lines


def _existing(tmp_path, lines):
    p = tmp_path / "0001_id_gemini-2.0-flash_aligned.json"
    p.write_text(json.dumps({"lines": lines, "unmatched_gemini": []}), encoding="utf-8")
    return p


def _manual(bbox, text):
    return {
        "bbox": bbox,
        "canvas_fragment": f"https://example.org/c1#xywh={bbox[0]},{bbox[1]},10,10",
        "confidence": "manual",
        "gemini_text": text,
    }


def _machine(bbox, text):
    return {
        "bbox": bbox,
        "canvas_fragment": f"https://example.org/c1#xywh={bbox[0]},{bbox[1]},10,10",
        "confidence": "line",
        "gemini_text": text,
    }


def test_manual_line_survives_realignment(tmp_path):
    """A hand-placed line absent from the new alignment is carried forward."""
    p = _existing(tmp_path, [_manual([0, 100, 50, 120], "Rev. James Coleman's Lodge")])
    fresh = [_machine([0, 200, 50, 220], "113 S. Gila Street")]

    lines, unmatched, n = _preserve_manual_lines(p, fresh, [])

    assert n == 1
    texts = [ln["gemini_text"] for ln in lines]
    assert "Rev. James Coleman's Lodge" in texts
    assert "113 S. Gila Street" in texts


def test_manual_text_removed_from_unmatched(tmp_path):
    """A preserved line must not also be reported as unaligned text."""
    p = _existing(tmp_path, [_manual([0, 100, 50, 120], "YUMA")])

    lines, unmatched, n = _preserve_manual_lines(p, [], ["YUMA", "ARKANSAS"])

    assert n == 1
    assert unmatched == ["ARKANSAS"]


def test_manual_wins_over_machine_line_on_same_bbox(tmp_path):
    """Where both claim one Surya line, the human's reading is kept."""
    bbox = [0, 100, 50, 120]
    p = _existing(tmp_path, [_manual(bbox, "Hotel Robert Treat")])
    fresh = [_machine(bbox, "Hotels and Motels")]

    lines, _, n = _preserve_manual_lines(p, fresh, [])

    assert n == 1
    assert len(lines) == 1
    assert lines[0]["gemini_text"] == "Hotel Robert Treat"
    assert lines[0]["confidence"] == "manual"


def test_same_text_at_different_bbox_is_not_duplicated(tmp_path):
    """The machine re-boxing text a human already placed must not double it."""
    p = _existing(tmp_path, [_manual([0, 100, 50, 120], "TAVERNS")])
    fresh = [_machine([0, 400, 50, 420], "TAVERNS")]

    lines, _, n = _preserve_manual_lines(p, fresh, [])

    assert n == 1
    assert [ln["gemini_text"] for ln in lines] == ["TAVERNS"]
    assert lines[0]["bbox"] == [0, 100, 50, 120]


def test_lines_come_back_in_reading_order(tmp_path):
    """Preserved lines are merged into position, not appended at the end."""
    p = _existing(tmp_path, [_manual([0, 150, 50, 170], "middle")])
    fresh = [_machine([0, 50, 50, 70], "top"), _machine([0, 300, 50, 320], "bottom")]

    lines, _, _ = _preserve_manual_lines(p, fresh, [])

    assert [ln["gemini_text"] for ln in lines] == ["top", "middle", "bottom"]


def test_page_with_no_manual_work_is_untouched(tmp_path):
    """The common case must pass through byte-for-byte, including order."""
    p = _existing(tmp_path, [_machine([0, 100, 50, 120], "old machine text")])
    fresh = [_machine([0, 300, 50, 320], "b"), _machine([0, 50, 50, 70], "a")]

    lines, unmatched, n = _preserve_manual_lines(p, fresh, ["x"])

    assert n == 0
    assert lines is fresh  # not re-sorted, not copied
    assert unmatched == ["x"]


def test_first_alignment_of_a_page_is_untouched(tmp_path):
    """No existing file means nothing to preserve."""
    p = tmp_path / "missing_aligned.json"
    fresh = [_machine([0, 50, 50, 70], "a")]

    lines, unmatched, n = _preserve_manual_lines(p, fresh, ["x"])

    assert (n, lines, unmatched) == (0, fresh, ["x"])


def test_unreadable_existing_file_does_not_lose_the_new_alignment(tmp_path):
    """Corrupt JSON must not abort the re-align, and reports 0 preserved."""
    p = tmp_path / "0001_id_gemini-2.0-flash_aligned.json"
    p.write_text("{ not json", encoding="utf-8")
    fresh = [_machine([0, 50, 50, 70], "a")]

    lines, unmatched, n = _preserve_manual_lines(p, fresh, [])

    assert n == 0
    assert lines == fresh


def test_partially_corrected_page_keeps_both_kinds(tmp_path):
    """The common review pattern: a few targeted fixes on a machine-aligned page."""
    p = _existing(tmp_path, [
        _machine([0, 50, 50, 70], "stale machine line"),
        _manual([0, 150, 50, 170], "hand-fixed line"),
    ])
    fresh = [
        _machine([0, 50, 50, 70], "better machine line"),
        _machine([0, 150, 50, 170], "machine guess at the fixed line"),
        _machine([0, 250, 50, 270], "newly recovered line"),
    ]

    lines, _, n = _preserve_manual_lines(p, fresh, [])

    assert n == 1
    assert [ln["gemini_text"] for ln in lines] == [
        "better machine line",     # machine improvements still land
        "hand-fixed line",         # the human's correction wins its bbox
        "newly recovered line",    # coverage gains are kept
    ]
