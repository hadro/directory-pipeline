"""pipeline/align_ocr.py — pages whose boxes sit in a placeholder canvas are named.

When info.json cannot be fetched, align_image falls back to the manifest canvas
size, which on NYPL can be a 2560x2560 placeholder. A per-page warning was the
only signal; five 1958 Green Book pages (2026-10-20) went unnoticed until
rescale_canvas_fragments.py found them. The run summary now lists them.
"""

from pathlib import Path

from pipeline.align_ocr import _is_placeholder_canvas, _report_placeholder_canvases


def test_square_fallback_canvas_is_a_placeholder():
    # 0078 of the 1958 Green Book as align_ocr first wrote it.
    assert _is_placeholder_canvas(
        {"canvas_width": 2560, "canvas_height": 2560, "coords_from_fallback": True}
    )


def test_dims_from_info_json_are_trusted_even_when_square():
    """A genuinely square image whose info.json answered is not a fallback."""
    assert not _is_placeholder_canvas(
        {"canvas_width": 2560, "canvas_height": 2560, "coords_from_fallback": None}
    )


def test_non_square_fallback_is_not_flagged():
    """The manifest size can be right; only the square placeholder is the signature."""
    assert not _is_placeholder_canvas(
        {"canvas_width": 4150, "canvas_height": 5800, "coords_from_fallback": True}
    )


def test_missing_dims_are_not_flagged():
    assert not _is_placeholder_canvas({"coords_from_fallback": True})


def test_report_names_pages_and_the_repair():
    out: list[str] = []
    _report_placeholder_canvases(["0077_a_aligned.json", "0078_b_aligned.json"],
                                 Path("output/vol"), out.append, "gemini-x")
    text = "\n".join(out)
    assert "2 page(s)" in text
    assert "tools/rescale_canvas_fragments.py output/vol" in text
    assert "tools/patch_canvas_fragments.py output/vol --aligned-model gemini-x" in text
    assert "0078_b_aligned.json" in text


def test_report_is_silent_when_nothing_is_wrong():
    out: list[str] = []
    _report_placeholder_canvases([], Path("output/vol"), out.append)
    assert out == []
