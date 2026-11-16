"""utils/image_utils.py — blank-page detection heuristic.

A false positive here silently drops a real page: run_gemini_ocr writes no .txt
for it, so align and extract never see it. The not-blank fixtures below are
shaped after real pages the 512 px / 0.5-ratio check skipped
(docs/evidence/2026-10-16-blank-page/); the blank ones after the show-through
versos and toned endpapers that make up most of what it rightly skips.
"""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from utils.image_utils import image_is_complete, is_blank_page


def _save(tmp_path, im, name="page.jpg"):
    p = tmp_path / name
    im.save(p)
    return p


def _text_page(size, *, paper, ink, stroke=1, pitch=28, glyph=(11, 14),
               grain=4.0, seed=0):
    """A dense two-column page of glyph-sized outlines on noisy paper."""
    rng = np.random.default_rng(seed)
    w, h = size
    im = Image.new("L", size, paper)
    draw = ImageDraw.Draw(im)
    gw, gh = glyph
    col_w = (w - 300) // 2
    for col in range(2):
        x0 = 150 + col * col_w
        for y in range(160, h - 160, pitch):
            x, end = x0, x0 + int(col_w * rng.uniform(0.55, 0.95))
            while x < end:
                if rng.random() > 0.15:  # else a word space
                    draw.rectangle([x, y, x + gw - 3, y + gh], outline=ink, width=stroke)
                x += gw
    return _with_grain(im, grain, rng)


def _with_grain(im, sigma, rng):
    a = np.asarray(im, dtype=np.float32) + rng.normal(0, sigma, (im.height, im.width))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def test_pure_white_page_is_blank(tmp_path):
    im = Image.new("L", (1600, 2200), 255)
    assert is_blank_page(_save(tmp_path, im))


def test_white_page_with_scanner_noise_is_blank(tmp_path):
    # Slightly uneven paper tone plus a dark scanner edge (cropped by margin).
    im = Image.new("L", (1600, 2200), 250)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 30, 2200], fill=10)  # binding shadow at the edge
    assert is_blank_page(_save(tmp_path, im))


def test_toned_blank_page_with_paper_grain_is_blank(tmp_path):
    im = _with_grain(Image.new("L", (1800, 2865), 200), 6.0, np.random.default_rng(1))
    assert is_blank_page(_save(tmp_path, im))


def test_show_through_is_blank(tmp_path):
    # A blank verso showing the facing page's print through the paper. Measured
    # show-through sits at 0.85-0.95 x background; sending it to OCR invites the
    # model to transcribe mirror-image text.
    im = _text_page((1800, 2865), paper=220, ink=194)
    assert is_blank_page(_save(tmp_path, im))


def test_page_with_text_lines_is_not_blank(tmp_path):
    im = Image.new("L", (1600, 2200), 255)
    draw = ImageDraw.Draw(im)
    for y in range(300, 1900, 60):  # text-like dark line blocks
        draw.rectangle([200, y, 1400, y + 25], fill=20)
    assert not is_blank_page(_save(tmp_path, im))


def test_page_with_single_heading_is_not_blank(tmp_path):
    # One short centred line (e.g. a "PART SECOND." divider page).
    im = Image.new("L", (1600, 2200), 255)
    draw = ImageDraw.Draw(im)
    draw.rectangle([600, 1050, 1000, 1090], fill=20)
    assert not is_blank_page(_save(tmp_path, im))


def test_thin_dark_type_on_toned_paper_is_not_blank(tmp_path):
    # Doggett's 1850-51 directory, page 154: 154 lines of small type on paper
    # whose background reads 200. A 512 px thumbnail averaged the strokes to
    # mid-grey (darkest pixel 66), under the old 0.5 x background cutoff of 100
    # on almost no pixels, and the page was skipped as blank.
    im = _text_page((1800, 2865), paper=200, ink=50, stroke=1)
    assert not is_blank_page(_save(tmp_path, im))


def test_grey_type_is_not_blank(tmp_path):
    # Green Book scans print at 0.55-0.65 x background, so a 0.5 x background
    # cutoff missed dense listing pages at any resolution.
    im = _text_page((1700, 2600), paper=215, ink=130, stroke=2, pitch=34, glyph=(14, 18))
    assert not is_blank_page(_save(tmp_path, im))


def test_dark_scan_is_never_blank(tmp_path):
    # Heuristic does not apply to overall-dark images — must return False.
    im = Image.new("L", (1600, 2200), 40)
    assert not is_blank_page(_save(tmp_path, im))


def _encoded(fmt):
    buf = io.BytesIO()
    _text_page((600, 900), paper=230, ink=40).save(buf, fmt)
    return buf.getvalue()


@pytest.mark.parametrize("data, complete", [
    (_encoded("JPEG"), True),
    (_encoded("PNG"), True),
    (_encoded("JPEG") + b"\x00" * 16, True),        # padding after EOI still decodes
    (_encoded("JPEG")[:8192], False),                # stream cut mid-file
    (b"", False),                                    # request died before any bytes
    (b"<html><body>502 Bad Gateway</body></html>", False),
], ids=["jpeg", "png", "jpeg-trailing-bytes", "truncated", "empty", "html"])
def test_image_is_complete(tmp_path, data, complete):
    p = tmp_path / "page.jpg"
    p.write_bytes(data)
    assert image_is_complete(p) is complete
