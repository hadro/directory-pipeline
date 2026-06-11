"""utils/image_utils.py — blank-page detection heuristic."""

from PIL import Image, ImageDraw

from utils.image_utils import is_blank_page


def _save(tmp_path, im, name="page.jpg"):
    p = tmp_path / name
    im.save(p)
    return p


def test_pure_white_page_is_blank(tmp_path):
    im = Image.new("L", (1600, 2200), 255)
    assert is_blank_page(_save(tmp_path, im))


def test_white_page_with_scanner_noise_is_blank(tmp_path):
    # Slightly uneven paper tone plus a dark scanner edge (cropped by margin).
    im = Image.new("L", (1600, 2200), 250)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 30, 2200], fill=10)  # binding shadow at the edge
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


def test_dark_scan_is_never_blank(tmp_path):
    # Heuristic does not apply to overall-dark images — must return False.
    im = Image.new("L", (1600, 2200), 40)
    assert not is_blank_page(_save(tmp_path, im))
