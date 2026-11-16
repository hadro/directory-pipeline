"""Image-level page heuristics shared by pipeline stages."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def image_is_complete(path: Path) -> bool:
    """Return True if *path* holds a whole, decodable image.

    An interrupted download leaves an empty or truncated file, and every stage
    downstream then fails on it (a Surya batch fails as a whole). A JPEG that
    starts with SOI and ends with EOI is accepted without decoding, which is
    the cheap common case; anything else is decoded to decide, so PNGs and
    JPEGs with trailing bytes still pass.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            f.seek(0, 2)
            if f.tell() < 4:
                return False
            f.seek(-2, 2)
            tail = f.read(2)
    except OSError:
        return False
    if head == b"\xff\xd8" and tail == b"\xff\xd9":
        return True
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001 — truncated, empty, or not an image at all
        return False


def is_blank_page(
    image_path: Path,
    *,
    long_side: int = 1600,
    margin_frac: float = 0.08,
    dark_ratio: float = 0.75,
    max_dark_frac: float = 0.001,
) -> bool:
    """Return True if the page image contains essentially no dark content.

    Sent to an LLM, a blank page is a hallucination magnet: the model invents
    the text it expects from the prompt instead of returning nothing. This
    check lets OCR stages skip such pages before spending an API call.

    Method: grayscale at no more than ``long_side`` px, crop ``margin_frac``
    off each edge (scanner borders and binding shadows), estimate the paper
    background as the 90th-percentile pixel value, then measure the fraction
    of pixels darker than ``dark_ratio`` × background. Text pages run several
    percent dark; blank scans, show-through included, essentially zero.

    Both the resolution and the ratio are set by what real pages broke:

    * Measuring on a 512 px thumbnail averaged small, thin type into mid-grey
      (the darkest pixel left on a dense Doggett 1850 page was 66 against a
      toned-paper background of 200), so full pages read as blank. 1600 px
      keeps the strokes.
    * Green Book and other grey-printed scans have type at 0.55–0.65 ×
      background, which a 0.5 ratio misses at any resolution.

    Thresholds are conservative: a false negative (blank page still sent to
    OCR) is annoying, but a false positive (real page silently skipped) loses
    data — when in doubt this returns False. On the 18,471-page corpus the
    sparsest real page (a "NOTES" page, a printer's imprint, a title page)
    measured 0.0022 dark, over twice ``max_dark_frac``. A page whose only
    content is one short line of small type, such as a copyright notice,
    still reads as blank. Measurements: docs/evidence/2026-10-16-blank-page/.
    """
    with Image.open(image_path) as im:
        im.thumbnail((long_side, long_side))
        gray = ImageOps.grayscale(im)
    w, h = gray.size
    mx, my = int(w * margin_frac), int(h * margin_frac)
    if w - 2 * mx < 16 or h - 2 * my < 16:
        return False  # degenerate crop — judge nothing
    arr = np.asarray(gray.crop((mx, my, w - mx, h - my)), dtype=np.uint8)

    background = float(np.percentile(arr, 90))
    if background < 100:
        return False  # dark scan overall — heuristic does not apply
    dark_frac = float((arr < background * dark_ratio).mean())
    return dark_frac < max_dark_frac
