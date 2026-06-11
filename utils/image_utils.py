"""Image-level page heuristics shared by pipeline stages."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def is_blank_page(
    image_path: Path,
    *,
    margin_frac: float = 0.08,
    dark_ratio: float = 0.5,
    max_dark_frac: float = 0.002,
) -> bool:
    """Return True if the page image contains essentially no dark content.

    Sent to an LLM, a blank page is a hallucination magnet: the model invents
    the text it expects from the prompt instead of returning nothing. This
    check lets OCR stages skip such pages before spending an API call.

    Method: grayscale + downsample, crop ``margin_frac`` off each edge
    (scanner borders and binding shadows), estimate the paper background as
    the 90th-percentile pixel value, then measure the fraction of pixels
    darker than ``dark_ratio`` × background. Text pages run several percent
    dark; blank scans essentially zero.

    Thresholds are conservative: a false negative (blank page still sent to
    OCR) is annoying, but a false positive (real page silently skipped) loses
    data — when in doubt this returns False.
    """
    with Image.open(image_path) as im:
        gray = ImageOps.grayscale(im)
        gray.thumbnail((512, 512))
        w, h = gray.size
        mx, my = int(w * margin_frac), int(h * margin_frac)
        if w - 2 * mx < 16 or h - 2 * my < 16:
            return False  # degenerate crop — judge nothing
        gray = gray.crop((mx, my, w - mx, h - my))
        arr = np.asarray(gray, dtype=np.uint8)

    background = float(np.percentile(arr, 90))
    if background < 100:
        return False  # dark scan overall — heuristic does not apply
    dark_frac = float((arr < background * dark_ratio).mean())
    return dark_frac < max_dark_frac
