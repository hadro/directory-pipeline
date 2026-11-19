"""Find picture regions (maps, photos, artwork) from the ink a detector's lines don't explain.

Text lines cover the text. What ink is left once they are erased is rules, dot
leaders, borders and pictures. Rules and leaders are thin; a map's roads and
rivers, a photo's tones and an ad's artwork form large regions dense with ink.
So: erase the line boxes, close what is left into regions, and keep regions
that are large in both directions, dense, and clear of the image edge (where a
book scan's binding and cover sit). Lines whose centre falls in one are picture
text: map labels, captions inside artwork.

No model and no text: it works with any detector's lines. The thresholds are
first guesses, to be measured (entry_placement.py --mask-pictures).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WORK_SIDE = 1000      # px: the page is analysed at this long side
CLOSE = 0.012         # closing kernel, share of the long side
MIN_SIDE = 0.10       # a region spans at least this share of the page each way
MIN_INK = 0.06        # and at least this share of its box is non-text ink
EDGE = 0.01           # regions within this share of the image edge are scan borders


def picture_regions(image: Path, boxes) -> list[list[float]]:
    """Picture regions [x0, y0, x1, y1] in page pixels, given the page's line boxes."""
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []
    H, W = gray.shape
    s = WORK_SIDE / max(H, W)
    small = cv2.resize(gray, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v * s)) for v in b[:4])
        ink[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] = 0
    k = max(3, round(CLOSE * WORK_SIDE))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    edge = EDGE * max(h, w)
    out = []
    for x, y, bw, bh, _ in stats[1:n]:
        if bw < MIN_SIDE * w or bh < MIN_SIDE * h:
            continue
        if x <= edge or y <= edge or x + bw >= w - edge or y + bh >= h - edge:
            continue
        if ink[y:y + bh, x:x + bw].mean() / 255 < MIN_INK:
            continue
        out.append([x / s, y / s, (x + bw) / s, (y + bh) / s])
    return out


def outside_pictures(lines: list[dict], regions) -> list[dict]:
    """The lines whose centre is in no picture region."""
    def inside(ln):
        cx = (ln["bbox"][0] + ln["bbox"][2]) / 2
        cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
        return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in regions)
    return [ln for ln in lines if not inside(ln)]
