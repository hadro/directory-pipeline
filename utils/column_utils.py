#!/usr/bin/env python3
"""Shared column-layout primitives for multi-column page images.

Two stages need to agree about where a page's columns are: ``align_ocr``
re-sorts Surya lines into reading order before alignment, and ``surya_detect``
reports a column count for QA. When they disagree, the column report describes
a page the aligner never saw. Both the gutter detector and the thresholds that
decide what counts as a column therefore live here, defined once.

This module deliberately depends on nothing else in the project so either stage
can import it without a cycle.
"""

# Column detection via coverage valleys.  The page is divided into
# COLUMN_VALLEY_BINS vertical bins; each bin counts how many line bboxes cover
# it.  Body columns show up as plateaus and gutters as valleys, which survives
# the sparse indented/wrapped lines that defeat a consecutive-x1-gap rule.  A
# bin is part of a gutter when its coverage falls to COLUMN_VALLEY_RATIO of the
# page's peak coverage.
COLUMN_VALLEY_BINS  = 32
COLUMN_VALLEY_RATIO = 0.55

# What it takes for a detected layout to count as genuinely multi-column: every
# reported column holds at least MIN_COLUMN_FRAC of the page's lines, and the
# reported columns together account for at least MIN_COLUMN_COVERAGE of them.
# Both stages read these so a page cannot be three columns to one and two to
# the other.
MIN_COLUMN_FRAC     = 0.10
MIN_COLUMN_COVERAGE = 0.80


def column_breaks(lines: list[dict], page_width: int) -> list[float]:
    """X positions of the gutters between body columns, left to right.

    *lines* are dicts with a ``"bbox"`` of ``[x1, y1, x2, y2]`` in image pixels.
    Returns an empty list when the page is too sparse to judge (fewer than four
    lines covering the peak bin) or shows no interior valley.
    """
    bin_w = max(page_width // COLUMN_VALLEY_BINS, 16)
    n_bins = page_width // bin_w + 1
    profile = [0] * n_bins
    for ln in lines:
        lo = max(0, ln["bbox"][0] // bin_w)
        hi = min(n_bins - 1, ln["bbox"][2] // bin_w)
        for b in range(lo, hi + 1):
            profile[b] += 1
    peak = max(profile) if profile else 0
    if peak < 4:
        return []
    threshold = peak * COLUMN_VALLEY_RATIO
    valleys: list[tuple[int, int]] = []
    run: list[int] | None = None
    for i, count in enumerate(profile):
        if count <= threshold:
            run = [i, i] if run is None else [run[0], i]
        elif run is not None:
            valleys.append((run[0], run[1])); run = None
    if run is not None:
        valleys.append((run[0], run[1]))
    # Valleys touching either edge are the page margins, not gutters.
    return [((a + b + 1) / 2.0) * bin_w
            for a, b in valleys if a != 0 and b != n_bins - 1]
