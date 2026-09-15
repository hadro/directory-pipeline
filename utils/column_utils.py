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


# ---------------------------------------------------------------------------
# Geometric alignment quality
# ---------------------------------------------------------------------------
# These live here rather than in tools/compare_alignments.py because the
# aligner itself now uses them to choose between candidate alignments, and a
# tool must not be a dependency of the pipeline.

def bbox_centre(b) -> "tuple[float, float]":
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def assign_columns(placed: "list[tuple[float, float]]", n_cols_max: int = 4) -> "list[int]":
    """Assign each placed point a column index from x-centre gaps alone."""
    xs = sorted(p[0] for p in placed)
    if not xs:
        return [0] * len(placed)
    span = xs[-1] - xs[0]
    gaps = sorted(
        ((xs[i + 1] - xs[i], (xs[i + 1] + xs[i]) / 2.0) for i in range(len(xs) - 1)),
        reverse=True,
    )
    cuts = sorted(mid for gap, mid in gaps[: n_cols_max - 1] if span and gap > span * 0.12)
    return [sum(1 for c in cuts if p[0] >= c) for p in placed]


def inversion_rate(placed: "list[tuple[float, float]]") -> float:
    """Normalised Kendall-tau distance between list order and geometric order.

    This is the primary quality signal because it is *independent of the
    objective Needleman-Wunsch optimises*. NW maximises text similarity, so a
    text-similarity score rewards its mistakes as readily as its successes.
    Geometry does not collude: if alignment is correct, walking the aligned
    lines in list order visits boxes in column-major, top-to-bottom order.
    Every pair visited out of that order is an inversion.

    Returns inversions / total_pairs, so 0.0 is perfect and higher is worse.
    """
    n = len(placed)
    if n < 3:
        return 0.0
    cols = assign_columns(placed)
    order = sorted(range(n), key=lambda i: (cols[i], placed[i][1], placed[i][0]))
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r
    inv = 0
    for i in range(n):
        ri = rank[i]
        for j in range(i + 1, n):
            if rank[j] < ri:
                inv += 1
    total = n * (n - 1) // 2
    return inv / total if total else 0.0
