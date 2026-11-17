#!/usr/bin/env python3
"""Run another script in this folder with "floor off" replaced by the width-aware floor.

Wherever the script sets align_ocr._MIN_LINE_HEIGHT to 0 (--min-line-height 0, or
entry_placement.py's *_nofloor configurations), a line under 15 px is kept only if
it is at least K x the page's median line width; the 35%-of-median rule still
applies. The shipped floor (15) runs the shipped filter. K = 0.5 is the rule on
branch claude/width-aware-line-floor; once that merges, the scripts' default floor
is this rule and this wrapper is not needed.

    python docs/evidence/2026-11-17-indra-lines/width_floor.py 0.5 \\
        docs/evidence/2026-11-17-indra-lines/gold_panel.py --mirror-root ... --min-line-height 0
"""
import multiprocessing
import runpy
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pipeline.align_ocr as A  # noqa: E402

FLOOR = 15
K = float(sys.argv[1]) if __name__ == "__main__" else 0.5
_shipped = A.filter_short_lines


def width_aware(lines, min_height=None):
    if min_height is not None or not lines or A._MIN_LINE_HEIGHT:
        return _shipped(lines, min_height)
    h = lambda ln: ln["bbox"][3] - ln["bbox"][1]  # noqa: E731
    w = lambda ln: ln["bbox"][2] - ln["bbox"][0]  # noqa: E731
    rel = round(statistics.median(map(h, lines)) * 0.35)
    wide = K * statistics.median(map(w, lines))
    return [ln for ln in lines if h(ln) >= rel and (h(ln) >= FLOOR or w(ln) >= wide)]


A.filter_short_lines = width_aware
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)   # entry_placement's workers inherit the patch
    sys.argv = sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
