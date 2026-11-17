"""Shared by compare_detectors.py and gold_panel.py: run the live aligner on one
page's lines from any detector, in memory.

The preparation is placement_panel.prepare() from the 2026-10-17 panel, with the
line file as an argument instead of ``{stem}_surya.json`` beside the image, so
Surya's file in the volume and Indra's in the mirror go through the same code.
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PANEL = REPO / "docs" / "evidence" / "2026-10-17-column-alignment-panel"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PANEL))

import pipeline.align_ocr as A  # noqa: E402
from placement_panel import FRAG, head  # noqa: E402,F401  (re-exported)

TALL = 2.0          # a box taller than this × the page's median line height
DISAGREE = 0.5      # head similarity under this: bound text is not what the detector read
FAILING, BORDERLINE = 0.5, 0.2   # page buckets by disagreement, as in box_vs_surya.py


def read_gemini(gem: Path) -> tuple[list[str], set[int]]:
    """Gemini lines and the indices inside advertisement blocks (as prepare())."""
    gl, ads, in_ad = [], set(), False
    for raw in gem.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if re.match(r"^=== .+ ===$", ln):
            in_ad = ln.startswith("=== ADVERTISEMENT")
            continue
        if in_ad:
            ads.add(len(gl))
        gl.append(ln)
    return gl, ads


def pick_gemini(img: Path, model: str | None = None) -> Path | None:
    """The page's Gemini OCR text: *model* if given, else as the panel picks."""
    if model:
        p = img.parent / f"{img.stem}_{model}.txt"
        return p if p.exists() else None
    cands = list(img.parent.glob(img.stem + "_gemini*.txt"))
    cands = [c for c in cands if not c.name.endswith("_entries_error.txt")]
    pref = [c for c in cands if c.name.endswith("_gemini-3.1-flash-lite.txt")]
    return (pref or sorted(cands) or [None])[0]


def prepare(lines_json: Path, min_conf: float = 0.35):
    """align_image() up to its build_aligned_lines call, for one line file."""
    page_bbox, lines, _ = A.parse_surya(lines_json, min_confidence=min_conf)
    lines = A._expand_multiline_surya_lines(lines)
    img_w = page_bbox[2]
    page_w = img_w - page_bbox[0]
    lines = A.filter_margin_lines(A.filter_short_lines(lines), img_w)
    plan = A.plan_columns(lines, page_w)
    if plan is not None:
        groups = [plan[0]] + plan[1]
        fallback = A._legacy_reading_order(lines, page_w)
        ordered = plan[0] + [ln for c in plan[1] for ln in c]
    else:
        groups, fallback, ordered = None, None, A._legacy_reading_order(lines, page_w)
    return lines, ordered, groups, fallback


def align(lines_json: Path, gl: list[str], ads: set[int], min_conf: float = 0.35):
    """The shipped alignment of *gl* onto one detector's lines.

    Returns (lines, bound, unmatched, column_path_ran): *bound* is
    [(bbox tuple, gemini_text)]. The column path ran when build_aligned_lines
    reached its cost comparison; the cost function is wrapped, not replaced, so
    the result is exactly what production would write.
    """
    lines, ordered, groups, fallback = prepare(lines_json, min_conf)
    real, calls = A._alignment_cost, {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    A._alignment_cost = counting
    try:
        out, unmatched = A.build_aligned_lines(ordered, gl, FRAG, ads,
                                               column_groups=groups, fallback_order=fallback)
    finally:
        A._alignment_cost = real
    bound = [(tuple(e["bbox"]), e.get("gemini_text", "")) for e in out]
    return lines, bound, unmatched, calls["n"] > 0


def page_stats(lines, bound, unmatched, n_gem: int) -> dict:
    """The per-detector numbers both scripts report for one page."""
    heights = [ln["bbox"][3] - ln["bbox"][1] for ln in lines]
    med = statistics.median(heights) if heights else 0
    own = {tuple(ln["bbox"]): ln.get("text", "") for ln in lines}
    sims = [head(t, own.get(b, "")) for b, t in bound]
    return {
        "lines": len(lines),
        "per_gem": len(lines) / n_gem if n_gem else 0.0,
        "tall": sum(1 for h in heights if med and h > TALL * med),
        "empty": (sum(1 for ln in lines if not (ln.get("text") or "").strip()) / len(lines)
                  if lines else 0.0),
        "unboxed": len(unmatched) / n_gem if n_gem else 0.0,
        "disagree": sum(s < DISAGREE for s in sims) / len(sims) if sims else None,
    }


def bucket(x) -> str:
    if x is None:
        return "none"
    return "failing" if x > FAILING else ("borderline" if x > BORDERLINE else "clean")
