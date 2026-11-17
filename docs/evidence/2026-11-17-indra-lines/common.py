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


JOIN_SHARE = 0.75   # share of each piece's words that must appear in the one Gemini line


def _words(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def join_rows(lines: list[dict], gl: list[str]) -> list[dict]:
    """Join side-by-side pieces of one printed row that Gemini wrote as one line.

    Detectors box a house number apart from its name (Tulsa's street pages), and a
    name apart from its address across a dot leader (the Green Book). Gemini writes
    each row as one line, and alignment pairs one line with one box. Two pieces on
    the same row, with no box between them, are joined when most of each piece's
    words appear together in a single Gemini line: across a column gutter they never
    do, since two columns' words do not share a Gemini line.
    """
    gem = [_words(g) for g in gl]
    by_word: dict[str, set[int]] = {}
    for i, ws in enumerate(gem):
        for w in ws:
            by_word.setdefault(w, set()).add(i)

    def together(a: set, b: set) -> bool:
        if not a or not b:
            return False
        cands = set().union(*(by_word.get(w, set()) for w in a | b))
        return any(len(a & gem[i]) >= JOIN_SHARE * len(a) and len(b & gem[i]) >= JOIN_SHARE * len(b)
                   for i in cands)

    lines = sorted(lines, key=lambda ln: (ln["bbox"][0], ln["bbox"][1]))
    out, used = [], set()
    for i, ln in enumerate(lines):
        if i in used:
            continue
        cur = dict(ln)
        while True:
            x0, y0, x1, y1 = cur["bbox"]
            right = [j for j, m in enumerate(lines) if j not in used and j != i and m["bbox"][0] >= x1 - 2
                     and min(y1, m["bbox"][3]) - max(y0, m["bbox"][1])
                     >= 0.6 * min(y1 - y0, m["bbox"][3] - m["bbox"][1])]
            if not right:
                break
            j = min(right, key=lambda j: lines[j]["bbox"][0])
            nxt = lines[j]
            if not together(_words(cur.get("text", "")), _words(nxt.get("text", ""))):
                break
            used.add(j)
            b = nxt["bbox"]
            cur["bbox"] = [x0, min(y0, b[1]), max(x1, b[2]), max(y1, b[3])]
            cur["text"] = f"{cur.get('text', '')} {nxt.get('text', '')}".strip()
        out.append(cur)
    return out


def prepare(lines_json: Path, min_conf: float = 0.35, mask_pictures: bool = False,
            join_with: list[str] | None = None):
    """align_image() up to its build_aligned_lines call, for one line file.
    *mask_pictures* drops lines inside picture regions (picture_mask.py) first;
    *join_with* (the page's Gemini lines) joins pieces of one row (join_rows)."""
    page_bbox, lines, _ = A.parse_surya(lines_json, min_confidence=min_conf)
    lines = A._expand_multiline_surya_lines(lines)
    if mask_pictures:
        from picture_mask import outside_pictures, picture_regions
        img = lines_json.with_name(lines_json.name[: -len("_surya.json")] + ".jpg")
        lines = outside_pictures(lines, picture_regions(img, [ln["bbox"] for ln in lines]))
    if join_with:
        lines = join_rows(lines, join_with)
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


def align(lines_json: Path, gl: list[str], ads: set[int], min_conf: float = 0.35,
          mask_pictures: bool = False, join: bool = False):
    """The shipped alignment of *gl* onto one detector's lines.

    Returns (lines, bound, unmatched, column_path_ran): *bound* is
    [(bbox tuple, gemini_text)]. The column path ran when build_aligned_lines
    reached its cost comparison; the cost function is wrapped, not replaced, so
    the result is exactly what production would write.
    """
    lines, ordered, groups, fallback = prepare(lines_json, min_conf, mask_pictures,
                                               gl if join else None)
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


def cascade_pick(surya_disagree, indra_disagree) -> str:
    """Indra unless its page is not clean; then whichever detector disagrees less.
    Production would run Surya only on the pages that reach the second test."""
    i = 1.0 if indra_disagree is None else indra_disagree
    s = 1.0 if surya_disagree is None else surya_disagree
    return "indra" if i <= BORDERLINE or i <= s else "surya"
