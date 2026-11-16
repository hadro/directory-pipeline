#!/usr/bin/env python3
"""Score alignment placement against city-directory-extraction's hand-gold panel.

For each gold row, recover the Surya box it was seeded from, run the aligner,
and ask whether the Gemini line bound to that box matches the verified
raw_line. Neither the box nor the gold text can move with the alignment
choice, so the judge cannot flatter any candidate.

Gold boxes are recovered by ORDER: make_gold_tool.py builds each page's rows
by walking its Surya lines in column-major order (_colidx/_order_lines, copied
below), and labellers only delete rows, so the gold is an order-preserving
subsequence of that walk. A monotonic DP alignment against it is far more
reliable than matching each row independently, which repeated surnames defeat.

Candidates, all from the live pipeline/align_ocr.py:
  whole     whole-page alignment over the legacy reading order
  column    the per-column path with its guards, no selection
  coverage  selection by unboxed count alone (the rule the evidence rejected)
  shipped   build_aligned_lines as it stands

The gold panel is unpublished and stays in city-directory-extraction; this
script reads it from there and prints counts and page names only.

    ./.venv/bin/python docs/evidence/2026-10-17-column-alignment-panel/placement_panel.py
    ... placement_panel.py --gold-dir PATH/TO/city-directory-extraction/data
"""
import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import pipeline.align_ocr as A  # noqa: E402
from utils.column_utils import column_breaks  # noqa: E402

OUT = REPO / "output"
# Sets in that data/ directory that are not the hand-built panel: harvested
# silver sets, other datasets, or boxes taken from this aligner (tulsa, lain).
NOT_PANEL = ("1906BPL", "bands_", "ftd", "nyu", "micro13", "minneapolis", "tulsa", "lain_eval")
DROP = ("polk1933bk",)               # region-level Surya: unusable boxes
SUSPECT = ("mb1931", "polk1917")     # partly region-level Surya
MATCH_MIN, GEM_MIN, CORRECT_MIN = 0.55, 0.75, 0.60
FRAG = lambda b: f"#xywh={b[0]},{b[1]},{b[2] - b[0]},{b[3] - b[1]}"  # noqa: E731


def norm(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def head(a: str, b: str) -> float:
    a, b = norm(a)[:60], norm(b)[:60]
    return ratio(a, b) if a and b else 0.0


def sim(gold: str, text: str) -> float:
    """Similarity tolerant of an entry wrapped across two OCR lines."""
    g, t = norm(gold), norm(text)
    if not g or not t:
        return 0.0
    r = ratio(g, t)
    if 0.4 * len(g) <= len(t) < len(g):
        r = max(r, ratio(g[: len(t)], t))
    if 0.4 * len(t) <= len(g) < len(t):
        r = max(r, ratio(t[: len(g)], g))
    return r


def _colidx(bbox, width, cols):                     # verbatim from make_gold_tool.py
    if cols <= 1 or not width:
        return 0
    x1, _, x2, _ = bbox
    return max(0, min(cols - 1, int(((x1 + x2) / 2) / (width / cols))))


def _order_lines(lines, width, cols):               # verbatim from make_gold_tool.py
    if cols <= 1:
        return sorted(range(len(lines)), key=lambda i: lines[i]["bbox"][1])
    return sorted(range(len(lines)),
                  key=lambda i: (_colidx(lines[i]["bbox"], width, cols), lines[i]["bbox"][1]))


def monotonic(gold, cand):
    """Best order-preserving assignment of gold rows to candidate lines."""
    m, n = len(gold), len(cand)
    S = [[head(g, c) for c in cand] for g in gold]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    bt = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best, how = dp[i][j - 1], 0
            if dp[i - 1][j] > best:
                best, how = dp[i - 1][j], 1
            s = S[i - 1][j - 1]
            if s >= MATCH_MIN and dp[i - 1][j - 1] + s > best:
                best, how = dp[i - 1][j - 1] + s, 2
            dp[i][j], bt[i][j] = best, how
    out, i, j = [None] * m, m, n
    while i > 0 and j > 0:
        if bt[i][j] == 2:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif bt[i][j] == 1:
            i -= 1
        else:
            j -= 1
    return dp[m][n], out


def recover_boxes(surya_path, raws):
    """Raw Surya line for each gold row; the editor's column count is not
    stored, so 1..6 are tried and the best-scoring order kept."""
    data = json.loads(surya_path.read_text(encoding="utf-8"))
    width, lines = data.get("image_width") or 0, data.get("lines", [])
    best = None
    for cols in range(1, 7):
        order = [i for i in _order_lines(lines, width, cols) if (lines[i].get("text") or "").strip()]
        total, assign = monotonic(raws, [lines[i]["text"] for i in order])
        if best is None or total > best[0]:
            best = (total, cols, [None if a is None else lines[order[a]] for a in assign])
    return best[1], best[2], width


def prepare(img, gem):
    """align_image() up to its build_aligned_lines call."""
    page_bbox, lines, _ = A.parse_surya(img.parent / f"{img.stem}_surya.json", min_confidence=0.35)
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
    return lines, ordered, groups, fallback, gl, ads, page_w


def candidates(ordered, groups, fallback, gl, ads):
    real, calls = A._alignment_cost, {"n": 0}

    def prefer_column(lines, unmatched):     # ties go to the column candidate
        calls["n"] += 1
        return 0.0

    out = {"whole": A._build_line_aligned_lines(fallback or ordered, gl, FRAG, ads)}
    try:
        A._alignment_cost = prefer_column
        out["column"] = A.build_aligned_lines(ordered, gl, FRAG, ads,
                                              column_groups=groups, fallback_order=fallback)
        ran = calls["n"] > 0                  # 0 calls: the column path reverted early
        A._alignment_cost = lambda lines, unmatched: float(len(unmatched))
        out["coverage"] = A.build_aligned_lines(ordered, gl, FRAG, ads,
                                                column_groups=groups, fallback_order=fallback)
    finally:
        A._alignment_cost = real
    out["shipped"] = A.build_aligned_lines(ordered, gl, FRAG, ads,
                                           column_groups=groups, fallback_order=fallback)
    return out, ran


def revert_reason(groups, gl):
    """Replay build_aligned_lines' checks to name why the column path reverted."""
    if not groups:
        return "no plan (1 col)"
    n_surya, n_gem = sum(len(g) for g in groups), len(gl)
    if n_surya and len(groups[0]) / n_surya > A._PARTITION_MAX_HEADER_FRAC:
        return f"guard1 header {len(groups[0])}/{n_surya}"
    if n_gem and n_surya / n_gem >= A._PARTITION_MAX_MERGE_RATIO:
        return f"guard2 ratio {n_surya / n_gem:.2f}"
    norms, cuts, prev = [A._normalize(g) for g in gl], [], 0
    for j, col in enumerate(groups[1:]):
        lo = 0 if j == 0 else prev + 1
        cut = (A._locate_column_start(col, norms, lo, n_gem) if (j or groups[0]) else 0)
        if cut is None:
            return f"locate failed col {j + 1}"
        if cut < prev:
            return f"cut order col {j + 1}"
        cuts.append(cut)
        prev = cut
    bounds = [0] + cuts + [n_gem]
    for j, (grp, (a, b)) in enumerate(zip(groups[1:], list(zip(bounds, bounds[1:]))[1:])):
        if grp and (b - a == 0 or not (A._PARTITION_SPAN_LO <= (b - a) / len(grp) <= A._PARTITION_SPAN_HI)):
            return f"span col {j + 1}"
    return "ran"


def pick_gemini(img):
    cands = list(img.parent.glob(img.stem + "_gemini*.txt"))
    pref = [c for c in cands if c.name.endswith("_gemini-3.1-flash-lite.txt")]
    return (pref or sorted(cands) or [None])[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-dir", type=Path, default=REPO.parent / "city-directory-extraction" / "data")
    args = ap.parse_args()
    files = sorted(p for p in args.gold_dir.glob("*_eval.jsonl") if not any(k in p.name for k in NOT_PANEL))
    if not files:
        sys.exit(f"No gold panel under {args.gold_dir}. It lives in the city-directory-extraction "
                 "repo and is not published; pass --gold-dir.")
    pages = defaultdict(list)
    for f in files:
        vol = f.stem.replace("_eval", "")
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                pages[(vol, d["context"].get("image", ""))].append(d["raw_line"])
    index = {}
    for f in OUT.rglob("*.jpg"):
        if "/_" not in str(f) and "/.claude/" not in str(f):
            index.setdefault(f.name, []).append(f)

    names = ["whole", "column", "coverage", "shipped"]
    agg = {k: {n: defaultdict(int) for n in names} for k in ("all", "clean", "c1", "c2")}
    skipped, reasons = defaultdict(int), defaultdict(int)
    print(f"{'page':30s} cols fair c2 | " + " | ".join(f"{n:>8s} ok/wr/ub" for n in names)
          + " | box!=surya | column path")
    for (vol, img_name), raws in sorted(pages.items()):
        if any(d in vol for d in DROP):
            skipped["dropped page (region Surya)"] += len(raws)
            continue
        found = [c for c in index.get(img_name, [])
                 if (c.parent / f"{c.stem}_surya.json").exists() and pick_gemini(c)]
        if not found:
            skipped["no Surya/Gemini inputs"] += len(raws)
            continue
        img = found[0]
        gem = pick_gemini(img)
        if gem.read_text(encoding="utf-8", errors="replace").lstrip().startswith("[OCR FAILED"):
            skipped["Gemini OCR failed on the page"] += len(raws)
            continue
        lines, ordered, groups, fallback, gl, ads, page_w = prepare(img, gem)
        res, ran = candidates(ordered, groups, fallback, gl, ads)
        why = revert_reason(groups, gl)
        reasons[" ".join(why.split()[:2]) if why.startswith(("guard", "locate", "span")) else why] += 1
        ecols, raw_hits, width = recover_boxes(img.parent / f"{img.stem}_surya.json", raws)

        fair = []
        for r, raw in zip(raws, raw_hits):
            if raw is None:
                skipped["no monotonic match"] += 1
                continue
            rb = raw["bbox"]
            inside = [ln for ln in lines
                      if rb[0] - 2 <= (ln["bbox"][0] + ln["bbox"][2]) / 2 <= rb[2] + 2
                      and rb[1] - 2 <= (ln["bbox"][1] + ln["bbox"][3]) / 2 <= rb[3] + 2]
            if not inside:
                skipped["box dropped by aligner preprocessing"] += 1
                continue
            if max((sim(r, g) for g in gl), default=0.0) < GEM_MIN:
                skipped["text not in Gemini output"] += 1
                continue
            proc = max(inside, key=lambda ln: head(r, ln.get("text", "")))
            fair.append((r, tuple(proc["bbox"]), _colidx(rb, width, ecols)))

        row = []
        for n in names:
            bound = {tuple(e["bbox"]): e.get("gemini_text", "") for e in res[n][0]}
            c = defaultdict(int)
            for r, box, col in fair:
                t = bound.get(box)
                k = "unboxed" if t is None else ("correct" if sim(r, t) >= CORRECT_MIN else "wrong")
                c[k] += 1
                agg["all"][n][k] += 1
                if not any(s in vol for s in SUSPECT):
                    agg["clean"][n][k] += 1
                agg["c2" if col >= 1 else "c1"][n][k] += 1
            row.append(f"{c['correct']:4d}/{c['wrong']:2d}/{c['unboxed']:2d}")
        # The no-gold signature: Gemini text bound to a box vs Surya's own reading of it.
        surya_at = {tuple(ln["bbox"]): ln.get("text", "") for ln in lines}
        sims = [head(e.get("gemini_text", ""), surya_at.get(tuple(e["bbox"]), "")) for e in res["shipped"][0]]
        dis = sum(s < 0.5 for s in sims) / max(len(sims), 1)
        flag = "*" if any(s in vol for s in SUSPECT) else " "
        c2 = sum(1 for *_, col in fair if col >= 1)
        print(f"{flag}{(vol + '/' + img_name[:12])[:29]:29s} {len(column_breaks(lines, page_w)) + 1:4d} "
              f"{len(fair):4d} {c2:3d} | " + " | ".join(f"{x:>17s}" for x in row) + f" | {dis:9.1%} | {why}")

    def fmt(d):
        n = sum(d.values())
        return ("-" if not n else f"n={n:5d}  correct {d['correct'] / n * 100:5.1f}%  "
                f"wrong {d['wrong'] / n * 100:4.1f}%  unboxed {d['unboxed'] / n * 100:4.1f}%")
    for key, title in (("all", "All scored rows"), ("clean", "Excluding suspect pages (mb1931, polk1917)"),
                       ("c1", "Gold in column 1"), ("c2", "Gold in column 2")):
        print(f"\n{title}")
        for n in names:
            print(f"  {n:9s} {fmt(agg[key][n])}")
    print("\nColumn path outcome per page:", dict(reasons))
    print("Not scored:", dict(skipped))


if __name__ == "__main__":
    main()
