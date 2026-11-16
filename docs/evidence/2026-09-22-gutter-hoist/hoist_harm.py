"""Does hoisting a mid-page gutter-crossing body line mis-bind its text?

Mirrors align_image() up to build_aligned_lines(), then re-runs each affected
page with a counterfactual plan in which mid-page hoisted lines stay in the
column their x1 falls in. For each affected line, compares Surya's own text to
the Gemini text the aligner bound to its box (difflib ratio, normalised).

Run from the repo root:
    ./.venv/bin/python docs/evidence/2026-09-22-gutter-hoist/hoist_harm.py

The "current" arm imports the live pipeline.align_ocr, so after a fix lands it
measures the fixed code. The numbers in README.md are from commit 8740c4a.
"""
import difflib
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pipeline.align_ocr as A  # noqa: E402
from utils.collections import eval_collections  # noqa: E402
from utils.column_utils import column_breaks  # noqa: E402

OK = 0.6  # text-agreement floor for "bound to its own text"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def gemini_for(surya: Path):
    # First model alphabetically. Any Gemini transcription serves here: the
    # question is geometric, not which model reads better.
    stem = surya.name[: -len("_surya.json")]
    cands = sorted(surya.parent.glob(f"{stem}_*.txt"))
    return cands[0] if cands else None


def load_gemini(p: Path):
    # Same parse as align_image(): drop delimiters, remember ad-block lines.
    lines, ads, in_ad = [], set(), False
    for raw in p.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if re.match(r"^=== .+ ===$", ln):
            in_ad = ln.startswith("=== ADVERTISEMENT")
            continue
        if in_ad:
            ads.add(len(lines))
        lines.append(ln)
    return lines, ads


def run(path: str):
    sp = Path(path)
    gp = gemini_for(sp)
    if gp is None:
        return None
    try:
        (_, _, w, _h), lines, _ = A.parse_surya(sp)
    except Exception:
        return None
    lines = A._expand_multiline_surya_lines(lines)
    lines = A.filter_short_lines(lines)
    lines = A.filter_margin_lines(lines, w)
    plan = A.plan_columns(lines, w)
    if plan is None:
        return None
    header, cols = plan
    breaks = column_breaks(lines, w)
    col_tops = [c[0]["bbox"][1] for c in cols if c]
    body_top, body_bot = min(col_tops), max(c[-1]["bbox"][3] for c in cols if c)
    # "Mid-page" = the line starts inside the column body, not above it.
    mid = [ln for ln in header if body_top < ln["bbox"][1] < body_bot]
    if not mid:
        return None
    gem, ads = load_gemini(gp)
    if not gem:
        return None
    frag = lambda b: ""
    fallback = A._legacy_reading_order(lines, w)

    def bind(groups):
        order = [ln for g in groups for ln in g]
        res, um = A.build_aligned_lines(order, gem, frag, ads,
                                        column_groups=groups, fallback_order=fallback)
        return {tuple(r["bbox"]): r.get("gemini_text", "") for r in res}, len(um)

    cur, cur_um = bind([header] + cols)

    mid_ids = {id(ln) for ln in mid}
    new_header = [ln for ln in header if id(ln) not in mid_ids]
    new_cols = [list(c) for c in cols]
    for ln in mid:
        ci = min(sum(1 for b in breaks if ln["bbox"][0] >= b), len(new_cols) - 1)
        new_cols[ci].append(ln)
    new_cols = [sorted(c, key=lambda ln: ln["bbox"][1]) for c in new_cols]
    cf, cf_um = bind([new_header] + new_cols)

    out = [(sim(ln["text"], cur.get(tuple(ln["bbox"]), "")),
            sim(ln["text"], cf.get(tuple(ln["bbox"]), "")),
            ln["text"][:60], cur.get(tuple(ln["bbox"]), "")[:60]) for ln in mid]
    base = [sim(ln["text"], cur.get(tuple(ln["bbox"]), "")) for c in cols for ln in c]
    return (sp.relative_to(ROOT / "output").parts[0], sp.name,
            len(header) / len(lines), len(new_header) / len(lines),
            out, base, cur_um, cf_um)


def main():
    kept, _ = eval_collections(ROOT / "output")
    files = [str(f) for d in kept for f in d.rglob("*_surya.json")
             if not any(p.startswith("_") for p in f.relative_to(ROOT / "output").parts)]
    print(f"{len(files)} surya files in {len(kept)} kept collections")
    t = Counter()
    fixed_by_coll = Counter()
    examples = []
    with ProcessPoolExecutor() as ex:
        for r in ex.map(run, files, chunksize=32):
            if r is None:
                continue
            coll, name, hfrac, cf_hfrac, out, base, cur_um, cf_um = r
            if hfrac > A._PARTITION_MAX_HEADER_FRAC:
                # Guard 1 reverts this page to whole-page alignment today.
                # Would it still, once the mid-page hoists are gone?
                t["guard1_pages"] += 1
                t["guard1_would_clear"] += cf_hfrac <= A._PARTITION_MAX_HEADER_FRAC
                continue
            t["pages"] += 1
            t["base_lines"] += len(base)
            t["base_ok"] += sum(s >= OK for s in base)
            for cs, fs, st, gt in out:
                t["mid_lines"] += 1
                t["cur_ok"] += cs >= OK
                t["cf_ok"] += fs >= OK
                if cs < OK <= fs:
                    t["fixed"] += 1
                    fixed_by_coll[coll] += 1
                    if len(examples) < 400:
                        examples.append((coll, name, st, gt))
                if cs >= OK > fs:
                    t["broken"] += 1
            t["cur_unmatched"] += cur_um
            t["cf_unmatched"] += cf_um
            t["pages_cf_better"] += cf_um < cur_um
            t["pages_cf_worse"] += cf_um > cur_um
    print(dict(t))
    if t["mid_lines"]:
        print(f"mid-page hoisted lines bound to own text: current {t['cur_ok']/t['mid_lines']:.1%}"
              f"  counterfactual {t['cf_ok']/t['mid_lines']:.1%}")
    if t["base_lines"]:
        print(f"baseline (column lines, same pages): {t['base_ok']/t['base_lines']:.1%}")
    print("\nlines fixed by counterfactual, by collection:")
    for c, n in fixed_by_coll.most_common(12):
        print(f"  {n:5d}  {c}")
    random.seed(1)
    print("\nexamples (surya text  ->  gemini text bound today):")
    for e in random.sample(examples, min(20, len(examples))):
        print("  ", e)


if __name__ == "__main__":
    main()
