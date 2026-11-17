#!/usr/bin/env python3
"""Score Surya's and Indra's line boxes against the hand-gold panel, one judge for both.

The 2026-10-17 judge asks, of each gold row's Surya box, whether the Gemini text
bound to it matches the gold. It cannot score another detector, whose boxes are
not Surya's. This judge asks the reverse, the same way for every detector:

  correct   a box carrying a Gemini line that matches the gold row (similarity
            >= 0.60) lies on the gold row's line
  wrong     matching text is bound, but only to boxes elsewhere
  unboxed   no bound Gemini line matches the gold row

"On the line": the box's vertical centre lies within the gold box's vertical
extent (+-2 px), and the two overlap horizontally by at least half the narrower
width, which separates the columns of a page.

The gold boxes are Surya's (recovered as placement_panel.py recovers them), so
the frame is Surya's segmentation. Where Surya's gold box is a tall region,
"on the line" is lenient for both detectors: such pages are flagged ``T`` and
reported apart, as are the panel's suspect pages (``*``). The panel's
polk1933bk pages, whose Surya boxes are all regions, cannot be scored this way
at all; look at them instead.

The same rows are scored three ways: Surya by the published judge (box to text;
it should reproduce the panel's "shipped" column, which checks the wiring),
Surya by this judge, and Indra by this judge.

    ./.venv/bin/python docs/evidence/2026-11-17-indra-lines/gold_panel.py --list-pages > /tmp/gold_pages.txt
    ./.venv/bin/python tools/indra_lines.py $(cat /tmp/gold_pages.txt)
    ./.venv/bin/python docs/evidence/2026-11-17-indra-lines/gold_panel.py

The gold panel is unpublished and stays in city-directory-extraction; this
script prints counts and page names only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BORDERLINE, REPO, TALL, align, cascade_pick, page_stats, pick_gemini, read_gemini,
)
from placement_panel import (  # noqa: E402
    CORRECT_MIN, DROP, GEM_MIN, NOT_PANEL, SUSPECT, head, recover_boxes, sim,
)

sys.path.insert(0, str(REPO))
from tools.indra_lines import DEFAULT_MIRROR  # noqa: E402

TOL = 2
JUDGES = ("surya_pub", "surya", "indra", "cascade")


def on_line(box, gold) -> bool:
    cy = (box[1] + box[3]) / 2
    if not (gold[1] - TOL <= cy <= gold[3] + TOL):
        return False
    overlap = min(box[2], gold[2]) - max(box[0], gold[0])
    return overlap >= 0.5 * min(box[2] - box[0], gold[2] - gold[0])


def text_to_box(row: str, gold_box, bound) -> str:
    """This judge: where did the Gemini text matching *row* get its box?"""
    boxes = [b for b, t in bound if sim(row, t) >= CORRECT_MIN]
    if not boxes:
        return "unboxed"
    return "correct" if any(on_line(b, gold_box) for b in boxes) else "wrong"


def box_to_text(row: str, proc_box, bound) -> str:
    """The published judge: what Gemini text did the gold row's box get?"""
    t = dict(bound).get(proc_box)
    if t is None:
        return "unboxed"
    return "correct" if sim(row, t) >= CORRECT_MIN else "wrong"


def load_gold(gold_dir: Path, output_root: Path):
    files = sorted(p for p in gold_dir.glob("*_eval.jsonl") if not any(k in p.name for k in NOT_PANEL))
    if not files:
        sys.exit(f"No gold panel under {gold_dir}. It lives in the city-directory-extraction "
                 "repo and is not published; pass --gold-dir.")
    pages = defaultdict(list)
    for f in files:
        vol = f.stem.replace("_eval", "")
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                pages[(vol, d["context"].get("image", ""))].append(d["raw_line"])
    index = {}
    for f in output_root.rglob("*.jpg"):
        if "/_" not in str(f) and "/.claude/" not in str(f):
            index.setdefault(f.name, []).append(f)
    return pages, index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-dir", type=Path, default=REPO.parent / "city-directory-extraction" / "data")
    ap.add_argument("--output-root", type=Path, default=REPO / "output")
    ap.add_argument("--mirror-root", type=Path, default=REPO / DEFAULT_MIRROR)
    ap.add_argument("--indra-min-conf", type=float, default=0.35,
                    help="parse_surya's floor for Indra lines (default 0.35, the pipeline's)")
    ap.add_argument("--min-line-height", type=int, metavar="PX",
                    help="Override align_ocr._MIN_LINE_HEIGHT (15) for both detectors. The floor "
                         "drops lines shorter than it; Indra's tight boxes on small type are 12-14 px")
    ap.add_argument("--list-pages", action="store_true",
                    help="Print the panel's page images, for tools/indra_lines.py, and stop")
    args = ap.parse_args()
    if args.min_line_height is not None:
        import pipeline.align_ocr as A
        A._MIN_LINE_HEIGHT = args.min_line_height   # read at call time by filter_short_lines

    pages, index = load_gold(args.gold_dir, args.output_root)
    agg = {k: {j: defaultdict(int) for j in JUDGES} for k in ("all", "clean", "flagged")}
    skipped = defaultdict(int)
    picks = defaultdict(int)
    listed = []
    if not args.list_pages:
        print(f"{'page':31s} rows | " + " | ".join(f"{j:>9s} ok/wr/ub" for j in JUDGES))
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
        if args.list_pages:
            listed.append(str(img))
            continue
        gem = pick_gemini(img)
        if gem.read_text(encoding="utf-8", errors="replace").lstrip().startswith("[OCR FAILED"):
            skipped["Gemini OCR failed on the page"] += len(raws)
            continue
        surya_json = img.parent / f"{img.stem}_surya.json"
        indra_json = (args.mirror_root / img.parent.resolve().relative_to(args.output_root.resolve())
                      / f"{img.stem}_surya.json")
        if not indra_json.exists():
            skipped["no Indra lines (run tools/indra_lines.py on --list-pages)"] += len(raws)
            continue

        gl, ads = read_gemini(gem)
        s_lines, s_bound, s_unm, _ = align(surya_json, gl, ads)
        i_lines, i_bound, i_unm, _ = align(indra_json, gl, ads, args.indra_min_conf)
        pick = cascade_pick(page_stats(s_lines, s_bound, s_unm, len(gl))["disagree"],
                            page_stats(i_lines, i_bound, i_unm, len(gl))["disagree"])
        picks[pick] += 1
        ecols, raw_hits, width = recover_boxes(surya_json, raws)

        # The panel's fair rows, so the published judge reproduces its numbers.
        fair = []
        for r, raw in zip(raws, raw_hits):
            if raw is None:
                skipped["no monotonic match"] += 1
                continue
            rb = raw["bbox"]
            inside = [ln for ln in s_lines
                      if rb[0] - 2 <= (ln["bbox"][0] + ln["bbox"][2]) / 2 <= rb[2] + 2
                      and rb[1] - 2 <= (ln["bbox"][1] + ln["bbox"][3]) / 2 <= rb[3] + 2]
            if not inside:
                skipped["box dropped by aligner preprocessing"] += 1
                continue
            if max((sim(r, g) for g in gl), default=0.0) < GEM_MIN:
                skipped["text not in Gemini output"] += 1
                continue
            proc = max(inside, key=lambda ln: head(r, ln.get("text", "")))
            fair.append((r, tuple(rb), tuple(proc["bbox"])))

        heights = [ln["bbox"][3] - ln["bbox"][1] for ln in s_lines] or [0]
        med = statistics.median(heights)
        tall_gold = med and any(g[3] - g[1] > TALL * med for _, g, _ in fair)
        suspect = any(s in vol for s in SUSPECT)
        part = "flagged" if (suspect or tall_gold) else "clean"

        cells = []
        for j in JUDGES:
            c = defaultdict(int)
            for r, gold_box, proc_box in fair:
                if j == "surya_pub":
                    k = box_to_text(r, proc_box, s_bound)
                else:
                    det = pick if j == "cascade" else j
                    k = text_to_box(r, gold_box, s_bound if det == "surya" else i_bound)
                c[k] += 1
                agg["all"][j][k] += 1
                agg[part][j][k] += 1
            cells.append(f"{c['correct']:4d}/{c['wrong']:2d}/{c['unboxed']:2d}")
        flag = ("*" if suspect else " ") + ("T" if tall_gold else " ")
        print(f"{flag}{(vol + '/' + img_name[:12])[:29]:29s} {len(fair):4d} | "
              + " | ".join(f"{x:>18s}" for x in cells) + f" {pick[0]}")

    if args.list_pages:
        print("\n".join(listed))
        return

    def fmt(d):
        n = sum(d.values())
        return ("-" if not n else f"n={n:5d}  correct {d['correct'] / n * 100:5.1f}%  "
                f"wrong {d['wrong'] / n * 100:4.1f}%  unboxed {d['unboxed'] / n * 100:4.1f}%")
    for key, title in (("all", "All scored rows"),
                       ("clean", "Unflagged pages"),
                       ("flagged", "Flagged pages (* suspect Surya, T tall gold boxes): judge is lenient")):
        print(f"\n{title}")
        for j in JUDGES:
            print(f"  {j:10s} {fmt(agg[key][j])}")
    print(f"\ncascade: Indra, then Surya where Indra's disagreement is over {BORDERLINE} and "
          f"Surya's is lower. Pages taking Indra {picks['indra']}, Surya {picks['surya']} "
          "(last column: i/s)")
    print("Not scored:", dict(skipped))


if __name__ == "__main__":
    main()
