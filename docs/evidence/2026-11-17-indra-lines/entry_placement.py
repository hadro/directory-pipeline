#!/usr/bin/env python3
"""Where each extracted entry's box lands under each line detector, judged by what is printed there.

tools/audit_entry_bindings.py compares an entry with the Gemini text bound to its
box, so it checks the binder, not the detector: when alignment puts a Gemini line
on the wrong printed line, the entry follows the text and the audit calls it
right. This script judges the box against the page instead.

For every page with Surya lines, Indra lines (tools/indra_lines.py), Gemini text
and NER entries, Gemini's text is aligned onto each detector's lines in memory
(common.align), and the page's entries are bound with the shipped binder
(utils.entry_binding.bind_entries). Each bound entry is then scored by the share
of its identity words printed inside its box: the text of the lines whose centres
fall in the box, read by Surya and by Indra, whichever reading shares more. Using
both readings keeps either detector's reading quality from deciding the result.

  good      at least 2/3 of the entry's identity words are printed in its box
  partial   1/3 to 2/3 (an entry wrapped across two lines, or a neighbour with the
            same surname: directories repeat surnames, so this band is lenient)
  weak      under 1/3
  wrong     none
  unplaced  the binder refused the entry (counted apart; the bands are shares of
            the placed entries)

A bound box taller than 2x its detector's median line height is also counted as
tall: such a box holds several printed lines, so its score is lenient.

Configurations:

  published      the entry's own box as published: the aligned line on disk whose
                 canvas_fragment it carries, bound at extraction by the binder of
                 the day (Tulsa 1921's predate the 0.3 binder)
  surya          Surya at the aligner's shipped floor: the baseline. Since PR #14
                 that is the width-aware floor; result files from 2026-11-17 used the
                 15 px floor and matched the aligned JSON on disk (the wiring line)
  surya_nofloor  Surya with filter_short_lines' 15 px floor off
  indra_nofloor  Indra, floor off
  cascade        indra_nofloor, or surya_nofloor where common.cascade_pick says so

    ./.venv/bin/python docs/evidence/2026-11-17-indra-lines/entry_placement.py output/tulsa_1921/ \\
        --mirror-root output/_indra_lines_2048_nosplit
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FRAG, REPO, TALL, align, cascade_pick, page_stats, read_gemini  # noqa: E402
from compare_detectors import eligible_pages  # noqa: E402
from gold_panel import on_line  # noqa: E402

sys.path.insert(0, str(REPO))
import pipeline.align_ocr as A  # noqa: E402
from tools.indra_lines import DEFAULT_MIRROR, mirror_root_for  # noqa: E402
from utils.entry_binding import bind_entries, identity_tokens, tokens  # noqa: E402

SHIPPED_FLOOR = A._MIN_LINE_HEIGHT
CONFIGS = ("published", "surya", "surya_nofloor", "indra_nofloor", "cascade")
BANDS = ("good", "partial", "weak", "wrong", "unplaced")


def band(share: float | None) -> str:
    if share is None:
        return "unplaced"
    return ("wrong" if share == 0 else "weak" if share < 1 / 3
            else "partial" if share < 2 / 3 else "good")


def printed_share(entry: dict, box, readings) -> float:
    """Share of the entry's identity words in the best reading of *box*."""
    ent = set(identity_tokens(entry))
    best = 0.0
    for lines in readings:
        text = " ".join(ln.get("text", "") for ln in lines if on_line(ln["bbox"], box))
        best = max(best, len(ent & set(tokens(text))) / len(ent))
    return best


def aligned_on_disk(img: Path, entries: list[dict]) -> list[dict]:
    """Lines of the aligned JSON holding most of the page's published fragments
    (as audit_entry_bindings --resolve overlap)."""
    frags = {e.get("canvas_fragment") for e in entries}
    best, best_n = [], -1
    for p in img.parent.glob(f"{img.stem}_*_aligned.json"):
        try:
            lines = json.loads(p.read_text(encoding="utf-8")).get("lines", [])
        except (OSError, ValueError):
            continue
        n = len(frags & {ln.get("canvas_fragment") for ln in lines})
        if n > best_n:
            best, best_n = lines, n
    return best


def detector_boxes(path: Path, gl, ads, entries, width: int, floor: int, mask: bool = False, join: bool = False):
    A._MIN_LINE_HEIGHT = floor   # read at call time by filter_short_lines
    lines, bound, unmatched, _ = align(path, gl, ads, mask_pictures=mask, join=join)
    disagree = page_stats(lines, bound, unmatched, len(gl))["disagree"]
    out = [{"bbox": list(b), "gemini_text": t, "canvas_fragment": FRAG(b)} for b, t in bound]
    binds = bind_entries(entries, out, canvas_width=width)
    med = statistics.median([ln["bbox"][3] - ln["bbox"][1] for ln in lines]) if lines else 0
    boxes = [out[b.line_index]["bbox"] if b.line_index is not None else None for b in binds]
    return boxes, disagree, med, bound


def score_page(job):
    key, surya, indra, gem, ent_path, mask, join = job
    entries = [e for e in json.loads(ent_path.read_text(encoding="utf-8")).get("entries", [])
               if identity_tokens(e)]
    gl, ads = read_gemini(gem)
    if not entries or not gl:
        return None
    img = surya.parent / (surya.name[: -len("_surya.json")] + ".jpg")
    page_bbox, s_read, _ = A.parse_surya(surya, min_confidence=0.0)
    _, i_read, _ = A.parse_surya(indra, min_confidence=0.0)
    width = page_bbox[2]

    boxes, meds = {}, {}
    disk = aligned_on_disk(img, entries)
    on_disk = {(tuple(ln["bbox"]), ln.get("gemini_text", "")) for ln in disk}
    by_frag = {ln.get("canvas_fragment"): ln["bbox"] for ln in disk}
    boxes["published"] = [by_frag.get(e.get("canvas_fragment")) for e in entries]
    boxes["surya"], _, meds["surya"], s15 = detector_boxes(surya, gl, ads, entries, width, SHIPPED_FLOOR, mask, join)
    meds["published"] = meds["surya"]
    boxes["surya_nofloor"], s_dis, meds["surya_nofloor"], _ = detector_boxes(surya, gl, ads, entries, width, 0, mask, join)
    boxes["indra_nofloor"], i_dis, meds["indra_nofloor"], _ = detector_boxes(indra, gl, ads, entries, width, 0, mask, join)
    pick = cascade_pick(s_dis, i_dis)
    boxes["cascade"] = boxes[f"{pick}_nofloor"]
    meds["cascade"] = meds[f"{pick}_nofloor"]

    res = {"page": key, "pick": pick, "n": len(entries),
           "wiring": (len(set(s15) & on_disk), len(s15))}
    for c in CONFIGS:
        bands = Counter(band(None if b is None else printed_share(e, b, (s_read, i_read)))
                        for e, b in zip(entries, boxes[c]))
        bands["tall"] = sum(1 for b in boxes[c] if b and meds[c] and b[3] - b[1] > TALL * meds[c])
        res[c] = bands
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volumes", nargs="+", type=Path)
    ap.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR)
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--pages", metavar="GLOB", help="Only image filenames matching this glob")
    ap.add_argument("--csv", type=Path, help="Write one row per page here")
    ap.add_argument("--ner-model", metavar="SLUG",
                    help="Score {stem}_{SLUG}_entries.json (default: the first entries file by name)")
    ap.add_argument("--join-rows", action="store_true",
                    help="Join pieces of one printed row that Gemini wrote as one line (common.join_rows)")
    ap.add_argument("--mask-pictures", action="store_true",
                    help="Drop lines inside picture regions (picture_mask.py) before aligning, every configuration")
    args = ap.parse_args()

    jobs = []
    for vol in args.volumes:
        mirror = mirror_root_for(vol, args.mirror_root, args.output_root)
        for key, surya, indra, gem in eligible_pages(vol, mirror, None):
            if args.pages and not fnmatch.fnmatch(Path(key).name, args.pages):
                continue
            stem = surya.name[: -len("_surya.json")]
            ents = sorted(surya.parent.glob(f"{stem}_{args.ner_model or '*'}_entries.json"))
            if ents:
                jobs.append((key, surya, indra, gem, ents[0], args.mask_pictures, args.join_rows))
    if not jobs:
        sys.exit("No page has Surya lines, Indra lines, Gemini text and entries.")

    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = [r for r in pool.map(score_page, jobs, chunksize=4) if r]

    n = sum(r["n"] for r in rows)
    print(f"{len(rows)} pages, {n} entries with identity words. Cascade took Surya on "
          f"{sum(r['pick'] == 'surya' for r in rows)} pages.")
    print(f"Wiring: {sum(r['wiring'][0] for r in rows)} of {sum(r['wiring'][1] for r in rows)} "
          "lines of the in-memory Surya alignment are in the aligned JSON on disk.\n")
    print("Shares of placed entries:")
    print(f"{'':14s}{'placed':>8s}" + "".join(f"{b:>9s}" for b in (*BANDS[:4], "tall")) + f"{'unplaced':>10s}")
    for c in CONFIGS:
        tot = Counter()
        for r in rows:
            tot.update(r[c])
        placed = n - tot["unplaced"]
        print(f"{c:14s}{placed:8d}" + "".join(f"{tot[b] / placed * 100:8.1f}%" for b in (*BANDS[:4], "tall"))
              + f"{tot['unplaced']:10d}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["page", "pick", "entries"] + [f"{c}_{b}" for c in CONFIGS for b in (*BANDS, "tall")])
            for r in rows:
                w.writerow([r["page"], r["pick"], r["n"]] + [r[c][b] for c in CONFIGS for b in (*BANDS, "tall")])

    def bad(r, c):
        return r[c]["wrong"] + r[c]["weak"]
    for c in ("published", "surya_nofloor", "indra_nofloor", "cascade"):
        diff = sorted(rows, key=lambda r: bad(r, c) - bad(r, "surya"))
        d = [bad(r, c) - bad(r, "surya") for r in rows]
        print(f"\n{c} against surya, wrong+weak entries per page: "
              f"pages better {sum(x < 0 for x in d)} ({-sum(x for x in d if x < 0)} entries), "
              f"worse {sum(x > 0 for x in d)} ({sum(x for x in d if x > 0)} entries)")
        print("  most improved: " + ", ".join(f"{Path(r['page']).name} ({bad(r, 'surya')}->{bad(r, c)})"
                                              for r in diff[:5] if bad(r, c) < bad(r, "surya")))
        print("  most worsened: " + ", ".join(f"{Path(r['page']).name} ({bad(r, 'surya')}->{bad(r, c)})"
                                              for r in diff[::-1][:5] if bad(r, c) > bad(r, "surya")))


if __name__ == "__main__":
    main()
