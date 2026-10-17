#!/usr/bin/env python3
"""Compare a volume's alignment on disk with a fresh one, without gold.

The judge is the box-vs-Surya disagreement: for each bound line, does the
Gemini text match what Surya itself read in that box? (head similarity < 0.5
counts as a disagreement). Surya read those pixels once, so its text at a box
is the same for every candidate alignment. Read the result per page against
the volume's own distribution: poor Surya text also scores low.

Three alignments per sampled page:
  disk     the *_aligned.json already in output/
  whole    a fresh whole-page alignment with the live code
  shipped  a fresh build_aligned_lines with the live code

shipped vs whole isolates the per-column feature; disk vs shipped is the net
effect of re-aligning. In memory only: nothing is written to output/.

    ./.venv/bin/python docs/evidence/2026-10-17-column-alignment-panel/box_vs_surya.py
    ... box_vs_surya.py --corpus 'output/tulsa_1922/**/*_aligned.json' --limit 150
"""
import argparse
import glob
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from placement_panel import REPO, candidates, head, prepare  # noqa: E402


def disagreement(lines_out, surya_at):
    sims = [head(e.get("gemini_text", ""), surya_at[tuple(e["bbox"])])
            for e in lines_out if tuple(e["bbox"]) in surya_at]
    return (sum(s < 0.5 for s in sims) / len(sims), len(sims)) if sims else (None, 0)


def bucket(x):
    return "failing" if x > 0.5 else ("borderline" if x > 0.2 else "clean")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="output/tulsa_1921/**/*_aligned.json")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    aligned = sorted(p for p in glob.glob(str(REPO / args.corpus), recursive=True)
                     if "_gemini" in Path(p).name)
    random.seed(args.seed)
    random.shuffle(aligned)
    rows = []
    for ap_ in aligned:
        if len(rows) >= args.limit:
            break
        ap_ = Path(ap_)
        stem = ap_.name[: -len("_aligned.json")]
        page = stem.rsplit("_gemini", 1)[0]
        img, gem = ap_.parent / f"{page}.jpg", ap_.parent / f"{stem}.txt"
        surya = ap_.parent / f"{page}_surya.json"
        if not (img.exists() and gem.exists() and surya.exists()):
            continue
        try:
            disk = json.loads(ap_.read_text(encoding="utf-8")).get("lines", [])
            lines, ordered, groups, fallback, gl, ads, _ = prepare(img, gem)
        except Exception:  # noqa: BLE001 — unreadable page: not a sample
            continue
        if len(gl) < 5:
            continue
        raw = json.loads(surya.read_text(encoding="utf-8")).get("lines", [])
        surya_at = {tuple(ln["bbox"]): ln.get("text", "") for ln in raw if ln.get("bbox")}
        surya_at.update({tuple(ln["bbox"]): ln.get("text", "") for ln in lines})
        res, ran = candidates(ordered, groups, fallback, gl, ads)
        d_disk, n_disk = disagreement(disk, surya_at)
        d_whole, n_whole = disagreement(res["whole"][0], surya_at)
        d_ship, n_ship = disagreement(res["shipped"][0], surya_at)
        if None in (d_disk, d_whole, d_ship):
            continue
        rows.append(dict(page=page, disk=d_disk, whole=d_whole, ship=d_ship, n_disk=n_disk,
                         n_whole=n_whole, n_ship=n_ship, plan=groups is not None, ran=ran))

    print(f"{args.corpus}: {len(rows)} pages sampled (seed {args.seed})\n")
    for key, nkey, label in (("disk", "n_disk", "on disk"), ("whole", "n_whole", "fresh whole-page"),
                             ("ship", "n_ship", "fresh shipped")):
        b = {"failing": 0, "borderline": 0, "clean": 0}
        for r in rows:
            b[bucket(r[key])] += 1
        print(f"  {label:18s} failing {b['failing']:3d}  borderline {b['borderline']:3d}  "
              f"clean {b['clean']:3d}   mean disagreement {statistics.mean(r[key] for r in rows):.3f}"
              f"   mean bound lines {statistics.mean(r[nkey] for r in rows):.0f}")
    print("\nPaired, per page (5-point margin):")
    for a, b, label in (("whole", "ship", "shipped vs fresh whole-page"),
                        ("disk", "ship", "shipped vs on disk")):
        better = sum(1 for r in rows if r[b] < r[a] - 0.05)
        worse = sum(1 for r in rows if r[b] > r[a] + 0.05)
        print(f"  {label:30s} better {better:3d}  worse {worse:3d}  same {len(rows) - better - worse:3d}")
    print(f"\nColumn plan on {sum(r['plan'] for r in rows)} pages; "
          f"column path ran on {sum(r['ran'] for r in rows)}.")


if __name__ == "__main__":
    main()
