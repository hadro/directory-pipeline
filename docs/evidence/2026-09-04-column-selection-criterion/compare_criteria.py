#!/usr/bin/env python3
"""Compare the coverage-only and inversion-aware selection criteria.

For each corpus, scores multi-column pages three ways — the raw per-column
candidate, the whole-page candidate, and whatever the current
``build_aligned_lines`` actually returns — and reports how often each rule
picks whole-page, and how often that pick was the geometrically worse of the
two candidates.

Run over every default corpus:

    ./.venv/bin/python docs/evidence/2026-09-04-column-selection-criterion/compare_criteria.py

Or point it somewhere specific:

    ... compare_criteria.py --corpus 'output/tulsa_1922/**/*_aligned.json' --limit 400

Defaults span several genres on purpose. The first version of this evidence set
measured only Green Books, and the criterion it endorsed turned out to behave
quite differently on city directories — see "Where it does not hold" in
README.md. A single-corpus run is what produced that blind spot, so the default
is now plural.
"""
import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import FIXED, FRAG, RAW, WHOLE, bbox_centre, inversion_rate, load, prep  # noqa: E402

# (label, glob). Kept short enough to run in a couple of minutes.
DEFAULT_CORPORA = [
    ("green books (travel guides)", "output/green_books_and_related/**/*_aligned.json"),
    ("morticians (national directory)",
     "output/the_national_directory_of_morticians_nationaldirectornati/**/*_aligned.json"),
    ("woods (business directory)", "output/woods_directory_73644404/**/*_aligned.json"),
    ("tulsa 1921 (city directory)", "output/tulsa_1921/**/*_aligned.json"),
    ("tulsa 1922 (city directory)", "output/tulsa_1922/**/*_aligned.json"),
    ("hearne's brooklyn (city directory)",
     "output/hearnes_brooklyn_city_directory_for_hearnesbrooklync1852/**/*_aligned.json"),
]


def score(pattern: str, limit: int, seed: int = 9) -> dict | None:
    """Score up to *limit* multi-column pages under *pattern*."""
    random.seed(seed)
    pages = glob.glob(pattern, recursive=True)
    random.shuffle(pages)

    n = ran = drift = 0
    cov_whole = cov_bad = new_whole = new_bad = unm_cov = unm_new = 0
    inv_c_sum = inv_w_sum = 0.0

    for p in pages:
        if n >= limit:
            break
        d = load(p)
        if not d:
            continue
        sl, W, gl, ad = d
        if len(gl) < 20:
            continue
        pr = prep(sl, W)
        if pr is None:          # plan_columns saw a single column
            continue
        n += 1
        lines, groups, fb = pr

        col, col_um = RAW(lines, gl, FRAG, ad, column_groups=groups, fallback_order=fb)
        whole, whole_um = WHOLE(fb, gl, FRAG, ad)
        if col == whole:
            # The column path reverted (a guard tripped, or a column could not
            # be located), so neither rule had a choice to make on this page.
            continue
        ran += 1
        inv_c = inversion_rate([bbox_centre(x["bbox"]) for x in col])
        inv_w = inversion_rate([bbox_centre(x["bbox"]) for x in whole])
        inv_c_sum += inv_c
        inv_w_sum += inv_w
        # "Geometrically worse" means the whole-page candidate is the more
        # disordered of the two, by a margin wide enough not to be noise.
        whole_is_worse = inv_w > inv_c + 0.01

        if len(whole_um) < len(col_um):     # a coverage-only rule takes whole-page
            cov_whole += 1
            unm_cov += len(whole_um)
            if whole_is_worse:
                cov_bad += 1
        else:
            unm_cov += len(col_um)

        fixed, fixed_um = FIXED(lines, gl, FRAG, ad, column_groups=groups, fallback_order=fb)
        unm_new += len(fixed_um)
        # The current code returns one candidate or the other; compare the
        # alignments themselves. Matching counts instead (as before 2026-10-17)
        # miscounted every page where two different alignments bound the same
        # number of lines.
        if fixed == whole:
            new_whole += 1
            if whole_is_worse:
                new_bad += 1
        elif fixed != col:
            drift += 1      # neither candidate: the pinned module has drifted

    if not n:
        return None
    return dict(pages=n, ran=ran, drift=drift, found=len(pages),
                inv_col=inv_c_sum / max(ran, 1), inv_whole=inv_w_sum / max(ran, 1),
                cov_whole=cov_whole, cov_bad=cov_bad, unm_cov=unm_cov,
                new_whole=new_whole, new_bad=new_bad, unm_new=unm_new)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", metavar="GLOB",
                    help="corpus glob; repeatable. Defaults to all six.")
    ap.add_argument("--limit", type=int, default=120,
                    help="max multi-column pages to score per corpus (default 120)")
    args = ap.parse_args()

    corpora = ([(g, g) for g in args.corpus] if args.corpus else DEFAULT_CORPORA)

    print(f"{'corpus':36} {'pages':>6} {'ran':>5} {'coverage-only':>16} {'inversion-aware':>16}")
    print(f"{'':36} {'':>6} {'':>5} {'bad/picks':>16} {'bad/picks':>16}")
    print("-" * 84)
    rows = []
    for label, pattern in corpora:
        r = score(pattern, args.limit)
        if r is None:
            print(f"{label:36} {'—':>6}   (no multi-column pages found)")
            continue
        rows.append((label, r))
        cov = "{}/{}".format(r["cov_bad"], r["cov_whole"])
        new = "{}/{}".format(r["new_bad"], r["new_whole"])
        print(f"{label:36} {r['pages']:>6} {r['ran']:>5} {cov:>16} {new:>16}")

    print()
    for label, r in rows:
        print(f"{label}:")
        if r["drift"]:
            print(f"    WARNING: {r['drift']} page(s) matched neither candidate (module drift)")
        print(f"    mean inversion   column {r['inv_col']:.3f}   whole-page {r['inv_whole']:.3f}"
              "   (pages where the column path ran)")
        print(f"    total unmatched  coverage-only {r['unm_cov']}   inversion-aware {r['unm_new']}")


if __name__ == "__main__":
    main()
