"""Does feeding plan_columns' known layout into inversion_rate fix city directories?

assign_columns re-derives the column split from x-centre gaps alone, with a
12%-of-span threshold. On dense city-directory pages that re-derivation
collapses to a single column on a third of Tulsa 1922 pages that plan_columns
had already resolved into two or more. When the metric thinks the page is one
column, "correct" order is pure top-to-bottom — so a genuine column-major
alignment (down column 1, back up to column 2) looks maximally disordered, and
the read-across whole-page candidate looks better. The metric inverts.

But build_aligned_lines already HAS the true layout: column_groups. Use it.
"""
import sys, glob, random

sys.path.insert(0, "/Users/joshhadro/github/directory-pipeline/docs/evidence/2026-09-04-column-selection-criterion")
sys.path.insert(0, "/Users/joshhadro/github/directory-pipeline")

from utils.column_utils import bbox_centre, inversion_rate
from harness import load, prep, RAW, FRAG


def ranges_from_groups(groups):
    """(x_lo, x_hi) of each column group's line centres; header group first."""
    out = []
    for g in groups:
        xs = [bbox_centre(l["bbox"])[0] for l in g if "bbox" in l]
        out.append((min(xs), max(xs)) if xs else None)
    return out


def assign_from_groups(centres, ranges):
    """Assign each centre to the group whose x-range contains it, else nearest."""
    idx = []
    for x, _y in centres:
        best, best_d = 0, None
        for i, r in enumerate(ranges):
            if r is None:
                continue
            lo, hi = r
            d = 0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))
            if best_d is None or d < best_d:
                best, best_d = i, d
        idx.append(best)
    return idx


def inversion_known(lines, ranges):
    """inversion_rate, but with the column assignment supplied rather than guessed."""
    placed = [bbox_centre(l["bbox"]) for l in lines if l.get("bbox")]
    if len(placed) < 3:
        return 0.0
    cols = assign_from_groups(placed, ranges)
    order = sorted(range(len(placed)), key=lambda i: (cols[i], placed[i][1], placed[i][0]))
    rank = [0] * len(placed)
    for r, i in enumerate(order):
        rank[i] = r
    inv = sum(1 for i in range(len(placed)) for j in range(i + 1, len(placed))
              if rank[j] < rank[i])
    total = len(placed) * (len(placed) - 1) // 2
    return inv / total if total else 0.0


def cost(lines, unmatched, inv):
    return len(unmatched) + inv * len(lines)


CORPORA = [
    ("green books   (control)", "output/green_books_and_related/**/*_aligned.json"),
    ("morticians    (control)",
     "output/the_national_directory_of_morticians_nationaldirectornati/**/*_aligned.json"),
    ("tulsa 1921    (city dir)", "output/tulsa_1921/**/*_aligned.json"),
    ("tulsa 1922    (city dir)", "output/tulsa_1922/**/*_aligned.json"),
    ("hearne's bkln (city dir)",
     "output/hearnes_brooklyn_city_directory_for_hearnesbrooklync1852/**/*_aligned.json"),
]
LIMIT = 120

print(f"{'corpus':26} {'pages':>6} {'guessed cols':>14} {'known cols':>14}")
print(f"{'':26} {'':>6} {'bad/picks':>14} {'bad/picks':>14}")
print("-" * 68)

for label, pattern in CORPORA:
    random.seed(9)
    pages = glob.glob(pattern, recursive=True)
    random.shuffle(pages)
    n = g_whole = g_bad = k_whole = k_bad = 0
    for p in pages:
        if n >= LIMIT:
            break
        d = load(p)
        if not d:
            continue
        sl, W, gl, ad = d
        if len(gl) < 20:
            continue
        pr = prep(sl, W)
        if pr is None:
            continue
        n += 1
        lines, groups, fb = pr
        ranges = ranges_from_groups(groups)

        col, col_um = RAW(lines, gl, FRAG, ad, column_groups=groups, fallback_order=fb)
        whole, whole_um = RAW(lines, gl, FRAG, ad, column_groups=None, fallback_order=fb)

        # ground-truth-ish judge: the known-column inversion of each candidate
        truth_c = inversion_known(col, ranges)
        truth_w = inversion_known(whole, ranges)
        whole_is_worse = truth_w > truth_c + 0.01

        # rule A: current, guessed columns
        gc = cost(col, col_um, inversion_rate([bbox_centre(x["bbox"]) for x in col]))
        gw = cost(whole, whole_um, inversion_rate([bbox_centre(x["bbox"]) for x in whole]))
        if gw < gc:
            g_whole += 1
            if whole_is_worse:
                g_bad += 1

        # rule B: known columns
        kc = cost(col, col_um, truth_c)
        kw = cost(whole, whole_um, truth_w)
        if kw < kc:
            k_whole += 1
            if whole_is_worse:
                k_bad += 1

    if n:
        print(f"{label:26} {n:>6} {f'{g_bad}/{g_whole}':>14} {f'{k_bad}/{k_whole}':>14}")
