# Evidence — choosing between column and whole-page alignment

**Date:** 2026-09-04
**Code:** `pipeline/align_ocr.py` — `_alignment_cost()` and the selection at the end of
`build_aligned_lines()`.
**Corpus:** `output/green_books_and_related/` (Green Books and related travel guides).

The headline, because it will be re-proposed by someone reasonable in a year:

> **Selecting between candidate alignments by coverage — "keep whichever boxes more
> text" — silently traded placement on 1 in 6 multi-column pages, putting text on the
> wrong line at more than four times the geometric disorder.**

**Erratum, 2026-10-17.** The harness that produced the tables below was flawed in two
ways:

- Its "whole-page" candidate called `build_aligned_lines(column_groups=None)`. That branch
  ignores `fallback_order` and aligns over column-major order. Production's whole-page
  path aligns over the legacy reading order, so the harness measured a candidate
  production never computes.
- It inferred which candidate the shipped rule picked from equal line counts.

Both are fixed in `harness.py`, `compare_criteria.py` and `known_cols_experiment.py`.

Read with that in mind:

- The 1-in-6 figure, the 250-page and 120-page tables, the six-corpus table and the
  known-column numbers are superseded by the corrected run in
  [Corrected results](#corrected-results-2026-10-17).
- The conclusion for travel guides and national/business directories stands, and is
  stronger.
- The city-directory "does not hold" finding was an artifact. On city directories the
  column path almost never completes, so the selection rule almost never runs.
- [Outcome on placement](#outcome-on-placement) measured a real re-align and is unaffected.

---

## Why a selection step exists at all

`plan_columns()` splits a page into columns and each column is aligned against its own
slice of the Gemini text. That exists because whole-page Needleman-Wunsch on a
two-column page binds lines *across* the gutter.

But column detection is not always right, and the static guards (an over-large hoisted
header, a Surya:Gemini line-ratio check) do not catch every bad split. On
`0030_5207647` in the 1947 Green Book (`4bea2040-1118-0132-673e-58d385a7b928`), same
inputs, three code paths:

| path | lines | unmatched |
|---|---|---|
| legacy whole-page order | **118** | **0** |
| new reading order, no columns | 115 | 3 |
| per-column path | **88** | **24** |

`plan_columns` sees two clean columns and the per-column alignment then wrecks a page
that aligned perfectly whole. Neither guard fires: the header group is small and the
Surya:Gemini ratio is 119:118, sitting exactly at 1.0 where no ratio guard would flag
it. Across that volume the effect was unmatched 62 → 259.

**Static prediction cannot settle this.** The page looks cleanly two-column by every
structural signal available *before* alignment runs. So compute both candidates and
measure the outcome — alignment is CPU-bound and fast, so this is cheap.

## The trap: measuring the outcome by coverage

The obvious rule is "return whichever leaves less Gemini text unboxed". It is wrong,
and not marginally.

Whole-page NW on a genuinely two-column page boxes *almost everything* — onto the wrong
lines. That scores well on unboxed count and badly on placement, which is the exact
failure per-column alignment was introduced to prevent. A coverage rule therefore
prefers, on real pages, the alignment the feature exists to avoid.

Measured over 250 multi-column pages, comparing the raw column candidate against the
whole-page candidate:

| | value |
|---|---|
| pages where a coverage rule picks whole-page | **42 / 250** |
| ...of those, whole-page is geometrically worse | **34 (81%)** |
| mean inversion rate, column candidate | 0.042 |
| mean inversion rate, whole-page candidate | **0.182** |

More than four times the disorder, on a sixth of multi-column pages.

## The fix

```
_alignment_cost(lines, unmatched) = len(unmatched) + inversion_rate(centres) * len(lines)
```

`inversion_rate` (normalised Kendall-tau distance between list order and column-major
geometric order) is the right second term because **it is independent of the objective
Needleman-Wunsch maximises**. NW maximises text similarity, so a text-similarity score
rewards its mistakes as readily as its successes. Geometry does not collude: if
alignment is correct, walking the aligned lines in list order visits boxes in
column-major, top-to-bottom order.

Scaling the rate by line count converts it into the same unit as the unboxed count, so
the two terms are commensurable. A bare rate would let a 400-line page's disorder count
the same as a 20-line page's.

Result on 120 multi-column pages:

| rule | picks whole-page | of those, geometrically worse | total unmatched |
|---|---|---|---|
| coverage-only | 46 | **34** (74%) | 237 |
| inversion-aware | 30 | **8** (27%) | 326 |

Bad picks down 76%, for 89 more unboxed lines across 120 pages. That is the intended
trade: give up a little coverage to avoid putting text on the wrong line.

## Corrected results (2026-10-17)

This is `compare_criteria.py` with the harness fixed. The whole-page candidate is now
production's revert path (`_build_line_aligned_lines` over `fallback_order`), and the
shipped rule's pick is read by comparing alignments, not counts. Only pages where the column
path actually ran can involve a choice; *ran* counts them. The same six corpora and sample
sizes were used.

| corpus | genre | multi-col pages | ran | coverage-only | inversion-aware | unmatched |
|---|---|---|---|---|---|---|
| green books | travel guides | 120 | 29 | 3/20 | **1/21** | 60 → 69 |
| morticians | national directory | 120 | 97 | 35/37 | **1/4** | 31 → 96 |
| woods directory | business directory | 120 | 74 | 15/34 | **1/23** | 93 → 121 |
| hearne's brooklyn | city directory | 34 | 4 | 0/3 | **0/4** | 2 → 2 |
| tulsa 1921 | city directory | 120 | 1 | 1/1 | **0/0** | 16 → 21 |
| tulsa 1922 | city directory | 120 | 1 | 1/1 | **0/0** | 2 → 16 |

Read the two rule columns as *bad picks / whole-page picks*, on the pages that ran.

- **Where the column path runs**, the coverage-only rule takes the more disordered whole-page
  alignment again and again: 35 of 37 picks on the morticians directory. The inversion-aware
  rule almost never does, in any corpus.
- **On city directories the column path completes on almost no page.** Guard 1 (the
  gutter-hoist over-collection) and failure to locate a column's start send them back to
  whole-page first. So the rule has nothing to choose between.

A gold-backed placement check on city directories reaches the same conclusion:
`docs/evidence/2026-10-17-column-alignment-panel/`.

## Where it does not hold

*Superseded by [Corrected results](#corrected-results-2026-10-17): this section's table and its
city-directory conclusion came from the flawed harness. The mechanism analysis that follows is
about `inversion_rate` itself and remains accurate, but it describes pages where, in
production, the column path rarely runs at all.*

**Everything above was measured on Green Books alone.** That corpus is 8 travel-guide
volumes, and the result does not carry to every genre in `output/`. Re-running the same
comparison across six collections (2026-09-14):

| corpus | genre | pages | coverage-only | inversion-aware | unmatched |
|---|---|---|---|---|---|
| green books | travel guides | 120 | 34/46 (74%) | **8/30 (27%)** | 237 → 326 |
| morticians | national directory | 120 | 11/27 (41%) | **2/68 (3%)** | 68 → 119 |
| woods directory | business directory | 120 | 30/39 (77%) | **4/46 (9%)** | 324 → 388 |
| hearne's brooklyn | **city directory** | 34 | 9/15 (60%) | **7/9 (78%)** | 17 → 37 |
| tulsa 1921 | **city directory** | 318 | 73/100 (73%) | **63/102 (62%)** | 976 → 1317 |
| tulsa 1922 | **city directory** | 400 | 65/109 (60%) | **90/168 (54%)** | 2911 → 3254 |

Read as *bad picks / total whole-page picks*. On travel guides and the two
national/business directories the criterion does what this document claims. On city
directories it largely does not: Tulsa 1921 improves marginally while costing 341 more
unboxed lines, and **Tulsa 1922 makes more bad picks in absolute terms than the coverage
rule it replaces** (90 against 65), because it reaches for whole-page far more often.

Two things make that worse rather than better as a signal:

- The judge here is `inversion_rate`, which is also a term the criterion optimises. A
  rule tuned on that signal ought to score well on this metric almost by construction,
  so underperforming on it is a strong negative result, not a marginal one.
- "Inversion-aware picks whole" is inferred from output-count equality, so it can
  miscount by a page or two. It cannot account for a 65 → 90 swing.

City directories are the genre CLAUDE.md singles out for Gemini reading *across*
columns, so this is not a corner case — it is close to the headline use case. Until it
is resolved, treat the criterion as **validated for travel guides and
national/business directories, unproven for city directories.**

### The mechanism, and a fix that does not survive checking

`inversion_rate` re-derives the column split itself, from x-centre gaps alone, keeping
gaps wider than 12% of the span. On a dense city-directory page — full-width headings,
continued lines, indented sub-entries — the centres form a continuum and no gap clears
12%, so **the metric reports one column for a page `plan_columns` already resolved into
two or more**. When it thinks the page is one column, correct order is pure
top-to-bottom, so a genuine column-major alignment (down column 1, back up to column 2)
looks maximally disordered and the read-across candidate wins. The metric inverts on
exactly the pages it most needs to be right about.

Measured 2026-09-14, multi-column pages per `plan_columns` where the metric sees one
column: green books **1/80**, Tulsa 1921 **11/80**, Tulsa 1922 **27/80**.

Two candidate explanations, one of which is wrong:

- **The `n_cols_max=4` cap.** It is not the cause. Raising it 4 → 8 changes the
  assignment on **0 of 60** Tulsa 1922 pages; the 12% gap threshold binds first, and
  the cap is never reached. (Verified with a positive control — the cap does change
  synthetic 6-column input — because identical output across three caps looks exactly
  like a patch that never applied.)
- **The 12% threshold.** This is the real constraint.

The obvious fix is to stop guessing: `build_aligned_lines` already holds the true layout
in `column_groups`, so feed it in rather than re-deriving it. Judged by geometry that
looks decisive — bad picks fall from 11/38 to 3/36 on green books, 10/33 to 1/42 on
Tulsa 1921, 9/50 to 1/57 on Tulsa 1922.

**Do not adopt that on this evidence.** The judge there is the known-column inversion
rate, which is the very quantity the fixed rule optimises — it is being scored by its
own metric, the trap this whole document exists to warn about. Scored instead on
text-vs-box agreement (the placement metric commit 2c66654 reported, differently shaped
from geometry), on the pages where the two rules disagree the *current* guessed-column
rule wins more often: green books 17 vs 9, Tulsa 1921 18 vs 15, Tulsa 1922 19 vs 10.

So the mechanism is real, the cap is a red herring, and the natural fix fails an
independent check. Settling this needs the non-circular judge — a placement audit via
`tools/audit_entry_bindings.py` (strong bindings ≥ 0.6 against the text at the assigned
box) over a re-aligned city-directory volume, the same way the "Outcome on placement"
table above was produced. That is hours of compute, not a quick check, and it is the
next thing worth doing here.

## Reproducing

```bash
# all six corpora above
./.venv/bin/python docs/evidence/2026-09-04-column-selection-criterion/compare_criteria.py

# or one, with a bigger sample
... compare_criteria.py --corpus 'output/tulsa_1922/**/*_aligned.json' --limit 400
```

The default is plural on purpose. The single-corpus default this script shipped with is
what produced the blind spot in "Where it does not hold".

`harness.py` mirrors `align_image()`'s call into `build_aligned_lines()`. **Both details
matter and both were got wrong first time:**

1. `align_image` passes `fallback_order=_legacy_reading_order(lines, page_w)` and
   `sorted_surya_lines = header + flattened columns`. An earlier harness passed
   column-major order as the fallback, which made every number meaningless — it
   reported the 1947 page as 115/3 and claimed the new rule picked whole-page on 221 of
   250.
2. The raw column candidate cannot be obtained from the current
   `build_aligned_lines`, because the selection is *inside* it — calling it measures the
   choice, not the candidates. `harness.py` materialises the pre-fix module out of git
   to get an honest column-only result. The commit is **pinned** (`PREFIX_COMMIT`), not
   `HEAD`, precisely because `HEAD` moves once the fix is committed — which it now has.
3. *(Added 2026-10-17.)* The whole-page candidate must be production's revert path:
   `_build_line_aligned_lines(fallback_order, …)`, which is `WHOLE` in `harness.py`.
   `build_aligned_lines(column_groups=None)` ignores `fallback_order` and aligns over the
   column-major order it was handed. That is item 1's mistake by another route, and it
   stood until 2026-10-17. Read the shipped rule's pick by comparing alignments, never
   counts, because two different alignments often bind the same number of lines.
   `results-2026-10-17.txt` holds the corrected run.

## The general lesson

Three separate results on this work were overturned, each time because a coverage-shaped
metric flattered a change that a placement- or geometry-shaped metric showed was neutral
or harmful. Coverage is the cheap metric and the easy one to fool. When judging an
alignment change, measure where boxes land, not how many there are.

---

## Outcome on placement

Coverage was never the thing to optimise; placement was. Measured against the
pre-re-align tree, bindings scoring >= 0.6 against the text at their assigned box:

| criterion | strong bindings | vs original | fragment churn |
|---|---|---|---|
| rollback (no re-align) | 108,312 | +0 | 0 |
| coverage-only selection | 108,042 | **-270** | ~17,800 |
| inversion-aware selection | 108,456 | **+144** | 15,400 |

The coverage-only rule made placement *worse than not re-aligning at all*, while
reporting a larger coverage gain (+3,507 substantive lines against +3,141). That is the
entire case for this evidence folder in two rows.

By detected column count (labelled from the OLD tree's geometry, so the criterion under
test cannot move a page between buckets): 1-column +24, 2-column +68, 3-column -22,
4+-column +74.

## A note on verification tooling

Two failures on this work came from trusting a signal that could not fail loudly:

- `find -newermt "-60 seconds"` is rejected by `bfs` (the `find` on this machine), which
  wants ISO 8601. It writes an error to stderr and the surrounding pipeline still reports
  `0` matches. A quiescence check built on it reported "quiet" while 2,750 files had been
  written in the preceding 15 minutes. Use `-mmin`, or compare `os.path.getmtime` in
  Python — and verify the check *discriminates* (a positive control that returns nonzero)
  rather than only that it returns zero when you expect zero.
- Completion was read off a `DONE` marker in a log rather than off the filesystem, and a
  downstream consumer was told the tree was settled while it was still being written.

A verification tool that errors to stderr while the pipeline reports success is worse
than no tool, because it manufactures confidence.

- A third instance, found on 2026-09-14: `compare_criteria.py` as first committed did
  `from validate2 import *`, and no `validate2` existed anywhere in the repo. The
  documented reproduction command had never run as written. The numbers it reported were
  real — they reproduce exactly once the import is corrected to `harness` — but nobody
  could have checked that, and the script sat in an evidence folder whose entire purpose
  is to let someone re-derive the result. **Run the reproduction command from the
  README, from a clean checkout, before committing an evidence set.**
