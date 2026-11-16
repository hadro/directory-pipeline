# Evidence — mid-page body lines hoisted as page-spanning headers

**Date:** 2026-09-22
**Code:** `pipeline/align_ocr.py` — `plan_columns()` (`spans_gutter`, `is_header`) and the
header-slice alignment in `build_aligned_lines()`; `utils/column_utils.py` —
`column_breaks()`. Measured at commit `8740c4a`.
**Corpus:** every production `*_surya.json` under `output/` (10,406 pages, 54 collections,
selected by `utils/collections.py`).
**Status:** measured, not fixed. The follow-up plan is `docs/plans/gutter-hoist-fix.md`.

The headline:

> **On 1,676 multi-column pages, 7,346 lines that sit inside the column body are treated
> as page-spanning headers. Leaving them in their column gives 450 of them back their own
> text (43 lose it) and boxes 453 more Gemini lines. Separately, 960 pages that Guard 1
> reverts to whole-page alignment only trip it because of these hoists.**

## How this was found

The COTe layout metric (Bourne, Simbeye & Govia, arXiv 2603.12718; `cotescore` on PyPI) was
assessed as an evaluation metric and rejected. The reasons are recorded in
`historical-ocr-eval/docs/plan.md`. Its **trespass** component, a box that crosses a
semantic boundary such as a column gutter, needs no ground truth when the boundary is the
page's own detected gutter, so it was run against `plan_columns` as a diagnostic.

## The mechanism

1. `column_breaks()` places each gutter at the **midpoint of a coverage valley**: a run of
   bins whose line coverage is at most 55% of the page's peak coverage. That is not a
   whitespace gap. On ragged-right text the right-hand tail of a column is already below
   55% coverage, so the valley starts inside the column and its midpoint can fall where
   long entries still have ink.
2. `spans_gutter()` is true for any line whose box straddles that midpoint. So a long
   body entry "crosses the gutter".
3. `is_header()` hoists any gutter-crossing line that is ≥30% of page width
   (`_HEADER_MIN_WIDTH_FRAC`), wherever it sits vertically. A long left-column entry on a
   two-column page is routinely 30–40% of page width.
4. `build_aligned_lines()` aligns the header group only against the Gemini lines that
   come **before the first column's start**. A hoisted entry from mid-page has no copy of
   its own text in that slice, so it is either left unboxed or bound to a heading. The
   entry's real Gemini line, in the column's slice, loses its box or takes a neighbour's.

## Measurement

`hoist_harm.py` mirrors `align_image()` exactly up to `build_aligned_lines()`: the same
`parse_surya`, `_expand_multiline_surya_lines`, `filter_short_lines`,
`filter_margin_lines` and ad-block parse. It then aligns each affected page twice:

- **current**: the plan as `plan_columns` returns it.
- **counterfactual**: mid-page header lines (top edge between the first column's top and
  the last column's bottom) moved back into the column their `x1` falls in.

The per-line score is `difflib` similarity between the line's **Surya** text and the
**Gemini** text bound to its box, with 0.6 or higher counting as "bound to its own text". Full output is in
`results.txt`.

| | current | counterfactual |
|---|---|---|
| mid-page hoisted lines bound to their own text (n = 7,346) | 70.1% | **75.6%** |
| lines whose own text is gained / lost | — | **+450 / −43** |
| Gemini lines left unboxed (all 1,676 pages) | 12,326 | **11,873** |
| pages with fewer / more unboxed lines | — | 184 / 67 |

Lines fixed, by collection: `green_books_and_related` 196, national directory of
morticians 119, `tulsa_1921` 32, `tulsa_1922` 27, Lain's Brooklyn street directory 17,
New Harlem Blue Book 17, Woods 15, the rest ≤11.

Typical fixed lines, as Surya text → Gemini text bound today:
`Miss W. A. Brown, 1306 Tatnall St.` → *(unboxed)*;
`Underwood—1303 S. Yellow Spring St.` → `OHIO`;
`HEYL FUNERAL HOME` → `OHIO—Continued 325`.

### The Guard 1 interaction

997 further pages hoist mid-page lines **and** exceed `_PARTITION_MAX_HEADER_FRAC` (10%),
so Guard 1 sends them to whole-page alignment today and the table above excludes them.
With the mid-page hoists removed, **960 of the 997 would pass Guard 1** and switch to the
per-column path, subject to the `_alignment_cost` selection. Whether that helps is
**not measured here**. The 2026-09-04 evidence says whole-page alignment on genuine
multi-column pages carries about 4× the geometric disorder, which suggests it would, but
the pages Guard 1 catches are by construction the ones whose layout was already confusing
the planner.

## Caveats

- **Small in absolute terms.** 450 lines across the corpus. The Guard-1 pages are the
  larger and unmeasured part.
- **The yardstick is noisy.** Surya's text is the reference, and it contains OCR errors.
  Ordinary column lines on the same pages score only 68.7% by the same rule. Read the
  table as a before/after difference on identical inputs, not as an accuracy.
- **The counterfactual is naive.** It un-hoists every mid-page line. The 43 lines that
  get worse are most likely genuine mid-page headings (a state name or section banner that
  spans both columns) that belong in the header group. A real fix has to keep them.
- **Unboxed count is coverage, not placement.** It cannot show whether a box landed on the
  right line. Any fix must also be run through `tools/audit_entry_bindings.py`.
- **One Gemini model per page** is read (the first `*_{model}.txt` alphabetically). The
  question is geometric, so the choice of model should not matter much, but it is not
  necessarily the model production used.

## Reproduce

```bash
./.venv/bin/python docs/evidence/2026-09-22-gutter-hoist/hoist_harm.py
```

About three minutes on all cores. It needs the gitignored `output/` data. It imports the
live `pipeline.align_ocr`, so after a fix the "current" column measures the fix.
