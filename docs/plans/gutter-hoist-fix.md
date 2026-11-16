# Deferred: stop hoisting mid-page body lines as page-spanning headers

**Created 2026-09-23.** Suggested follow-up branch (working name `gutter-hoist`). Not
started. The measurement behind every number below is
`docs/evidence/2026-09-22-gutter-hoist/`. Read its README first.

## The problem in one paragraph

`plan_columns()` hoists any line that crosses a detected gutter and is ≥30% of page width
into the header group, **wherever it sits on the page**. Gutters are placed at the midpoint
of a coverage valley rather than in actual whitespace, so on ragged-right pages long body
entries "cross" them. The header group is aligned only against the Gemini text before the
first column, so a hoisted mid-page entry cannot find its own text.

## What is already known, and what is new

The over-collection itself was already known. `test_narrow_head_declined_when_hoist_already_over_collects`
says in so many words that Green Book listing pages "already hoist body lines past the
partition guard". The design relied on Guard 1 (`_PARTITION_MAX_HEADER_FRAC`, 10%) to
catch it. What was not known:

1. **Below Guard 1 it is silent.** 1,676 pages hoist 7,346 mid-page lines without
   tripping the guard. Keeping those lines in their column gives 450 of them back their
   own text (43 lose it) and boxes 453 more Gemini lines.
2. **Above Guard 1 it costs the page per-column alignment.** 960 of the 997 pages Guard 1
   reverts would pass it once the spurious hoists are gone. Nobody has measured what
   those pages would gain.
3. **The existing test name overstates its coverage.** `test_mid_page_gutter_crossing_line_is_not_hoisted`
   exercises a 27.8%-wide line, which is below the strict rule's 30% threshold. Only the
   narrow-head path is guarded against mid-page lines. The synthetic `_page()` fixture also
   uses uniform-width columns, so it cannot produce a ragged-right valley at all.

## Suggested branch work, in order

### 1. Pin the failure in tests first

In `tests/test_column_partition.py`:

- A two-column page with **ragged-right** line widths, plus one long left-column entry
  (≥30% page width) in mid-page that reaches past the valley midpoint, with a right-column
  line level with it. It must stay in column 0. This fails today.
- A genuine mid-page heading: wide and centred, with both columns ending above it and
  resuming below it. It must still be hoisted. This protects the 43 lines the naive
  counterfactual broke.
- Rename or tighten `test_mid_page_gutter_crossing_line_is_not_hoisted` so its name
  matches the narrow case it actually covers.

### 2. Fix `is_header()`: require a band of its own (option A, recommended)

A genuine page-spanning heading has **nothing beside it**: the columns stop above it and
resume below it. A long body entry has a line from the neighbouring column level with it.
So for a mid-page gutter-crossing line (top edge below `body_top`), hoist only when no
line from any other column overlaps it vertically.

Why this rather than a threshold:
- It adds no new constant to tune. It is a geometric property of what a heading *is*.
- It separates the two populations the naive counterfactual lumped together. That is
  where the 43 regressions should come back.
- It should also treat the Guard-1 pages differently from the counterfactual. On a
  genuinely single-column listing mistaken for two columns, wide lines really are alone
  in their band, so they would still be hoisted and Guard 1 would still fire. Only the
  spurious hoists stop.

Edge case to check against real pages: the bottom of a column where the neighbouring
column has already ended. A long entry there has nothing beside it and would be
hoisted. One candidate refinement: also require body lines *below* it in at least two
columns, because a new section starts after a real heading. Measure before adding it.

Leave the top-band paths (`body_top`, `is_narrow_running_head`) unchanged. They are
tested and not implicated.

### 3. Measure, in this order

1. `./.venv/bin/python -m pytest tests/test_column_partition.py tests/test_column_utils.py tests/test_alignment_cost.py`
2. `docs/evidence/2026-09-22-gutter-hoist/hoist_harm.py` before and after. The "current"
   arm imports live code, so after the fix it measures the fix. Expect `mid_lines` to
   fall sharply. What remains should be genuine headings, which is worth spot-checking.
3. **The Guard-1 pages, as a separate question.** For the ~960 pages that would switch
   from whole-page to per-column alignment, score both candidates with the inversion-rate
   harness in `docs/evidence/2026-09-04-column-selection-criterion/`. This is the
   larger and riskier population. If the per-column path is worse there, the fix
   should ship with Guard 1 still measured on the *old* hoist set, and that decision
   should be recorded.
4. `tools/audit_entry_bindings.py` after re-aligning the most affected volumes
   (`green_books_and_related`, the morticians directory, `tulsa_1921`/`tulsa_1922`), using
   `--resolve coverage` because the fragments will have moved. Unboxed count is coverage, not
   placement, and cannot sign off a binding change on its own.

### 4. Roll out (no Gemini spend)

Re-alignment and re-binding are both CPU-only. No OCR or NER re-run is needed.

```bash
python pipeline/align_ocr.py output/<collection> --force
python tools/patch_canvas_fragments.py output/<collection> --aligned-model <slug> --dry-run
python tools/patch_canvas_fragments.py output/<collection> --aligned-model <slug>
pipeline postprocess output/<collection>
```

Then republish the combined CSV to the companion `green-books` repo, since
`canvas_fragment` values change.

## Options considered and not recommended first

- **B: fix the gutter position** in `column_breaks()`, for example at the deepest bin of the valley
  or at the right edge of the left column's ink. That addresses the root cause, but
  `column_breaks` is shared with `surya_detect`'s column-count report by design
  (`utils/column_utils.py` docstring), so the blast radius is every multi-column page and
  both stages. Revisit only if option A leaves a residue that traces back to gutter
  placement.
- **C: horizontal bands.** Even a correctly identified mid-page heading is emitted *ahead
  of all columns*, while Gemini reads it in place. The faithful model splits the page at
  full-width headings into bands, each with its own columns. This is the long-term shape
  and a much larger change. Out of scope here, but option A's "band of its own" test is
  the first building block of it.

## Acceptance

- The new tests pass. Existing column, alignment-cost and column-utils tests are unchanged.
- `hoist_harm.py` shows the mid-page hoist count falling, with no loss on the lines that
  remain hoisted.
- The Guard-1 population is measured and a decision recorded either way.
- `audit_entry_bindings.py` on the re-aligned volumes shows no net increase in
  `text_zero`.
- Findings go into `docs/evidence/` next to the 2026-09-22 measurement.
