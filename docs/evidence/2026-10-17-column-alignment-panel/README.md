# Evidence — per-column alignment on city directories

**Date:** 2026-10-17
**Code:** `pipeline/align_ocr.py`: `plan_columns()`, `build_aligned_lines()` and its
guards, and `_alignment_cost()`.
**Corpus:**

- The hand-gold panel in `city-directory-extraction` (`data/*_eval.jsonl`): 39 pages of
  New York and Brooklyn directories, 1786–1933, 1,766 scored lines.
- A 150-page sample of `output/tulsa_1921`.

The headline:

> **On city directories the per-column path rarely runs. It completed on 4 of 19
> multi-column panel pages and on 1 of 59 sampled Tulsa 1921 pages. Where it ran it was
> neutral to slightly worse: 4 of 217 lines. The placement errors that remain sit on dense
> pages it never reached, and those pages are Surya segmentation failures.**

This settles the 0.3 question for city directories: shipping it is low-risk and, so far,
without upside. The 2026-09-04 column-selection evidence had suggested the opposite, from
a flawed harness; see its erratum.

## Method

**Ground truth.** Each panel row is a human-verified transcription of one printed line.
The gold editor (`make_gold_tool.py`) builds rows by walking a page's Surya lines in
column-major order, and labellers only delete rows. So the gold is an order-preserving
subsequence of that walk.

Each row's Surya box is recovered by a monotonic DP alignment against the walk. The
editor's column-ordering code is copied verbatim, and its column count is chosen by best
fit. Independent per-row matching recovered only 40–80% of rows on dense pages, because of
repeated surnames.

**The judge.** Did the aligner bind, to that box, the Gemini line matching the verified
text (similarity ≥ 0.6, tolerant of wrapped entries)? Neither the box nor the gold text
can move with the alignment choice, so the judge cannot favour a candidate. The gold is
Surya-seeded, which rules it out for ranking Surya's text, but not for judging placement:
Surya's boxes are the fixed frame here.

**Candidates.** All four come from the live code, with `align_image()`'s preprocessing
reproduced exactly:

- whole-page over the legacy reading order
- the per-column path with its guards and no selection
- selection by unboxed count alone
- `build_aligned_lines()` as shipped

**Not scored:**

- both `polk1933bk` pages (region-level Surya boxes)
- six pages whose Gemini OCR failed: their `.txt` is an `[OCR FAILED]` placeholder
- rows whose text is absent from Gemini's output, which tests OCR, not alignment
- rows whose Surya line `align_image()`'s filters drop

`mb1931` and `polk1917` are scored but flagged, because their Surya output is partly
region-level.

## Results

**Placement**, as share of scored gold lines placed correctly:

| candidate | all 1,766 lines | without mb1931 / polk1917 (1,621) |
|---|---|---|
| whole-page | 96.4% | 99.0% |
| per-column, no selection | 96.1% | 98.8% |
| coverage-only selection | 96.3% | 99.0% |
| **shipped** | **96.1%** | **98.8%** |

**Why the column path rarely ran,** on the 19 multi-column pages with usable OCR:

| outcome | pages |
|---|---|
| Guard 1: hoisted "header" group above 10% of lines (11–23% here) | 9 |
| a column's start could not be located in the Gemini text | 6 |
| the column path ran | 4 |

All nine Guard 1 pages are two-column directories. The cause is the gutter-hoist
over-collection measured in `docs/evidence/2026-09-22-gutter-hoist/`, seen here on an
independent corpus. Five of the six location failures are the dense four-to-six-column
pages.

Where it ran (217 lines), the shipped rule kept the column result every time:

- `mb1931` 0022 and `trow1907`: identical to whole-page.
- `lain1876` 0022: 3 lines left unboxed.
- `polk1933si`: 1 line misplaced.

**The errors are on dense pages that never reached the column path**, and their Surya
output is under-segmented: tall region boxes, and fewer usable boxes than Gemini lines.

| page | correct | tall boxes | usable Surya boxes per Gemini line | box-vs-Surya disagreement |
|---|---|---|---|---|
| polk1917 | 0/48 | 16 | 0.75 | 84.5% |
| queens1933 | 20/31 | 26 | 0.82 | 61.5% |
| polk1925 | 11/14 | 26 | 0.49 | 53.0% |
| clean pages | ≈99% | 0–3 | ≈0.9 or more | under 5% |

**A no-gold signal.** The last column of that table compares the Gemini text bound to
each box with Surya's own reading of the box: the share of bound lines whose first 60
characters are under 0.5 similar. It ranks pages the same way the gold does.

The unmatched-rate outlier report cannot see these pages, because they box nearly
everything, onto the wrong lines. Mean Surya line length does *not* separate them:
polk1917 averages 30 characters, inside the clean pages' 26–38. `city-directory-extraction`
found the opposite failure, over-segmentation, on Lain 1897 p.0032: 317 micro-boxes
averaging 16 characters.

**Tulsa 1921**, 150 pages, the alignment on disk against a fresh one with the 0.3 code,
judged by the box-vs-Surya signal:

| | failing (>50%) | borderline (20–50%) | clean |
|---|---|---|---|
| on disk | 0 | 51 | 99 |
| fresh whole-page | 0 | 51 | 99 |
| fresh shipped | 0 | 51 | 99 |

The fresh shipped alignment matches the tree on disk on 149 of 150 pages. The column path
ran on 1 of 59 multi-column pages.

Spot checks of borderline lines found headings, short fragments and Surya boxes spanning
two columns, not displaced entries. So the borderline band is weak evidence of
mis-binding.

## Limits

- **The gold never reaches column 3.** Rows were sampled from the left of each page, and
  there are no full-page transcriptions. This says nothing about deep columns or reading
  order. `results.txt` splits rows into column 1 and column 2 by the recovered editor
  bucketing, which is approximate on the dense Polk/Queens pages.
- **Tulsa's own eval set is not used**, because its boxes come from this aligner.
- **39 pages.** Enough to show the path rarely runs, not to rank small differences.

## Reproduce

The gold panel is unpublished and stays in `city-directory-extraction`. The script reads
it from a sibling checkout and prints only counts and page names.

```bash
./.venv/bin/python docs/evidence/2026-10-17-column-alignment-panel/placement_panel.py
#   --gold-dir PATH  if city-directory-extraction is not beside this repo
./.venv/bin/python docs/evidence/2026-10-17-column-alignment-panel/box_vs_surya.py
#   --corpus 'output/<volume>/**/*_aligned.json' --limit N --seed S
```

Both need the gitignored `output/` data. Neither writes to it. Together they take a few
minutes.
