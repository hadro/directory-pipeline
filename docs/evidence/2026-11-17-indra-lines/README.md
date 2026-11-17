# Evidence — Indra OCR as the line detector

**Date:** 2026-11-17
**Status:** the harness only. **Not run yet.** Indra's models download from Hugging
Face, and the volumes and the gold panel are on the machine that holds `output/`, so
the run happens there. Results go at the end of this file.

**Code:**

- `tools/indra_lines.py`: runs Indra and writes its lines as `*_surya.json` into a mirror
  of the volume.
- `common.py`, `compare_detectors.py`, `gold_panel.py`: this folder.

## The question

Surya 0.17.1 supplies the line boxes that alignment, entry binding, ALTO export and
the review UI are built on. Three reasons to look for another source of boxes:

- **Accuracy.** The 2026-10-17 panel found that the placement errors left after 0.3 sit
  on dense pages where **Surya under-segments**: tall region boxes, and fewer usable
  boxes than Gemini lines. polk1917 placed 0 of 48 lines correctly, queens1933 20 of 31,
  and polk1925 11 of 14 (`../2026-10-17-column-alignment-panel/`).
- **Speed.** Surya took about 61 s per page on an M2 MacBook in the 0.3 smoke run.
- **No upgrade path.** Surya 0.20 removed `surya.foundation`, and 0.24 (Surya 2.5)
  removed line detection altogether: it is a layout VLM that returns blocks.

[Indra OCR](https://github.com/datalab-to/indra) (Datalab; `indra-ocr` 1.1.2 on PyPI,
2026-10-20) is a CPU engine on ONNX Runtime. It returns `Line(box, text, conf)` in
reading order, in image pixels, which is the shape of a Surya line.

**Does Indra give better boxes on the dense pages, without losing the clean ones? And
how fast is it on an M2?**

## What Datalab claims (not verified here)

- About 212 ms per page (median) on 4 x86 vCPUs, on OmniParseBench; about 44 pages/s
  on 32 cores.
- Its own stated limitations include **old and degraded scans: 29.5% on the degraded-scan
  tests**, and complex tables.
- The code is Apache 2.0. The weights use a modified OpenRAIL-M licence: free for
  research, personal use and organisations under $20M, but not for use competing with
  Datalab's API.

## Differences that matter for alignment

- **Alignment matches on the detector's text.** Anchors and the Needleman-Wunsch
  scoring compare each Gemini line with the detector's reading of each box. A weak
  reading weakens the match even when the box is right, and Indra is weakest on old
  scans.
- **Confidence is on another scale.** Indra's `conf` is the mean CTC peak probability.
  A line under 0.5 keeps its box and comes back with no text. `parse_surya()`'s floor of
  0.35 keeps lines between 0.35 and 0.5 as boxes without text. Both scripts take
  `--indra-min-conf 0.5` to drop them instead.
- **Indra splits a line at a wide horizontal gap** (`split_line`). That may separate
  columns cleanly, or break up an entry that runs to a dot leader.
- Surya's `--blank-threshold`, the fix for dot-leader index pages, has no equivalent.
- **Pages are read upright** (`turn=False`), as Surya reads them. Otherwise Indra's
  orientation model would return boxes in a turned frame for any page it judged
  sideways.

## How the mirror works

`tools/indra_lines.py output/<vol>/` writes `output/_indra_lines/<vol>/`. That folder
holds:

- `{stem}_surya.json` and `{stem}_surya.txt` from Indra, with Indra-only fields under
  `detector` and `blocks`, which `parse_surya()` ignores;
- symlinks to the page images;
- copies of each page's Gemini text and the volume's metadata.

Files are copied rather than linked because later stages write some of them, and a
write through a symlink would land in the real volume. The `_` prefix keeps the mirror
out of every cross-collection tool. `align_ocr.py` and `extract_entries.py` run on the
mirror unchanged.

## The judges

**`gold_panel.py`** scores both detectors against `city-directory-extraction`'s
hand-gold panel with one judge. For each gold row, it finds the box carrying the Gemini
line that matches the row (similarity ≥ 0.60), and asks whether that box is on the
gold row's line:

- its vertical centre lies within the gold box's extent (±2 px);
- it overlaps the gold box horizontally by at least half the narrower width.

The 2026-10-17 judge went the other way: from the gold box to the text bound to it.
That cannot score Indra, whose boxes are not Surya's. The same rows are also scored for
Surya by that published judge. On unchanged inputs it should reproduce the panel's
**96.1%** (shipped, all rows), which checks the wiring. Pages whose Gemini OCR was
redone after 2026-10-17 will move it.

Limits:

- **The gold boxes are Surya's.** Where a gold box is a tall region, "on the line" is
  lenient for both detectors. Those pages are flagged `T` and reported apart, with the
  panel's suspect pages (`*`).
- **The gold rows were built by walking Surya's lines**, so a printed line that Surya
  merged into a neighbour has no gold row. The judge sees misplaced lines, not missing
  ones. `compare_detectors.py`'s lines-per-Gemini-line covers that.
- **polk1933bk** (all region boxes) cannot be scored. Look at it instead.

**`compare_detectors.py`** needs no gold and runs on any volume. Per page and per
detector it reports:

- lines per Gemini line;
- tall boxes (over 2× the page's median line height);
- the share of lines without text;
- the share of Gemini lines left unboxed;
- the box-vs-own-text disagreement, the no-gold signal that ranked the panel's pages as
  the gold did.

It ends with a Surya-bucket × Indra-bucket table and the pages that moved between
clean and failing. Disagreement is judged against each detector's *own* text, so use it
to find pages to look at, not as a verdict.

## Running it

From the repo root, on the machine with `output/`:

```bash
uv sync --inexact --extra indra       # adds Indra; --inexact keeps Surya and torch installed

# (Not the four Trow pages in docs/evidence/2026-09-02-ocr-token-burn/pages/: they are
#  mirror-image, faint microfilm, and Indra finds no lines on them. See Results.)

# 1. The gold panel: run Indra on its pages only, then score. Indra's default detection
#    size (1024 px), then 2048 px into a second mirror.
python docs/evidence/2026-11-17-indra-lines/gold_panel.py --list-pages > /tmp/gold_pages.txt
tr '\n' '\0' < /tmp/gold_pages.txt | xargs -0 python tools/indra_lines.py -q
python docs/evidence/2026-11-17-indra-lines/gold_panel.py > docs/evidence/2026-11-17-indra-lines/gold_results.txt
python docs/evidence/2026-11-17-indra-lines/gold_panel.py --indra-min-conf 0.5 > docs/evidence/2026-11-17-indra-lines/gold_results_conf05.txt
tr '\n' '\0' < /tmp/gold_pages.txt | xargs -0 python tools/indra_lines.py -q \
    --det-long-side 2048 --mirror-root output/_indra_lines_2048
python docs/evidence/2026-11-17-indra-lines/gold_panel.py --mirror-root output/_indra_lines_2048 \
    > docs/evidence/2026-11-17-indra-lines/gold_results_side2048.txt
# The configuration that passes: lines kept whole, and the aligner's 15 px floor off for both
# detectors (see Results). Slow to score: about 1,200 lines a page on the densest pages.
tr '\n' '\0' < /tmp/gold_pages.txt | xargs -0 python tools/indra_lines.py -q \
    --det-long-side 2048 --no-split --mirror-root output/_indra_lines_2048_nosplit
python docs/evidence/2026-11-17-indra-lines/gold_panel.py --mirror-root output/_indra_lines_2048_nosplit \
    --min-line-height 0 > docs/evidence/2026-11-17-indra-lines/gold_results_side2048_nosplit_floor0.txt

# 2. A production volume, no gold: the same 150-page Tulsa sample size as 2026-10-17.
python tools/indra_lines.py output/tulsa_1921/
python docs/evidence/2026-11-17-indra-lines/compare_detectors.py output/tulsa_1921/ \
    --limit 150 --seed 0 --csv docs/evidence/2026-11-17-indra-lines/tulsa_1921.csv \
    > docs/evidence/2026-11-17-indra-lines/tulsa_1921.txt

# 3. Look at the pages that moved: align the mirror, then open it in the review UI.
python pipeline/align_ocr.py output/_indra_lines/tulsa_1921/
pipeline review output/_indra_lines/tulsa_1921/
```

**Speed:** `indra_run.json` at each mirror root holds every page's wall time, including
image load, and Indra's own timer. `tools/indra_lines.py` prints the median.
`--threads` defaults to 4, as in Datalab's benchmark. Try 8 on the M2 too.

## The bar

Indra goes forward only if all of these hold:

1. **Unflagged gold pages:** Indra places at least as well as Surya by the same judge,
   within 0.5 points.
2. **The dense pages** (polk1917, queens1933, polk1925): Indra fixes most of Surya's
   misplacements. Read the flagged rows, and look at the pages.
3. **Tulsa 1921:** lines per Gemini line near 1, and no more than a handful of pages
   going from clean to failing.
4. **Speed on the M2:** a median in seconds per page, against Surya's 61.

If it passes, the next step is a line-detector option in the OCR stage, on its own
branch. If it is close, the obvious hybrid is Indra's boxes with Surya's text where
Indra returned none.

## Results

Run on 2026-11-16 on an M2 MacBook (8 cores, 16 GB), with Indra 1.1.2 on 4 threads. Output:
the `gold_results*.txt` files beside this README (counts and page names only).

**The wiring holds.** Surya, scored by the published judge, gives 96.1%, the panel's figure
from 2026-10-17.

**The four Trow 1907 pages are not a fair sample.** They were kept because Gemini looped
on them. They are faint microfilm, and mirror images (the header reads backwards). Indra
finds no lines on the raw scans at any detection size. With contrast stretched it finds
lines, and it reads them only once the page is flipped left to right.

**Three things held Indra back, and none of them was finding lines:**

1. **Detection size.** Indra shrinks the page to 1,024 px on its long side before finding
   lines, and small type on dense pages disappears (polk1917: 114 lines for 756 Gemini
   lines). Its adaptive retry needs 8 lines from the first pass, and stops at 2,048 px.
   Fixed with `--det-long-side 2048`.
2. **Line cutting.** Indra cuts a line wherever a gap lines up with other lines' edges. On a
   directory page, where addresses line up, that separates name from address, and
   alignment pairs one line with one box. The gap rule (over 3 line heights) adds almost
   nothing; the gutter rule does it all. Fixed with `--no-split`.
3. **The aligner's 15 px floor.** `filter_short_lines()` drops every line shorter than
   `_MIN_LINE_HEIGHT` = 15 px, a constant set in March for Surya's boxes. Indra's boxes fit
   the type: its median line on polk1925 is 13 px, and the floor dropped 1,033 of 1,183
   lines. Surya loses its short lines there too, but its tall region boxes survive.
   `gold_panel.py --min-line-height 0` turns the floor off for both detectors, leaving the
   filter's 35%-of-median rule.

**Gold panel**, share of gold lines placed correctly. With the floor off, 1,813 rows are
scored instead of 1,766, because Surya's short lines are no longer dropped first:

| Indra | floor | Surya | Indra | unflagged, Surya / Indra | flagged, Surya / Indra |
|---|---|---|---|---|---|
| 1024 px (default) | 15 px | 96.2% | 86.0% | 99.6% / 91.6% | 72.5% / 47.3% |
| 2048 px | 15 px | 96.2% | 95.5% | 99.6% / 99.5% | 72.5% / 68.0% |
| 2048 px, no split | 15 px | 96.2% | 97.5% | 99.6% / 99.5% | 72.5% / 83.3% |
| 1024 px (default) | off | 96.2% | 83.9% | 99.6% / 91.6% | 76.6% / 39.8% |
| 2048 px | off | 96.2% | 95.3% | 99.6% / 99.5% | 76.6% / 71.4% |
| **2048 px, no split** | **off** | **96.2%** | **98.5%** | **99.6% / 99.5%** | **76.6% / 92.6%** |

The dense pages, correct / scored. The last column scores more rows, because the floor is off:

| page | Surya, floor 15 | Indra 1024 | Indra 2048 | 2048, no split | Surya / Indra, 2048, no split, floor off |
|---|---|---|---|---|---|
| polk1917 | 0/48 | 0/48 | 39/48 | 48/48 | 0/59 / **48/59** |
| polk1925 | 11/14 | 0/14 | 0/14 | 0/14 | 30/32 / 25/32 |
| queens1933 | 21/31 | 2/31 | 8/31 | 8/31 | 45/47 / 45/47 |
| mb1931 p.21 | 33/33 | 9/33 | 9/33 | 33/33 | 33/33 / 33/33 |
| polk1933si | 55/56 | 17/56 | 56/56 | 55/56 | 55/56 / 55/56 |
| trow1913 | 92/92 | 36/92 | 92/92 | 92/92 | 92/92 / 92/92 |

`--indra-min-conf 0.5` changes nothing (85.8% against 86.0%).

**Speed**, wall time per page including image load: median 0.58 s at 1024 px and 0.71 s
at 2048 px (slowest 3.4 s), against about 61 s for Surya on the same machine.

**Against the bar**, at 2048 px, no split, floor off:

1. Unflagged pages: 99.5% against 99.6%. **Met.**
2. Dense pages: polk1917 goes from 0 to 48 of 59, queens1933 and mb1931 tie, and polk1925
   is 5 lines worse (25 against 30 of 32). **Mostly met.** These pages are all flagged, where
   the judge is lenient, so look at polk1917 and polk1925 before trusting either.
3. Tulsa 1921 without gold: **not run yet.**
4. Speed: **met**, by about 85 times.

**The floor is the catch.** Indra needs it off, and the pipeline has it on. Turning it off
costs Surya nothing on this panel (96.2% either way), but it changes every Surya alignment
in production, so it needs its own check (`tools/audit_entry_bindings.py`, or Tulsa below)
before it moves.

**Next:**

1. Look at polk1917 and polk1925 with boxes drawn.
2. Tulsa 1921 without gold (`compare_detectors.py`), at 2048 px with `--no-split`. It needs
   a `--min-line-height` option too.
3. If both hold: a line-detector option for the OCR stage, and a decision on the floor.
