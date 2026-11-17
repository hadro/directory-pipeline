# Evidence — Indra OCR as the line detector

**Date:** 2026-11-17
**Status:** run on 2026-11-16 and 17. **Indra alone does not pass**: it trails Surya on
Tulsa 1921. **Indra first, with Surya on the pages where Indra binds badly, does**: 98.8%
of gold lines placed against Surya's 96.2%, with Surya needed on about a third of Tulsa's
pages. At the entry level on Tulsa it cuts wrong boxes by a third, but most of that gain
comes from turning off the aligner's 15 px line floor, which helps Surya alone too.
Results are at the end of this file.

**Code:**

- `tools/indra_lines.py`: runs Indra and writes its lines as `*_surya.json` into a mirror
  of the volume.
- `common.py`, `compare_detectors.py`, `gold_panel.py`, `entry_placement.py`: this folder.

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

# 2. A production volume, no gold: every Tulsa 1921 page with Surya lines (834 of 1,250).
for j in output/tulsa_1921/*_surya.json; do echo "${j%_surya.json}.jpg"; done > /tmp/tulsa_pages.txt
tr '\n' '\0' < /tmp/tulsa_pages.txt | xargs -0 python tools/indra_lines.py -q \
    --det-long-side 2048 --no-split --mirror-root output/_indra_lines_2048_nosplit
python docs/evidence/2026-11-17-indra-lines/compare_detectors.py output/tulsa_1921/ \
    --mirror-root output/_indra_lines_2048_nosplit --min-line-height 0 \
    > docs/evidence/2026-11-17-indra-lines/tulsa_1921.txt
# tulsa_1921_floor15.txt: the same without --min-line-height 0. tulsa_1921_split.txt:
# Indra without --no-split, into and from output/_indra_lines_2048.

# 3. Look at the pages that moved: align the mirror, then open it in the review UI.
python pipeline/align_ocr.py output/_indra_lines_2048_nosplit/tulsa_1921/
pipeline review output/_indra_lines_2048_nosplit/tulsa_1921/
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

**Looking at the dense pages.** A page has a few dozen gold rows, and these pages are
flagged, so the boxes were drawn for the whole page and read beside the no-gold check (the
share of bound lines whose Gemini text disagrees with the detector's own reading of the
box, floor off):

- **polk1917:** Indra's gain is real but partial. Surya's bound text is shifted by several
  lines from near the top of the page. Indra's is right for stretches, then drifts. Over the
  whole page 72% of Indra's bound lines disagree, against 87% of Surya's. Even with
  splitting off, Indra returns short pieces of lines ("21", "11th", "Frank"), and each
  takes a Gemini line. The 59 gold rows fell mostly in its good stretches.
- **polk1925:** both detectors mis-bind parts of the middle column. Surya is better over
  the whole page (41% against 73%), as on the gold rows.

**Tulsa 1921** (`tulsa_1921*.txt`): all 834 pages with Surya lines and Gemini text, Indra
at 2,048 px, no split. Pages by the no-gold check (clean: at most 20% disagree; failing:
over 50%):

| | floor | clean | borderline | failing |
|---|---|---|---|---|
| Surya | 15 px | 539 | 263 | 32 |
| Indra | 15 px | 487 | 294 | 52 |
| Surya | off | 662 | 134 | 38 |
| Indra | off | 571 | 221 | 41 |
| Indra, splitting on | off | 363 | 353 | 117 |

Indra alone trails Surya: 571 clean pages against 662 at the same floor, with 14 pages
going from clean to failing and 13 the other way. The check penalises a detector that
reads worse, but these failures are real misplacements. The pages carry display
advertisements, Indra's boxes there run two printed lines together (median box height
28 px against Surya's 20 on page 0062 left), and binding drifts. It is not the detection
size (the tall boxes are there at 1,024, 1,672 and 2,048 px), and not skew (Indra trails
on the pages it found level too). With splitting on it is much worse.

**Indra first, Surya where Indra binds badly.** The check needs no gold, so production can
compute it per page. Run Indra on every page; where more than 20% of its bound lines
disagree, run Surya as well, and keep whichever disagrees less (`cascade_pick()` in
`common.py`; both scripts report it):

| | gold panel, placed correctly | Tulsa clean / borderline / failing | Surya runs on |
|---|---|---|---|
| Surya | 96.2% | 662 / 134 / 38 | every page |
| Indra | 98.5% | 571 / 221 / 41 | none |
| **Indra, then Surya** | **98.8%** | **719 / 97 / 17** | 4 of 33 gold pages, 263 of 834 Tulsa pages |

On the gold panel: unflagged pages 99.5% (Surya 99.6%), flagged pages 94.4% (Surya 76.6%).
The Tulsa row is optimistic, because the same check picks and judges; the gold row is not.
On Tulsa it would take about 834 × 0.9 s + 263 × 61 s, roughly 4.7 hours against 14 for
Surya alone. The check also sends pages that Indra reads poorly to Surya when its boxes
are fine (trow1907: both place all 67 gold rows). That costs time, not accuracy.

**The floor, for Surya alone.** On Tulsa, turning it off takes Surya from 539 clean pages
to 662, and from 32 failing to 38: 11 failing pages come clean and 7 clean pages fail.
The new failures are wrapped continuation lines ("bell av.", "Main av.") with no tall
letters, 10–14 px high. The floor was dropping them. Gemini runs them into the line
above, so once kept, each takes a Gemini line, and binding drifts (on one page from 13
disagreeing lines to 102). A floor relative to the page's median line height (0.5, 0.6,
0.7 and 0.8 ×) did no better than off on either set: the fragments are 13–14 px, the
lines 18–20 px. On the gold panel the floor changes nothing for Surya (96.2% either way).

**Entry boxes, Tulsa 1921** (`entry_placement.py`, `entries_tulsa_1921.txt`). Lines are
not what the data publishes; entries are, through a binder that refuses weak matches. So
each configuration's alignment was bound with today's binder, in memory, and each entry's
box judged by the share of its identity words printed inside it, read by Surya or Indra,
whichever shares more. `tools/audit_entry_bindings.py` cannot answer this: it compares an
entry with the *Gemini text* bound to its box, so when alignment puts that text on the
wrong printed line, the entry follows it and the audit calls it right.

800 pages, 69,843 entries; about 48,300 placed under every configuration (the rest are
the street section, below). Shares of placed entries:

| | wrong (no word in the box) | weak (under a third) | tall box |
|---|---|---|---|
| Surya, 15 px floor (today) | 6.6% | 4.5% | 2.2% |
| Surya, floor off | 4.5% | 4.1% | 1.9% |
| Indra, floor off | 6.4% | 5.5% | 1.3% |
| **Cascade**, floor off | **3.6%** | **4.1%** | **1.3%** |

Against today's Surya, counting wrong and weak entries page by page:

| | pages better (entries) | pages worse (entries) | net entries |
|---|---|---|---|
| Surya, floor off | 448 (2,108) | 118 (900) | −1,208 |
| Indra, floor off | 367 (2,108) | 311 (2,497) | +389 |
| **Cascade** | 436 (2,428) | 191 (784) | **−1,644** |

The cascade cuts wrong and weak entry boxes by about a third (11.1% to 7.7%), but **three
quarters of that is the floor**: Surya alone with the floor off gets −1,208. Indra adds
about 440 more, and a cascade that runs Surya on 203 of the 800 pages. The pages that get
worse are mostly the floor's: the wrapped continuation lines above (0406 right: 8 to 90).
The judge is lenient in two ways: a box holding several lines, and a neighbouring entry
with the same surname, both score as partly right.

The wiring holds: 115,899 of 116,021 lines of the in-memory Surya alignment are in the
aligned JSON on disk.

**A binder bug, found on the way (not Indra's).** Today's binder refuses all 21,902
entries of Tulsa's street-directory section (150 pages), under every configuration.
Their identity words come from `identity_tokens()`, which takes the first field whose
name ends `_name`: here `street_name`, the section heading ("MAIN AV—NORTH"), so every
entry on the page looks the same. Dropping `street_name` binds 162 of 162 on page 0337
right; as extracted, 1. It is the Green Book `category` trap in another field. The
published entries predate this binder and keep their boxes, but re-extracting Tulsa, or
running `tools/patch_canvas_fragments.py` over it, would strip them.

**Against the bar**, at 2,048 px, no split, floor off:

1. Unflagged gold pages: Indra 99.5%, Surya 99.6%. **Met.**
2. The dense pages: **partly.** polk1917 improves but stays mostly wrong for both, and
   polk1925 is worse with Indra. The cascade takes the better detector on both.
3. Tulsa 1921: **not met by Indra alone** (91 fewer clean pages, 14 from clean to
   failing). **Met by the cascade.**
4. Speed: Indra **met**, 70 to 85 times faster (median 0.88 s a page on Tulsa). The
   cascade is about 3 times faster than Surya on Tulsa.
5. Entry boxes on Tulsa (added after the bar was set): the cascade is **better on net**
   (1,644 fewer wrong or weak entries), but 784 entries get worse on 191 pages, mostly
   from the floor. Not "a handful".

**Next:**

1. **The floor, on its own branch, first.** It is most of the entry-level gain, with or
   without Indra, and Indra needs it. Its regressions (wrapped continuation lines kept
   as lines) need a fix before it ships; `entry_placement.py` is the before/after check.
2. **The street-section binder bug**, on its own branch: 21,902 Tulsa entries. Before
   anyone re-extracts or re-patches Tulsa.
3. **A second collection** (Green Books) through `compare_detectors.py` and
   `entry_placement.py`, before cutting a cascade branch. Tulsa says Indra's own
   contribution there is modest; the dense pages (polk1917) are where it is large.
4. **The cascade in the pipeline**, if 3 holds: Indra lines in the line stage, Gemini
   OCR and align unchanged, then a Surya fallback stage for the pages over the threshold.
   Surya 0.17.1 and Indra 1.1.2 install and import together in the project's `.venv`
   (Python 3.12).
5. **Indra's fragments** (polk1917): join pieces of one printed line before alignment.
   Not tried.
