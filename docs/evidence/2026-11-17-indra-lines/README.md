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

# 0. A first look (a minute or two): four dense 1907 Trow pages tracked in this repo.
#    Mirrored to output/_indra_lines/pages/. Read the timings and eyeball the boxes.
python tools/indra_lines.py docs/evidence/2026-09-02-ocr-token-burn/pages/

# 1. The gold panel: run Indra on its pages only, then score.
python docs/evidence/2026-11-17-indra-lines/gold_panel.py --list-pages > /tmp/gold_pages.txt
python tools/indra_lines.py $(cat /tmp/gold_pages.txt)
python docs/evidence/2026-11-17-indra-lines/gold_panel.py > docs/evidence/2026-11-17-indra-lines/gold_results.txt
python docs/evidence/2026-11-17-indra-lines/gold_panel.py --indra-min-conf 0.5 > docs/evidence/2026-11-17-indra-lines/gold_results_conf05.txt

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

Not run yet.
