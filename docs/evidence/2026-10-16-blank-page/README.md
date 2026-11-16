# Evidence — the blank-page check skipped real pages

**Date:** 2026-10-16
**Code:** `utils/image_utils.py` — `is_blank_page()`, which `pipeline/run_gemini_ocr.py`
`process_image()` calls before every OCR request. The legacy rule is the one at commit
`8740c4a`.
**Corpus:** every page scan under `output/`. That is 18,501 files in 192 directories,
probes and slices included, because the check runs on whatever gets OCR'd. 30 are
unreadable, leaving 18,471 measured.
**Status:** fixed in the same change. `results.txt` measures the fixed code.

The headline:

> **The legacy check called 1,786 pages blank, and 1,142 of them have measurable ink. They
> include full pages of entries: Doggett 1850–51 p. 154 (154 lines), every page of the
> Polk 1933 microfilm slices, and dense Green Book listings. One production run lost real
> pages: 5 in-scope pages of *The New York Directory for 1786*, OCR'd 2026-07-24 and
> recovered 2026-10-16. The fixed
> check calls 644 pages blank, all of them blank under the old rule too. Every one was
> inspected. Beyond library marks and a one-word printed header ("NOTES", "MEMO"), the
> only content any of them carries is one copyright line.**

A skipped page writes no `.txt`, so align and extract never see it and nothing reports
the loss. The run summary's "N blank" count is the only trace, and it is not kept.

## How this was found

`historical-ocr-eval` ran its multi-column bench through `run_gemini_ocr` and two gold
pages, `nypl_doggett_s_1850_51_4adf9ec0/0021_56750619.jpg` and `0022_56750889.jpg`
(154 and 149 gold lines), came back "Blank page — no OCR".

## The mechanism

The legacy rule: grayscale, `thumbnail((512, 512))`, crop 8% margins, take p90 as the
paper background, and call the page blank when fewer than 0.2% of pixels are darker
than 0.5 × background. Two separate failures:

1. **The thumbnail averages thin type into grey.** At full resolution Doggett 0021 has
   4.2% of pixels below 64. At 512 px the darkest pixel left on the page is 66, and only
   0.12% of pixels fall under the cutoff.
2. **The cutoff is absolute darkness, not contrast.** Toned paper puts p90 at about 200,
   so the cutoff is 100. Grey-printed scans, Green Books especially, put their type at
   0.55–0.65 × background, so a 0.5 cutoff misses them at any resolution. A quarter of
   the legacy-blank pages with OCR or Surya evidence of text still measure under 0.2% at
   2,048 px, and the contact sheets show dense Green Book listings among them.

| page | thumbnail | background (p90) | cutoff | dark frac | p10 / p50 / min | legacy |
|---|---|---|---|---|---|---|
| Doggett 1850 0021 | 322×512 | 200 | 100 | 0.0012 | 139 / 188 / 66 | blank |
| Doggett 1850 0022 | 318×512 | 203 | 102 | 0.0009 | 143 / 190 / 63 | blank |
| control: Trow 1884 0021 | 300×512 | 252 | 126 | 0.1198 | 118 / 229 / 0 | text |

The fix measures at a 1,600 px long side and counts pixels darker than **0.75** ×
background, blank below **0.1%**. The method, margin crop, p90 background and dark-scan
guard are otherwise unchanged. Doggett 0021 and 0022 now measure 16% dark.

## Measurement

**There is no ground truth, and neither obvious proxy works:**

- **OCR text from earlier runs is not a label.** Most of the corpus was OCR'd before the
  check existed (2026-06-11), and Gemini hallucinates on blank pages. 46 pages the fixed
  check calls blank carry over 1,000 characters of Gemini output, and every one is blank
  on inspection. See *Hallucinated pages* below.
- **Surya line counts are not a label either.** Surya detects lines in show-through.

The labels are therefore **by inspection**. Contact sheets, autocontrasted so that faint
ink shows, were made of:

- every page the fixed rule calls blank (644);
- every page from half to four times the cutoff (dark 0.0005–0.004, 137 pages).

Any page showing print was zoomed and checked for mirror reversal. Mirror-reversed text is
show-through from the other side of the leaf, so the page itself is blank.

**Choosing the ratio.** Pages called blank at 1,600 px, by ratio *r* and cutoff, with
how many of those the legacy rule had *passed*:

| r | cutoff 0.0005 | 0.001 | 0.002 | 0.004 | legacy-passed among them, at 0.001 |
|---|---|---|---|---|---|
| 0.50 | 865 | 903 | 969 | 1,051 | 27 |
| 0.60 | 753 | 786 | 827 | 879 | 10 |
| 0.70 | 663 | 697 | 750 | 803 | 0 |
| **0.75** | 614 | **644** | 683 | 751 | **0** |
| 0.80 | 546 | 578 | 627 | 695 | 0 |
| 0.85 | 466 | 511 | 543 | 592 | 0 |

At 0.6 and below, the rule starts skipping pages the legacy rule sent to OCR. Above 0.75 it
sends more show-through to OCR, which invites the model to transcribe mirror-image text.

**Choosing the cutoff.** The real content nearest the line, in the 0.0005–0.004 band:

| dark | page |
|---|---|
| 0.00226 | a printed "NOTES" page (heading, rules, folio) with faint text beneath (`e77514b9…`) |
| 0.00235 | the facing "NOTES" page, with a faint advertisement (`e77514b9…`) |
| 0.00244 | a title page with a printed reprint note (`passenger_lists…`) |
| 0.00247 | one short printed line (`flower_lore…`) |
| 0.00248 | a map section with a Comptroller's stamp (`nypl_trow_1890_91`) |
| 0.00275 | a title page (`register_of_artists_engravers…`) |
| 0.00289 | a handwritten name and address on a cover (`ia_longworth_1839`) |
| 0.00294 | a printer's imprint (`the_illustrated_language_of_flowers`) |
| 0.00311 | a handwritten docket, "Charles Douglass" (`frederick_douglass_papers…`) |

Everything else in the band is blank paper, mirror-reversed show-through, library and
copyright stamps, barcodes, bookplates, typed labels, pencil shelf marks, Internet Archive
"Digitized by" placeholders, or a printed border. A cutoff of 0.001 sits 2.2× under the
sparsest real page. 0.002 would sit 10% under it, and would skip 39 more pages, all blank
with marks.

**What the fixed rule still skips.** Of the 644 pages under 0.001, 47 are uniform white
(background 255, nothing darker). The other 597 are blank paper and show-through, many of
them the backs of advertisement leaves in the IA Trow and Upington volumes. Some also
carry library marks: collection stamps, barcode labels, a conservation-folder label,
pencil shelf marks, a patterned Library of Congress endpaper. A few carry a one-word
printed header on an otherwise empty page ("NOTES", "MEMO"). **One** carries real
content: a Green Book verso whose only print is "Copyright 1955 by Victor H. Green", at
0.00002 dark. No threshold in this family separates one short line of small type from
foxing, and the legacy rule skipped that page too.

`pipeline/run_gemini_ocr.py --no-blank-skip` still OCRs every page. `main.py` and the
`pipeline` subcommands do not forward that flag. `harvest_slice.sh` in the 2026-09-02
evidence calls the leaf script directly for that reason.

## Blast radius

**Production losses.** These are full-volume runs after 2026-06-11 where pages the legacy
rule flagged are the only pages without a `.txt`. Each such page was inspected. The
count ignores `included_pages.txt`, so it includes pages that page selection had excluded
from the run anyway:

| volume | last OCR | legacy-blank, no `.txt` | real pages among them |
|---|---|---|---|
| `the_new_york_directory_for_1786_containi_96203733` | 2026-07-24 | 35 | **5 in the run's scope**: name lists A–B (0032), a bank notice (0077), the additional name lists (0094, 0102), the Brooklyn directory title (0103). **19 more outside `included_pages.txt`**: a pasted newspaper clipping (0004); contents, the twelve almanac months, eclipses, and the dollar, gold-coin and guinea tables (0013–0030) |
| `miller_s_new_york_as_it_is_or_stranger_s_16012298` | 2026-07-24 | 45 | 0: blank leaves and show-through |
| `1869_geo_p_rowell_and_co_s_american_news…` | 2026-09-19 | 17 | 0: pure white |
| `the_national_directory_of_morticians…` | 2026-08-24 | 8 | 0: blank, one library barcode |
| `the_artistic_language_of_flowers_cu31924074093760` | 2026-06-11 | 5 | 0: pure white |
| `afro-american-travel-guide-1957` | 2026-08-11 | 2 | 0: a "NOTES" header, show-through |
| `the_negro_motorist_green_book_2016298176` | 2026-07-19 | 1 | 0 |

All 24 real 1786 pages measure at least 0.022 dark under the fixed rule, 22× the cutoff.
The 5 in scope were recovered on 2026-10-16. A re-run of `run_gemini_ocr` under the fixed
check used the run's original model (`gemini-3.1-flash-lite`) and the volume's own
`ocr_prompt.md`. It sent those 5 pages and nothing else; in-scope 0007 is blank and stayed
skipped. Page 0077 needed one RECITATION retry. Align and extract have not been re-run for
them. The 19 out-of-scope pages are a page-selection question, not a loss to this check.

**Volumes last OCR'd before 2026-06-11** were never exposed to the check, whatever gaps
they have. **Probe directories** (`nypl_*`, `ia_*`, `loc_*`, `harvest_*`, 104 of them)
OCR a few chosen pages, so a missing `.txt` there usually means the page was not
selected. 296 of their pages that the legacy rule called blank now pass, among them every
page of the `harvest_polk1933*` microfilm slices. Their thumbnails match the old code
comment's description of NYPL microfilm: darkest pixel 77–132 against a background of
210–230. They now measure 11–23% dark. The 2026-09-02 token-burn harness ran those
slices with `--no-blank-skip`.

**Exposure that had not yet become loss.** Every Green Book page was OCR'd before the
check existed. 286 of them were legacy-blank and 214 of those now pass. A re-OCR of the
collection under the legacy rule, including the deferred degenerate-page work in
`docs/plans/degenerate-page-reocr.md`, would have silently dropped real listing pages.

## Hallucinated pages

46 pages that the fixed check calls blank have at least 1,000 characters of Gemini output
from runs before the check existed. All 46 are blank on inspection, so the text was
invented. 41 are Green Book pages OCR'd by `gemini-2.0-flash`. A typical output opens
with "ALABAMA | NICKNAME: The Cotton State" or "Page 1xx | ALABAMA | BIRMINGHAM |
HOTELS | The Booker T. Washington Hotel…", and the largest runs to 1,048,042 characters.
This is the failure the check exists to prevent.

Three of these pages reached extraction: **47 rows** in
`green_books_and_related/the_green_book_9ea5d5b0/c4391ba0-1a76-0132-7cae-58d385a7bbd0/entries_gemini-3.1-flash-lite-preview_text_on*.csv`
are attributed to pages 0003_5205891 (1), 0106_5205994 (1) and 0112_5206000 (45).

**Followed up 2026-10-17.**

- **Not published.** The 47 rows sit only in that volume's text-only experiment CSV and in the
  three pages' cached `*_entries.json`. They are not in:
  - the main or fixed CSV
  - `green_book_entries_all.csv`
  - the green-books repo's CSVs, Hugging Face export, annotations or git history
- **Still at risk.** 14 of the blank pages carrying invented text would still have reached NER on
  their volumes' next extraction run. 11 had never been extracted, and 3 had caches that the
  2026-09-04 re-align made stale.
- **Fixed.** All 44 pages this check now calls blank that carry ≥1,000 bytes of Gemini OCR have
  had their OCR, alignment and entries files moved to
  `output/_quarantine/hallucinated-blank-pages-2026-10-17/`. Its README gives the undo steps,
  and an exclude list for restoring the 2026-09-04 alignment backup. The 47 rows were removed
  from the text-only CSV.

## Caveats

- **Inspection was at contact-sheet scale.** Autocontrast makes faint ink visible, and
  every page with visible print was zoomed, but a faint mark smaller than a few pixels at
  that scale could be missed.
- **The corpus is mostly 18th–20th century New York directories, Green Books and IA
  scans.** Bitonal scans, colour plates and very small source images (under ~1,000 px)
  were not examined as separate groups.
- **Thin grey type is the remaining weak spot.** This is reasoned, not measured. A 1 px
  grey stroke at full resolution blurs further when downsampled to 1,600 px. No inspected
  text page fell into this case, but the margin there is smaller than for dark type.
- **The 30 unreadable files** were interrupted downloads: empty files, bodies cut at a
  chunk boundary, and IA responses stopping at 108,620 bytes. All six sampled came back
  whole on one re-request. `is_blank_page` still raises on such a file. Since the same
  change, the Gemini and Surya OCR stages check `image_is_complete` first and download a
  corrupt page again, at its siblings' width. The download stage does the same on
  `--resume`, and writes through a `.part` file so it no longer leaves partial images.

## Reproduce

```bash
./.venv/bin/python docs/evidence/2026-10-16-blank-page/blank_census.py [--csv PATH]
```

Each page is decoded three times, so expect 20–30 minutes on all cores. It needs the
gitignored `output/` data. It imports the live `utils.image_utils`, so after any change
the "live" column measures the change.
