# Deferred: re-OCR the 21 degenerate Green Book pages

**Created 2026-09-04.** Deferred by Josh during the alignment-repair work — do the
re-align and the entry-binder rewrite first, then come back to this.

## What these pages are

21 pages in `output/green_books_and_related/` whose Gemini OCR output is a
runaway: `_output_issue()` in `pipeline/run_gemini_ocr.py` classifies them as
`long_line`, `block_loop` or `repetition`. They are 0.4% of the collection's
5,838 aligned pages but carry a hugely disproportionate share of unaligned text —
one 9,484-line page is a single sheet of a directory.

Because the text is garbage, alignment cannot box it, NER still reads it (see
`_page_text_from_aligned()`, which feeds NER the aligned lines *plus*
`unmatched_gemini`), and the resulting CSV rows are unboxed or wrong.

## The blocker: gemini-2.0-flash is retired

Every page in this collection was OCR'd with `gemini-2.0-flash`. As of
2026-09-04 that model returns 404 from the API:

> This model models/gemini-2.0-flash is no longer available. Please update your
> code to use models/gemini-3.6-flash for the latest features and improvements.

So these pages **cannot** be re-OCR'd with the model that produced the rest of the
volume. A re-OCR necessarily introduces a second model's `.txt` into directories
that are otherwise single-model.

## Why that is not a drop-in fix

`_detect_aligned_slug()` in `pipeline/extract_entries.py` picks **one model per
directory**, ranked by expected useful boxed lines. A new model covering only 21
pages would lose that ranking to the incumbent covering all of them, so those
pages would still be read from the degenerate `gemini-2.0-flash` alignment and
nothing would improve.

Doing this properly therefore requires one of:

1. **Per-page model selection** — let extraction choose the best alignment for
   each page rather than for the whole directory. Correct, and the natural home
   for it is the same ranking function; it is a real change to how pages are
   resolved, so it wants its own tests.
2. **Re-OCR whole volumes on a current model** — clean and uniform, but it
   obsoletes every existing alignment, every `canvas_fragment`, and every cached
   thumbnail in the affected volumes. Real API cost and a full thumbnail rebuild.

Option 1 is much cheaper and is the recommended path.

## Recovery / safety notes

- A first attempt at re-OCR on 2026-09-04 deleted these 21 `.txt` files before
  discovering the model was retired. They were restored from
  `output/_ocr_backups/green_books_degenerate_txt_2026-09-04/` and verified
  byte-identical. **Probe model availability with a single page before clearing
  any `.txt`** — `run_gemini_ocr.py` skips pages that already have text, so
  clearing files is the only way to force a re-run, and it is destructive.
- Full aligned-JSON backups from the same day:
  `output/_ocr_backups/green_books_aligned_2026-09-04/` (tarball, 6,208 files)
  and `output/_ocr_backups/green_books_and_related_aligned_2026-09-04/`.

## The pages


**n_h_a_directory_and_guide_to_travelers_b33397d0**

| page | failure | OCR lines |
|---|---|---|
| `0002_58022239_gemini-2.0-flash.txt` | long_line | 9,484 |
| `0003_58022240_gemini-2.0-flash.txt` | long_line | 6,565 |
| `0070_58022307_gemini-2.0-flash.txt` | long_line | 6,452 |
| `0071_58022308_gemini-2.0-flash.txt` | block_loop | 1,790 |

**smith_s_tourist_guide_of_necessary_infor_e20bf5b0**

| page | failure | OCR lines |
|---|---|---|
| `0095_58051339_gemini-2.0-flash.txt` | block_loop | 1,534 |
| `0098_58051342_gemini-2.0-flash.txt` | block_loop | 539 |

**the_green_book_9ea5d5b0**

| page | failure | OCR lines |
|---|---|---|
| `0084_5214071_gemini-2.0-flash.txt` | block_loop | 9,464 |
| `0022_5207547_gemini-2.0-flash.txt` | block_loop | 3,212 |
| `0036_5206486_gemini-2.0-flash.txt` | repetition | 10,139 |
| `0028_5205916_gemini-2.0-flash.txt` | block_loop | 3,607 |
| `0113_5206001_gemini-2.0-flash.txt` | repetition | 5,245 |
| `0003_5488760_gemini-2.0-flash.txt` | block_loop | 8,802 |

**the_travelers_guide_e088efa0**

| page | failure | OCR lines |
|---|---|---|
| `0078_58019133_gemini-2.0-flash.txt` | long_line | 205 |
| `0080_58019135_gemini-2.0-flash.txt` | repetition | 2,538 |
| `0081_58019136_gemini-2.0-flash.txt` | repetition | 2,991 |

**travelguide_634f3af0**

| page | failure | OCR lines |
|---|---|---|
| `0008_58017316_gemini-2.0-flash.txt` | block_loop | 2,787 |
| `0004_58030241_gemini-2.0-flash.txt` | block_loop | 466 |
| `0013_58030360_gemini-2.0-flash.txt` | block_loop | 1,508 |
| `0026_58030473_gemini-2.0-flash.txt` | block_loop | 3,222 |
| `0058_58030505_gemini-2.0-flash.txt` | block_loop | 2,401 |
| `0120_58030772_gemini-2.0-flash.txt` | block_loop | 8,322 |

Total: 21 pages
