# Known failure modes and gotchas

## 1. Several OCR models' alignments in one volume

**Symptom:** Boxes land on the wrong lines, or come from an older alignment, after a
volume was re-OCR'd with a different model.

**Cause:** Extraction picks which `*_{ocr_model}_aligned.json` set to read. If no files
exist for the expected model, it auto-detects one from the files present and prints
"Auto-detected OCR model …". Where several models' alignments exist, the one with the best
line coverage wins. (Before this, a plain alphabetical glob picked `gemini-2.0-flash`
over newer models deterministically.)

**Detection:** The `"aligned_model"` field in each page's `*_entries.json` names the
alignment that produced its boxes.

**Fix:** To use a specific alignment:
```bash
pipeline extract output/{slug} --aligned-model {ocr_model} --force
```

---

## 2. Scope filter extension mismatch

**Symptom:** Aligned files exist but extraction processes no pages, or falls back to
raw text.

**Cause:** `included_pages.txt` (written by `--select-pages`) stores image filenames
ending in `.jpg`. The scope filter in `extract_entries.py` strips the last extension
to get the stem (e.g. `foo.jp2.jpg` → stem `foo.jp2`). But aligned JSON filenames
use `.jp2` as their stem (e.g. `foo_{ocr_model}_aligned.json`). When the scope
filter compares `foo.jp2` (from `included_pages.txt`) against aligned file stems, it
may fail to match if the `.jp2` is in one place but not the other.

**Detection:** `len(passing_aligned_files) == 0` even though aligned files exist.
Compare stem format in `included_pages.txt` against `*_aligned.json` filename stems.

**Fix:** Delete `included_pages.txt` to process all pages, or edit it to match the
exact stem format used in the aligned JSON filenames.

---

## 3. Cached entries, and what `"mode"` means

**Symptom:** Re-running extraction does not change a page's entries.

**Cause:** `extract_entries.py` reuses a page's `*_entries.json` unless the aligned JSON
it was built from is newer, so re-aligning a page re-extracts it automatically. A page
extracted from raw text before alignment existed is re-extracted once its alignment is
written.

**Detection:** Don't use the `"mode"` field. It is the NER input mode, `"text-only"`
(the default) or `"multimodal"`, and is unrelated to alignment. A page extracted from
aligned JSON carries `"aligned_model"`; one extracted from raw text does not.

**Fix:** `--force` re-runs NER on every page. That costs API calls, so use it only when
the cache really is stale.

---

## 4. Entries without a box: the binder refused them

**Symptom:** After a correct alignment, some entries still have a page-level
`canvas_fragment` (no `#xywh=`), so the explorer shows the full page for them.

**Cause:** `utils/entry_binding.py` lets only identity fields supply a box: `line_text`,
`name`, `subject`, `title`, `caption`, anything ending `_name`, plus address-like
fields. An entry with no confident match gets no box rather than a best guess. The old
matcher fell through to any string field, so a Green Book entry matched its own
`category` to the printed section heading. That was 489 of about 900 misplacements.
Context fields named after the set (`section_name`, `list_name`, `index_name`, …) are
excluded too.

**Detection:** The page's `*_entries.json` lists the entry under `unplaced_entries`, with
its nearest miss and score.

**Fix:** Usually none needed. A refusal is a result, not an error. For a volume-wide
review list:
```bash
python tools/collate_review_queue.py output/{slug}/
```
To measure placement before and after a change: `python tools/audit_entry_bindings.py`.

---

## 5. Explorer shows no thumbnails (CSV outside the volume directory)

**Symptom:** The data explorer HTML opens but shows no images.

**Cause:** `explore_entries.py` finds `manifest.json` through `--output-dir`, which
defaults to the CSV's parent directory. A CSV that lives elsewhere (a combined CSV, a
copy) finds no manifest.

**Fix:**
```bash
python -m pipeline.explore_entries path/to/entries.csv --output-dir output/{slug}/
```

---

## 6. `--select-pages` scope tab not used

**Symptom:** Extraction processes frontmatter, title pages, index sections, ads —
producing many junk entries.

**Cause:** The `--select-pages` UI has two tabs: Sample (for prompt calibration) and
Scope (for filtering which pages get processed). If the Scope tab was never used,
all pages including frontmatter are processed.

**Fix:** Re-run `--select-pages`, use the Scope tab to exclude frontmatter/back-matter,
save, then re-run `--extract-entries --force`.

---

## 7. Prompt reuse not applied to subsequent volumes

**Symptom:** Entry quality drops on a second volume in the same series because the
wrong (generic) NER prompt was used.

**Cause:** `extract_entries.py` auto-detects a nearby `ner_prompt.md` but only if it
exists in the same directory. For a new volume with a different output slug, no prompt
is found and the global default is used.

**Fix:** Always pass `--ner-prompt output/{first-slug}/ner_prompt.md` when processing
additional volumes in the same series.

---

## 8. Pages silently missing: the blank-page check

**Symptom:** Some pages have an image but no `*_{model}.txt`, and never reach align or
extract.

**Cause:** `is_blank_page()` skips pages it judges blank before any Gemini call, and a
skipped page writes nothing. Until 2026-10-16 it skipped 1,142 pages that had ink: small
type on toned paper, grey-printed Green Books, light microfilm. The fixed rule keeps any
page with measurable ink. Details: `docs/evidence/2026-10-16-blank-page/`.

**Detection:** The OCR run's summary line counts "blank" pages. Compare images against
`.txt` files.

**Fix:** Re-run `pipeline ocr output/{slug}`, which OCRs any page the fixed check passes.
`--no-blank-skip` forces every page.

**The reverse case:** OCR written before the check existed (2026-06-11) can be text
Gemini *invented* for a blank page. 44 such pages were quarantined under
`output/_quarantine/hallucinated-blank-pages-2026-10-17/`. Restoring the 2026-09-04
alignment backup needs that folder's `exclude-from-restore.txt` (`tar -X`).

---

## 9. `[OCR FAILED: …]` placeholder text files

**Symptom:** A page's `.txt` is one line starting `[OCR FAILED`, and its entries are
nonsense or empty.

**Cause:** Versions before 0.3 wrote that placeholder when every attempt failed. A
non-empty `.txt` counted as done, so the page was never retried, and align and extract
read the placeholder as page text.

**Fix:** In 0.3 and later, `pipeline ocr` deletes and retries placeholders. Then re-run
`pipeline extract`.

---

## 10. Thumbnails broken only for narrow crops, on NYPL or Internet Archive images

**Symptom:** Some entry thumbnails are broken images; others work.

**Cause:** Explorers and maps built before 0.3 requested size `full` for crops narrower
than the display width. IIIF Image API 3 servers such as iiif.nypl.org answer 400 to
`full`; they need `max`, and v2 servers such as tile.loc.gov need `full`.

**Fix:** Rebuild the explorer (or map) with 0.3 or later, which picks the keyword per
service.

---

## 11. Two different "section" columns

**Symptom:** A CSV has both `section` and `volume_section`, or the explorer's section
switch shows unexpected values.

**Cause:** `volume_section` is the structural section from `sections.txt`, written by
extraction and used by the explorer's section switch. `section` is whatever the NER schema
defines; Tulsa's and others' prompts use that name for their own field.

**Fix:** None needed. They are separate on purpose. The switch appears only for volumes
with a `sections.txt` and two or more sections.

---

## 12. Every box on a page offset: a placeholder canvas

**Symptom:** All the crops on a page land in the wrong region, usually on NYPL images.

**Cause:** When `info.json` cannot be fetched, alignment falls back to the manifest's
canvas size. On NYPL that can be a 2560×2560 placeholder, so every `#xywh=` on the page
is in the wrong coordinate space. Until 0.3 the only signal was one warning line per page;
five 1958 Green Book pages went unnoticed that way.

**Detection:** Since 0.3 the align summary lists these pages ("boxes scaled to a square
placeholder canvas"), even under `--quiet`, and prints the repair commands. It scans every
aligned file on disk, not only that run's. Aligned files written before 2026-09-20 lack
the flag it reads; check those with
`python tools/rescale_canvas_fragments.py output/{slug} --check`.

**Fix:** No API calls needed:
```bash
python tools/rescale_canvas_fragments.py output/{slug} --aligned-model {ocr_model}
python tools/patch_canvas_fragments.py output/{slug} --aligned-model {ocr_model}
```
Then rebuild the explorer. Both tools default `--aligned-model` to the commonest model
among the aligned files under the directory (before 0.3, the retired `gemini-2.0-flash`),
so pass it when a volume has several.

---

## 13. Pages lost without a report before 0.3

**Symptom:** An older volume is missing pages or entries, but its runs reported success.

**Cause:** Three stages could lose a page and still exit 0 before 0.3:
- **Download:** a page that timed out was a warning line, and the stage reported done.
- **Gemini OCR:** a response that a safety or content filter blocked was counted
  "empty", not failed. It was not retried and wrote no `.txt`.
- **NER:** a page whose call errored or returned unparseable JSON was one log line. The
  stage was recorded complete, and the next page carried on with the context of the page
  before the lost one.

0.3 retries timeouts, sends blocked responses through the retry ladder, and makes all
three stages fail with the page names. `main.py` now exits 1 when any target fails.

**Detection:** Compare images with `manifest.json` canvases, `.txt` files with images, and
`*_entries.json` files with `.txt` or aligned files. Each gap is a lost page.

**Fix:** Re-run the stage. A lost page has no output file, so it is not skipped: re-run the
download with the same source, then `pipeline ocr output/{slug}` and
`pipeline extract output/{slug}`. No `--force` needed.

---

## 14. `--infer-categories` changed nothing

**Symptom:** `fix_entries.py --infer-categories` (or `pipeline postprocess --infer-categories`)
runs, but the fixed CSV's categories are unchanged.

**Cause:** Before 0.3, `detect_schema()` did not recognise a category column named
`section` (the 1958 Green Book's NER prompt uses that name). Inferred categories went to a
`category` key the CSV did not have, and the writer dropped them silently: 1,816 rows on
that volume.

**Fix:** Re-run `pipeline postprocess output/{collection}/ --infer-categories` with 0.3 or
later. A CSV with both columns still takes `category` as the category column.
