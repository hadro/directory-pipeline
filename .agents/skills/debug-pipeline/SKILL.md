---
name: debug-pipeline
description: |
  Diagnoses problems with the directory-pipeline. Trigger this skill when the user
  reports something isn't working: "no thumbnails", "empty CSV", "no bounding boxes",
  "canvas_fragment has no xywh", "crops show the wrong part of the page", "entries are
  missing", "pipeline failed", or any symptom suggesting a stage produced wrong or
  missing output. The skill asks for the symptom, runs targeted file checks, and gives
  the exact fix command.
---

When this skill is invoked, follow these steps exactly. Run all commands from the
project root directory. See `references/gotchas.md` for a full catalogue of known
failure modes.

## Step 1: Ask for the symptom

Present these numbered options and ask the user which best describes their problem:

1. No image thumbnails / snippets in the data explorer
2. Thumbnails show the full page, not cropped to the entry
3. Entries CSV is empty or missing many entries
4. Pipeline ran but `canvas_fragment` column has no `#xywh=` bounding box
5. Crops show the wrong part of the page
6. Pipeline failed with an error (ask the user to paste the error message)
7. Other (ask user to describe)

## Step 2: Ask for the output directory

Ask: "What is the path to your output directory (e.g. `output/green-book-1956/`)?"

## Step 3: Run targeted diagnostics

Use Bash, Glob, and Read tools to inspect files. Run only the checks relevant to
the reported symptom.

**One thing to know first:** the `"mode"` field in a page's `*_entries.json` is the NER
input mode (`"text-only"` or `"multimodal"`), and `"text-only"` is the default. It says
nothing about alignment. A page extracted from aligned JSON has an `"aligned_model"` field;
a page extracted from raw OCR text does not.

### Symptom 1: No thumbnails in explorer

Checks:
- Glob `{output_dir}/manifest.json` and `{output_dir}/*/manifest.json`
- Where does the CSV live? The explorer's `--output-dir` defaults to the CSV's parent
  directory, which only finds the manifest if the CSV sits in the volume directory
- Are thumbnails broken only for *some* entries, on NYPL or Internet Archive material?

Report:
- If `manifest.json` is missing: "The explorer needs `manifest.json` to serve
  thumbnails. Re-run with `--download`, or point `--output-dir` at the directory that
  holds it."
- If the CSV is outside the volume directory: "Re-run with `--output-dir output/{slug}/`."
- If only narrow crops are broken on NYPL/IA images: the explorer was built before 0.3,
  which asked IIIF Image API 3 servers for size `full` (they answer 400). Rebuild it:
  `python -m pipeline.explore_entries output/{slug}/entries_{model}.csv`.

### Symptom 2: Thumbnails show full page, not cropped

Checks:
- Glob `{output_dir}/*_aligned.json` and `{output_dir}/*/*_aligned.json` — count files
- Read a sample of `*_entries.json` files: is `"aligned_model"` present?
- In those files, count `unplaced_entries`

Report:
- If no `*_aligned.json` files: "Alignment has not been run. Run
  `pipeline ocr output/{slug}`, then `pipeline extract output/{slug}`." Re-aligned pages
  are re-extracted automatically (the cache is invalidated when the aligned JSON is
  newer), so `--force` is not needed.
- If aligned JSON exists but `*_entries.json` lack `"aligned_model"`: those pages were
  extracted from raw text. `pipeline extract output/{slug}` re-extracts them if their
  alignment is newer; otherwise add `--force` (this re-runs NER on every page and costs
  API calls).
- If `"aligned_model"` is present but some entries still show the full page: the binder
  refused to place them. That is by design; see Symptom 4, Check E.

### Symptom 3: Entries CSV empty or missing entries

Checks:
- The end of the extract run's output. Since 0.3 it names every page whose NER call
  failed ("NER failed on N page(s): …") and exits 1, even under `--quiet`
- Pages with a `*_{model}.txt` or `*_aligned.json` but no `*_entries.json`. Before 0.3 a
  failed NER call was one log line and the run exited 0, so an older volume can have
  these with no other record
- Glob `{output_dir}/*_entries_error.txt` — the raw response of a page whose NER output
  could not be parsed
- Glob `{output_dir}/ner_prompt.md` — missing prompt means generic prompting
- Bash: `grep -l GEMINI_API_KEY .env 2>/dev/null` — confirm key is set (do not print value)
- Compare the number of images on disk with the number of canvases in `manifest.json`.
  A page that never downloaded reaches no later stage
- Pages with an image but no `*_{model}.txt`. List them; each is a page no later
  stage ever saw
- Bash: `grep -l '^\[OCR FAILED' {output_dir}/*_gemini-*.txt` — old failure placeholders
- Glob `{output_dir}/included_pages.txt` — the scope filter excludes pages from OCR and
  extraction

Report:
- If extraction named failed pages, or pages lack a `*_entries.json`: re-run
  `pipeline extract output/{slug}`. A failed page writes no entries file, so it is retried
  without `--force`; a page with an `*_entries_error.txt` is also retried automatically.
  If one page keeps failing, report its `*_entries_error.txt` (a parse failure) or the
  error logged beside its name (an API failure).
- If `*_entries_error.txt` exists: read it and report what Gemini returned
- If API key not set: "Add `GEMINI_API_KEY=...` to `.env` in the project root."
- If `ner_prompt.md` missing: "No custom NER prompt found — results may be poor.
  Run `--generate-prompts` first, or pass `--ner-prompt` pointing to a prompt
  from a related collection."
- If images are missing: re-run the download with the same source, e.g.
  `pipeline ingest <URL>` (plus `--slug` if the first run used one). Pages already on disk
  are skipped, so only the missing ones are requested. Since 0.3 the download summary
  says "N page(s) failed to download" and the stage fails; before, a page that never
  arrived was a warning line and the run reported success.
- Pages without a `.txt` have one of four causes:
  - **The blank-page check skipped them.** The OCR run's summary counts them as
    "blank". The check was fixed on 2026-10-16: before that it skipped pages with
    faint or grey type. Re-running `pipeline ocr output/{slug}` OCRs any page the
    fixed check passes. `--no-blank-skip` forces every page.
  - **OCR failed on them.** The run exits non-zero and the stage is not recorded as
    complete. Re-running retries them.
  - **Gemini returned no text.** The OCR summary counts these as "empty". Before 0.3
    that included responses a safety or content filter blocked, which were neither
    retried nor counted as failures. 0.3 sends them through the retry ladder and fails
    the page if no attempt returns text. Re-running `pipeline ocr output/{slug}` requests
    every page without a `.txt` again.
  - **The image is corrupt.** The OCR stages re-download a corrupt page first, and
    report it as "corrupt" if that fails.
- If `[OCR FAILED` placeholders exist: versions before 0.3 wrote them, and align and
  extract read them as page text. A 0.3 `pipeline ocr` run deletes and retries them;
  then re-run `pipeline extract`.

### Symptom 4: `canvas_fragment` has no `#xywh=`

This is the most common complex failure. Run all checks below in order.

**Check A — aligned files present?**
- Glob `{output_dir}/*_aligned.json` and `{output_dir}/*/*_aligned.json`
- If none: alignment was never run. Fix: `pipeline ocr output/{slug}`, then
  `pipeline extract output/{slug}`

**Check B — which alignment did extraction use?**
- Extraction auto-detects the OCR model from the files present. When it has to, it
  prints "Auto-detected OCR model …". If several models' alignments exist, the one with
  the best line coverage wins. The `"aligned_model"` field in each `*_entries.json`
  records which one was used.
- To force a specific alignment:
  `pipeline extract output/{slug} --aligned-model {ocr_model} --force`

**Check C — scope filter mismatch?**
- Glob `{output_dir}/included_pages.txt`
- If it exists, read it and note the file extension of listed stems (`.jpg`, `.jp2`, etc.)
- Glob `*_aligned.json` and note the extension in the filename stems
- If extensions differ (e.g. `included_pages.txt` lists `.jpg` stems but aligned
  files use `.jp2` stems), the scope filter silently eliminates all aligned files.
- Fix: edit `included_pages.txt` so stems match the aligned file naming, or delete
  it to process all pages.

**Check D — cached entries?**
- A page's `*_entries.json` is reused unless its aligned JSON is newer. If the entries
  lack `"aligned_model"` and the alignment is older than them, pass `--force` (re-runs
  NER on every page).

**Check E — entries the binder refused (most common since 0.3)**
- `utils/entry_binding.py` gives an entry a box only when an identity field
  (`line_text`, `name`, `subject`, `title`, `caption`, `*_name`, plus address-like fields)
  matches an aligned line confidently. Otherwise
  the entry keeps the page-level fragment. This is deliberate: a confidently wrong box is
  the failure it prevents.
- Refusals are listed under `unplaced_entries` in each page's `*_entries.json`. Collate
  them for a volume with `python tools/collate_review_queue.py output/{slug}/`. Entries
  scoring just under the floor usually have a real line; entries at 0.00 usually do not.
- To measure placement across a volume: `python tools/audit_entry_bindings.py output/{slug}`.

**Check F — Surya returned region blocks?**
- `pipeline ocr` / `align_ocr.py` ends by listing pages that left the most Gemini text
  unboxed. A page near 90% unboxed usually means Surya read a dense index or table as
  page-sized blocks.
- Fix: delete those pages' `*_surya.json` and `*_aligned.json`, then run
  `pipeline ocr output/{slug} --blank-threshold 0.5`. Surya redoes only the deleted
  pages, Gemini skips pages already transcribed, and align fills in the gaps.

### Symptom 5: Crops show the wrong part of the page

Checks:
- Are all the boxes on a page off, or only some entries?
- The align run's summary. Since 0.3 it ends with "⚠ N page(s) have boxes scaled to a
  square placeholder canvas (info.json could not be fetched)" and lists the pages, even
  under `--quiet`. It reads every aligned file on disk, so re-running
  `pipeline ocr output/{slug}` reports pages aligned by earlier runs too
- Aligned files written before 2026-09-20 lack the flag that report reads. For those:
  `python tools/rescale_canvas_fragments.py output/{slug} --check` compares each page's
  stored canvas size with its live `info.json` and exits 1 on a mismatch

Report:
- If pages were aligned against a placeholder canvas: when `info.json` could not be
  fetched, alignment fell back to the manifest's canvas size. On NYPL that can be a
  2560×2560 placeholder, which puts every box on the page in the wrong coordinate space.
  Fix in place, with no API calls:
  ```
  python tools/rescale_canvas_fragments.py output/{slug} --aligned-model {ocr_model}
  python tools/patch_canvas_fragments.py output/{slug} --aligned-model {ocr_model}
  ```
  The first rescales the aligned JSON. The second rebinds the entries cache and the
  volume CSV. Both default `--aligned-model` to the commonest model among the aligned
  files on disk, so pass it when the volume has several. Then rebuild the explorer.
- If only some entries on a multi-column page are off, by a line or a column: this is
  likely the known gutter-hoist bug, deferred to 0.4 (`docs/plans/gutter-hoist-fix.md`).
  Measure it with `python tools/audit_entry_bindings.py output/{slug}`. To correct
  individual lines, run `pipeline review output/{slug}`, then `pipeline extract output/{slug}`.
- If the volume was re-OCR'd with another model and the boxes come from the old
  alignment: see gotcha 1 in `references/gotchas.md`.

### Symptom 6: Pipeline failed with error

Ask the user to paste the error message or traceback. Then:
- `ModuleNotFoundError` for surya: "Run `uv sync --extra gpu` (requires GPU or Apple Silicon)."
- `ModuleNotFoundError` for geocoder/geopy: "Run `uv sync --extra geo`."
- `FileNotFoundError` for `.json` or `.txt`: check if a prior stage was skipped.
- Gemini API error: check `.env` for `GEMINI_API_KEY`.
- A 404 for `gemini-2.0-flash`: that model is retired. Use the defaults in
  `utils/models.py`, or `--ocr-model gemini-3.8-flash` for higher accuracy.
- A 400 about thinking settings after changing a model: see `thinking_config_for()` in
  `utils/gemini.py`. Lite and non-lite 3.x models need different settings.
- "the Gemini API refused a call for lack of credit": the account's prepaid credit is
  spent (an HTTP 402). Since 0.3, OCR and NER stop at the first refusal and exit 1;
  before, each page sat through the rate-limit backoff and then failed. Add credit, then
  re-run the same command: pages already done are skipped.
- "N page(s) failed … stage not recorded as complete" (OCR) or "NER failed on N
  page(s): …" (extract): both exit non-zero and leave the stage unrecorded when any page
  failed. Align also leaves its stage unrecorded on a failed page, though it exits 0.
  Re-run to retry; failed pages write no output, so they are not skipped.
- "N page(s) failed to download": re-run the download with the same source. Pages
  already on disk are skipped, so only the missing ones are requested.
- `✗ Failed: <target>` in `main.py`'s final summary: one of that target's stages failed,
  and since 0.3 the run exits 1. Later stages still ran on whatever the failed stage left,
  so their output is partial. Look further up for the failed stage's own error.

## Step 4: Present findings and fix command

Summarize what was found. Give a precise, copy-pasteable fix command. If multiple
issues were found, list them in order of likely impact.
