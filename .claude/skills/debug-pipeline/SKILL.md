---
name: debug-pipeline
description: |
  Diagnoses problems with the directory-pipeline. Trigger this skill when the user
  reports something isn't working: "no thumbnails", "empty CSV", "no bounding boxes",
  "canvas_fragment has no xywh", "entries are missing", "pipeline failed", or any
  symptom suggesting a stage produced wrong or missing output. The skill asks for
  the symptom, runs targeted file checks, and gives the exact fix command.
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
5. Pipeline failed with an error (ask the user to paste the error message)
6. Other (ask user to describe)

## Step 2: Ask for the output directory

Ask: "What is the path to your output directory (e.g. `output/green-book-1956/`)?"

## Step 3: Run targeted diagnostics

Use Bash, Glob, and Read tools to inspect files. Run only the checks relevant to
the reported symptom.

### Symptom 1: No thumbnails in explorer

Checks:
- Glob `{output_dir}/manifest.json` and `{output_dir}/*/manifest.json`
- Confirm whether `--output-dir` was passed when running the explorer

Report:
- If `manifest.json` is missing: "The explorer needs `manifest.json` to serve
  thumbnails. Re-run with `--download`, or run from the output directory that
  contains the downloaded images."
- If `--output-dir` was not passed: "Re-run with `--output-dir output/{slug}/`
  to point the explorer at the images directory."

### Symptom 2: Thumbnails show full page, not cropped

Checks:
- Glob `{output_dir}/*_aligned.json` and `{output_dir}/*/aligned.json` — count files
- Read one `entries_*.json` file and check the `"mode"` field on a sample of records

Report:
- If no `*_aligned.json` files: "Alignment has not been run. Run:
  `python main.py output/{slug} --surya-ocr --gemini-ocr --align-ocr`
  then re-extract with `--force`."
- If `*_aligned.json` exists but `entries_*.json` has `"mode": "text-only"`:
  "Entries were extracted before alignment, or `--aligned-model` was not passed.
  Re-run: `python main.py output/{slug} --extract-entries --aligned-model {ocr_model} --force`"

### Symptom 3: Entries CSV empty or missing entries

Checks:
- Glob `{output_dir}/ner_prompt.md` — missing prompt means generic prompting
- Glob `{output_dir}/*_entries_error.txt` — API error logs
- Bash: `grep -l GEMINI_API_KEY .env 2>/dev/null` — confirm key is set (do not print value)
- Glob `{output_dir}/entries_*.json` and read a small sample

Report:
- If `*_entries_error.txt` exists: read it and report the Gemini error
- If API key not set: "Add `GEMINI_API_KEY=...` to `.env` in the project root."
- If `ner_prompt.md` missing: "No custom NER prompt found — results may be poor.
  Run `--generate-prompts` first, or pass `--ner-prompt` pointing to a prompt
  from a related collection."

### Symptom 4: `canvas_fragment` has no `#xywh=`

This is the most common complex failure. Run all checks below in order.

**Check A — aligned files present?**
- Glob `{output_dir}/*_aligned.json` and `{output_dir}/*/aligned.json`
- If none: alignment was never run. Fix:
  `python main.py output/{slug} --surya-ocr --gemini-ocr --align-ocr`
  then re-extract with `--force`

**Check B — aligned model mismatch?**
- Read the filename stems of `*_aligned.json`. Example:
  `foo_gemini-2.0-flash_aligned.json` → OCR model is `gemini-2.0-flash`
- Ask the user what command they ran for extraction. If `--aligned-model` was not
  passed, that is the bug.
- Fix: `python main.py output/{slug} --extract-entries --aligned-model {ocr_model} --force`

**Check C — scope filter mismatch?**
- Glob `{output_dir}/included_pages.txt`
- If it exists, read it and note the file extension of listed stems (`.jpg`, `.jp2`, etc.)
- Glob `*_aligned.json` and note the extension in the filename stems
- If extensions differ (e.g. `included_pages.txt` lists `.jpg` stems but aligned
  files use `.jp2` stems), the scope filter silently eliminates all aligned files,
  forcing text-only fallback.
- Fix: edit `included_pages.txt` so stems match the aligned file naming, or delete
  it to process all pages.

**Check D — entries cache blocking fresh run?**
- Glob `{output_dir}/entries_*.json` and read a sample, check `"mode"` field
- If `"mode": "text-only"`: "Cached entries used text-only mode. Pass `--force` to
  overwrite with aligned mode."

### Symptom 5: Pipeline failed with error

Ask the user to paste the error message or traceback. Then:
- `ModuleNotFoundError` for surya: "Run `uv sync --extra gpu` (requires GPU or Apple Silicon)."
- `ModuleNotFoundError` for geocoder/geopy: "Run `uv sync --extra geo`."
- `FileNotFoundError` for `.json` or `.txt`: check if a prior stage was skipped.
- Gemini API error: check `.env` for `GEMINI_API_KEY`.

## Step 4: Present findings and fix command

Summarize what was found. Give a precise, copy-pasteable fix command. If multiple
issues were found, list them in order of likely impact.
