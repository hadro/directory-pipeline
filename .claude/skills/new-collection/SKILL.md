---
name: new-collection
description: |
  Walks the user through first-time calibration for a new collection type in the
  directory-pipeline. Trigger this skill when the user says "I have a new collection",
  "starting fresh", "never run this before", "how do I calibrate", "first time setup",
  or asks about `--select-pages` or `--generate-prompts`. Explains the "calibrate
  once, run many" pattern and walks through the three setup steps in sequence.
---

When this skill is invoked, follow these steps exactly. Run all commands from the
project root directory.

## Intro: the calibrate-once pattern

Start by explaining the approach:

The pipeline is designed so that you calibrate once for a collection type (e.g.
"NYPL Green Book 1950s") and then reuse those calibration outputs for every
additional volume in that series. Calibration produces two reusable files:

- `ner_prompt.md` — a Gemini-generated prompt customized to the layout and content
  of your specific collection type
- `ocr_prompt.md` — OCR instructions tuned to the collection's formatting

Once you have these from volume 1, every subsequent volume in the same series can
skip calibration and go straight to OCR + extraction.

## Step 1: Download the images

Ask the user for the IIIF manifest URL or collection URL. Show:

```bash
python main.py <URL> --download
```

Explain:
- This fetches all IIIF images to `output/{slug}/` where `{slug}` is auto-derived
  from the collection metadata. Override with `--slug my-name`.
- It also writes `manifest.json`, which maps image filenames to IIIF canvas URIs
  (needed for image thumbnails in the explorer and IIIF viewer).
- Accepted URL formats: NYPL digital collections, Library of Congress, Internet
  Archive, or any direct IIIF manifest URL.

Tell the user to run it and note the output directory path (e.g. `output/green-book-1947/`).

## Step 2: Select representative pages

```bash
python main.py output/{slug} --select-pages
```

This opens a browser UI. Walk through each of its two tabs:

**Sample tab** — pick 4–10 representative pages:
- Choose pages that show the typical layout: dense listing columns, city/state
  headings, any recurring special formatting.
- Avoid: all-caps section headers, blank pages, unusual one-off layouts.
- Selections are saved to `selection.txt`.

**Scope tab** — exclude sections that should not be processed:
- Exclude frontmatter (title page, preface, table of contents), back-matter indexes,
  advertisement sections.
- This creates `included_pages.txt`, which filters which pages get OCR'd and extracted.

Tell the user: "Complete BOTH tabs before closing the browser. The Sample tab drives
prompt generation; the Scope tab prevents junk pages from being processed."

## Step 3: Generate prompts

```bash
python main.py output/{slug} --generate-prompts
```

Explain:
- Sends the selected sample pages to Gemini, which analyzes the layout and generates:
  - `ocr_prompt.md` — OCR instructions for this collection's formatting
  - `ner_prompt.md` — extraction field definitions and edge cases for this directory type
- Both files are saved in `output/{slug}/`.
- Requires `GEMINI_API_KEY` in `.env`.

## Step 4: Show the reuse pattern

After calibration, show how to process additional volumes:

```bash
# First volume (after calibration above):
python main.py output/{slug-vol1} --extract \
  --ner-prompt output/{slug-vol1}/ner_prompt.md

# Subsequent volumes — reuse prompts from volume 1:
python main.py <URL-vol2> --extract \
  --ner-prompt output/{slug-vol1}/ner_prompt.md
```

`--extract` is shorthand for `--download --gemini-ocr --extract-entries --explore`.

For the full precision path (bounding boxes + manual alignment review), use
`--guided` instead of `--extract`. This adds Surya OCR, NW alignment, and pauses
for interactive review before extracting entries.

Notes:
- `--select-pages` and `--generate-prompts` do not need to be re-run for later
  volumes unless the layout changes significantly.
- If a later volume has a very different format, run calibration again for it.

## Output file summary

| File | Stage | Purpose |
|------|-------|---------|
| `output/{slug}/manifest.json` | `--download` | Maps filenames to IIIF canvas URIs; needed for thumbnails |
| `output/{slug}/selection.txt` | `--select-pages` (Sample tab) | Sample pages for prompt generation |
| `output/{slug}/included_pages.txt` | `--select-pages` (Scope tab) | Page filter applied during extraction |
| `output/{slug}/ocr_prompt.md` | `--generate-prompts` | Custom OCR prompt for this collection type |
| `output/{slug}/ner_prompt.md` | `--generate-prompts` | Custom NER prompt; reuse across volumes |
