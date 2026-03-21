---
name: explore-entries
description: |
  Builds the command to launch the interactive data explorer for the directory-pipeline.
  Trigger this skill when the user asks "how do I explore the data", "open the explorer",
  "view the entries", "create the HTML explorer", "browse the CSV", or mentions
  `explore_entries.py`. The skill inspects the output directory to detect available
  files and builds the right command with the correct flags for thumbnails, palette,
  and translated variants.
---

When this skill is invoked, follow these steps exactly. Run all commands from the
project root directory.

## Step 1: Ask which output to explore

Ask: "Which output directory or CSV file do you want to explore? For example:
`output/green-book-1956/` or `output/green-book-1956/entries_gemini-2.0-flash.csv`."

## Step 2: Resolve the target

If the user gave a directory path, use Glob to find entries CSV files:
- Pattern: `{output_dir}/entries_*.csv` and `{output_dir}/*/entries_*.csv`

If multiple CSVs are found, list them and ask which to use. Common variants:
- `entries_{model}.csv` — standard extraction
- `entries_{model}_geocoded.csv` — after geocoding
- `entries_translated.csv` — after translation

If the user gave a CSV path directly, use that.

## Step 3: Check for supporting files

Use Glob to look for the following alongside the CSV (or in the output directory):

- `manifest.json` — enables image thumbnails if present
- `palette.json` — auto-applied for color theming if present (no extra flag needed)
- `*.jpg` or `*.jp2` — confirms images are local

## Step 4: Build the command

Base command:
```bash
python pipeline/explore_entries.py output/{slug}/entries_{model}.csv
```

**Add `--output-dir` for thumbnails:** If `manifest.json` exists in `output/{slug}/`:
```bash
python pipeline/explore_entries.py output/{slug}/entries_{model}.csv \
  --output-dir output/{slug}/
```
Without `--output-dir`, the explorer has no images. With it, entries that have
`#xywh=` in their `canvas_fragment` show cropped snippets; others show the full page.

**Palette:** If `palette.json` is found, tell the user: "A `palette.json` was
detected — color theming will be applied automatically."

**Translated variant:** If the user wants to explore a translated CSV:
```bash
python pipeline/explore_entries.py output/{slug}/entries_translated.csv \
  --output-dir output/{slug}/
```
To create a translated CSV first, see the translation workflow (ask about `/pipeline-run`).

## Step 5: Present the command and explain output

Show the complete command in a code block. Explain:
- Output is written to `output/{slug}/entries_{model}_explorer.html` (or a name
  derived from the CSV filename). Open in any browser — no server needed.
- The HTML file must stay in the same relative location as the output directory
  for local image links to work correctly.
- If `canvas_fragment` values contain `#xywh=` bounding boxes (from `--align-ocr`),
  thumbnails are cropped to the exact entry region. Without alignment, the full
  page is shown. To add bounding boxes, run `--surya-ocr --align-ocr` first.
- To generate a custom color palette from cover pages, run `--generate-palette` first.
