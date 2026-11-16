---
name: explore-entries
description: |
  Builds the command to launch the interactive data explorer for the directory-pipeline.
  Trigger this skill when the user asks "how do I explore the data", "open the explorer",
  "view the entries", "create the HTML explorer", "browse the CSV", or mentions
  `explore_entries.py`. The skill inspects the output directory to detect available
  files and builds the right command with the correct flags for thumbnails, crops,
  sections and hosted viewers.
---

When this skill is invoked, follow these steps exactly. Run all commands from the
project root directory.

## Step 1: Ask which output to explore

Ask: "Which output directory or CSV file do you want to explore? For example:
`output/green-book-1956/` or `output/green-book-1956/entries_gemini-3.1-flash-lite.csv`."

## Step 2: Resolve the target

The explorer accepts a CSV, a directory containing one, or several of either.
Several volumes become one explorer with a volume selector.

If the user gave a directory path, use Glob to find entries CSV files:
- Pattern: `{output_dir}/entries_*.csv` and `{output_dir}/*/entries_*.csv`

If multiple CSVs are found, list them and ask which to use. Common variants:
- `entries_{model}.csv` — standard extraction
- `entries_{model}_fixed.csv` — after `pipeline postprocess` / `fix_entries.py`
- `entries_{model}_geocoded.csv` — after geocoding

If the user gave a CSV path directly, use that.

## Step 3: Check for supporting files

Use Glob to look for the following alongside the CSV (or in the output directory):

- `manifest.json` — enables IIIF thumbnails
- `*_aligned.json` — alignment was run, so entries can have `#xywh=` crops
- `sections.txt` — the volume is split into sections. Its rows carry a
  `volume_section` column, and the explorer adds a section switch

## Step 4: Build the command

Base command:
```bash
python -m pipeline.explore_entries output/{slug}/entries_{model}.csv
```

**Thumbnails:** `--output-dir`, where `manifest.json` is looked for, defaults to the CSV's
parent directory. Add it only when the CSV lives somewhere else, such as a combined or
copied CSV:
```bash
python -m pipeline.explore_entries path/to/entries.csv --output-dir output/{slug}/
```

Other options, when they fit the user's goal:
- `--context-lines N` — lines of surrounding text shown above and below an entry in the
  detail crop (default 3)
- `--full-page-thumbs` — show whole pages instead of crops, when bounding boxes are absent
  or unreliable
- `--title TEXT` — header title (otherwise derived from the path)
- `--viewer-url URL` (and optionally `--manifest-url URL`) — for an explorer published
  alongside a self-hosted IIIF viewer; adds a "View in Mirador" link at the exact canvas
- `--out FILE` — output path (default `<csv_stem>_explorer.html` next to the CSV)

## Step 5: Present the command and explain output

Show the complete command in a code block. Explain:
- Output is a single self-contained HTML file, by default
  `<csv_stem>_explorer.html` next to the CSV. Open it in any browser; no server needed.
- Entries whose `canvas_fragment` has `#xywh=` show a crop of their line. Entries without
  one show the full page, for one of two reasons:
  - alignment was never run (`pipeline ocr`, then `pipeline extract`)
  - the binder declined to place that entry, which is deliberate; it is listed under
    `unplaced_entries` in the page's `*_entries.json`
- For a volume with `sections.txt`, the section switch narrows the table and its facets to
  one structural section at a time.
- If thumbnails are broken only for some entries on NYPL or Internet Archive images, the
  HTML was built before 0.3. Rebuild it with this command.
