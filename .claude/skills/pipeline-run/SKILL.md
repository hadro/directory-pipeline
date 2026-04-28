---
name: pipeline-run
description: |
  Builds the exact `python main.py` command for the directory-pipeline project.
  Trigger this skill when the user asks "what command should I run", "how do I run
  the pipeline", "what flags do I need", "run the pipeline", or describes a goal
  like "extract entries" or "add bounding boxes". The skill inspects the output
  directory to detect what has already been run, then outputs a precise,
  ready-to-paste command with an explanation of what each flag does.
---

When this skill is invoked, follow these steps exactly. Always run commands from
the project root directory.

## Step 1: Ask the user's goal

Present the following numbered choices and ask the user to pick one:

1. Run full pipeline from scratch (download → OCR → CSV + explorer)
2. Run full pipeline with precision bounding boxes and manual review (`--guided`)
3. Add precision bounding boxes to an existing download (Surya OCR + align)
4. Extract or re-extract structured entries from existing OCR
5. Create a data explorer (interactive HTML from an existing entries CSV)

## Step 2: Ask for the source

Ask: "What is your source? Provide either a IIIF manifest URL (or NYPL/LOC/IA
collection URL), or the path to an existing output directory (e.g. `output/green-book-1956/`)."

## Step 3: Inspect the output directory

If the user provided an output directory path (not a URL), use Glob and Read tools
to detect pipeline state. Check for these files inside the directory (or its item
subdirectories):

- `manifest.json` — download has been run
- `*_surya.json` — Surya OCR has been run
- `*_gemini.txt` — Gemini OCR has been run
- `*_aligned.json` — alignment has been run; note the filename stem to detect the
  OCR model (e.g. `foo_gemini-2.0-flash_aligned.json` → model is `gemini-2.0-flash`)
- `selection.txt` — select-pages has been run
- `ner_prompt.md` — prompts have been generated
- `entries_*.csv` — extraction has been run
- `entries_*.json` — per-page extraction cache; check `"mode"` field inside for
  `"text-only"` vs `"aligned"`

## Step 4: Build the command

Use the detected state and the user's goal to construct the command.

### Goal 1: Full pipeline from scratch

```
python main.py <URL> --extract
```

`--extract` is shorthand for `--download --gemini-ocr --extract-entries --explore`.
If `ner_prompt.md` exists from a prior collection run, append:
```
--ner-prompt output/{prior-slug}/ner_prompt.md
```

### Goal 2: Full pipeline with precision bounding boxes and manual review

```
python main.py <URL> --guided
```

`--guided` is shorthand for `--download --select-pages --surya-ocr --gemini-ocr
--align-ocr --review-alignment --extract-entries --geocode --map`. It pauses at
`--select-pages` (to scope which pages to process) and again at `--review-alignment`
(to correct unmatched lines) before continuing. Requires GPU or Apple Silicon for Surya.

If `ner_prompt.md` exists from a prior collection run, append:
```
--ner-prompt output/{prior-slug}/ner_prompt.md
```

### Goal 3: Add precision bounding boxes

Requires Surya (GPU or Apple Silicon). Command:
```
python main.py output/{slug} --surya-ocr --gemini-ocr --align-ocr
```

If `*_{model}.txt` files already exist, `--gemini-ocr` can be omitted (output is cached).

After alignment, `--review-alignment` opens a Flask browser UI to manually fix
unmatched lines.

### Goal 4: Extract or re-extract entries

Base command:
```
python main.py output/{slug} --extract-entries
```

Conditional additions:
- If `*_aligned.json` files exist, detect the OCR model from the filename stem.
  If the model differs from the NER model (or is unclear), append:
  `--aligned-model gemini-2.0-flash` (use the detected model slug)
- If `ner_prompt.md` exists in the output dir, append:
  `--ner-prompt output/{slug}/ner_prompt.md`
- If `entries_*.json` already exists and the user wants a fresh run, or if mode is
  `"text-only"` but alignment now exists, append `--force`
- To use a specific NER model: `--model MODEL`

### Goal 5: Create data explorer

Base command:
```
python pipeline/explore_entries.py output/{slug}/entries_{model}.csv
```

If `manifest.json` exists in `output/{slug}/`, append `--output-dir output/{slug}/`
to enable image thumbnails. If `palette.json` exists, it is applied automatically.

## Step 5: Present the command and explain it

Show the final command in a code block. Provide a brief explanation of each flag
used, and describe what output file(s) to expect.

All stages run in fixed order regardless of flag order on the command line.
See `main.py` lines 111–135 for the full PIPELINE stage list.
