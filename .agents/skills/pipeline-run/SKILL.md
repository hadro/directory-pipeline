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

Ask: "What is your source? Provide either a IIIF manifest URL (or LoC/IA
collection URL), or the path to an existing output directory (e.g. `output/green-book-1956/`)."

## Step 3: Inspect the output directory

If the user provided an output directory path (not a URL), use Glob and Read tools
to detect pipeline state. Check for these files inside the directory (or its item
subdirectories):

- `manifest.json` — download has been run
- `*_surya.json` — Surya OCR has been run
- `*_gemini-*.txt` (`*_{model}.txt`) — Gemini OCR has been run
- `*_aligned.json` — alignment has been run; the filename stem names the OCR model
  (e.g. `foo_gemini-3.1-flash-lite_aligned.json` → model is `gemini-3.1-flash-lite`)
- `pipeline_state.json` — the models used and the stages completed; downstream stages
  read it, so `--model` flags are rarely needed
- `selection.txt` — select-pages has been run
- `ner_prompt.md` — prompts have been generated
- `entries_*.csv` — extraction has been run
- `*_entries.json` — per-page extraction cache. An `"aligned_model"` field means the
  page was extracted from aligned JSON (boxes possible); without it, from raw text. The
  `"mode"` field is the NER input mode (`"text-only"` default, or `"multimodal"`) and
  says nothing about alignment

## Step 4: Build the command

Use the detected state and the user's goal to construct the command.

### Goal 1: Full pipeline from scratch

```
pipeline run <URL>
```

(equivalent to `python main.py <URL> --extract`, shorthand for
`--download --gemini-ocr --extract-entries --explore`; Flex inference is on
by default — add `--no-flex` for time-sensitive runs).
If `ner_prompt.md` exists from a prior collection run, append:
```
--ner-prompt output/{prior-slug}/ner_prompt.md
```

### Goal 2: Full pipeline with precision bounding boxes and manual review

```
pipeline guided <URL>
```

(equivalent to `python main.py <URL> --guided`). `--guided` is shorthand for `--download --select-pages --surya-ocr --gemini-ocr
--align-ocr --review-alignment --extract-entries --explore` — the `pipeline run` arc plus the
human-in-the-loop stages. It pauses at
`--select-pages` (to scope which pages to process) and again at `--review-alignment`
(to correct unmatched lines) before continuing. Requires GPU or Apple Silicon for Surya.
Geo stages are not included; for materials with address fields, run
`pipeline geo output/{slug}/` (or `--geocode --map`) afterwards.

If `ner_prompt.md` exists from a prior collection run, append:
```
--ner-prompt output/{prior-slug}/ner_prompt.md
```

### Goal 3: Add precision bounding boxes

Requires Surya (GPU or Apple Silicon). Command:
```
pipeline ocr output/{slug}
```

(equivalent to `python main.py output/{slug} --surya-ocr --gemini-ocr --align-ocr`;
already-completed OCR files are skipped automatically).

Options worth knowing:
- `--blank-threshold 0.5` — for pages Surya reads as tables (dot-leader indexes), which
  otherwise come back as page-sized blocks; the align run lists such pages at the end
- `--no-blank-skip` — OCR every page, including ones the blank-page check would skip
- `--max-output-tokens N` — per-page output cap (default 24576); raise if dense pages
  come back truncated

After alignment, `pipeline review output/{slug}` opens a Flask browser UI to
manually fix unmatched lines.

### Goal 4: Extract or re-extract entries

Base command:
```
pipeline extract output/{slug}
```

(equivalent to `python main.py output/{slug} --extract-entries --explore`;
add `--no-explore` to skip the explorer build). Fine-grained flags below pass
through unchanged.

Conditional additions:
- The OCR model is auto-detected from the aligned files (best line coverage wins when
  several models' alignments exist). Append `--aligned-model {ocr_model}` only to force
  a specific one.
- If `ner_prompt.md` exists in the output dir, append:
  `--ner-prompt output/{slug}/ner_prompt.md`
- Re-aligned pages are re-extracted automatically: a page's cache is invalidated when its
  aligned JSON is newer. Append `--force` only for a genuinely fresh run of every page,
  which costs NER API calls for all of them.
- To use a specific NER model: `--model MODEL`

### Goal 5: Create data explorer

Base command:
```
python -m pipeline.explore_entries output/{slug}/entries_{model}.csv
```

`--output-dir` (where `manifest.json` is found, for thumbnails) defaults to the CSV's
parent directory. Append `--output-dir output/{slug}/` only when the CSV lives elsewhere.
`--context-lines N` sets how many lines of surrounding text a thumbnail crop shows
(default 3).

## Step 5: Present the command and explain it

Show the final command in a code block. Provide a brief explanation of each flag
used, and describe what output file(s) to expect.

All stages run in fixed order regardless of flag order on the command line.
See `STAGES` in `pipeline/stages.py` for the full stage order, and "Artifacts and naming conventions" in `docs/pipeline-stages.md` for the filename contract.
