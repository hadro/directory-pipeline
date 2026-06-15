# Plan: automatic section detection for multi-section directories

**Status:** Planned — Phase 0 not yet started; best implemented in a local session.

> **What this is:** an implementation plan (no code yet) for a `detect_sections`
> stage that analyzes a volume's per-page OCR output and drafts a `sections.txt`
> marking the structural sections of a city directory (alphabetical name list,
> street/reverse directory, classified business directory, front matter, ads).
> Created 2026-06-15. Intended to be implemented by a local instance with the
> actual `output/` data on disk (it is gitignored, so cloud sessions can't see it).

---

## TL;DR for the implementer

- **The consumer side already exists and is fully wired.** `sections.txt` is
  consumed today by `extract_entries.py`, `run_gemini_ocr.py`, `generate_prompt.py`,
  and `select_pages.py` via `utils/section_utils.py`. The *only* missing piece is a
  **producer** that drafts the file. Build that producer. Do **not** redesign the
  consumer contract.
- **Mirror `pipeline/detect_columns.py` end-to-end.** It is the closest twin: an
  opt-in "analyze every page → write a report CSV → don't touch images" stage,
  declarative in `stages.py`, with a `--force` guard. Copy its shape.
- **Heuristic-first, model-at-the-margins.** A deterministic feature scorer over
  artifacts you *already produce* gets ~80–90%. A small model (or one cheap Gemini
  call) only adjudicates low-confidence boundary pages. The deterministic core ships
  first and has no API/network dependency.
- **Never silently overwrite `sections.txt`.** Write `sections_draft.txt` +
  `sections_report.csv`. The user reviews and promotes the draft to `sections.txt`.
- **Validate before wiring (Phase 0).** Run the read-only feature dump on
  `output/tulsa_1921/` and `output/hearnes_brooklyn_city_directory_for_hearnesbrooklync1852/`
  first to confirm the sections actually separate on these features and to calibrate
  thresholds. That script is not throwaway — it becomes the stage's feature extractor.

---

## Background: the gap

`sections.txt` marks the first page of each structural section in a volume and
routes per-section prompts:

```
# filename-of-first-page    label
0015_p15020coll12:2453.jpg  alphabetical
0181_p15020coll12:2619.jpg  street
0311_p15020coll12:2749.jpg  business
```

Today this file is **written by hand** (see the artifact table in
`docs/pipeline-stages.md`). Everything downstream already reads it:

| Consumer | What it does with sections | Reference |
|---|---|---|
| `utils/section_utils.py` | parse file → label per page, per-section prompt path, boundary test | `load_sections`, `section_for_page`, `prompt_for_page`, `is_section_boundary` |
| `pipeline/extract_entries.py` | resets carry-forward context at each boundary; switches to `ner_prompt_{label}.md` | `extract_entries.py:1112-1123`, `:1167-1177` |
| `pipeline/run_gemini_ocr.py` | switches to `ocr_prompt_{label}.md` per section | (section lookup via `section_utils`) |
| `pipeline/generate_prompt.py` | "sections mode": samples pages from each run, generates per-section prompts | `generate_prompt.py` (sections branch) |
| `pipeline/select_pages.py` | seeds selection from section boundaries | (section lookup) |
| `main.py` / `cli/main.py` | `--sections` flag + `_resolve_sections()` plumbing to declarative stages via `from_ctx` | `main.py:208 _resolve_sections`, `stages.py` `Opt("sections", "--sections", from_ctx=True)` |

So the feature is unusually well-scoped: **produce a draft `sections.txt`; consume
nothing new.**

---

## Why this is worth doing (the payoff)

Separate NER prompts matter because a city directory stacks sections whose entry
schemas genuinely differ — sometimes *inverted*:

| Section | Entry shape | Context model | Why one global prompt fails it |
|---|---|---|---|
| **alphabetical** (resident/business name list) | `Surname, Given (spouse), occupation, employer, h/r/bds address` — a *person* record, dense abbreviations | alphabetical letter (≈irrelevant to fields) | Green-Book-style `STATE→CITY→CATEGORY` prompt has no purchase here |
| **classified** (business / buyers' guide) | `business_name, address, phone` under ALL-CAPS category headings | CATEGORY carries down | Maps almost directly onto `prompts/examples/ner_prompt_greenbook.md` |
| **street** (reverse directory) | `house_number → occupant` under a street heading — **address-first, name-second** | street heading + cross-streets | An *inverse* of the alphabetical prompt; a name-first prompt mis-segments it, and carry-forward context from the alphabetical section actively corrupts it |
| **frontmatter / register** | officials, churches, lodges — many micro-formats | — | Usually *excluded* from extraction, not NER'd |

**Value scales with schema divergence, not with boundary count.** Two flavors of
business list want one prompt; alphabetical-vs-street want two. Tulsa 1921 (mature
20th-c. Polk-style directory: alphabetical + classified + street + register, likely
4 runs) is a strong case. Hearne's Brooklyn 1852 (earlier, flatter: register/ads +
one big alphabetical list, likely 2 runs) is a weaker but real case. **Display ads,
if sprinkled one page at a time, are a per-page *flag* (`is_advertisement`), not a
contiguous section** — don't force them into runs.

A second payoff beyond prompt quality: **scoping**. Knowing the front-matter/register
run lets you exclude it from extraction (cleaner CSV, lower cost) the same way
`included_pages.txt` already trims frontmatter.

---

## Goal / non-goals

**Goal.** A `pipeline detect-sections <DIR>` stage that reads per-page OCR artifacts,
classifies each page into a section type, smooths the per-page labels into contiguous
runs, and writes:
1. `sections_report.csv` — one row per page: predicted label, confidence, and every
   feature (so a human can eyeball and tune).
2. `sections_draft.txt` — the existing `sections.txt` format, boundaries only, with a
   `# DRAFT — review then rename to sections.txt` header.
3. (optional) `sections_review.html` — self-contained thumbnails-per-boundary report,
   mirroring `tools/review_ocr.py`.

**Non-goals.**
- Not changing the `sections.txt` schema or any consumer.
- Not pixel-level layout segmentation — the granularity is *first page of each run*.
- Not auto-promoting the draft or auto-running in the default `pipeline run` chain.
  This is an **opt-in calibration step**, like `detect-columns` / `select-pages` /
  `generate-prompts` ("calibrate once, run many").
- Not building the per-section prompt generator (it largely exists in
  `generate_prompt.py`'s sections mode) — only feeding it.

---

## Where it sits: the two-pass calibration workflow

Detection needs OCR text, and per-section prompt generation needs detected sections,
so a sectioned volume is calibrated in two passes (this is expected and matches the
existing calibrate-once pattern):

```
Pass 1 (calibrate):
  download → (surya/gemini) OCR → [align] → detect-sections → sections_draft.txt
                                              │
                                       human reviews / promotes → sections.txt
                                              │
                             generate-prompts --sections   → ner_prompt_{label}.md ×N
Pass 2 (run):
  extract-entries --sections sections.txt   → per-section routing + context resets
```

Place the `StageDef` **after `align_ocr` and before `extract_entries`** in
`stages.py` STAGES order. Like `detect_columns`, it is **opt-in** (only runs when
`--detect-sections` is passed); it is *not* added to the default `run`/`guided`
chains, so automated runs are unaffected. It writes a draft artifact and changes
nothing about extraction unless the user later passes `--sections`.

---

## Input tiers (degrade gracefully)

Read the richest artifact available per page; fall back cleanly:

1. **`{stem}_{model}_aligned.json`** (best) — `lines[].bbox` (geometry) + `gemini_text`
   (clean text). All features available. Parse with the same shape `align_ocr.py`
   writes (see its output schema in `docs/pipeline-stages.md`).
2. **`{stem}_surya.json`** — `lines[].bbox` + `text` (uncorrected, fine for structure).
   All geometry + text features available.
3. **`{stem}_{model}.txt`** (Gemini plain text) — text features + line-count density
   only; no per-line geometry. For `col_count`, fall back to **`columns_report.csv`**
   if present (it is produced by `detect-columns`/`surya-detect`, which run on *images*
   independent of OCR — a free geometry signal even in this tier).

Auto-detect the model slug exactly as the other scripts do (`utils/models.py`
`discover_ocr_slug()` + `pipeline_state.json`); accept `--model` override. Respect
`included_pages.txt` scope if present (see `tools/review_ocr.py:_load_scope`).

---

## Per-page features (the meat)

Put these in importable pure functions so they unit-test without images/network
(mirror how `tests/test_align_parse.py` imports `parse_surya` from `align_ocr.py`).

**Geometry (needs bbox; tier 1–2, or `columns_report.csv` for `col_count`):**
- `n_lines` — OCR line count on the page.
- `col_count` — text columns. Prefer `columns_report.csv`; else cluster line `x1`
  values (reuse `detect_columns.analyze_image` or a simple x-histogram with the same
  8%-of-width gap rule `align_ocr.py` uses for column breaks).
- `median_line_height` — `median(y2 - y1)`.
- `tall_line_frac` — fraction of lines with height > `1.4 × median_line_height`
  (heading-size proxy).
- `centered_frac` — fraction of lines whose bbox horizontal center is within ±10% of
  page center **and** width < ~60% of page width (centered-heading proxy).

**Text (needs line text; all tiers):**
- `pct_digit_leading` — fraction of body lines whose first non-space char is a digit.
  **Primary street/reverse-directory tell.**
- `pct_allcaps_short` — fraction of lines that are ALL-CAPS and ≤ ~4 words.
  **Classified category-heading / register-heading tell.**
- `alpha_monotone_frac` — over lines that look like surnames (leading alpha token),
  fraction whose leading letter is ≥ the previous such line's. **Alphabetical
  name-list cadence tell.**
- `abbrev_density` — per-line hits of resident-directory abbreviations, e.g.
  `\b(h|r|bds|b|wid|emp|Mrs|Miss|av|st|rd)\b` and dotted variants. **Resident-list tell.**
- `mean_line_len`, `pct_short_lines`.
- `digit_density`, `phone_frac` (regex for `\(?\d{3}\)?[ -]?\d{3,4}` / period-form
  phone), `dollar_frac`. **Ads/classified tell.**
- `running_head` classification of the top 1–2 lines: guide-word pattern
  (two short tokens, or `ABB-ACK`-style) → alphabetical; a street name → street.

Keep all thresholds as **named module constants** (repo convention — cf.
`fix_entries.py:_INFER_RULES`), so Phase 0 can tune them in one place.

---

## Label taxonomy

The controlled vocabulary the stage emits (these become `ner_prompt_{label}.md`
suffixes, so keep them short and filename-safe):

`frontmatter` · `alphabetical` · `classified` · `street` · `advertisements` · `unknown`

Starting classification rules (calibrate in Phase 0 — these are hypotheses, not law):
- `street` — `pct_digit_leading > ~0.45` and `col_count ≥ 2`.
- `classified` — `pct_allcaps_short > ~0.12` spread down the page, not digit-dominant.
- `alphabetical` — `alpha_monotone_frac > ~0.6` and `abbrev_density` high, dense narrow cols.
- `advertisements` — very low `n_lines` + high `tall_line_frac` + sparse (prefer the
  per-page `is_advertisement` flag if the run is non-contiguous / single pages).
- `frontmatter` — low density, prose-like, low alpha cadence / low digit-leading /
  low caps-heading (often the residual at the volume's front/back).
- else `unknown`.

Score each candidate; `confidence = top_score - second_score` (margin). Low margin →
flag for human review and/or model adjudication (Phase 5).

---

## Phases

### Phase 0 — Validate & calibrate (read-only; do this first, ~1–2 hrs)
Write `scripts/dump_section_features.py` (or a notebook cell) that runs the **feature
extractor only** over a volume dir and prints/plots the per-page series:
`col_count`, `density` (n_lines vs neighbor median), `pct_digit_leading`,
`pct_allcaps_short`, `alpha_monotone_frac`, plus the first ~3 lines of each page as a
content sniff. Run on:
- `output/tulsa_1921/` — expect ≥3 clearly separable runs (alphabetical / classified / street / register).
- `output/hearnes_brooklyn_city_directory_for_hearnesbrooklync1852/` — expect ~2 runs.

**Gate:** the section runs should be visually obvious in those series. If they are,
the deterministic path is confirmed and the model is just an adjudicator. If they are
*not* cleanly separable, lean harder on Phase 5 and report back before wiring. This
script is reused verbatim as the stage's feature extractor — not throwaway.

### Phase 1 — Feature extractor module
`pipeline/detect_sections.py` with importable pure functions:
- `load_page_text_and_geometry(item_dir, slug) -> list[PageInput]` (tiered loader).
- `extract_page_features(page: PageInput) -> dict` (all features above).
Unit-test these on synthetic line dicts (no images, no network).

### Phase 2 — Classify, smooth, emit
- `classify_page(features: dict) -> tuple[str, float]` (label, confidence).
- `smooth_runs(labels: list[str], confidences, min_run=3, window=3) -> list[str]`
  — median-filter labels, merge sub-`min_run` runs into neighbors; corroborate
  boundaries with density change-points (reuse `tools/review_ocr.py:_window_median`).
- `emit_boundaries(filenames, labels) -> list[(filename, label)]` — one line per run.
- Write `sections_report.csv` (per-page features + label + confidence) and
  `sections_draft.txt` (boundaries only, `# DRAFT` header). **Never** write
  `sections.txt` directly; if it already exists, leave it untouched and say so.
- CLI/`main()` mirrors `detect_columns.py`: positional `output_dir`, `--model`,
  `--force` (guard on `sections_report.csv` existing), `--quiet`, threshold overrides.

### Phase 3 — (optional) HTML review report
`sections_review.html`, self-contained, mirroring `tools/review_ocr.py`: a per-page
label strip (color per label) + a thumbnail card for each detected boundary page so
the user can confirm/correct the first-page-of-run picks at a glance. This is the
human-confirmation surface that makes a 90%-right draft a 30-second edit.

### Phase 4 — Wire as an opt-in stage
Mirror `detect_columns` at every touchpoint (grep `detect_columns` / `detect-columns`
to find them all — current list):
- `pipeline/stages.py` — add `StageDef("detect_sections", "pipeline/detect_sections.py",
  "--detect-sections", declarative=True, opts=(Opt("model","--model",when="not_none"),
  Opt("force","--force",kind="switch"), …))`, positioned after `align_ocr`,
  before `extract_entries`. Not added to default run/guided chains.
- `main.py` — argparse `--detect-sections` flag + `dest` + help; add to the help banner
  block near `main.py:21-22`.
- `cli/main.py` — expose under a subcommand (group with `ocr` or `calibrate`; it is a
  calibration step, so `calibrate` is the better fit).
- `app.py` — dashboard button (driven by `stages.py`; add to the appropriate group).
- `tests/test_stages_registry.py` — add an option-order test like
  `test_detect_columns_option_order`.

### Phase 5 — (optional) Model adjudication at the margins
For pages with low classification margin or near a candidate boundary only, escalate to
a `section_type` enum classification. Two interchangeable backends, gated so the
deterministic core never depends on either:
- **Cheap now:** one Gemini `flash-lite` call per ambiguous page (OCR already exists —
  you're just labeling). Fractions of a cent for the handful of transition pages.
- **Local (per `docs/plans/huggingface-uv-scripts.md`):** the NuExtract3 `image_type` enum
  idea generalizes directly to a `section_type` enum — page-level classification is the
  cheapest possible VLM task. Register it the same way a local OCR/NER backend would.

### Phase 6 — (optional) Per-section prompt generation polish
Confirm `generate_prompt.py --sections sections.txt` samples pages from each run and
emits `ner_prompt_{label}.md` (+ `ocr_prompt_{label}.md`) well for these volumes;
tune the per-section meta-prompt if the street/reverse section needs an address-first
schema hint. Mostly exists — verify and refine, don't rebuild.

---

## Verification ground rules

- **Backward compatibility is sacred.** Volumes that never run `--detect-sections`
  behave exactly as today. The consumer side already no-ops without a `sections.txt`.
- **Never overwrite a curated `sections.txt`.** Only ever write `sections_draft.txt`
  and `sections_report.csv`. State clearly in stdout where the draft is and that the
  user must promote it.
- **Empirical check on real volumes.** After Phase 2, run on `tulsa_1921` and the
  Hearne's volume; confirm the emitted runs match a manual skim (Tulsa ≥3 runs incl.
  a `street` run with high `pct_digit_leading`; Hearne's ~2). Eyeball
  `sections_report.csv` for obviously mislabeled pages.
- **End-to-end smoke.** With a confirmed `sections.txt`, run
  `generate-prompts --sections` then `extract-entries --sections` on a slice and
  confirm: (a) context resets log at each boundary (`[section boundary] … context
  reset`), (b) the right `ner_prompt_{label}.md` is loaded per page.
- **No network in the core.** Phases 0–4 must run with no API key. Phase 5 is the only
  part allowed to call a model, and it must be optional/flagged.
- **`uv run pytest -q` green** before pushing each phase.

## Tests to add

- `tests/test_detect_sections.py`:
  - `extract_page_features` on synthetic line dicts → expected feature values
    (a digit-leading "street" page, an ALL-CAPS-heading "classified" page, an
    alpha-cadence "alphabetical" page).
  - `classify_page` returns the expected label for each synthetic archetype.
  - `smooth_runs` merges a 1-page blip inside a long run; preserves a real boundary;
    respects `min_run`.
  - `emit_boundaries` yields one line per run with the correct first-page filename.
- `tests/test_stages_registry.py`: option-order test for the new `StageDef`.

## File / function map

```
pipeline/detect_sections.py
  PageInput                      # dataclass: filename, idx, lines[{bbox,text}], page_w, page_h
  load_page_text_and_geometry()  # tiered loader: aligned.json > surya.json > .txt (+columns_report.csv)
  extract_page_features()        # -> dict of the features above
  classify_page()                # -> (label, confidence)
  smooth_runs()                  # per-page labels -> contiguous runs
  emit_boundaries()              # runs -> [(first_page_filename, label)]
  write_report()                 # sections_report.csv
  write_draft()                  # sections_draft.txt (never sections.txt)
  build_review_html()            # (Phase 3) sections_review.html
  main()                         # CLI mirroring detect_columns.py

scripts/dump_section_features.py # (Phase 0) read-only feature dump; reuses the extractor

tests/test_detect_sections.py    # pure-function unit tests
```

Constants (one block, top of module): `TALL_LINE_RATIO=1.4`, `CENTERED_TOL=0.10`,
`STREET_DIGIT_FRAC=0.45`, `CLASSIFIED_CAPS_FRAC=0.12`, `ALPHA_MONOTONE_FRAC=0.6`,
`MIN_RUN=3`, `SMOOTH_WINDOW=3`, … (all tuned in Phase 0).

## Risks & caveats

- **Calibration is data-dependent.** Thresholds tuned on Tulsa/Hearne's may not
  transfer to every directory series; expose them as flags and keep the report CSV so
  re-tuning is cheap. This is why Phase 0 precedes wiring.
- **Type collisions geometry can't resolve** (2-col alphabetical vs 2-col classified)
  are exactly what Phase 5's text-reading adjudication is for — don't over-tune
  heuristics trying to nail them deterministically.
- **Ads are a flag, not always a section.** If `advertisements` pages are
  non-contiguous, leave them to the existing per-entry `is_advertisement` and don't
  emit a run.
- **Boundary precision is page-level.** A run that starts mid-page is acceptable —
  `sections.txt` only needs the first *page*; `extract_entries.py` resets context at
  the page boundary. Don't chase sub-page splits.
- **Don't run in the default chain.** Keep it opt-in so `pipeline run` stays fast and
  surprise-free.

---

## Session log

_(Append one entry per working session: date, phase touched, what landed, what's next.)_

- 2026-06-15 — Plan written (Phase 0 not yet started). Consumer side confirmed already
  wired (`utils/section_utils.py` + `extract_entries.py:1112`). Next: implement the
  Phase 0 read-only feature dump and run it on `output/tulsa_1921/` and the Hearne's
  volume to confirm separability and calibrate thresholds before any stage wiring.
