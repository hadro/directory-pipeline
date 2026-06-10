# directory-pipeline

Extracts structured CSVs from digitized historical directories (city directories, phone books, Green Books) sourced from the Library of Congress, Internet Archive, and any public IIIF endpoint.

## Setup

```bash
uv sync                    # core deps
uv sync --extra gpu        # add Surya OCR
uv sync --extra geo        # add geocoding
uv sync --all-extras       # everything
```

Requires `.env` with `GEMINI_API_KEY` (and optionally `GOOGLE_MAPS_API_KEY`).

## Running the pipeline

After `uv sync`, a `pipeline` CLI command is available:

```bash
pipeline run    <URL>          # automated: download → OCR → extract → explore
pipeline guided <URL>          # human-in-loop: pauses for page selection + alignment review
pipeline ingest <URL>          # download only
pipeline calibrate <DIR>       # select sample pages + generate prompts (once per collection type)
pipeline ocr    <DIR>          # Surya OCR + Gemini OCR + align bboxes
pipeline extract <DIR>         # NER extraction + build explorer
pipeline review  <DIR>         # interactive Flask alignment review
pipeline postprocess <DIR>     # fix + combine + build explorer (post-extraction)
```

Run `pipeline <subcommand> --help` for per-subcommand flags.

The underlying `python main.py <URL> [--flags]` interface still works for advanced use.
Stages always execute in fixed order regardless of flag order. All stages are optional.

## Key stages

| Flag | What it does | Output |
|------|-------------|--------|
| `--download` | Fetch IIIF images | `output/{slug}/` images + `manifest.json` |
| `--surya-ocr` | Surya line-level bbox detection | `*_surya.json` |
| `--gemini-ocr` | Gemini text extraction | `*_gemini.txt` |
| `--align-ocr` | NW alignment of Gemini text to Surya bboxes | `*_aligned.json` |
| `--review-alignment` | Interactive Flask UI to fix bad alignments | updates `*_aligned.json` |
| `--extract-entries` | NER → structured entries | `entries_{model}.csv` |
| `--geocode` | Geocode entries | `*_geocoded.csv` |
| `--map` | Leaflet HTML map | `*_map.html` |
| `--select-pages` | Browser UI to pick sample pages (once per volume) | `selection.txt` |
| `--generate-prompts` | Gemini-generated OCR + NER prompts (once per collection type) | `ocr_prompt.md`, `ner_prompt.md` |

## Gemini OCR cost options

`--gemini-ocr` supports two cost-saving modes via the Gemini API:

- **`--flex`** — Flex inference (`service_tier="flex"`): ~50% cheaper than standard pricing, with 1–15 min latency per request. Best for large volumes where real-time throughput isn't needed. Use with `--gemini-ocr` in `main.py` or directly with `pipeline/run_gemini_ocr.py`.
- **Batch API** (not yet implemented) — Submit all pages as a single async job; 50% cheaper, up to 24-hour turnaround. Planned for future implementation.

Default model is `gemini-3.1-flash-lite` for both OCR and NER. For higher accuracy use `--ocr-model gemini-2.0-flash`. Combine with `--flex` to cut cost ~50%.

## Surya and alignment

- `--surya-ocr` requires GPU or Apple Silicon; slow on CPU
- `--align-ocr` uses Needleman-Wunsch with city/state headings as anchors; runs a second pass on unmatched lines to catch missed columns
- `--review-alignment` is a Flask server — access via `localhost:5000` locally or ngrok/Colab proxy in Colab
- Aligned JSON confidence values: `"line"` (Surya), `"manual"` (user-confirmed via review UI)

## Supported sources

| Source | How to use |
|--------|-----------|
| Library of Congress | Pass any `loc.gov` URL; `--loc-csv` exports metadata |
| Internet Archive | Pass any `archive.org` URL; `--ia-csv` exports metadata |
| Generic IIIF | Pass any IIIF manifest or collection URL directly; `--iiif-csv` exports metadata |
| CONTENTdm | Run `sources/build_contentdm_manifest.py` first to generate a IIIF manifest, then use as generic IIIF |
| Pre-built CSV | Pass a CSV path directly to `--download` |

## Calibrate once, run many

Run `--select-pages` and `--generate-prompts` once per collection type. For additional volumes in the same series, reuse with `--ner-prompt output/{first-slug}/ner_prompt.md`.

## Output schema

Every row in the CSV has a `canvas_fragment` column — a IIIF URI pointing back to the exact page (with `#xywh=` bounding box if alignment was run). The slug is auto-derived from collection metadata; override with `--slug`.

After each run, `pipeline_state.json` is written to `output/{slug}/` recording the model used and stages completed. Downstream scripts (`align_ocr.py`, `fix_entries.py`) read this automatically — no `--model` flag needed.

## Post-extraction workflow

After `pipeline run` or `pipeline extract`, use:

```bash
pipeline postprocess output/<collection>/   # fix + combine volumes + build explorer
pipeline postprocess output/<vol>/ --no-combine   # single volume, no combine step
```

This runs `fix_entries.py` → `combine_volumes.py` → `explore_entries.py` in sequence.
The model slug is read from `pipeline_state.json` automatically — no `--model` flag needed.

## Analysis scripts (`analysis/`)

These scripts operate on already-extracted CSVs — they do not invoke the pipeline.

| Script | Purpose |
|--------|---------|
| `fix_entries.py` | Normalize, deduplicate, and patch per-volume CSVs. Key flags: `--all` (all volumes in collection), `--infer-categories` (keyword-based category assignment). Model is auto-detected from `pipeline_state.json`. |
| `combine_volumes.py` | Merge fixed per-volume CSVs into one combined CSV with `volume_title`, `volume_year` columns added from each `manifest.json`. |
| `compare_ocr.py` | Side-by-side Gemini model comparison. Use `"surya"` token to include Surya output. |
| `compare_extraction.py` | Compare text-only vs multimodal NER extraction quality. |
| `visualize_alignment.py` | Draw alignment bounding boxes on images (`*_viz.jpg`). |
| `visualize_entries.py` | Draw entry bounding boxes colored by category. |
| `review_entries.py` | Send extracted CSV to Gemini for systematic data-quality review. |
| `model_eval.py` | Evaluate NER quality metrics (fill rate, state consistency, drift rate, cost per entry) across models and modes. |
| `check_manifest_staleness.py` | Compare cached `manifest.json` against live IIIF manifests to detect canvas-count drift after collection updates. |

### Collection-specific scripts (`collections/`)

Collection-specific scripts live under `collections/`:

- **`collections/greenbook/`** — Green Book explorer, visualization, and geographic drift detection. Working notes and design docs in `collections/greenbook/docs/`.
- **`collections/tulsa/`** — Tulsa 1921 City Directory research: NYTimes comparison, Greenwood district geocoding and spatial analysis.

To extend for a new collection type, follow the pattern in `collections/greenbook/README.md`.

### Category inference (`fix_entries.py --infer-categories`)

Applies deterministic keyword rules (`_INFER_RULES`, ~line 292) to entries with ambiguous or empty categories. Only rewrites entries whose category is `""`, `"General"`, or `"Hotels - Motels - Tourist Homes - Restaurants"`. Does not force-assign a default for non-matching entries.

```bash
python analysis/fix_entries.py --all --infer-categories
```

## Flask dashboard

`app.py` provides a local web UI for running pipeline stages without the CLI. Launch with:

```bash
python app.py
```

Accessible at `http://localhost:5001`. Stages are grouped by subcommand (ingest, calibrate, ocr, extract) and show completion status based on output files present.

## Companion repo: green-books

Published data and explorer UI live at `hadro.github.io/green-books` (repo: `github.com/hadro/green-books`). The pipeline produces CSVs and IIIF manifests; green-books consumes them. Key handoff artifacts: `green_book_entries_all.csv`, `image_to_volume.json`, per-volume `manifest.json` files under `manifests/`.

## Colab

For Surya OCR and alignment review on GPU: `colab/ocr-align-review.ipynb`
