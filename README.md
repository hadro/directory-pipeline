# directory-pipeline

A pipeline for ingesting, OCR-ing, aligning, and extracting structured data from
digitized print directories and similar historical collections. Works with items from
the [Library of Congress](https://www.loc.gov/collections/), [Internet Archive](https://archive.org/), and [NYPL Digital Collections](https://digitalcollections.nypl.org/). The primary test collection is the
*Negro Motorist Green Book* (1936–1966), though the pipeline accepts any IIIF
Presentation v2 or v3 manifest.

## What it does

The pipeline takes a collection CSV (from NYPL, LoC, or Internet Archive), a
pre-built CSV from any source, or a single IIIF manifest URL, and runs any
combination of the following stages:

1. **Export collection metadata** to CSV (via LoC, IA, or  NYPL APIs)
2. **Download images** from IIIF manifests at full resolution
3. **Detect double-page spreads** (common in microfilm digitization)
4. **Split spreads** into separate left/right page files
5. **Detect column layout** per image to guide Tesseract settings
6. **Run Tesseract OCR** to get word-level bounding boxes (hOCR output)
7. **Run Gemini OCR** to get accurately transcribed text
8. **Compare OCR models** side-by-side in an HTML report
9. **Align** Gemini text to Tesseract bounding boxes using anchored Needleman-Wunsch
10. **Improve alignment** by retrying Tesseract with alternative PSM settings for
    poorly-matched pages
11. **Visualize** alignment quality by drawing color-coded boxes on images
12. **Extract structured entries** (name, address, city, state, category) via Gemini NER
13. **Geocode entries** to lat/lon using Google Maps (address-level) or Nominatim (city-level)
14. **Generate an interactive map** from geocoded entries

The end result is a per-page JSON file pairing Gemini's corrected text with
Tesseract's pixel-level word coordinates, a structured entries CSV, and an
interactive Leaflet map — suitable for search indexing, IIIF annotation,
and geospatial analysis.

---

## Pipeline stages

Stages always run in the fixed order below, regardless of flag order on the
command line. All stages are optional — run only what you need.

```
--nypl-csv            sources/nypl_collection_csv.py  → collection_csv/{slug}.csv
--loc-csv             sources/loc_collection_csv.py   → collection_csv/{slug}.csv
--ia-csv              sources/ia_collection_csv.py    → collection_csv/{slug}.csv
--download            download_images.py              → images/{slug}/
--detect-spreads      detect_spreads.py               → images/{slug}/spreads_report.csv
--split-spreads       split_spreads.py                → *_left.jpg, *_right.jpg, *_split.json
--detect-columns      detect_columns.py               → images/{slug}/columns_report.csv
--tesseract           run_ocr.py                      → *_tesseract.hocr, *_tesseract.txt
--gemini-ocr          run_gemini_ocr.py               → *_{model}.txt
--compare-ocr         compare_ocr.py                  → *_comparison.html
--align-ocr           align_ocr.py                    → *_{model}_aligned.json
--improve-alignment   improve_alignment.py            → updated *_{model}_aligned.json
--visualize           visualize_alignment.py          → *_{model}_viz.jpg
--extract-entries     extract_entries.py              → entries_{model}.csv, *_{model}_entries.json
--geocode             geocode_entries.py              → entries_{model}_geocoded.csv
--map                 map_entries.py                  → entries_{model}.html
```

### Stage descriptions

#### `sources/loc_collection_csv.py` — Export LoC metadata
Exports items from a Library of Congress collection or single item URL to the
same CSV schema. No API token required — the LoC JSON API is publicly accessible.
Paginates collections automatically and derives a readable filename slug from the
item title.

```bash
python sources/loc_collection_csv.py https://www.loc.gov/item/01015253/
python sources/loc_collection_csv.py https://www.loc.gov/collections/civil-war-maps/
```

#### `sources/ia_collection_csv.py` — Export Internet Archive metadata
Exports items from an Internet Archive collection or single item URL. Handles both
IA collections (lists of items) and individual item identifiers. Derives a slug
from the IA item title and identifier. No authentication required.

```bash
python sources/ia_collection_csv.py https://archive.org/details/ldpd_11290437_000/
python sources/ia_collection_csv.py https://archive.org/details/durstoldyorklibrary
```

#### `sources/nypl_collection_csv.py` — Export NYPL metadata
Walks the NYPL Digital Collections API hierarchy for a collection UUID, recursively
descending into sub-collections. For each item found, fetches capture metadata and
writes one row to a CSV. Requires a `NYPL_API_TOKEN` environment variable (or `--token`).
Responses are cached locally as JSON to avoid redundant API calls on re-runs.

All three source scripts produce the same CSV schema, which feeds into `download_images.py`:

| Column | Description |
|---|---|
| `item_id` | Item identifier — used as the per-item image directory name |
| `item_title` | Human-readable title — used for progress display |
| `iiif_manifest_url` | IIIF Presentation manifest URL — used to download images |
| `microform` | `True` if the item is a microfilm/microform scan — used by `detect_spreads.py` |

#### `download_images.py` — Download images
Fetches full-resolution images from IIIF manifests. Two input modes:

**CSV mode** (default): reads a collection CSV and downloads every item's images.
```
images/{slug}/{item_id}/{page:04d}_{image_id}.jpg
```

**Manifest mode** (`--manifest URL`): downloads a single IIIF manifest directly — no CSV needed. Accepts any public IIIF Presentation v2 or v3 manifest URL.
```
images/{item_id}/{page:04d}_{image_id}.jpg
```

The IIIF manifest is cached alongside the images for downstream use by `align_ocr.py`.
Handles HTTP 429/503 with exponential backoff and retry. Falls back to an alternative
IIIF endpoint on repeated HTTP 403 errors. Safe to re-run — existing files are skipped.

**LoC-specific:** `www.loc.gov/item/{id}/manifest.json` endpoints are blocked by
Cloudflare for non-browser clients. The downloader automatically falls back to the
LoC item JSON API (`?fo=json`) to build a synthetic IIIF manifest, then downloads
images from `tile.loc.gov`. The requested download width is capped at the native
image resolution to avoid upscaling artifacts from the tile pyramid.

#### `detect_spreads.py` — Spread detection
Analyzes each image to determine whether it contains two facing pages (a spread
captured in a single microfilm frame) vs. a single page. Uses pixel-projection
analysis: finds the content boundary within the dark microfilm border, checks the
aspect ratio, then scans the central vertical band for a persistent gutter/seam.

Handles three physical gutter forms: dark shadow (spine), white gap (pages splaying
open), tonal boundary (dark cover beside white content). Outputs
`spreads_report.csv` — does not modify images.

If a collection CSV is available (via `--csv`), items flagged as microform get a
looser detection threshold.

#### `split_spreads.py` — Split spreads
Reads `spreads_report.csv` and, for each spread, crops the image at the detected
gutter column into `{stem}_left.jpg` and `{stem}_right.jpg`. Also writes a
`{stem}_split.json` sidecar with the pixel offsets. Original images are untouched.

#### `detect_columns.py` — Column detection
Uses a vertical pixel-projection profile (dark-pixel density per column) to count
text columns per image and detect gutter positions. Outputs `columns_report.csv`
with a `recommended_psm` (Tesseract page segmentation mode) for each image:
PSM 4 for single-column pages, PSM 1 for multi-column. `run_ocr.py` reads this
CSV to apply per-image PSM automatically.

#### `run_ocr.py` — Tesseract OCR
Runs Tesseract on each image and saves:
- `{stem}_tesseract.hocr` — full hOCR with word-level bounding boxes
- `{stem}_tesseract.txt` — plain text

Tesseract dictionary correction is **disabled by default** (`load_system_dawg=0`,
`load_freq_dawg=0`). This preserves proper nouns, street names, and abbreviations
as-written, which is critical for accurate NW alignment with Gemini text. Use
`--dict` to re-enable. Runs in parallel using `--workers`.

Per-image PSM is loaded from `columns_report.csv` if present; the global `--psm`
flag always overrides it.

#### `run_gemini_ocr.py` — Gemini OCR
Sends each image to the Gemini API using a system prompt from `prompts/ocr_prompt.md`
and saves the plain-text response as `{stem}_{model_slug}.txt`. Handles HTTP 429
rate limits with exponential backoff (up to 5 retries, starting at 10s delay).
Skips images where the output file already exists and is non-empty.

#### `compare_ocr.py` — Model comparison
Calls multiple models (any mix of Gemini model names and the special token
`tesseract`) on each image and produces:
- `{stem}_comparison.html` — side-by-side panel view of each model's output
- `ocr_comparison_stats.csv` — character-level similarity stats across all images

Useful for evaluating which Gemini model performs best on a collection before
committing to a full run.

#### `align_ocr.py` — NW alignment
The core output stage. For each image that has both a Gemini `.txt` file and a
Tesseract `.hocr` file, aligns the two using anchored
[Needleman-Wunsch](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
global sequence alignment and saves `{stem}_{model_slug}_aligned.json`.

**Approach:** Runs a single global word-level NW pass over all Tesseract words and
all Gemini tokens on a page. Before the NW pass, city/state headings and category
lines that appear verbatim in both sources are committed as fixed anchors. The NW
problem is then split into independent segments at those anchors, preventing
misalignment drift across long pages where Tesseract's reading order diverges
from Gemini's.

**Reading-order correction:** Tesseract on multi-column pages reads across columns
rather than down each column. Lines are re-sorted before alignment:
- *Row-major (default):* lines are grouped into 50 px horizontal bands and sorted
  left-column-first within each band. Correctly places centered section headings
  (state names, category lines) before the body columns they head.
- *Column-major (true two-column pages):* when lines cluster into exactly two columns
  each holding ≥ 20% of page lines, the left column is emitted top-to-bottom followed
  by the right column top-to-bottom, matching Gemini's reading order for pages with
  independent side-by-side city sections.

Column breaks are detected as gaps in the x1 distribution exceeding 10% of page width.

**Output JSON schema:**

```json
{
  "image": "0001_58030238.jpg",
  "model": "gemini-2.0-flash",
  "canvas_uri": "https://...",
  "canvas_width": 2048,
  "canvas_height": 3000,
  "lines": [
    {
      "bbox": [x1, y1, x2, y2],
      "canvas_fragment": "canvas_uri#xywh=x,y,w,h",
      "confidence": "word",
      "gemini_text": "corrected line text",
      "words": [
        {
          "bbox": [x1, y1, x2, y2],
          "canvas_fragment": "canvas_uri#xywh=...",
          "confidence": "word",
          "text": "word"
        }
      ]
    }
  ],
  "unmatched_gemini": ["lines with no Tesseract match"]
}
```

IIIF canvas URIs and dimensions are read from the `manifest.json` cached by
`download_images.py`. Bounding boxes are expressed in both pixel space and IIIF
canvas fragment space.

#### `improve_alignment.py` — Alignment improvement
Retries Tesseract with alternative page segmentation modes (PSM 11, PSM 6) for
pages where more than 5% of Gemini lines are unmatched. Re-runs `align_ocr` with
the new hOCR and keeps the result only if it reduces the unmatched count. Writes
updated `*_aligned.json` files in-place.

#### `visualize_alignment.py` — Alignment visualization
Reads each `*_aligned.json` and draws color-coded bounding boxes on the source
image, saving `{stem}_{model_slug}_viz.jpg`:

- **Green** — word-confidence boxes (per word)
- **Red** — unmatched Gemini lines (listed in the margin, no coordinates)

Visualization output files are automatically excluded from all OCR and alignment
stages (files ending in `_viz.jpg` are skipped).

#### `extract_entries.py` — Entry extraction
Reads each `*_aligned.json` file and calls Gemini with a NER prompt
(`prompts/ner_prompt.md`) to identify structured directory entries. Each entry is
extracted with the following fields when present:

| Field | Description |
|---|---|
| `establishment_name` | Business or individual name |
| `raw_address` | Street address as it appears in the source |
| `city` | City |
| `state` | State (full name or abbreviation) |
| `category` | Type of establishment (e.g. Hotels, Restaurants, Beauty Parlors) |
| `canvas_fragment` | IIIF `#xywh=` fragment pointing to the source line on the canvas |

Outputs a per-page `*_entries.json` sidecar and an aggregate `entries_{model}.csv`
for the entire collection.

#### `geocode_entries.py` — Geocoding
Reads an `entries_{model}.csv` (or an images directory containing one) and resolves
each entry to geographic coordinates:

- **Address-level** (requires `GOOGLE_MAPS_API_KEY`): calls the Google Maps Geocoding
  API for entries with a `raw_address`. Accurate to building level.
- **City-level fallback**: calls Nominatim with city + state for entries without an
  address, or when no Google API key is set.

Results are cached in `geocache.json` (keyed by query string) so re-runs only hit
the network for new queries. Writes `entries_{model}_geocoded.csv` with added
`lat`, `lon`, and `geocode_level` (`"address"` | `"city"` | `""`) columns.

```bash
GOOGLE_MAPS_API_KEY=... python geocode_entries.py images/green_book_1962_9ab2e8f0/ \
    --model gemini-2.0-flash
```

#### `map_entries.py` — Interactive map
Reads a `entries_{model}_geocoded.csv` and generates a self-contained Leaflet HTML
map (`entries_{model}.html`) with:
- Clustered markers (MarkerCluster) color-coded by establishment category
- Sidebar with live search, state dropdown, and category checkboxes
- Live count of entries shown

Entries geocoded at city level are jittered slightly to avoid stacking. Requires the
geocoded CSV from `geocode_entries.py`.

---

## Directory layout

```
directory-pipeline/
├── main.py                       # Pipeline orchestrator
├── download_images.py            # Download images from IIIF manifests
├── detect_spreads.py             # Spread detection
├── split_spreads.py              # Spread splitting
├── detect_columns.py             # Column layout detection
├── run_ocr.py                    # Tesseract OCR
├── run_gemini_ocr.py             # Gemini OCR
├── compare_ocr.py                # Model comparison
├── align_ocr.py                  # NW alignment
├── improve_alignment.py          # Alignment improvement (PSM retry)
├── visualize_alignment.py        # Alignment visualization
├── extract_entries.py            # Structured entry extraction (NER)
├── geocode_entries.py            # Entry geocoding
├── map_entries.py                # Interactive map generation
│
├── sources/                      # Collection metadata exporters
│   ├── loc_collection_csv.py     # Library of Congress
│   └── ia_collection_csv.py      # Internet Archive
│   ├── nypl_collection_csv.py    # NYPL Digital Collections
│
├── utils/                        # Shared utilities
│   └── iiif_utils.py             # IIIF v2/v3 manifest parsing
│
├── prompts/                      # Gemini system prompts
│   ├── ocr_prompt.md             # OCR transcription prompt
│   └── ner_prompt.md             # NER / entry extraction prompt
│
├── analysis/                     # Standalone analysis tools
│   ├── compare_extraction.py     # Compare entry extraction across models
│   ├── visualize_entries.py      # Draw entry bounding boxes on images
│   └── repatch_fragments.py      # Re-sync canvas_fragment coords after re-alignment
│
├── pyproject.toml                # Python project config and dependencies
├── collection_csv/               # Output of --*-csv stages (one CSV per collection)
└── images/
    └── {slug}/                   # e.g. the_negro_motorist_green_book_1947_4bea2040/
        └── {item_id}/            # NYPL UUID or LoC/IA identifier
            ├── manifest.json
            ├── 0001_{image_id}.jpg
            ├── 0001_{image_id}_left.jpg              # if spread-split
            ├── 0001_{image_id}_right.jpg             # if spread-split
            ├── 0001_{image_id}_split.json            # split coordinate sidecar
            ├── 0001_{image_id}_tesseract.hocr
            ├── 0001_{image_id}_tesseract.txt
            ├── 0001_{image_id}_{model}.txt           # Gemini plain text
            ├── 0001_{image_id}_{model}_aligned.json  # NW alignment output
            ├── 0001_{image_id}_{model}_viz.jpg       # alignment visualization
            ├── 0001_{image_id}_{model}_entries.json  # per-page entries
            ├── 0001_{image_id}_comparison.html       # OCR model comparison
            ├── spreads_report.csv
            ├── columns_report.csv
            ├── entries_{model}.csv                   # aggregate entries for collection
            ├── entries_{model}_geocoded.csv          # entries with lat/lon
            ├── entries_{model}.html                  # interactive Leaflet map
            └── geocache.json                         # geocoding cache
```

For NYPL collections, `{slug}` is derived as `{title_words}_{uuid8}`, e.g.
`the_negro_motorist_green_book_1940_feb978b0`. For LoC items it is derived from
the item title and numeric ID. For IA items it is derived from the item title and
IA identifier. Pass `--slug` to override.

---

## Installation

Requires Python 3.11+ and Tesseract.

```bash
# Install Tesseract
brew install tesseract          # macOS
apt install tesseract-ocr       # Debian/Ubuntu

# Install Python dependencies
uv sync                         # or: pip install -e .
```

Set environment variables:

```bash
export NYPL_API_TOKEN=your_token_here      # from https://api.repo.nypl.org/sign_up
                                            # (not needed for LoC or IA)
export GEMINI_API_KEY=your_key_here
export GOOGLE_MAPS_API_KEY=your_key_here   # optional; enables address-level geocoding
```

---

## Usage

### Library of Congress items

```bash
# Export metadata and download images for a single LoC item
python main.py https://www.loc.gov/item/01015253/ \
    --loc-csv --download --tesseract --gemini-ocr --model gemini-2.0-flash

# Export metadata and download images for a full LoC collection
python main.py https://www.loc.gov/collections/civil-war-maps/ \
    --loc-csv --download --gemini-ocr --model gemini-2.0-flash

# Override slug if desired
python main.py https://www.loc.gov/item/01015253/ \
    --loc-csv --download --slug brooklyn_directory_1852
```

### Internet Archive

```bash
# Download a single IA item
python main.py https://archive.org/details/ldpd_11290437_000/ \
    --ia-csv --download --tesseract --gemini-ocr --model gemini-2.0-flash

# Download an IA collection
python main.py https://archive.org/details/durstoldyorklibrary \
    --ia-csv --download --gemini-ocr --model gemini-2.0-flash
```

### NYPL collections

```bash
# Export metadata and download images
python main.py collections.txt --nypl-csv --download

# Detect and split spreads, then run both OCR engines
python main.py collections.txt --detect-spreads --split-spreads \
    --tesseract --gemini-ocr --model gemini-2.0-flash

# Run the full alignment pipeline
python main.py collections.txt --align-ocr --improve-alignment --visualize \
    --model gemini-2.0-flash --force

# Extract entries, geocode, and build a map
python main.py collections.txt --extract-entries --geocode --map \
    --model gemini-2.0-flash
```

### Any IIIF manifest

```bash
# Download a single manifest directly (no CSV needed)
python download_images.py --manifest https://example.org/iiif/item/manifest.json \
    --output-dir images/my-item
```

### Other options

```bash
# Compare two Gemini models before committing to a full run
python main.py collections.txt --compare-ocr \
    --models gemini-2.0-flash gemini-2.5-pro

# Dry run — show commands without executing
python main.py collections.txt --download --tesseract --gemini-ocr \
    --model gemini-2.0-flash --dry-run
```

Individual scripts can also be run directly:

```bash
python align_ocr.py images/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash --force

python extract_entries.py images/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash

python geocode_entries.py images/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash

python map_entries.py images/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash
```

Analysis tools (not in the main pipeline) can be run from the `analysis/` directory:

```bash
python analysis/visualize_entries.py images/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash

python analysis/repatch_fragments.py images/the_negro_motorist_green_book_1947_4bea2040 \
    --model gemini-2.0-flash
```

---

## Key design decisions

**Gemini for accuracy, Tesseract for coordinates.** Gemini produces far more
accurate transcriptions of historical print than Tesseract, especially for proper
nouns, abbreviations, and damaged text. Tesseract's value is its hOCR output:
word-level bounding boxes. The alignment step combines both.

**Anchored Needleman-Wunsch alignment.** The aligner runs a single global
word-level NW pass (gap penalty −40, similarity 0–100 based on character edit
distance). Before the NW pass, city/state headings and category lines that appear
verbatim in both Gemini and Tesseract are committed as fixed *anchors*. The
sequence is then split into independent segments at each anchor and each segment
is aligned separately. This prevents misalignment drift on long pages where a
mismatched heading would otherwise pull all subsequent entries to wrong coordinates.

**Tesseract dictionary correction disabled.** By default Tesseract silently
"corrects" street names, proper nouns, and abbreviations toward dictionary words
(e.g. `Mound` → `Wound`, `Innesfallen` → `Innisfallen`). This is disabled via
`load_system_dawg=0 load_freq_dawg=0` to preserve names as-written for alignment.

**Column reading-order correction.** Tesseract on multi-column pages reads across
columns (left-to-right by y position) while Gemini reads column-by-column. Lines
are re-sorted before alignment using two strategies:
- *Row-major:* the default; lines are grouped into 50 px y-bands and sorted
  left-column-first within each band. Correct for pages where a centered heading
  precedes two-column body text.
- *Column-major:* used when lines cluster cleanly into two columns each with
  ≥ 20% of all lines. Emits the entire left column top-to-bottom, then the entire
  right column, matching Gemini's reading order for pages with independent
  side-by-side city sections.

**Two-tier geocoding.** `geocode_entries.py` uses Google Maps (building-level
accuracy) for entries with a street address and Nominatim city/state centroids as
a fallback. Both are cached in `geocache.json` so repeated runs over the same data
are instant. City-level points are jittered on the map to prevent stacking.

**IIIF v2/v3 support.** `utils/iiif_utils.py` provides a single version-agnostic
interface for walking IIIF Presentation manifests. IIIF Image API URLs work
identically for both API versions. The downloader caps requested width at the
service's advertised `maxWidth` to prevent upscaling artifacts on tile-pyramid
servers.

**IIIF-native output.** The aligned JSON includes `canvas_uri` and
`canvas_fragment` (IIIF `#xywh=` fragment) for every word and line, making the
output directly consumable by IIIF annotation tools and viewers.

---

## Prior work and inspirations

**Greif, Griesshaber & Greif (2025) — "Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents"** ([arXiv:2504.00414](https://arxiv.org/abs/2504.00414)) / [pipeline](https://github.com/niclasgriesshaber/gemini_historical_dataset_pipeline) / [benchmarking code](https://github.com/niclasgriesshaber/llm_historical_dataset_benchmarking)

The foundational paper for this pipeline's architecture. Benchmarks multimodal LLMs against Tesseract and Transkribus on 18th–19th century German city directories, finding that feeding both the source image *and* noisy conventional OCR into a multimodal LLM produces far lower error rates than either alone (0.84% CER with Gemini 2.0 Flash). The paper directly motivates the two-stage design (Tesseract for coordinates → Gemini for accuracy), temperature 0.0, and the separation of OCR, post-correction, and NER into distinct pipeline stages.

**Bell, Marlow, Wombacher et al. (2020) — *directoreadr*** ([PLOS ONE 15(8): e0220219](https://doi.org/10.1371/journal.pone.0220219)) / [code](https://github.com/brown-ccv/directoreadr)

The closest prior work: an end-to-end pipeline for extracting geocoded business data from scanned Polk city directory yellow pages (Providence, RI, 1936–1990) using classical computer vision, Tesseract, fuzzy street matching, and ArcGIS geocoding. Achieves 94.4% automated page processing. Documents the brittle year-specific heuristics required for header detection and the historical street change problem that dominates geocoding failures — both of which motivate the mLLM-based approach here.

**Fleischhacker, Kern & Göderle (2025) — "Enhancing OCR in historical documents with complex layouts through machine learning"** ([Int. J. Digital Libraries 26:3](https://doi.org/10.1007/s00799-025-00413-z))

Demonstrates that layout detection as a preprocessing step improves OCR accuracy by over 15 percentage points on multi-column historical documents (Habsburg civil service directories). The key mechanism: without layout detection, Tesseract reads across columns rather than down them, scrambling the text. This directly motivates the column reading-order correction in `align_ocr.py` and the `detect_columns.py` + `improve_alignment.py` stages.

**Cook, Jones, Rosé & Logan (2020) — "The Green Books and the Geography of Segregation in Public Accommodations"** ([NBER Working Paper 26819](https://www.nber.org/papers/w26819))

The canonical prior digitization of the Green Books. Establishes that the pre-mLLM state of the art was entirely manual data entry (OCR was explicitly rejected due to irregular formatting and ad placement), uses the US Census Geocoder as a national baseline (~50% exact match), and produces the canonical six-category establishment taxonomy used in the NER schema here. Also documents the cross-year identity matching problem and calls for city directory cross-referencing as a research next step.

**Smith & Cordell (2018) — "A Research Agenda for Historical and Multilingual Optical Character Recognition"** ([Northeastern University / NEH](https://repository.library.northeastern.edu/files/neu:m043p093w))

A practitioner-consensus research agenda identifying layout analysis as the top barrier to historical OCR progress and OCR post-correction as high-leverage and underinvested. Validates line-level sequence alignment for ground truth creation (the same approach as Needleman-Wunsch alignment used here) and argues that "how dirty is too dirty" is a task-specific empirical question — informing the pipeline's decision to expose confidence metrics rather than hard-filter at a fixed threshold.

**Berenbaum, Deighan, Marlow et al. (2016) — *georeg*** ([arXiv:1612.00992](https://arxiv.org/abs/1612.00992)) / [code](https://bitbucket.org/brown-data-science/georeg)

Predecessor to *directoreadr*, applying a morphological contour merging + k-means column clustering approach to Rhode Island manufacturing registries. Documents the cross-page context inheritance pattern (nearest heading above, including the last heading from the prior page) and the sobering finding that geocoding success compounds OCR errors, parse failures, and historical street changes — even with 99% record identification, geocoding reached only 61%.

**Carlson, Bryan & Dell (2023) — *EffOCR*: "Efficient OCR for Building a Diverse Digital History"** ([arXiv:2304.02737](https://arxiv.org/abs/2304.02737)) / [code](https://github.com/dell-research-harvard/effocr)

Provides the key OCR benchmarks on historical US newspapers: off-the-shelf Tesseract at ~10.6% CER, fine-tuned TrOCR at 1.3% CER. These establish both the baseline noisy-input quality for the alignment stage and the comparison target for evaluating whether mLLM post-correction is competitive with conventional fine-tuned OCR. Also documents Google Cloud Vision's failure on full-page historical newspaper scans — reinforcing the decision to use Tesseract as the noisy-input stage rather than a cloud OCR API.

**HuggingFace (2025) — "Supercharge your OCR Pipelines with Open Models"** ([huggingface.co/blog/ocr-open-models](https://huggingface.co/blog/ocr-open-models))

A practitioner survey of the current open-weight VLM-based OCR landscape (Nanonets-OCR2, PaddleOCR-VL, dots.ocr, OlmOCR-2, Granite-Docling, DeepSeek-OCR, Chandra, Qwen3-VL) that introduces "locality awareness" — the ability to produce corrected text paired with bounding boxes — as a first-class capability distinction. Models with grounding support (Chandra, OlmOCR-2, dots.ocr, Granite-Docling) could in principle replace the Tesseract → Needleman-Wunsch alignment pipeline with a single-pass architecture. None has been tested on degraded historical scans; Granite-Docling (258M parameters, CPU-runnable, DocTags structured output) is the most tractable starting point for empirical evaluation on Green Books pages.
