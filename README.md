# directory-pipeline

A pipeline for ingesting, OCR-ing, and structurally aligning digitized print
directories and similar historical collections. Works with items from
[NYPL Digital Collections](https://digitalcollections.nypl.org/) and the
[Library of Congress](https://www.loc.gov/collections/). The primary test
collection is the *Negro Motorist Green Book* (1936–1966), though the pipeline
accepts any IIIF Presentation v2 or v3 manifest.

## What it does

The pipeline takes a collection CSV (from NYPL or LoC), a pre-built CSV from
any source, or a single IIIF manifest URL, and runs any combination of the
following stages:

1. **Export collection metadata** to CSV (via NYPL or LoC APIs)
2. **Download images** from IIIF manifests at full resolution
3. **Detect double-page spreads** (common in microfilm digitization)
4. **Split spreads** into separate left/right page files
5. **Detect column layout** per image to guide Tesseract settings
6. **Run Tesseract OCR** to get word-level bounding boxes (hOCR output)
7. **Run Gemini OCR** to get accurate transcribed text
8. **Compare OCR models** side-by-side in an HTML report
9. **Align** Gemini text to Tesseract bounding boxes using Needleman-Wunsch
10. **Visualize** alignment quality by drawing color-coded boxes on images

The end result is a JSON file per page that pairs Gemini's corrected text with
Tesseract's pixel-level word coordinates, suitable for search indexing, IIIF
annotation, or structured data extraction.

---

## Pipeline stages

Stages always run in the fixed order below, regardless of flag order on the
command line. All stages are optional — run only what you need.

```
--nypl-csv        nypl_collection_csv.py   → collection_csv/{slug}.csv  (NYPL only)
--loc-csv         loc_collection_csv.py    → collection_csv/{slug}.csv  (LoC only)
--download        download_images.py       → images/{slug}/
--detect-spreads  detect_spreads.py        → images/{slug}/spreads_report.csv
--split-spreads   split_spreads.py         → *_left.jpg, *_right.jpg, *_split.json
--detect-columns  detect_columns.py        → images/{slug}/columns_report.csv
--tesseract       run_ocr.py               → *_tesseract.hocr, *_tesseract.txt
--gemini-ocr      run_gemini_ocr.py        → *_{model}.txt
--compare-ocr     compare_ocr.py           → *_comparison.html
--align-ocr       align_ocr.py             → *_{model}_aligned.json
--visualize       visualize_alignment.py   → *_{model}_viz.jpg
```

### Stage descriptions

#### `nypl_collection_csv.py` — Export NYPL metadata
Walks the NYPL Digital Collections API hierarchy for a collection UUID, recursively
descending into sub-collections. For each item found, fetches capture metadata and
writes one row to a CSV. Requires a `NYPL_API_TOKEN` environment variable (or `--token`).
Responses are cached locally as JSON to avoid redundant API calls on re-runs.

#### `loc_collection_csv.py` — Export LoC metadata
Exports items from a Library of Congress collection or single item URL to the
same CSV schema. No API token required — the LoC JSON API is publicly accessible.
Paginates collections automatically and derives a readable filename slug from the
item title.

```bash
# Export a single LoC item
python loc_collection_csv.py https://www.loc.gov/item/01015253/

# Export a full LoC collection
python loc_collection_csv.py https://www.loc.gov/collections/civil-war-maps/
```

Both scripts produce the same CSV schema, which feeds directly into `download_images.py`:

| Column | Description |
|---|---|
| `item_id` | Item identifier (UUID for NYPL, numeric ID for LoC) — used as the per-item image directory name |
| `item_title` | Human-readable title — used for progress display |
| `iiif_manifest_url` | IIIF Presentation manifest URL — used to download images |
| `microform` | `True` if the item is a microfilm/microform scan — used by `detect_spreads.py` to apply a looser gutter threshold |

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
images from `tile.loc.gov` (which is not Cloudflare-protected). The requested
download width is capped at the native image resolution to avoid upscaling artifacts
from the tile pyramid.

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
`{stem}_split.json` sidecar with the pixel offsets, so downstream scripts can map
coordinates back to the original image. Original images are untouched.

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
Sends each image to the Gemini API using a system prompt from `ocr_prompt.md`
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
Tesseract `.hocr` file, aligns the two using the
[Needleman-Wunsch](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
global sequence alignment algorithm and saves `{stem}_{model_slug}_aligned.json`.

**Reading-order correction:** Tesseract on multi-column pages reads across columns
rather than down each column. Lines are re-sorted by detected column (left→right,
then top-to-bottom within each column) before alignment. Column breaks are detected
as gaps in the distribution of line left-edges (x1) that exceed 10% of page width —
large enough to skip indentation variation within a column but small enough to catch
the true inter-column gutter.

**hOCR line classes:** All Tesseract line types that contain word-level text are
ingested: `ocr_line`, `ocr_caption`, `ocr_textfloat`, `ocr_header`.

**Confidence tiers in output:**

| Tier | Meaning |
|---|---|
| `word` | Per-word NW alignment succeeded; each word has its own bbox |
| `line` | Line matched but word alignment was weak; bbox covers the whole line |
| `none` | Gemini line had no Tesseract match; no coordinates available |

Output JSON schema:

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
`download_images.py`. Bounding boxes are expressed in both pixel space (for
local use) and IIIF canvas fragment space (for annotation tools).

#### `visualize_alignment.py` — Alignment visualization
Reads each `*_aligned.json` and draws color-coded bounding boxes on the source
image, saving `{stem}_{model_slug}_viz.jpg`:

- **Green** — word-confidence boxes (per word)
- **Orange** — line-confidence boxes (whole line)
- **Red** — unmatched Gemini lines (listed in the margin, no coordinates)

Visualization output files are automatically excluded from all OCR and alignment
stages (files ending in `_viz.jpg` are skipped).

---

## Directory layout

```
directory-pipeline/
├── main.py                    # Pipeline orchestrator
├── nypl_collection_csv.py     # Stage 1a: NYPL metadata export
├── loc_collection_csv.py      # Stage 1b: Library of Congress metadata export
├── iiif_utils.py              # Shared IIIF v2/v3 manifest parsing
├── download_images.py         # Stage 2: image download
├── detect_spreads.py          # Stage 3: spread detection
├── split_spreads.py           # Stage 4: spread splitting
├── detect_columns.py          # Stage 5: column detection
├── run_ocr.py                 # Stage 6: Tesseract OCR
├── run_gemini_ocr.py          # Stage 7: Gemini OCR
├── compare_ocr.py             # Stage 8: model comparison
├── align_ocr.py               # Stage 9: NW alignment
├── visualize_alignment.py     # Stage 10: alignment visualization
├── ocr_prompt.md              # System prompt sent to Gemini
├── collections.txt            # List of NYPL collection URLs/UUIDs
├── pyproject.toml             # Python project config and dependencies
├── collection_csv/            # Output of --nypl-csv / --loc-csv (one CSV per collection)
└── images/
    └── {slug}/                # e.g. the_negro_motorist_green_book_1940_feb978b0/
        └── {item_id}/         # NYPL UUID or LoC numeric ID, e.g. feb978b0-… or 01015253
            ├── manifest.json
            ├── 0001_{image_id}.jpg
            ├── 0001_{image_id}_left.jpg          # if spread-split
            ├── 0001_{image_id}_right.jpg         # if spread-split
            ├── 0001_{image_id}_split.json        # split coordinate sidecar
            ├── 0001_{image_id}_tesseract.hocr
            ├── 0001_{image_id}_tesseract.txt
            ├── 0001_{image_id}_{model}.txt       # Gemini plain text
            ├── 0001_{image_id}_{model}_aligned.json
            ├── 0001_{image_id}_{model}_viz.jpg
            ├── 0001_{image_id}_comparison.html
            ├── spreads_report.csv
            └── columns_report.csv
```

For NYPL collections, `{slug}` is derived as `{title_words}_{uuid8}`, e.g.
`the_negro_motorist_green_book_1940_feb978b0`. For LoC items it is derived from
the item title and numeric ID, e.g. `the_brooklyn_city_directory_01015253`. For
CSV-as-source, it is the CSV filename stem. Pass `--slug` to override.

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
export NYPL_API_TOKEN=your_token_here   # from https://api.repo.nypl.org/sign_up
                                        # (not needed for LoC)
export GEMINI_API_KEY=your_key_here
```

---

## Usage

### NYPL collections

```bash
# Export metadata and download images
python main.py collections.txt --nypl-csv --download

# Detect and split spreads, then run both OCR engines
python main.py collections.txt --detect-spreads --split-spreads \
    --tesseract --gemini-ocr --model gemini-2.0-flash

# Run the full alignment and visualization pipeline
python main.py collections.txt --align-ocr --visualize \
    --model gemini-2.0-flash --force
```

### Library of Congress items

```bash
# Export metadata and download images for a single LoC item
python main.py https://www.loc.gov/item/01015253/ \
    --loc-csv --download --tesseract --gemini-ocr --model gemini-2.0-flash

# Export metadata and download images for a full LoC collection
python main.py https://www.loc.gov/collections/civil-war-maps/ \
    --loc-csv --download --gemini-ocr --model gemini-2.0-flash

# Slug is derived from the URL path (e.g. '01015253' or 'civil-war-maps')
# Override with --slug if desired
python main.py https://www.loc.gov/item/01015253/ \
    --loc-csv --download --slug brooklyn_directory_1852
```

`loc_collection_csv.py` can also be run standalone if you want a CSV without going through `main.py`:

```bash
python loc_collection_csv.py https://www.loc.gov/item/01015253/
# → collection_csv/the_brooklyn_city_directory_01015253.csv
```

### Any IIIF manifest

```bash
# Download a single manifest directly (no CSV needed)
python download_images.py --manifest https://www.loc.gov/item/01015253/manifest.json

# Or via main.py using the --manifest flag on download_images.py directly
python download_images.py \
    --manifest https://example.org/iiif/item/manifest.json \
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
python run_gemini_ocr.py images/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash --workers 8

python align_ocr.py images/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash --force

python visualize_alignment.py \
    images/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash
```

---

## Key design decisions

**Gemini for accuracy, Tesseract for coordinates.** Gemini produces far more
accurate transcriptions of historical print than Tesseract, especially for proper
nouns, abbreviations, and damaged text. Tesseract's value is its hOCR output:
word-level bounding boxes. The alignment step combines both.

**Needleman-Wunsch alignment.** Global sequence alignment (gap penalty −40,
similarity score 0–100 based on character edit distance) handles insertions and
deletions gracefully. The algorithm runs twice: first at line level to pair Gemini
lines with Tesseract lines, then at word level within each matched line pair.

**Tesseract dictionary correction disabled.** By default Tesseract silently
"corrects" street names, proper nouns, and abbreviations toward dictionary words
(e.g. `Mound` → `Wound`, `Innesfallen` → `Innisfallen`). This is disabled via
`load_system_dawg=0 load_freq_dawg=0` to preserve names as-written for alignment.

**Column reading-order correction.** Tesseract on multi-column pages reads across
columns (left-to-right by y position) rather than down each column. Lines are
re-sorted into correct reading order (left column top-to-bottom, then right column
top-to-bottom) by grouping hOCR line bounding boxes into column clusters using a
10% page-width gap threshold on x1 coordinates.

**IIIF v2/v3 support.** `iiif_utils.py` provides a single version-agnostic
interface for walking IIIF Presentation manifests. IIIF Image API URLs
(`{service_id}/full/{width},/0/default.jpg`) work identically for both API
versions. The downloader caps requested width at the service's advertised
`maxWidth` to prevent upscaling artifacts on tile-pyramid servers (notably LoC).

**IIIF-native output.** The aligned JSON includes `canvas_uri` and
`canvas_fragment` (IIIF `#xywh=` fragment) for every word and line, making the
output directly consumable by IIIF annotation tools and viewers.
