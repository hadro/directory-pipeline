# Pipeline stage full reference

Detailed documentation for each stage in the directory-pipeline. See the [main README](../README.md) for an overview and quick start.

---

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

All three source scripts produce the same CSV schema, which feeds into `pipeline/download_images.py`:

| Column | Description |
|---|---|
| `item_id` | Item identifier — used as the per-item image directory name |
| `item_title` | Human-readable title — used for progress display |
| `iiif_manifest_url` | IIIF Presentation manifest URL — used to download images |
| `microform` | `True` if the item is a microfilm/microform scan — used by `detect_spreads.py` |

#### `pipeline/download_images.py` — Download images
Fetches images from IIIF manifests. Two input modes:

**CSV mode** (default): reads a collection CSV and downloads every item's images.
```
output/{slug}/{item_id}/{page:04d}_{image_id}.jpg
```

**Manifest mode** (`--manifest URL`): downloads a single IIIF manifest directly — no CSV needed. Accepts any public IIIF Presentation v2 or v3 manifest URL.
```
output/{item_id}/{page:04d}_{image_id}.jpg
```

The IIIF manifest is cached alongside the images for downstream use by `align_ocr.py`.
Handles HTTP 429/503 with exponential backoff and retry. Falls back to an alternative
IIIF endpoint on repeated HTTP 403 errors. Safe to re-run — existing files are skipped.

**LoC-specific:** `www.loc.gov/item/{id}/manifest.json` endpoints are blocked by
Cloudflare for non-browser clients. The downloader automatically falls back to the
LoC item JSON API (`?fo=json`) to build a synthetic IIIF manifest, then downloads
images from `tile.loc.gov`. The requested download width is capped at the native
image resolution to avoid upscaling artifacts from the tile pyramid.

#### `pipeline/detect_spreads.py` — Spread detection
Analyzes each image to determine whether it contains two facing pages (a spread
captured in a single microfilm frame) vs. a single page. Uses pixel-projection
analysis: finds the content boundary within the dark microfilm border, checks the
aspect ratio, then scans the central vertical band for a persistent gutter/seam.

Handles three physical gutter forms: dark shadow (spine), white gap (pages splaying
open), tonal boundary (dark cover beside white content). Outputs
`spreads_report.csv` — does not modify images.

If a collection CSV is available (via `--csv`), items flagged as microform get a
looser detection threshold.

#### `pipeline/split_spreads.py` — Split spreads
Reads `spreads_report.csv` and, for each spread, crops the image at the detected
gutter column into `{stem}_left.jpg` and `{stem}_right.jpg`. Also writes a
`{stem}_split.json` sidecar with the pixel offsets. Original images are untouched.

#### `pipeline/select_pages.py` — Page selector (interactive, two modes)
Generates a browser UI with two tabs. Opens the page in the default browser
automatically (pass `--no-open` to suppress).

**Tab 1 — Sample pages:** pick 4–10 representative pages for prompt calibration.
Saves `selection.txt`, consumed by `generate_prompt.py`.

**Tab 2 — Scope pages:** all pages start selected (green). Deselect pages to
exclude from OCR and entry extraction — useful for trimming frontmatter,
advertisements, and almanac-style sections that precede the actual directory
listings. Use the "Exclude first N" quick action for fast frontmatter trimming.
Saves `included_pages.txt`, consumed by `run_gemini_ocr.py` and
`extract_entries.py`. If absent, all pages are processed (backward compatible).

**Saving:** in server mode the Save buttons write files directly to `output/{slug}/`.
Press Ctrl+C when done. With `--no-open`, files are browser-downloaded instead.

```bash
python pipeline/select_pages.py output/the_travelers_guide_e088efa0/
python pipeline/select_pages.py output/the_travelers_guide_e088efa0/ --no-open
```

#### `pipeline/generate_prompt.py` — Volume-specific prompt generation
Sends selected sample page images to Gemini and generates two prompts tailored to
the specific volume's layout, typography, and entry format:

- `output/{slug}/ocr_prompt.md` — OCR transcription system prompt
- `output/{slug}/ner_prompt.md` — NER entry-extraction system prompt

Both prompts are **auto-discovered** by `run_gemini_ocr.py` and `extract_entries.py`
when they exist in the slug directory — no explicit `--prompt-file` flag needed.
Falls back to the global `prompts/ocr_prompt.md` / `prompts/ner_prompt.md` if no
volume-specific prompt is found.

Safe to re-run: if both output files already exist the script exits immediately
without calling the API. Pass `--force` to regenerate.

The NER meta-prompt instructs Gemini to define `page_context` fields and entry fields
appropriate to the document it sees. The only fixed requirement is the JSON response envelope:
`{"page_context": {...}, "entries": [...]}`. `extract_entries.py` infers CSV column
names dynamically from whatever fields the model returns, so **no code changes are needed for new collection types**.

Pass `--ner-template` to specify a reference prompt shown to Gemini as a structural
example (defaults to `prompts/ner_prompt.md`; use `prompts/ner_prompt_greenbook.md`
for a richer directory-style example).

Requires a `selection.txt` file (from `--select-pages`). Pass `--ocr-only` or
`--ner-only` to generate a single prompt. Prints both prompts to stdout for
immediate review in addition to saving them.

```bash
# Generate both prompts (typical workflow):
python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \
    --selection output/the_travelers_guide_e088efa0/selection.txt

# Use the Green Book prompt as a structural reference:
python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \
    --selection output/the_travelers_guide_e088efa0/selection.txt \
    --ner-template prompts/ner_prompt_greenbook.md

# OCR prompt only:
python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \
    --selection output/the_travelers_guide_e088efa0/selection.txt --ocr-only

# Use explicit page filenames instead of a selection file:
python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \
    --pages 0005_58019060.jpg 0012_58019067.jpg 0023_58019078.jpg 0041_58019096.jpg
```

#### `pipeline/surya_detect.py` — Surya column detection (preferred)
Runs Surya's neural text-line detection model on each image to count text columns
and detect gutter positions. Produces the same `columns_report.csv` format as
`detect_columns.py` (with `recommended_psm` per image), so either detector feeds
`old/run_ocr.py` unchanged. Surya's neural approach is more robust than pixel-projection
on degraded microfilm scans and pages with irregular layouts.

#### `pipeline/detect_columns.py` — Pixel-projection column detection (legacy)
Uses a vertical pixel-projection profile (dark-pixel density per column) to count
text columns per image and detect gutter positions. Outputs `columns_report.csv`
with a `recommended_psm` (Tesseract page segmentation mode) for each image:
PSM 4 for single-column pages, PSM 1 for multi-column. `old/run_ocr.py` reads this
CSV to apply per-image PSM automatically. Prefer `--surya-detect` for new runs.

#### `pipeline/run_surya_ocr.py` — Surya OCR (preferred)
Runs Surya's recognition model on each image and saves:
- `{stem}_surya.json` — line-level bounding boxes with text and confidence scores
- `{stem}_surya.txt` — plain text (one line per Surya line)

Surya produces line-level bounding boxes rather than word-level, which aligns
more cleanly with Gemini's line-oriented output format. Handles multi-column
pages via the reading-order correction in `align_ocr.py`. Runs in batches
(default 4 images per batch; reduce with `--batch-size` if OOM).

#### `old/run_ocr.py` — Tesseract OCR (legacy)
Runs Tesseract on each image and saves:
- `{stem}_tesseract.hocr` — full hOCR with word-level bounding boxes
- `{stem}_tesseract.txt` — plain text

Tesseract dictionary correction is **disabled by default** (`load_system_dawg=0`,
`load_freq_dawg=0`). This preserves proper nouns, street names, and abbreviations
as-written, which is critical for accurate NW alignment with Gemini text. Use
`--dict` to re-enable. Runs in parallel using `--workers`.

Per-image PSM is loaded from `columns_report.csv` if present; the global `--psm`
flag always overrides it. Prefer `--surya-ocr` for new runs.

#### `pipeline/run_gemini_ocr.py` — Gemini OCR
Sends each image to the Gemini API using a system prompt from `prompts/ocr_prompt.md`
(auto-discovered; see `--generate-prompts`) and saves the plain-text response as
`{stem}_{model_slug}.txt`. Handles HTTP 429 rate limits with exponential backoff
(up to 5 retries, starting at 10 s delay). Skips images where the output file
already exists and is non-empty; deletes and retries empty output files.

Split images (`_left.jpg`, `_right.jpg`) are preferred over the original spread
when both exist — the original is skipped to avoid redundant API calls on pages
that have already been split.

**Quality-failure retry.** When output fails a quality check (dot-leader runaway or
repetition loop), the page is retried up to three times with escalating temperatures
`[0.1, 0.3, 0.7]`. Dot-leader failures also trigger an additional `_NO_LEADER_INSTRUCTION`
appended to the system prompt. If all retries fail, dot-leader runs are collapsed to
a canonical five-dot sequence rather than discarding the page.

**Auto-resolution for handwriting.** If the OCR prompt contains any of the keywords
`handwrit`, `manuscript`, or `cursive`, high-resolution image mode is enabled
automatically. Override explicitly with `--high-res`.

**Ditto mark expansion.** Pass `--expand-dittos` to append a ditto-expansion
instruction to the system prompt, directing the model to interpret `"` ditto marks
as repeating the value from the line above. Useful for directories that use ditto
marks to avoid repeating city names or categories.

#### `analysis/compare_ocr.py` — Model comparison
Calls multiple models (any mix of Gemini model names and the special token
`surya`) on each image and produces:
- `{stem}_comparison.html` — side-by-side panel view of each model's output
- `ocr_comparison_stats.csv` — character-level similarity stats across all images

Useful for evaluating which Gemini model performs best on a collection before
committing to a full extract or guided run.

#### `pipeline/align_ocr.py` — NW alignment
The core output stage. For each image that has both a Gemini `.txt` file and a
Surya `.json` (preferred) or Tesseract `.hocr` (legacy) file, aligns the two using
anchored [Needleman-Wunsch](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
global sequence alignment and saves `{stem}_{model_slug}_aligned.json`.

**Approach:** Runs a single global NW pass over all Surya lines (or Tesseract words)
and all Gemini tokens on a page. Before the NW pass, city/state headings and category
lines that appear verbatim in both sources are committed as fixed anchors. The NW
problem is then split into independent segments at those anchors, preventing
misalignment drift across long pages where OCR reading order diverges from Gemini's.

**Reading-order correction:** OCR on multi-column pages may read across columns
rather than down each column. Lines are re-sorted before alignment:
- *Row-major (default):* lines are grouped into 50 px horizontal bands and sorted
  left-column-first within each band. Correctly places centered section headings
  (state names, category lines) before the body columns they head.
- *Column-major (true two-column pages):* when lines cluster into exactly two columns
  each holding ≥ 20% of page lines, the left column is emitted top-to-bottom followed
  by the right column top-to-bottom, matching Gemini's reading order for pages with
  independent side-by-side city sections.

Column breaks are detected in two stages: first as gaps in the x1 distribution
exceeding 8% of page width; then a bimodal histogram fallback for pages where a
page-number outlier creates a degenerate single-line split.

**Output JSON schema:**

```json
{
  "image": "0001_58030238.jpg",
  "model": "gemini-2.0-flash",
  "canvas_uri": "https://...",
  "canvas_width": 3316,
  "canvas_height": 4513,
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
  "unmatched_gemini": ["lines with no bbox match"]
}
```

IIIF canvas URIs are read from the `manifest.json` cached by
`download_images.py`. `canvas_width` and `canvas_height` reflect the **natural
image pixel dimensions** fetched from the IIIF Image API `info.json` — not the
dimensions declared in the manifest, which may differ (e.g. NYPL manifests
declare 2560×2560 square canvases for portrait images). All `bbox` and
`canvas_fragment` coordinates are in this natural pixel space, which is required
for IIIF annotation tools and Mirador to place boxes correctly.

#### `analysis/visualize_alignment.py` — Alignment visualization
Reads each `*_aligned.json` and draws color-coded bounding boxes on the source
image, saving `{stem}_{model_slug}_viz.jpg`:

- **Green** — word-confidence boxes (per word)
- **Red** — unmatched Gemini lines (listed in the margin, no coordinates)

Visualization output files are automatically excluded from all OCR and alignment
stages (files ending in `_viz.jpg` are skipped).

#### `pipeline/review_alignment.py` — Interactive alignment review
A local Flask web UI for manually correcting pages where automatic alignment left
unmatched Gemini entries. Run after `--align-ocr` (and optionally `--visualize`)
to work through problematic pages before proceeding to `--extract-entries`.

**Workflow:**
1. The sidebar lists all pages in the images directory, sorted by unmatched-entry
   count. Filter by volume, search term, or minimum unmatched count.
2. Click a page to load it. Matched lines are overlaid in green.
3. The *Unmatched Gemini text* panel shows entries that have no bounding box.
4. Draw one or more bounding boxes over the regions containing those entries —
   each drag adds a box. Boxes are numbered on the canvas when multiple are drawn.
   Use *Undo last* to remove the most recent box.
5. Click *Run Surya on N boxes*. Surya re-runs OCR on each crop; all detected
   lines are merged top-to-bottom and Needleman-Wunsch-aligned against the
   unmatched entries.
6. The *Proposed matches* panel shows Surya → Gemini pairs with similarity scores.
   Pairs ≥ 40% are pre-checked in green. Uncheck or add as needed.
7. Click *Save accepted*. Matched pairs are written back to `*_aligned.json`
   with `"confidence": "manual"` and the page's unmatched count updates immediately.

**Keyboard navigation:** press ↑/↓ arrow keys to move between pages in the sidebar
(when focus is not in the search field). The selected page scrolls into view
automatically.

Surya models are pre-loaded at server startup (~30 s) so all subsequent
annotation requests are fast.

```bash
# Run standalone (recommended for iterative review):
python pipeline/review_alignment.py output/ --model gemini-2.0-flash

# Or via the pipeline (blocks until Ctrl+C):
python main.py collections.txt --review-alignment --model gemini-2.0-flash
```

#### `pipeline/extract_entries.py` — Entry extraction
Reads each `*_aligned.json` file and calls Gemini with the volume-specific NER
prompt (auto-discovered from `output/{slug}/ner_prompt.md`, falling back to
`prompts/ner_prompt.md`) to identify structured entries.

**Schema-agnostic:** entry fields are defined entirely by the NER prompt and inferred
dynamically from the model's output. The aggregate `entries_{model}.csv` columns
reflect exactly what the prompt asks for. Context is carried between pages using
whatever `page_context` fields the prompt defines. No code changes are needed for
a new collection type — only a new NER prompt.

For Green Book collections the typical fields are `establishment_name`, `raw_address`,
`city`, `state`, `category`, and `canvas_fragment`. For natural history volumes or
other document types the fields will match that volume's NER prompt.

Outputs a per-page `*_entries.json` sidecar and an aggregate `entries_{model}.csv`.

By default uses `gemini-3.1-flash-lite-preview` (fast and cheap). Dense pages that exceed the
model's output token limit automatically fall back to `gemini-2.5-flash` (higher
output limit), and then to partial JSON recovery if needed. Previously failed pages
(where a `*_entries_error.txt` sidecar exists) are auto-retried without `--force`.

#### `pipeline/geo/geocode_entries.py` — Geocoding
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
GOOGLE_MAPS_API_KEY=... python pipeline/geo/geocode_entries.py \
    output/green_book_1962_9ab2e8f0/ --model gemini-2.0-flash
```

#### `pipeline/geo/map_entries.py` — Interactive map
Reads a `entries_{model}_geocoded.csv` and generates a self-contained Leaflet HTML
map (`entries_{model}.html`) with:
- Clustered markers (MarkerCluster) color-coded by establishment category
- Sidebar with live search, state dropdown, and category checkboxes
- Live count of entries shown
- Source-scan thumbnails in map popups (when IIIF manifests are available)
- IIIF Content State deep-links that open a IIIF viewer directly to the correct page and region (when `--viewer-url` is supplied)

Entries geocoded at city level are jittered slightly to avoid stacking. Requires the
geocoded CSV from `geocode_entries.py`.

When IIIF manifests are present alongside the images (as cached by `download_images.py`),
each marker popup includes a thumbnail of the exact page region where the entry appears,
fetched directly from the source institution's IIIF image server. Pass `--output-dir`
to specify the output root if the CSV has been moved; by default the script searches
the CSV's parent directory for manifests.

**IIIF Content State deep-links.** Pass `--viewer-url` to embed a link in each popup
that opens a IIIF viewer directly at the correct page and zooms to the entry region.
The link encodes a [IIIF Content State 1.0](https://iiif.io/api/content-state/1.0/)
annotation as a Base64url `?iiif-content=` URL parameter. Any IIIF viewer that
supports Content State (including the demo self-hosted Mirador viewer at
`hadro.github.io/green-book-iiif-test`) will navigate directly to the right canvas
and scroll the entry into view.

Pass `--manifest-url` to specify the manifest explicitly; if omitted the script
derives it as `{viewer-url}/manifest.json`.

```bash
python pipeline/geo/map_entries.py output/green_book_1940_feb978b0/ --model gemini-2.0-flash
python pipeline/geo/map_entries.py path/to/entries.csv --output-dir output/
python pipeline/geo/map_entries.py output/green_book_1947_xxx/ \
    --model gemini-2.0-flash \
    --viewer-url https://hadro.github.io/green-book-iiif-test \
    --manifest-url https://hadro.github.io/green-book-iiif-test/manifest.json
```

#### `pipeline/iiif/export_annotations.py` — IIIF Annotation Pages export
Converts `*_{model}_aligned.json` files to W3C Annotation Pages (JSON-LD), the
standard format for overlaying transcription text on IIIF images in viewers like
Mirador, Universal Viewer, and Clover.

Produces two annotation files per page:
- `*_{model}_annotations.json` — line-level transcription (`motivation: supplementing`),
  one annotation per aligned line with the Gemini-corrected text as the body and the
  IIIF `#xywh=` canvas fragment as the target
- `*_{model}_entry_annotations.json` — entry-level structured data
  (`motivation: describing`) when `*_{model}_entries.json` sidecars are present;
  body is `name — address, city, state` as plain text

Annotation pages are valid without server-hosted IDs (omit `--base-url` for
self-contained local files). Add `--base-url` to embed persistent URIs so viewers
can reload annotations from a known endpoint.

```bash
python pipeline/iiif/export_annotations.py output/green_book_1940_feb978b0/uuid/
python pipeline/iiif/export_annotations.py output/green_book_1940_feb978b0/uuid/ \
    --base-url https://example.org/annotations --model gemini-2.0-flash
python pipeline/iiif/export_annotations.py output/green_book_1940_feb978b0/uuid/ \
    --no-entries   # line-level transcription only
```

#### `pipeline/iiif/export_entry_boxes.py` — IIIF colored bounding boxes
Reads `*_{model}_entries.json` files and produces `*_{model}_box_annotations.json`
— W3C Annotation Pages with colored rectangular overlays, one annotation per
entry. Color coding matches `analysis/visualize_entries.py`:

| Category | Color |
|---|---|
| Hotels / Motels | Blue |
| Tourist Homes | Teal |
| Restaurants / Bars | Red |
| Barber / Beauty | Purple |
| Service Stations | Amber |
| Other / Unknown | Grey |

Advertisements receive a stroke twice as thick as regular entries.

Each annotation uses a simple `FragmentSelector` (`#xywh=` canvas fragment) as its
target, which Mirador 3.3 and Universal Viewer handle most reliably. Coordinates are
converted to natural image pixel space at export time (see *IIIF canvas coordinate
space* under Key design decisions).

With `--update-manifest`, the script also:
1. Adds an `annotations` property to each canvas in `manifest.json` referencing its
   box annotation page, so IIIF viewers load the colored boxes automatically on open.
2. Corrects each canvas `width`/`height` in the manifest to the natural image
   dimensions (fetched from the IIIF image service `info.json`), which is required
   for Mirador to map annotation coordinates to the correct pixel positions.
   Requires `--base-url`. Original manifest is backed up as `manifest_bak.json`.

```bash
python pipeline/iiif/export_entry_boxes.py output/green_book_1947_xxx/uuid/
python pipeline/iiif/export_entry_boxes.py output/green_book_1947_xxx/uuid/ \
    --model gemini-2.0-flash
python pipeline/iiif/export_entry_boxes.py output/green_book_1947_xxx/uuid/ \
    --base-url https://hadro.github.io/green-book-iiif-test/annotations \
    --update-manifest
```

#### `pipeline/iiif/build_ranges.py` — IIIF table of contents (directory collections)
Builds a IIIF Presentation API v3 `structures` array (Range hierarchy) from a
geocoded entries CSV, grouping entries by State → City → Category. Each range
node points to the first canvas where that group appears in document order.

Designed for directory-style collections with `state`, `city`, and `category`
fields. Outputs a standalone `ranges_{model}.json` that can be loaded by IIIF
viewers, or merged directly into `manifest.json` with `--update-manifest`.

```bash
python pipeline/iiif/build_ranges.py output/green_book_1947_4bea2040/uuid/
python pipeline/iiif/build_ranges.py output/green_book_1947_4bea2040/uuid/ \
    --model gemini-2.0-flash --depth 2 --update-manifest \
    --base-url https://hadro.github.io/green-book-iiif-test
```

---

#### `pipeline/explore_entries.py` — Interactive data explorer
Reads an `entries_{model}.csv` (or an output directory containing one) and writes
a **self-contained HTML file** — no server or dependencies required. Open it in
any browser.

The explorer auto-introspects the CSV schema, so it works for any document type:

- **Facet filters** — auto-generated for low-cardinality fields (category, state, etc.); clicking a bar in a chart filters the table
- **Full-text search** across all fields
- **Page density strip** — entry count per canvas, showing document structure at a glance
- **Field fill-rate overview** — which columns have reliable data across the volume
- **Detail panel** — click any row to see all fields plus a live IIIF thumbnail of the source page region (when `manifest.json` files are present)
- **CSV export** — download the current filtered subset

No geocoding or alignment required; runs directly after `--extract-entries`.

```bash
python pipeline/explore_entries.py output/green_book_1947_4bea2040/
python pipeline/explore_entries.py output/green_book_1947_4bea2040/ --out my_explorer.html
python pipeline/explore_entries.py output/green_book_1947_4bea2040/uuid/entries_gemini-2.0-flash.csv
```

#### `pipeline/run_chandra_ocr.py` — Chandra OCR (local model, GPU)
Runs [Chandra](https://github.com/datalab-to/chandra-ocr) (a 5B vision-language
model) on each image and saves `{stem}_chandra-ocr-2.txt`. Converts Chandra's
Markdown output (tables, bold headers) to plain text compatible with `align_ocr.py`.
Safe to re-run — already-processed images are skipped.

Chandra is fully local — no API key or per-call cost — and produces layout-aware
transcriptions well-suited to complex multi-column pages. GPU strongly recommended
(T4 16 GB fits the BF16 model with headroom). Supports two inference backends:

- `--method hf` — HuggingFace Transformers (default; works on Colab T4)
- `--method vllm` — vLLM server (faster; requires H100 80 GB)

To use Chandra output for alignment, pass `--model chandra-ocr-2` to `align_ocr.py`.

```bash
python pipeline/run_chandra_ocr.py output/travelguide/
python pipeline/run_chandra_ocr.py output/travelguide/ --method hf --batch-size 4
python pipeline/align_ocr.py output/travelguide/ --model chandra-ocr-2
```

#### `pipeline/review_ocr.py` — OCR anomaly triage report
Compares each page's line count and average line length to a rolling window of
neighboring pages and flags pages that deviate significantly. Writes a
**self-contained HTML report** with per-page thumbnails for quick visual triage.

Flagged pages are not necessarily errors — title pages, ad pages, or other
non-directory content will also be flagged. The thumbnail lets you tell at a glance
whether a flag is a legitimate layout difference or an OCR problem worth investigating.

Useful as a fast sanity check before running `--extract-entries` on a new volume.

```bash
python pipeline/review_ocr.py output/my-collection/item_dir/
python pipeline/review_ocr.py output/my-collection/item_dir/ --model gemini-2.0-flash
python pipeline/review_ocr.py output/my-collection/item_dir/ --threshold 0.5 --window 3
```

#### `analysis/visualize_entries.py` — Entry bounding box visualization
Reads each `*_entries.json` file and draws color-coded bounding boxes on the
corresponding image, saving `*_entries_viz.jpg`. Color assignment is driven by
category values present in the data — consistent across pages within a run. Pass
`--ner-prompt` to pre-seed all expected categories so colors stay stable even on
pages where a category is absent.

Entries with no spatial coordinates (`canvas_fragment` has no `#xywh`) are listed
in the right margin in their category color.

Distinct from `analysis/visualize_alignment.py` (which shows the raw NW alignment
result) — this script shows the final extracted entry footprints.

```bash
python analysis/visualize_entries.py output/green_book_1947_4bea2040/uuid/
python analysis/visualize_entries.py output/green_book_1947_4bea2040/uuid/ \
    --ner-prompt output/green_book_1947_4bea2040/ner_prompt.md
python analysis/visualize_entries.py output/green_book_1947_4bea2040/ \
    --model gemini-2.0-flash --force
```

#### `pipeline/patch_canvas_fragments.py` — Retroactive bounding box patching
Re-runs the canvas fragment matching logic against existing `*_aligned.json` files
and updates `*_entries.json` sidecars and the volume CSV — **without making any
Gemini API calls**. Useful when alignment has been improved (e.g. after a
`--review-alignment` session) and you want the entry coordinates updated to reflect
the better alignment without re-running the full extraction.

Uses the same three-strategy matching chain as `extract_entries.py` (exact →
substring → fuzzy).

```bash
python pipeline/patch_canvas_fragments.py output/my_volume/ \
    --aligned-model gemini-2.0-flash
python pipeline/patch_canvas_fragments.py output/collection/ \
    --aligned-model gemini-2.0-flash
```