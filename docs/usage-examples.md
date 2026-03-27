# Usage examples

See the [main README](../README.md) for the quick start. This page covers the full workflow in detail, source-specific notes, and advanced options.

---

## 1. First-time setup: calibrating for a new collection

Every new collection type needs a one-time calibration step that generates tailored OCR and NER prompts. Once generated, those prompts are reused automatically for every subsequent volume in the same series — no code changes needed.

```bash
# Step 1: Download images
python main.py https://archive.org/details/ldpd_11290437_000/ --download

# Step 2: Open the page selector in your browser — pick 4–10 representative pages,
#         then click "Save" and press Ctrl+C when done.
python main.py https://archive.org/details/ldpd_11290437_000/ --select-pages

# Step 3: Generate tailored prompts (Gemini analyzes your selected pages)
python main.py https://archive.org/details/ldpd_11290437_000/ --generate-prompts

# Step 4: Run the pipeline — prompts are auto-discovered from the output directory
python main.py https://archive.org/details/ldpd_11290437_000/ --extract
```

To also include page scoping and the precision bounding-box upgrade, use `--guided` instead of `--extract` in step 4. This will pause at `--select-pages` (to scope which pages to process) and again at `--review-alignment` (to correct unmatched lines) before proceeding.

For microfilm or bound-volume scans that contain double-page spreads, run `--detect-spreads --split-spreads` first, before either `--extract` or `--guided`.

---

## 2. Running on additional volumes (same series)

After calibrating once, point subsequent volumes to the first volume's NER prompt:

```bash
python main.py https://archive.org/details/ldpd_11290437_001/ --extract \
  --ner-prompt output/ldpd_11290437_000/ner_prompt.md
```

The OCR prompt is auto-discovered from the same directory. If both prompts live in the first volume's output folder, `--extract` will find them automatically without any explicit flags.

---

## 3. Source URLs at a glance

The pipeline accepts any of these URL forms directly — no preprocessing needed.

| Source | Example URL | Notes |
|---|---|---|
| Internet Archive item | `https://archive.org/details/ldpd_11290437_000/` | |
| Internet Archive collection | `https://archive.org/details/durstoldyorklibrary` | |
| Library of Congress item | `https://www.loc.gov/item/01015253/` | |
| Library of Congress collection | `https://www.loc.gov/collections/civil-war-maps/` | |
| NYPL item or collection | UUID or collection URL | Requires `NYPL_API_TOKEN` |
| Any IIIF manifest | `https://example.org/iiif/manifest.json` | Accepts v2 or v3 |

NYPL requires an API token. Set it in your environment or `.env` file:

```bash
export NYPL_API_TOKEN=your_token_here
python main.py collections.txt --nypl-csv --download
```

To download a single IIIF manifest directly without a CSV:

```bash
python pipeline/download_images.py \
    --manifest https://example.org/iiif/item/manifest.json \
    --output-dir output/my-item
```

---

## 4. Precision upgrade: bounding boxes per entry

The default `--extract` path gives each entry a canvas URI pointing to its page. Adding Surya OCR and alignment upgrades every `canvas_fragment` to a `#xywh=` bounding box — the exact line on the page.

```bash
# Add to any run:
python main.py URL --surya-ocr --align-ocr

# Then optionally review and correct unmatched lines:
python pipeline/review_alignment.py output/ --model gemini-2.0-flash
```

`--guided` runs this full path automatically, pausing at the review step.

---

## 5. Geocoding and maps

```bash
# Geocode entries (Nominatim city-level by default; Google Maps with API key)
python main.py collections.txt --geocode --model gemini-2.0-flash

# Build the interactive Leaflet map
python main.py collections.txt --map --model gemini-2.0-flash

# Generate map with IIIF Content State deep-links (opens viewer at exact page and region)
python pipeline/geo/map_entries.py output/green_book_1947_xxx/ \
    --model gemini-2.0-flash \
    --viewer-url https://hadro.github.io/green-book-iiif-test \
    --manifest-url https://hadro.github.io/green-book-iiif-test/manifest.json
```

---

## 6. IIIF annotation export

```bash
# Line-level and entry-level W3C Annotation Pages
python pipeline/iiif/export_annotations.py \
    output/green_book_1947_xxx/uuid/ --model gemini-2.0-flash

# Colored entry bounding boxes as annotation overlays
python pipeline/iiif/export_entry_boxes.py \
    output/green_book_1947_xxx/uuid/ --model gemini-2.0-flash

# Export boxes and update the manifest so IIIF viewers load them automatically
python pipeline/iiif/export_entry_boxes.py \
    output/green_book_1947_xxx/uuid/ \
    --base-url https://hadro.github.io/green-book-iiif-test/annotations \
    --update-manifest
```

---

## 7. Running the review tool on Google Colab

The alignment review tool is a local Flask app. Colab's built-in port proxy (`google.colab.kernel.proxyPort`) exposes it to your browser without any third-party tunnel.

**Prerequisite**

Copy your `output/` data into Colab's local filesystem first — Drive-mounted paths can cause Flask to hang on directory scans:

```python
!cp -r /content/drive/MyDrive/pipeline-output /content/output
```

**Step 1 — Start the server and get the URL**

Set your variables, then run this cell. The URL is printed immediately; the server loads models in the background (~30 s).

```python
import subprocess
from google.colab.output import eval_js

PORT   = 5000
MODEL  = "gemini-2.0-flash"
VOLUME = "/content/output/my_volume"

subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)

subprocess.Popen(
    ["python", "-m", "pipeline.review_alignment",
     VOLUME, "--host", "0.0.0.0", "--port", str(PORT), "--model", MODEL],
    stdout=open("/tmp/review.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

print(f"Server starting — run the next cell to watch the log.")
print(f"URL: {eval_js(f'google.colab.kernel.proxyPort({PORT})')}")
```

**Step 2 — Watch the log until ready**

Run this cell and wait for `Models ready.` (~30 s), then stop it — the server keeps running.

```python
!tail -f /tmp/review.log
```

Open the URL printed in Step 1.

**Notes**
- Server not responding: check `!cat /tmp/review.log`
- Clicking **Done reviewing** in the UI shuts the Flask server down — repeat from Step 1 to restart
- Copy results back to Drive when finished: `!cp -r /content/output /content/drive/MyDrive/pipeline-output`

---

## 8. Useful flags

```bash
# Preview every command the pipeline would run, without executing anything
python main.py https://archive.org/details/ldpd_11290437_000/ --extract --dry-run

# Compare two Gemini models on OCR quality before committing to a full run
python main.py collections.txt --compare-ocr \
    --models gemini-2.0-flash gemini-2.5-pro

# Force re-processing of already-completed files
python main.py collections.txt --align-ocr --model gemini-2.0-flash --force
```
