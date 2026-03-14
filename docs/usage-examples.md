# Usage examples

Full usage examples for directory-pipeline by source type and stage. See the [main README](../README.md) for the quick start.

---

## Standard end-to-end run

```bash
# Minimal automated path: download → OCR → CSV
python main.py collections.txt --to-csv

# Full pipeline: also includes Surya alignment, geocoding, and map
python main.py collections.txt --full-run
```

## Library of Congress items

```bash
# Export metadata and download images for a single LoC item
python main.py https://www.loc.gov/item/01015253/ \
    --loc-csv --download --gemini-ocr --model gemini-2.0-flash

# Export metadata and download images for a full LoC collection
python main.py https://www.loc.gov/collections/civil-war-maps/ \
    --loc-csv --download --gemini-ocr --model gemini-2.0-flash

# Override slug if desired
python main.py https://www.loc.gov/item/01015253/ \
    --loc-csv --download --slug brooklyn_directory_1852
```

## Internet Archive

```bash
# Download a single IA item
python main.py https://archive.org/details/ldpd_11290437_000/ \
    --ia-csv --download --gemini-ocr --model gemini-2.0-flash

# Download an IA collection
python main.py https://archive.org/details/durstoldyorklibrary \
    --ia-csv --download --gemini-ocr --model gemini-2.0-flash
```

## NYPL collections

```bash
# Export metadata and download images
python main.py collections.txt --nypl-csv --download

# Detect and split spreads, then run OCR
python main.py collections.txt --detect-spreads --split-spreads \
    --surya-ocr --gemini-ocr --model gemini-2.0-flash

# Run alignment and visualize results
python main.py collections.txt --align-ocr --visualize \
    --model gemini-2.0-flash

# Interactively review and correct unmatched entries
python pipeline/review_alignment.py output/ --model gemini-2.0-flash

# Run the review tool on Google Colab (see below for full setup)
# The key difference is --host 0.0.0.0 so the ngrok tunnel can reach it

# Extract entries, geocode, and build a map
python main.py collections.txt --extract-entries --geocode --map \
    --model gemini-2.0-flash
```

## Volume-specific prompt calibration

For any new collection type, generate tailored OCR and NER prompts before running OCR.
The generated NER prompt defines whatever entry fields are appropriate for the
document — no code changes are needed.

```bash
# Step 1: Download images
python main.py collections.txt --nypl-csv --download

# Step 2: Open the browser UI and pick 4–10 representative sample pages
#         Click "Save to output folder" — selection.txt is written automatically.
#         Press Ctrl+C in the terminal when done.
python main.py collections.txt --select-pages

# Step 3: Generate volume-specific prompts (Gemini analyzes the selected pages)
python main.py collections.txt --generate-prompts

# Step 4: Automated CSV run — prompts are auto-discovered
python main.py collections.txt --to-csv

# Or the full pipeline with precision upgrade, geocoding, and map:
python main.py collections.txt --full-run
```

Or run the prompt generation standalone:

```bash
# Standalone — with explicit selection file
python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \
    --selection output/the_travelers_guide_e088efa0/selection.txt

# OCR prompt only, using explicit page filenames instead of selection.txt
python pipeline/generate_prompt.py output/the_travelers_guide_e088efa0/ \
    --pages 0005_58019060.jpg 0012_58019067.jpg 0023_58019078.jpg 0041_58019096.jpg \
    --ocr-only
```

## Any IIIF manifest

```bash
# Download a single manifest directly (no CSV needed)
python pipeline/download_images.py \
    --manifest https://example.org/iiif/item/manifest.json \
    --output-dir output/my-item
```

## Other options

```bash
# Compare two Gemini models before committing to a full run
python main.py collections.txt --compare-ocr \
    --models gemini-2.0-flash gemini-2.5-pro

# Dry run — show every command the pipeline would execute, without running anything
python main.py https://archive.org/details/ldpd_11290437_000/ --to-csv --dry-run

# Force re-processing of already-completed files
python main.py collections.txt --align-ocr --model gemini-2.0-flash --force
```

## Running pipeline scripts directly

```bash
python pipeline/align_ocr.py output/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash --force

python pipeline/extract_entries.py output/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash

python pipeline/geo/geocode_entries.py output/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash

python pipeline/geo/map_entries.py output/the_negro_motorist_green_book_1940_feb978b0 \
    --model gemini-2.0-flash
```

Analysis tools can be run from the `analysis/` directory:

```bash
python analysis/visualize_alignment.py \
    output/the_negro_motorist_green_book_1940_feb978b0 --model gemini-2.0-flash

python analysis/compare_extraction.py \
    output/the_negro_motorist_green_book_1940_feb978b0
```

## IIIF annotation export and self-hosted viewer

```bash
# Export line-level and entry-level IIIF annotation pages
python pipeline/iiif/export_annotations.py \
    output/green_book_1947_xxx/uuid/ --model gemini-2.0-flash

# Export colored entry bounding boxes (standalone — not in --full-run)
python pipeline/iiif/export_entry_boxes.py \
    output/green_book_1947_xxx/uuid/ --model gemini-2.0-flash

# Export boxes and update manifest so viewers auto-load them
python pipeline/iiif/export_entry_boxes.py \
    output/green_book_1947_xxx/uuid/ \
    --base-url https://hadro.github.io/green-book-iiif-test/annotations \
    --update-manifest

# Generate map with IIIF Content State deep-links (open viewer at correct page/region)
python pipeline/geo/map_entries.py output/green_book_1947_xxx/ \
    --model gemini-2.0-flash \
    --viewer-url https://hadro.github.io/green-book-iiif-test \
    --manifest-url https://hadro.github.io/green-book-iiif-test/manifest.json
```

## Running the review tool on Google Colab

The review tool is a local Flask web app, so running it on Colab requires a tunnel
to expose the server to your browser. The steps below use [ngrok](https://ngrok.com)
(free tier is sufficient).

**Prerequisites**

- Your `output/` data must be accessible in Colab — either copy it from Google Drive
  or mount Drive and copy locally (Drive-mounted paths can cause Flask to hang on
  directory scans):
  ```python
  !cp -r /content/drive/MyDrive/pipeline-output /content/output
  ```
- Install pyngrok in your Colab environment:
  ```python
  !pip install pyngrok
  ```

**Step 1 — Start the server**

Run this cell. It starts Flask in the background via `nohup` so it survives cell
completion, and writes all output to `/tmp/review.log`.

```python
# Kill anything already on port 5000
!fuser -k 5000/tcp

!nohup python -m pipeline.review_alignment /content/output/my_volume \
    --host 0.0.0.0 --port 5000 --model gemini-2.0-flash \
    > /tmp/review.log 2>&1 &
```

**Step 2 — Wait for the server to be ready**

Run this cell and wait until you see `Models ready.` in the output (takes ~30 s).
Stop the cell (square button) once you see it — the server keeps running.

```python
!tail -f /tmp/review.log
```

**Step 3 — Open the ngrok tunnel**

```python
from pyngrok import ngrok

NGROK_TOKEN = "your_token_here"   # from https://dashboard.ngrok.com
ngrok.set_auth_token(NGROK_TOKEN)

public_url = ngrok.connect(5000)
print("Review tool URL:", public_url)
```

Open the printed URL in your browser.

**Notes**

- If you see `ERR_NGROK_324` (too many tunnels), kill old ones first:
  ```python
  for t in ngrok.get_tunnels():
      ngrok.disconnect(t.public_url)
  ```
- If you see `connection refused`, check that the server is still running:
  ```python
  !cat /tmp/review.log
  ```
- When you click **Done reviewing** in the UI the Flask server shuts down. To restart,
  repeat from Step 1.
- Saves are written back to the local `/content/output/` copy. Copy results back to
  Drive when finished:
  ```python
  !cp -r /content/output /content/drive/MyDrive/pipeline-output
  ```