# Usage examples

See the [main README](../README.md) for the quick start. This page covers the full workflow in detail, source-specific notes, and advanced options.

Commands below use the `pipeline` CLI installed by `uv sync`. Each subcommand wraps `python main.py` stage flags — see [pipeline-stages.md](pipeline-stages.md) for the flag-level reference.

---

## 1. First-time setup: calibrating for a new collection

Every new collection type needs a one-time calibration step that generates tailored OCR and NER prompts. Once generated, those prompts are reused automatically for every subsequent volume in the same series — no code changes needed.

```bash
# Step 1: Download images
pipeline ingest https://archive.org/details/ldpd_11290437_000/

# Step 2: Calibrate — opens the page selector in your browser (pick 4–10
#         representative pages, click "Done — continue pipeline"), then has
#         Gemini generate tailored OCR + NER prompts from your selection.
pipeline calibrate https://archive.org/details/ldpd_11290437_000/

# Step 3: Run the pipeline — prompts are auto-discovered from the output directory
pipeline run https://archive.org/details/ldpd_11290437_000/
```

To also include page scoping and the precision bounding-box upgrade, use `pipeline guided` instead of `pipeline run` in step 3. This pauses at page selection (to scope which pages to process) and again at alignment review (to correct unmatched lines) before proceeding.

For microfilm or bound-volume scans that contain double-page spreads, add spread handling to the ingest step:

```bash
pipeline ingest <URL> --detect-spreads --split-spreads
```

---

## 2. Running on additional volumes (same series)

After calibrating once, point subsequent volumes to the first volume's NER prompt:

```bash
pipeline run https://archive.org/details/ldpd_11290437_001/ \
  --ner-prompt output/ldpd_11290437_000/ner_prompt.md
```

The OCR prompt is auto-discovered from the same directory. If both prompts live in the first volume's output folder, `pipeline run` will find them automatically without any explicit flags.

---

## 3. Source URLs at a glance

The pipeline accepts any of these URL forms directly — no preprocessing needed.

| Source | Example URL | Notes |
|---|---|---|
| Internet Archive item | `https://archive.org/details/ldpd_11290437_000/` | |
| Internet Archive collection | `https://archive.org/details/durstoldyorklibrary` | |
| Library of Congress item | `https://www.loc.gov/item/01015253/` | |
| Library of Congress collection | `https://www.loc.gov/collections/civil-war-maps/` | |
| Any IIIF manifest | `https://example.org/iiif/manifest.json` | Accepts v2 or v3 |

To download a single IIIF manifest directly without a CSV:

```bash
python -m pipeline.download_images \
    --manifest https://example.org/iiif/item/manifest.json \
    --output-dir output/my-item
```

---

## 4. Precision upgrade: bounding boxes per entry

The default `pipeline run` path gives each entry a canvas URI pointing to its page. Adding Surya OCR and alignment upgrades every `canvas_fragment` to a `#xywh=` bounding box — the exact line on the page.

```bash
# Surya OCR + Gemini OCR + alignment on an already-downloaded volume
# (already-completed OCR files are skipped automatically)
pipeline ocr output/ldpd_11290437_000/

# Then optionally review and correct unmatched lines in a browser UI
# (the model slug is auto-detected from pipeline_state.json)
pipeline review output/ldpd_11290437_000/
```

`pipeline guided` runs this full path automatically, pausing at the review step.

---

## 5. Geocoding and maps

```bash
# Geocode entries (Nominatim city-level by default; Google Maps with API key),
# then build the interactive Leaflet map. The model slug is auto-detected.
python main.py output/ldpd_11290437_000/ --geocode --map

# Generate map with IIIF Content State deep-links (opens viewer at exact page and region)
python -m pipeline.geo.map_entries output/green_book_1947_xxx/ \
    --viewer-url https://hadro.github.io/green-book-iiif-test \
    --manifest-url https://hadro.github.io/green-book-iiif-test/manifest.json
```

---

## 6. IIIF annotation export

```bash
# Line-level and entry-level W3C Annotation Pages
python -m pipeline.iiif.export_annotations \
    output/green_book_1947_xxx/uuid/ --model gemini-3.1-flash-lite

# Colored entry bounding boxes as annotation overlays
python -m pipeline.iiif.export_entry_boxes \
    output/green_book_1947_xxx/uuid/ --model gemini-3.1-flash-lite

# Export boxes and update the manifest so IIIF viewers load them automatically
python -m pipeline.iiif.export_entry_boxes \
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
MODEL  = "gemini-3.1-flash-lite"
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
pipeline run https://archive.org/details/ldpd_11290437_000/ --dry-run

# Disable Flex inference (on by default; ~50% cheaper but 1–15 min/request)
# when you need real-time throughput
pipeline run <URL> --no-flex

# Compare two Gemini models on OCR quality before committing to a full run
python main.py collections.txt --compare-ocr \
    --models gemini-3.1-flash-lite gemini-2.0-flash

# Force re-processing of already-completed OCR + alignment
pipeline ocr output/ldpd_11290437_000/ --force
```

---

## 9. Using pieces as a library

Everything in `pipeline/api.py` is supported for direct use from notebooks and
external scripts — no CLI required. (Anything *not* re-exported there is
internal and may change without notice; the API itself is provisional until a
1.0 release.) A runnable tour lives in `colab/library-cookbook.ipynb`.

```python
# Walk any public IIIF manifest (v2 or v3) — works with no pipeline run at all
import json, urllib.request
from pipeline.api import iter_canvases, image_url

manifest = json.load(urllib.request.urlopen("https://www.loc.gov/item/01015253/manifest.json"))
for canvas in iter_canvases(manifest):
    print(canvas["canvas_id"], image_url(canvas["service_id"], width=1024))
```

```python
# Find what a pipeline run produced, then load the entries
from pathlib import Path
import csv
from pipeline.api import get_ocr_model, discover_ocr_slug

vol = Path("output/my_volume")
model = get_ocr_model(vol) or discover_ocr_slug(vol)
with next(vol.rglob(f"entries_{model}*.csv")).open() as fh:
    entries = list(csv.DictReader(fh))
```

```python
# Parse Surya OCR output; clean and merge entries CSVs
from pipeline.api import parse_surya, process_csv, combine_volumes

page_bbox, lines, median_conf = parse_surya(vol / "item" / "0001_x_surya.json")
stats = process_csv(src_csv, dst_csv, infer_categories=True)
combine_volumes(collection_dir, collection_dir / "combined.csv")
```

```python
# Call Gemini yourself, with the pipeline's retry/backoff behavior
from pipeline.api import get_client, generate_with_retry

client = get_client()   # reads GEMINI_API_KEY from the environment / .env
```
