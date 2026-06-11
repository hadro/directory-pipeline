"""Curated public API for using pipeline pieces as a library.

Everything importable from this module is supported for use in notebooks and
external scripts; anything not re-exported here is internal and may change
without notice. (Signatures here are provisional until a 1.0 release, but
changes will be deliberate rather than incidental.)

Quick tour
----------
Walk any public IIIF manifest (v2 or v3) without touching the pipeline:

    import json, urllib.request
    from pipeline.api import iter_canvases, image_url

    manifest = json.load(urllib.request.urlopen("https://…/manifest.json"))
    for canvas in iter_canvases(manifest):
        print(canvas["canvas_id"], image_url(canvas["service_id"], width=1024))

Find what a pipeline run produced:

    from pipeline.api import get_ocr_model, discover_ocr_slug, read_state

    model = get_ocr_model(vol_dir) or discover_ocr_slug(vol_dir)
    entries_csv = next(vol_dir.rglob(f"entries_{model}*.csv"))

Parse Surya OCR output, clean an entries CSV, merge volumes:

    from pipeline.api import parse_surya, process_csv, combine_volumes

    page_bbox, lines, median_conf = parse_surya(surya_json_path)
    stats = process_csv(entries_csv, fixed_csv, infer_categories=True)
    combine_volumes(collection_dir, combined_csv)

Call Gemini with the pipeline's retry/backoff behavior:

    from pipeline.api import get_client, generate_with_retry

    client = get_client()           # reads GEMINI_API_KEY from env/.env
    response = generate_with_retry(client, model=..., contents=..., config=...)

See docs/usage-examples.md ("Using pieces as a library") and
colab/library-cookbook.ipynb for worked examples.
"""

# IIIF manifest walking (works standalone — no pipeline run required)
from utils.iiif_utils import (
    image_url,
    iter_canvases,
    manifest_item_id,
    manifest_version,
)

# Model names and output-file discovery
from utils.models import (
    DEFAULT_NER_MODEL,
    DEFAULT_OCR_MODEL,
    discover_ocr_slug,
    model_slug,
)

# Per-volume run state (pipeline_state.json)
from pipeline.state import (
    get_ner_model,
    get_ocr_model,
    read_state,
)

# Surya OCR output parsing
from pipeline.align_ocr import parse_surya

# Entries CSV post-processing
from analysis.fix_entries import process_csv
from analysis.combine_volumes import combine as combine_volumes

# Gemini client + transient-error retry
from utils.gemini import generate_with_retry, get_client

__all__ = [
    # IIIF
    "manifest_version",
    "iter_canvases",
    "image_url",
    "manifest_item_id",
    # models / discovery
    "DEFAULT_OCR_MODEL",
    "DEFAULT_NER_MODEL",
    "model_slug",
    "discover_ocr_slug",
    # run state
    "read_state",
    "get_ocr_model",
    "get_ner_model",
    # OCR parsing
    "parse_surya",
    # CSV post-processing
    "process_csv",
    "combine_volumes",
    # Gemini
    "get_client",
    "generate_with_retry",
]
