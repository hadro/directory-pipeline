# Key design decisions

Technical architecture notes for directory-pipeline. See the [main README](../README.md) for the overview.

---

**Gemini for accuracy, Surya for coordinates.** Gemini produces far more
accurate transcriptions of historical print than conventional OCR engines, especially
for proper nouns, abbreviations, and damaged text. Surya provides line-level bounding
boxes that align more cleanly with Gemini's line-oriented output than Tesseract's
word-level hOCR. Tesseract remains supported as a legacy fallback for collections
where Surya has already been run or word-level granularity is needed.

**Anchored Needleman-Wunsch alignment.** The aligner runs a single global
word-level NW pass (gap penalty −40, similarity 0–100 based on character edit
distance). Before the NW pass, city/state headings and category lines that appear
verbatim in both Gemini and Surya are committed as fixed *anchors*. The
sequence is then split into independent segments at each anchor and each segment
is aligned separately. This prevents misalignment drift on long pages where a
mismatched heading would otherwise pull all subsequent entries to wrong coordinates.

**Tesseract dictionary correction disabled.** By default Tesseract silently
"corrects" street names, proper nouns, and abbreviations toward dictionary words
(e.g. `Mound` → `Wound`, `Innesfallen` → `Innisfallen`). This is disabled via
`load_system_dawg=0 load_freq_dawg=0` to preserve names as-written for alignment.

**Column reading-order correction.** OCR engines on multi-column pages may read
across columns (left-to-right by y position) while Gemini reads column-by-column.
Lines are re-sorted before alignment using two strategies:
- *Row-major:* the default; lines are grouped into 50 px y-bands and sorted
  left-column-first within each band. Correct for pages where a centered heading
  precedes two-column body text.
- *Column-major:* used when lines cluster cleanly into two columns each with
  ≥ 20% of all lines. Emits the entire left column top-to-bottom, then the entire
  right column, matching Gemini's reading order for pages with independent
  side-by-side city sections.

Column breaks are detected in two stages: an x1-gap threshold of 8% of page width
(stage 1), with a bimodal histogram fallback (stage 2) for pages where a
page-number outlier in the margin creates a degenerate one-line pseudo-column.

**Schema-agnostic entry extraction.** `extract_entries.py` does not hard-code
any field names. The NER prompt (volume-specific or global fallback) defines both
the `page_context` fields (heading values carried between pages) and the per-entry
schema. `extract_entries.py` forwards whatever context keys the model returns to
the next page's prompt, and infers CSV column names from the union of all keys
returned across all pages. Adding support for a new collection type requires only
a new `ner_prompt.md` in the output slug directory — generated automatically by
`--generate-prompts` or written by hand. Geographic downstream stages
(`--geocode`, `--map`, `build_ranges.py`, `export_entry_boxes.py`) expect Green
Book-style fields (`city`, `state`, `category`, `canvas_fragment`) and degrade
gracefully when those fields are absent.

**Fallback model for dense pages.** `extract_entries.py` defaults to
`gemini-2.5-flash-lite` (fast and cheap) but escalates to `gemini-3.1-flash-lite` for pages
that hit the output token limit, then falls back to partial JSON recovery (salvaging
complete entries before the truncation point) before writing an error sidecar.

**Two-tier geocoding.** `geocode_entries.py` uses Google Maps (building-level
accuracy) for entries with a street address and Nominatim city/state centroids as
a fallback. Both are cached in `geocache.json` so repeated runs over the same data
are instant. City-level points are jittered on the map to prevent stacking.

**IIIF v2/v3 support.** `utils/iiif_utils.py` provides a single version-agnostic
interface for walking IIIF Presentation manifests. IIIF Image API URLs work
identically for both API versions. The downloader caps requested width at the
service's advertised `maxWidth` to prevent upscaling artifacts on tile-pyramid
servers.

**IIIF canvas coordinate space.** IIIF manifest canvas dimensions do not
necessarily match the actual image pixel dimensions. NYPL manifests, for example,
declare all canvases as 2560×2560 (square) even for portrait images that are
3316×4513. Mirador 3 maps annotation `xywh` coordinates directly to image pixel
space regardless of the canvas dimensions in the manifest. As a result, all
bounding boxes and `canvas_fragment` values in this pipeline are stored in
**natural image pixel coordinates**, fetched at align time from the IIIF Image API
`info.json`. The `canvas_width`/`canvas_height` fields in `*_aligned.json` reflect
these natural dimensions and serve as the authoritative coordinate space for all
downstream scripts. `export_entry_boxes.py` converts coordinates to natural pixel
space during export, and `map_entries.py` reads canvas dimensions from
`*_aligned.json` (not the manifest) so that IIIF `pct:` thumbnail calculations
remain correct even after a manifest has been updated with natural canvas dims.

**IIIF-native output.** The aligned JSON includes `canvas_uri` and
`canvas_fragment` (IIIF `#xywh=` fragment) for every word and line, making the
output directly consumable by IIIF annotation tools and viewers.
