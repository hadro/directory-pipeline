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

**Fallback model for dense pages, with partial JSON recovery.** `extract_entries.py`
defaults to `gemini-3.1-flash-lite-preview` (fast and cheap) but escalates to
`gemini-2.5-flash` for pages that hit the output token limit. If the fallback also
produces malformed JSON, `_recover_partial_json()` salvages every complete entry that
appears before the truncation point rather than discarding the page entirely. An
`*_entries_error.txt` sidecar is written only when partial recovery also fails.
Previously-failed pages (where the error sidecar exists) are auto-retried without
`--force`, so a re-run after a network interruption picks up where it left off.

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

**Alignment confidence tiers.** Every line and word in `*_aligned.json` carries a
`"confidence"` field: `"line"` for Surya-aligned output (most reliable), `"word"`
for legacy Tesseract hOCR-aligned output (less precise), and `"manual"` for matches
accepted through the `--review-alignment` UI. Downstream scripts can filter or weight
by this tier — e.g. `export_entry_boxes.py` inherits whatever confidence level was
set during alignment, and `visualize_alignment.py` color-codes the three tiers
differently.

**Cross-page NER context persistence.** After each page is processed by
`extract_entries.py`, the current heading context (e.g. city, state, category) is
written to `extraction_context_{slug}.json`. When the pipeline is resumed or
restarted, this context is loaded first so subsequent pages inherit the correct
heading state without reprocessing earlier pages. This also means a single failed
page can be retried in isolation without losing context.

**OCR quality-failure retry with temperature stepping.** When Gemini OCR output
fails a quality check (dot-leader runaway detected, or repetitive output loop),
`run_gemini_ocr.py` retries up to three times with escalating temperatures
`[0.1, 0.3, 0.7]`. If the failure is dot-leader-specific, the
`_NO_LEADER_INSTRUCTION` clause is also appended to the system prompt on retry.
This separates the two failure modes: temperature diversity breaks repetition
loops, while the explicit instruction targets the dot-leader pathology specifically.
As a last resort, any remaining dot-leader run of 10+ characters is collapsed to a
canonical five-dot sequence rather than discarding the page.

**Split-spread sidecar coordinate mapping.** When `split_spreads.py` crops a
two-page spread, it writes a `{stem}_split.json` sidecar recording the pixel
offsets for each half. During alignment, `align_ocr.py` reads these offsets and
translates all bounding boxes back to the full-spread coordinate space before
generating `canvas_fragment` values. The IIIF manifest references the original
full-spread canvas, so IIIF viewers receive coordinates relative to that canvas —
not relative to the cropped half image.

**Heading-field exclusion in canvas fragment linking.** When `extract_entries.py`
searches aligned lines for a matching bounding box for a given entry, heading/context
fields (`state`, `city`, `section`) are explicitly excluded from the text-match
candidates. Without this, the first entry after a state heading could have its
`canvas_fragment` set to the heading's bounding box (e.g. the substring "ALABAMA"
matches both the heading line and the `state` field of the following entry).

**Three-strategy canvas fragment matching.** `_find_fragment()` tries strategies
in order: (1) exact match on `gemini_text`, (2) substring match — the entry text
is a substring of the aligned line, handling Green Book format where a city name
and several entries share one OCR line, (3) fuzzy match via
`difflib.get_close_matches()` at cutoff 0.6. Using a cutoff below ~0.5 produces
false matches on short strings; above 0.7 misses legitimate matches on pages with
OCR noise. The same three-strategy chain is used in `patch_canvas_fragments.py` for
retroactively adding bounding boxes to existing extraction runs.

**Edge-bleed rejection.** Two-page spread scans include a thin sliver of the facing
page at both edges. Surya detects these slivers as text lines and the NW aligner can
assign entries to them. Any bounding box narrower than 5% of canvas width whose left
or right edge falls within 5% of the canvas boundary is silently rejected. Both
thresholds are expressed as fractions of canvas width, so they scale correctly
regardless of scan resolution.

**Auto-resolution for handwritten and manuscript content.** If the OCR system prompt
contains any of the keywords `handwrit`, `manuscript`, or `cursive`,
`run_gemini_ocr.py` automatically switches to high-resolution image mode before
sending the request to Gemini. This behavior can also be triggered explicitly with
`--high-res`. The auto-detection means calibrated prompts generated by
`--generate-prompts` for manuscript collections carry the resolution requirement
with them without requiring a separate flag on every run.
