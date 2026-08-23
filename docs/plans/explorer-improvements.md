# Plan: make the generated data explorer publication-ready

**Status:** Proposed — no code written. Created 2026-08-23.

> **What this is:** a prioritized set of functional improvements to
> `pipeline/explore_entries.py` + `pipeline/explore_entries.html`, aimed at one
> goal: the artifact that falls out of `pipeline run` should be something a
> librarian, archivist, or researcher can push to GitHub Pages *as is* and have
> it stand up as a real data publication — not a preview they then have to
> rebuild by hand.
>
> The two flagship sites (`hadro.github.io/tulsa-city-directories`,
> `hadro.github.io/green-books`) both required "additional front-end design work
> beyond what the pipeline generates" (README, line 79). This plan is about
> closing that gap generically, so the next collection needs none.

---

## The design constraint: capability detection, never configuration

Everything below must stay source-agnostic. The rule the current code already
follows — introspect the CSV header, don't hardcode a schema — extends to every
new feature:

> **A feature activates when the data or the output directory can support it,
> and is silently absent otherwise. No collection-specific config file, no
> per-source branches in the template.**

| Feature | Activation condition |
|---|---|
| Map tab | `lat`/`lon` columns present and non-empty |
| Timeline / range filter | a column parses as numeric or as a year across ≥80% of rows |
| Cross-volume entity history | collection mode + a name-like and a place-like column |
| Page-context view | `*_aligned.json` or `*_annotations.json` on disk |
| Region crops in detail panel | `canvas_fragment` has `#xywh=` |
| Rights / attribution block | manifest carries `rights` or `requiredStatement` |
| Provenance block | `pipeline_state.json` present |

Two things in the current template violate this and should be fixed early:

- `explore_entries.html:254-267` hardcodes Green Book–specific display names
  (`FIELD_LABELS["is_new_firm"] = "New listing"`) and boolean value labels into
  the *generic* template. These belong in an optional, auto-discovered
  `field_labels.json` sidecar (or a `--labels` flag), with the built-in map
  reduced to type-level defaults (`True`→`Yes`).
- `explore_entries.py:255` hardcodes `_ID_FIELDS = {"canvas_fragment", "image",
  "line_text"}`. Should be a flag (`--hide-fields`) layered over that default.

---

## Tier 1 — Publish-readiness (highest value, lowest effort)

These are what separate "an HTML file" from "a citable data publication." None
require new dependencies.

### 1.1 Emit a site, not a file: `pipeline publish <DIR>`

Add a `--publish-dir` mode (and a `pipeline publish` subcommand) that writes a
complete, deployable directory instead of one loose HTML file:

```
site/
  index.html                  # the explorer
  404.html                    # copy of index.html — makes deep links survive
  .nojekyll                   # required: Pages/Jekyll otherwise mangles _-prefixed paths
  data/
    entries.csv               # the real data, downloadable and linkable
    entries.json              # same, machine-readable
    schema.json               # column names, types, fill rates, facet values
    datapackage.json          # Frictionless Data descriptor
  about.html                  # provenance + methodology (§1.3)
  CITATION.cff                # GitHub renders a "Cite this repository" button
  README.md                   # what this is, how it was made, how to reuse
  robots.txt
  sitemap.xml
  .github/workflows/pages.yml # ready-to-commit deploy workflow
```

Everything in `data/` is generated from what's already in memory at build time.
`.nojekyll` and `404.html` are one-liners that prevent the two most common
GitHub Pages surprises.

### 1.2 Rights, attribution, and citation — currently dropped entirely

`grep -i rights pipeline/explore_entries.py utils/iiif_utils.py` returns
nothing. IIIF v3 manifests carry `rights` (a license URI) and
`requiredStatement` (an attribution label/value pair the spec says viewers
**must** display). The pipeline reads these manifests already and throws both
away. For anything published, this is the gap that matters most.

- Extend `_extract_item_meta()` to pull `rights`, `requiredStatement`, and
  `provider` → surface as a footer block on every page and a `rights` field in
  `datapackage.json`.
- Generate `CITATION.cff` and a "Cite this dataset" panel: source institution,
  item title and date, item URL, extraction date, model, pipeline version.
- Per-entry citation: a "copy citation" button in the detail panel that emits a
  human-readable string plus the `canvas_fragment` IIIF URI as the locator.

### 1.3 An About page generated from provenance the pipeline already records

`pipeline_state.json` (`pipeline/state.py`) records `source_url`, `ocr_model`,
`ner_model`, `stages_completed`, `last_run`. The explorer ignores it. For
machine-extracted data, disclosing how it was made is not optional — it's the
thing that makes it usable by anyone else.

Generate `about.html` (and an in-app "About this data" panel) containing:

- Source item, institution, manifest URL, rights statement.
- Pipeline provenance: OCR model, NER model, stages run, run date, page count.
- **The NER prompt itself**, verbatim, from `ner_prompt.md` — this is the field
  definition document. Nothing else explains what `category` actually means.
- Field dictionary: every column, its fill rate, cardinality, and top values
  (all already computed in `_classify_fields()`).
- A plain-language accuracy caveat plus known-limitations list.

### 1.4 Discoverability metadata

The generated page has no `<meta name="description">`, no Open Graph tags, no
canonical URL, no structured data. Adding `schema.org/Dataset` JSON-LD gets the
published site into Google Dataset Search — a large, free reach win for
cultural-heritage data, and entirely derivable from the manifest metadata
already extracted.

- `<meta name="description">`, OG/Twitter card (thumbnail: the first canvas).
- `<link rel="canonical">` from a `--base-url` flag.
- JSON-LD `Dataset` with `creator`, `temporalCoverage`, `spatialCoverage` (from
  state/city columns when present), `distribution` → `data/entries.csv`,
  `license` → the IIIF `rights` URI.
- `sitemap.xml` listing `index.html` and `about.html`.

### 1.5 "Report an error" → prefilled GitHub issue

Machine-extracted historical data *will* be wrong, and readers of these
collections are often the people who can spot it. A one-click correction path
turns that into a contribution:

- Detail panel button → opens `https://github.com/{repo}/issues/new` with title,
  entry values, canvas URI, and a permalink to the entry prefilled via query
  params (`--issues-repo` flag; hidden when unset).
- Ship `.github/ISSUE_TEMPLATE/data-correction.yml` in the published site.

### 1.6 Fix the documented-but-missing page density strip

`explore_entries.py:9` and `docs/pipeline-stages.md:640` both advertise a "page
density strip — entry count per canvas, showing document structure at a
glance." It does not exist in the template. Either build it (it's a genuinely
useful structural view — spikes and gaps reveal section boundaries and pages the
NER stage silently dropped, and clicking a bar should filter to that page) or
remove the claim. Recommend building it: it doubles as an extraction-QA tool.

---

## Tier 2 — Scale and search quality

### 2.1 The 2,000-row ceiling is a correctness problem, not a perf nicety

`explore_entries.html:445` renders `sorted.slice(0, 2000)` with a footer reading
"Showing first 2,000 of N". For a combined multi-volume CSV, **entries past
2,000 in any filtered view are unreachable** — no pagination, no infinite
scroll. Replace with windowed/virtualized rendering (render ~60 rows around the
scroll position into a spacer-padded tbody). ~80 lines of vanilla JS, no
dependency, and it removes the cap entirely.

### 2.2 Rendering is O(fields x rows) per keystroke

Three compounding issues, all fixable without changing behavior:

- `getFiltered()` is called once by `renderTableBody`, once by
  `renderFacetSidebar`, once by `renderCharts`, and **once per column** inside
  `renderTableHead` (`explore_entries.html:576`). On a 40k-row, 12-column CSV
  that's ~15 full passes per render. Compute once per render cycle and pass it
  down.
- `ALL_ENTRIES.indexOf(row)` runs **three times per rendered row**
  (`:447`, `:451`, `:475`) — a linear scan of the full array each time. Assign a
  stable `_i` index at build time and use it everywhere (it's also the basis for
  entry permalinks, §3.4).
- The search haystack is rebuilt per row per keystroke
  (`:383`: `displayFields.map(...).join(" ").toLowerCase()`). Precompute one
  lowercase haystack per row at load.

### 2.3 Payload size: dictionary-encode and externalize

- **Collection mode currently ships volume 1 twice.** `build_html()` passes both
  `entries_json` (= first volume's rows) and `volumes_json` (= *all* volumes,
  including the first). Drop the duplicate.
- Facet columns (`state`, `category`, `volume_title`) repeat the same handful of
  strings across every row. Dictionary-encode them in the payload and rehydrate
  on load — typically a 2–4x size reduction on directory data.
- Add `--data-mode inline|external`. `external` writes `data/entries.json` and
  fetches it, keeping `index.html` small, cacheable, and diffable, and making
  the data directly reusable by third parties. Keep `inline` the default so the
  file:// single-file property survives, and note the CORS caveat in `--help`.

### 2.4 Search that survives OCR

Current search is `haystack.includes(query)` — a raw substring match, no
tokenization, no ranking, no error tolerance. On OCR'd historical text this is
the single biggest functional limitation: "Johnson" misses `Johnsou`,
"restaurant" ranks a `line_text` incidental mention identically to a name-field
hit.

Build a small inverted index at generation time (Python side, embedded as JSON):

- **Tokenized + prefix matching**, with normalization (case, diacritics,
  punctuation, `&`/`and`, common abbreviations `st`/`street`).
- **Field-weighted ranking** — name-like fields outrank address, address
  outranks `line_text`. Results ordered by score rather than table order.
- **Query syntax**: quoted phrases, `field:value`, `-negation`, `OR`.
- **Fuzzy fallback**: when a query returns zero results, retry against a
  character-trigram index and offer "Did you mean *Johnson*? (14 entries)".
  Only on zero results, so the fast path stays fast.

### 2.5 Range filters for numeric and date fields

`_classify_fields()` (`explore_entries.py:264`) has exactly two outcomes: ≤50
distinct values → facet checkboxes; more → free-text search only. So year, page
number, and any count column fall off a cliff. Add a third type, `range`,
detected by parseability, rendered as a brushable histogram (Observable Plot is
already loaded) that filters the table on drag.

---

## Tier 3 — Depth: multi-volume, source document, geography

### 3.1 "All volumes" mode

In collection mode the volume `<select>` is exclusive: you can only ever search
*one* volume, and switching resets all filters (`loadVolume()`,
`explore_entries.html:308`). The obvious missing option is **All volumes** as
the default: union the field schemas, add `volume_title`/`volume_year` as
facets, and let the volume selector become just another facet. Cross-volume
search is the main reason someone puts a whole series through the pipeline.

### 3.2 Cross-volume entity history

Given a name-like column and a place-like column, group entries whose normalized
key matches across volumes and show, in the detail panel, "this establishment
appears in 1938, 1940, 1947, 1955" with a jump link per appearance. This is
exactly what the Green Book site hand-built; the generic version keys off
`--link-fields name,city` with sensible auto-detection. Also enables a "listings
that appear/disappear in year X" view — the highest-value analytical question
for any directory series.

### 3.3 Page context: see the entry in the document

Currently the detail panel shows a padded crop of the entry region. Two
additions, both from artifacts already on disk:

- **Page view**: click "see in context" → full page image with *all* entry boxes
  overlaid (from `*_entry_annotations.json` / `*_box_annotations.json`, produced
  by `pipeline/iiif/export_entry_boxes.py`), each clickable. Optionally
  OpenSeadragon for pan/zoom, degrading to a static image.
- **Transcription panel**: the aligned OCR lines for that page from
  `*_aligned.json`, with the entry's lines highlighted — lets a reader verify the
  structured record against the transcription, and makes the OCR text itself
  searchable alongside the fields.

### 3.4 Stable entry permalinks

`?cf=<canvas_fragment>` is the only deep link, and it breaks when a row has no
bbox or when two rows share a canvas. Mint a stable `entry_id` at build time
(short hash of canvas_fragment + source row index), expose it as a column, use
it for permalinks (`#e=<id>`), for citation, and for "report an error" links.

### 3.5 Map as a tab, not a separate file

`pipeline/geo/map_entries.py` builds a *separate* Leaflet HTML. When `lat`/`lon`
columns exist, the explorer should offer a **Map tab** sharing the same filter
state: filter the table, the map follows; click a pin, the detail panel opens.
Add GeoJSON to the export menu. When the columns are absent the tab never
appears — the generic contract holds.

---

## Tier 4 — Worth doing, lower urgency

- **Vendored assets option.** `explore_entries.html:7-8` loads d3 and Observable
  Plot from jsdelivr. An offline or CSP-restricted copy silently loses every
  chart. Add `--vendor-assets` to inline them (and SRI hashes when staying on
  CDN).
- **Export menu**: CSV (current) + JSON + GeoJSON + IIIF Content State +
  clipboard TSV. Export currently reuses `FIELD_META` column order from the
  *initial* volume, which is wrong after a volume switch — fix alongside.
- **Facet ergonomics**: type-to-filter within a long facet list, sort by
  count/alpha, exclude (NOT) mode, AND/OR toggle for multi-select.
- **Missing-data filters**: "entries with no bounding box", "entries missing
  address" — turns the explorer into an extraction-QA tool, which is what you
  actually want between `extract` and `postprocess`.
- **Cross-tab view**: category x state heatmap, click a cell to filter.
- **Saved views**: a small bar of build-time-generated starter views (top facet
  values) plus a "copy link to this view" button.
- **`<noscript>` fallback** and a print stylesheet for the current result set.
- **Keyboard**: `/` focuses search, `j`/`k` move rows, `Esc` closes detail.

---

## Suggested sequencing

| Phase | Contents | Rough effort |
|---|---|---|
| 1 | §1.1 site emitter, §1.2 rights/citation, §1.3 about page, §1.4 metadata, §1.6 density strip | 1–2 days |
| 2 | §2.1 virtualization, §2.2 render perf, §2.3 payload | 1 day |
| 3 | §2.4 search index, §2.5 range filters | 1–2 days |
| 4 | §3.1 all-volumes, §3.4 permalinks, §1.5 corrections | 1 day |
| 5 | §3.3 page context, §3.5 map tab, §3.2 entity history | 2–3 days |
| 6 | Tier 4 | ongoing |

Phases 1 and 2 are independent and together deliver most of the "ready to
publish" outcome.

---

## Tests to add

The explorer has no test coverage today. Minimum viable:

- `tests/test_explore_entries.py` — build from a fixture CSV; assert the output
  contains no unsubstituted `{...}` placeholders, that embedded JSON parses,
  that row counts match, and that a CSV with only unknown columns still
  produces a working page (the generic-ness guarantee).
- Field classification: facet vs. range vs. search boundaries, empty CSV, single
  row, a column that is 100% empty, unicode and embedded `</script>` in values.
- Site emitter: asserts `.nojekyll`, `404.html`, `data/entries.csv`, and
  `datapackage.json` all exist and that `index.html` references only relative
  paths.
- Optional headless smoke test (Playwright): load the built page, type in the
  search box, assert the row count label changes.

## Risks and caveats

- **Don't lose the single-file property.** It is the reason the explorer is easy
  to hand to a colleague. Externalized data and the site emitter must stay
  opt-in; the default output of `pipeline run` stays one self-contained file.
- **Payload ceiling.** Even dictionary-encoded, a several-hundred-thousand-row
  collection will not fit comfortably in a browser tab. At that scale the honest
  answer is a prebuilt search index plus paged data shards, or handing off to
  Datasette — worth stating a documented supported ceiling (~100k entries)
  rather than degrading silently.
- **Scope creep into a CMS.** The explorer should stay a read-only publication
  artifact. Corrections go out to GitHub issues (§1.5); they never come back
  into the HTML.
- **GitHub Pages limits** (100 MB/file, 1 GB/site, 100 GB/month bandwidth) are
  worth documenting in the published README, since IIIF thumbnails are pulled
  from the source institution rather than the site itself — which is also worth
  saying out loud, because it means a published explorer depends on the source
  institution's image server staying up.

## File / function map

| Change | Where |
|---|---|
| Rights/attribution/provenance extraction | `explore_entries.py:_extract_item_meta`, new `_extract_provenance` reading `pipeline/state.py` |
| Field classification (`range` type, labels sidecar) | `explore_entries.py:_classify_fields`, `:255 _ID_FIELDS`, `:258 _FACET_MAX_CARDINALITY` |
| Search index construction | new `explore_entries.py:_build_search_index` |
| Site emitter | new `pipeline/publish_site.py` + `cli/main.py` subcommand + `pipeline/stages.py` StageDef |
| Virtualized table, perf fixes | `explore_entries.html:436 renderTableBody`, `:516 renderTableHead`, `:378 getFiltered` |
| Payload duplication | `explore_entries.py:build_html` |
| Labels/value maps | `explore_entries.html:254-267` → data-driven |
| Map tab | port `pipeline/geo/map_entries.py` rendering into the template, gated on `lat`/`lon` |
| Page context / transcription | `explore_entries.html:showDetail`, reading `*_aligned.json` + `*_entry_annotations.json` |
