# Plan: make the generated data explorer publication-ready

**Status:** Proposed — no code written. Created 2026-08-23. Revised 2026-08-24
to ground every section in a survey of the ~60 extracted collections actually on
disk (see "Grounding" below); activation conditions, priorities, and effort
notes updated where the real corpus contradicted the original assumptions.

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

## Grounding: what's actually in `output/` (surveyed 2026-08-24)

This plan is not hypothetical — `output/` already holds **~60 collections with
extracted `entries_*.csv`**, spanning city directories, Green Book–family travel
guides, professional and trade directories, and one-off books. Every priority
below is calibrated against that corpus. A full per-feature example map is in the
[appendix](#appendix-example-collections--test-corpus); the load-bearing facts:

**Scale is already past every soft limit in this document.** The largest outputs
are combined multi-volume CSVs, and the pre-built explorers on disk are the exact
artifact this plan improves:

| On-disk output | Rows | Pre-built `_explorer.html` |
|---|---|---|
| `green_books_and_related/…/green_book_entries_all.csv` (28 editions, 22 years 1937–1966) | 63,590 | **48 MB** |
| `green_books_and_related/travel_guides_all.csv` (unions several *different* series) | 42,111 | — |
| `tulsa_1921` / `tulsa_1922` (single city-directory volumes) | 69,843 / 69,669 | **39 MB / 30 MB** |
| `green_books_and_related/travelguide_634f3af0/combined.csv` (11 vols) | 28,581 | **41 MB** |
| `lain_healy_s_brooklyn_directory_…1897` | 10,301 | 3.8 MB |
| `green_books_and_related/go_guide…/combined.csv` (7 vols) | 9,206 | 3.8 MB |
| `hodges_new_york_city_nurses_directory` | 5,297 | 2.5 MB |
| `the_negro_motorist_green_book_2016298176` (carries drift scores) | 3,477 | 1.5 MB |
| `woods_directory_73644404` (only geocoded collection on disk) | 3,324 | 2.6 MB (map) |

A 48 MB self-contained HTML that then truncates its own table to 2,000 rows is
the status quo, not an edge case. ~40 further Green Book sub-volumes and the
`tovey`/`brewers` brewer series each clear 2,000 rows on their own.

**The corpus is a three-level hierarchy, not a flat volume list.**
`green_books_and_related/` is a *collection of collections*: 7 series, ~50
volumes, organized `series → volume → page`, with cross-series unions already
materialized (`travel_guides_all.csv`). §3.1's "all volumes" model assumes one
flat series; the real shape is deeper (see §3.1).

**Schemas are genuinely heterogeneous** — which is the whole argument for
capability detection: `name`/`address`/`category` (Green Book),
`surname`/`given_name`/`occupation` (Lain Brooklyn), `region`/`hospital`/
`qualifications` (Hodges nurses), `alphabetical_range`/`race_designation`
(Tulsa), `section`/`subsection`/`page` (London booksellers). Boolean columns
differ per collection (`is_advertiser`, `is_business`, `is_recommended`,
`is_advertisement`, `is_rear`, `is_special_contract`) and **none of them is
`is_new_firm`** — the one value the template currently hardcodes (see the
`field_labels.json` fix below).

**QA/validation columns are already computed and sitting unused in the CSVs —
but they're back-office signals, not reader content.** `analysis/fix_entries.py`
(postprocess) writes largely generic `flag_*` columns (`flag_state_invalid`,
`flag_header_row`, `flag_duplicate`, `flag_unanchored`, `flag_hallucinated`, …);
the Green Book additionally carries a **collection-specific, gitignored** `drift_*`
layer (`collections/greenbook/detect_geographic_drift.py`). These are genuinely
useful for *reviewing* an extraction, but they are operator metadata — so they
belong in a review mode, not the published explorer (new §1.7, deliberately not a
publish-readiness feature).

**Provenance is multi-model, and rights on disk are messier than assumed** —
details in §1.2/§1.3, but in short: the pipeline uses **up to three distinct
models — OCR, NER, and prompt-generation** — and all three should be surfaced.
`pipeline_state.json` (`pipeline/state.py`) records `ocr_model` and `ner_model`
(older on-disk files predate even those, so the CSV *filename* is the fallback),
but the **prompt-generation model is not persisted anywhere today.** Separately,
structured IIIF `rights`/`requiredStatement` exist for essentially only the
LOC-hosted v3 manifests (Tulsa) — NYPL v3 and ContentDM v2 need entirely
different extraction paths.

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
| Transcription panel (§3.3) | `*_aligned.json` on disk |
| Page view with entry boxes (§3.3) | `*_entry_annotations.json` / `*_box_annotations.json` on disk |
| Region crops in detail panel | `canvas_fragment` has `#xywh=` |
| QA / validation facets (§1.7) | `flag_*`/`drift_*` present **and** `--qa` passed (off by default; never in published output) |
| Rights / attribution block | manifest carries `rights`/`requiredStatement` (v3) **or** `attribution`/`license` (v2) **or** a rights-like `metadata` entry |
| Provenance block | any model role derivable (state `ocr_model`/`ner_model`, else CSV filename) + manifest present |

Three of these were written too narrowly for the actual corpus and are corrected
in the sections they gate:

- **Page-context** was one row (`*_aligned.json` *or* `*_annotations.json`); on
  disk `*_aligned.json` is common but no `*_entry_annotations.json` exists yet,
  so the transcription half is buildable today and the box-overlay half needs
  `export_entry_boxes.py` run first — hence two separate rows (see §3.3).
- **Rights** as originally written (`rights` *or* `requiredStatement`) activates
  for only the LOC/v3 manifests. Most of the corpus is ContentDM v2 or bare NYPL
  v3 (see §1.2).
- **Provenance** cannot gate on `pipeline_state.json` — it is usually absent and
  never carries the model. Derive from the filename instead (see §1.3).

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
require new dependencies. (§1.7 is the odd one out — an authoring aid grouped
here only because it's the same cheap surfacing work; it stays out of the
published artifact by default.)

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

**But the naive "pull `rights` + `requiredStatement`" only works for a small
slice of the corpus.** Surveying the manifests behind the extracted collections:

- **LOC / IIIF v3 (Tulsa):** the clean case — `rights` is a real license URI
  (`…/publicdomain/zero/1.0/`) and `requiredStatement` is a label/value
  attribution. This is the *only* pattern the original bullet handles.
- **NYPL / IIIF v3 (afro guide, green books family):** `rights`,
  `requiredStatement`, and `provider` are all **absent**. Provider/collection and
  rights live inside the `metadata` array as HTML (`<span><a href=…>`).
- **ContentDM / IIIF v2 (greenbook:88, London booksellers):** no v3 fields at
  all — v2 uses top-level `attribution` and `license`, and here `attribution`
  arrives as a **list containing empty strings and an embedded license URL in
  prose**.

So `_extract_item_meta()` needs a four-step resolver, not a two-field pull:

- Extend `_extract_item_meta()` to resolve rights/attribution/provider in
  priority order: (1) v3 `rights` + `requiredStatement` + `provider`; (2) v2
  `license` + `attribution`; (3) fallback scan of the `metadata` array for
  rights-/provider-like labels; **sanitizing embedded HTML and dropping
  empty-string list members at every step.** Surface as a footer block on every
  page and a `rights` field in `datapackage.json`. When nothing resolves, emit an
  explicit "rights unknown — see source manifest" rather than silence.
- Generate `CITATION.cff` and a "Cite this dataset" panel: source institution,
  item title and date, item URL, extraction date, model, pipeline version.
- Per-entry citation: a "copy citation" button in the detail panel that emits a
  human-readable string plus the `canvas_fragment` IIIF URI as the locator.

### 1.3 An About page generated from provenance the pipeline already records

For machine-extracted data, disclosing how it was made is not optional — it's
the thing that makes it usable by anyone else. And "how it was made" is **not one
model but up to three distinct roles — OCR, NER, and prompt generation** — each of
which can use a different model and all of which should be surfaced and
documented. Where each is recoverable today:

- **OCR + NER models — the reliable pair.** `pipeline_state.json`
  (`pipeline/state.py`) records `ocr_model` and `ner_model` (read via
  `get_ocr_model`/`get_ner_model`), and both are also encoded in filenames
  (`*_{model}.txt`, `entries_{model}.csv`) as a fallback. `last_run_args`
  additionally captures the invocation — e.g. `--mode multimodal` vs. text-only —
  worth surfacing. (Older on-disk state files, like the afro guide's, predate the
  model fields, so keep the filename fallback.)
- **Prompt-generation model — not persisted today.** The generated
  `ocr_prompt.md`/`ner_prompt.md` carry no record of which model wrote them. To
  make all three surface-able (per review comment), add a small provenance header
  to the generated prompt files (or a `prompt_model` field in state) at
  generation time — a tiny change to `generate_prompts.py`.

So build provenance from state (models + run flags) with a filename fallback,
plus manifest (source/rights) and `ner_prompt.md` (field definitions); treat run
date/stages as enrichment.

Generate `about.html` (and an in-app "About this data" panel) containing:

- Source item, institution, manifest URL, rights statement.
- Pipeline provenance **by role**: OCR model, NER model, and prompt-generation
  model (each labeled); run mode (multimodal / text-only) from `last_run_args`;
  stages run and run date; page count. List all model variants when a volume was
  extracted more than once.
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

### 1.6 Make the summary vizzes individually toggleable (and fix the phantom density strip)

Two related issues. First, `explore_entries.py:9` and `docs/pipeline-stages.md:640`
advertise a "page density strip — entry count per canvas" that **does not exist**
in the template — only a fill-rate chart (`renderFillRate`) and up to four facet
bar charts (`renderCharts`) do, inside a `#charts-row` with a runtime "Hide
summary" collapse toggle. Fix the false doc claim regardless.

Second — the more useful ask (per review comment): some vizzes earn their space
and some don't, and that judgment is per-collection. Rather than hardcoding which
exist, make each **individually toggleable**, with a mechanism that also lets you
kill one in an *already-generated* file without a rebuild:

- Emit a single build-time config object near the top of the embedded JS, e.g.
  `const VIZ = { fillRate: true, facetCharts: true, densityStrip: false };`. Each
  render function early-returns and hides its container when its flag is false.
- Because that object is plain text at the top of the generated HTML, a user can
  **flip one boolean by hand** to permanently disable a viz that "takes up room
  but offers no value" — no regeneration needed (this is exactly the review ask).
- Back it with a build flag for defaults (`--viz fillRate,facetCharts` /
  `--no-viz`) so the choice can also be made at generation time.
- The density strip stays **off by default** (limited value in practice):
  implement it only behind `VIZ.densityStrip`, or drop it and just remove the doc
  claim. Recommend shipping the toggle mechanism and leaving the density strip
  unbuilt-but-reserved.

### 1.7 QA/validation facets — a review-mode aid, *not* a published feature

These columns are useful, but they are **back-office signals, not reader
content**, so scope them to an authoring/review mode and keep them out of the
published artifact by default. (This is a deliberate demotion from an earlier
draft that mis-framed them as a headline publication feature — surfacing raw
`flag_hallucinated`/`drift_score` columns to a researcher browsing hotels is
clutter at best and bad optics at worst.)

What's on disk, and where it comes from — the distinction matters:

- `flag_*` are generated by `analysis/fix_entries.py` (tracked, largely generic):
  `flag_state_invalid`, `flag_state_eq_city`, `flag_name_address`,
  `flag_header_row`, `flag_duplicate`, `flag_unanchored`, `flag_hallucinated` —
  real heuristics, legible to an *operator*. They exist **only after
  `fix_entries` runs** (postprocess), so the raw `pipeline run` output has none,
  and several no-op on collections lacking `state`/`city`.
- `drift_*` (`drift_geonames`, `drift_window`, `drift_cross_volume`,
  `drift_alignment`, `drift_score`) come from
  `collections/greenbook/detect_geographic_drift.py` — **gitignored and
  Green-Book-specific, not a generic pipeline output.** Detect and show it
  opportunistically at most; don't architect around it.

The design is opt-in via `--qa` (confirmed with reviewer), off by default:

- **`--qa` passed:** a collapsed **"Data quality" facet group** over whatever
  `flag_*`/`drift_*` columns are present, plus "show only flagged" — the review
  surface you want *between* `extract` and `postprocess`.
- **Default (no `--qa`) and published build:** raw `flag_*`/`drift_*` columns are
  excluded from facets, table, and search. What survives for the reader is
  condensed and de-jargoned — one caveat line ("312 of 63,590 entries carry a
  validation flag — see About") plus a single optional "hide flagged entries"
  toggle.
- Either way these columns never enter the default table view or the search
  haystack; they are QA metadata, not directory content.

Cost stays near-zero (the computation exists); the value is real but accrues to
the *operator*, which is why this is an authoring aid rather than a
publish-readiness item.

---

## Tier 2 — Scale and search quality

### 2.1 The 2,000-row ceiling is a correctness problem, not a perf nicety

`explore_entries.html:445` renders `sorted.slice(0, 2000)` with a footer reading
"Showing first 2,000 of N". For a combined multi-volume CSV, **entries past
2,000 in any filtered view are unreachable** — no pagination, no infinite
scroll. This is not a tail-risk: the combined Green Book already on disk is
**63,590 rows** (its pre-built explorer a **48 MB** file), so the default view
hides **97%** of the data; Tulsa is 69,843. And it bites on meaningful filtered
views too — e.g. filtering 1921 Tulsa to `race_designation = "(c)"` returns
**5,455** entries, of which the table shows 2,000. Replace with
windowed/virtualized rendering (render ~60 rows around the scroll position into a
spacer-padded tbody). ~80 lines of vanilla JS, no dependency, and it removes the
cap entirely. The published flagship sites already run well past this ceiling —
see the reference-architecture note in §2.3.

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

The pre-built explorers on disk are **48 MB, 41 MB, 39 MB, and 30 MB** single
self-contained HTML files. The payload ceiling is not a future risk to note in
passing — it is a present condition, and it puts these files within sight of
GitHub Pages' 100 MB/file limit while making them slow to open and impossible to
diff. Treat dictionary-encoding + `--data-mode external` as **core Phase 2**, not
optional polish, and use these exact files as the before/after benchmark.

**Reference architecture — the published flagship already does this (reviewer
asked us to check).** The green-books explorer (`github.com/hadro/green-books`)
is **vanilla JS, no framework**, and keeps its page shell small (~358 KB
`all-volumes.html`) while **loading entries from an external ~21.5 MB CSV**
(`green_book_entries_all.csv`) plus small sidecar JSON (`canvas_map.json`,
`image_to_volume.json`) fetched at runtime — handling **~113,800 listings across
50 volumes** with a "still loading" state and live client-side filtering. That is
exactly the `--data-mode external` pattern below, validated at ~114k rows in
plain vanilla JS; and the per-view hand-authored pages
(`index.html`/`explorer.html`/`all-volumes.html`) are precisely the
generic-ization target this whole plan is chasing. Two caveats for the backport:
(a) it appears to load-then-render rather than virtualize, so pair it with §2.1
windowing to stay smooth at 100k+; (b) it uses **no prebuilt search index** (no
lunr/flexsearch file in the repo) — search is client-side over the loaded rows,
so §2.4's index would be an improvement *beyond* the flagship, not a backport.
(The render loop lives in the site's `.js`, which I could not read this session —
confirm virtualize-vs-render-all during implementation.)

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

**The real corpus is deeper than "a collection is a flat list of volumes," and
the design should account for it now.** `green_books_and_related/` is a
*collection of collections*: 7 series (`the_green_book_9ea5d5b0` = 28 volumes,
`travelguide_634f3af0` = 11, `go_guide…` = 7, …), each with its own
`combined.csv`, and a top-level `travel_guides_all.csv` (42,111 rows) that
**unions across series with different schemas**. So:

- The volume selector needs a **grouping level** (series → volume), not a flat
  list, once a directory nests series.
- The schema-union in "all volumes" is the *easy* case (one series, near-identical
  columns); the *hard* case already exists on disk — unioning Green Book,
  Travelguide, and Go Guide, whose columns only partially overlap. Decide the
  policy explicitly: union-all-columns with per-series fill, and let the series
  facet stand in for schema provenance.
- `volume_year` is the natural sort/facet key and is populated in the combined
  CSVs (Green Book spans 22 distinct years, 1937–1966) — but note it is added by
  `combine_volumes.py`, so "all volumes" mode depends on that stage having run,
  not just on loose per-volume `entries_*.csv`.

### 3.2 Cross-volume entity history

Given a name-like column and a place-like column, group entries whose normalized
key matches across volumes and show, in the detail panel, "this establishment
appears in 1938, 1940, 1947, 1955" with a jump link per appearance. This is
exactly what the Green Book site hand-built; the generic version keys off
`--link-fields name,city` with sensible auto-detection. Also enables a "listings
that appear/disappear in year X" view — the highest-value analytical question
for any directory series.

The combined Green Book is the demonstration corpus and shows the payoff is real:
**4,800 establishments recur across ≥5 distinct years**, with anchors like the
YMCA/YWCA appearing in 21 of the 22 editions. Two corpus facts to design around:

- **Name normalization is the whole game, and schemas disagree on how names are
  stored.** Green Book has a single `name`; `lain_healy` Brooklyn splits
  `surname` + `given_name`; Tulsa has `name` plus `spouse_name`. `--link-fields`
  must accept a *composite* key and normalize aggressively (case, punctuation,
  `Mrs`/`Mrs.`, `&`/`and`) — the raw keys already collide as
  `mrslwilliams`/`mrsvwilliams`.
- Gate the feature on `volume_year` being present (combined CSVs), not merely on
  "collection mode," so the appearance timeline has an axis.

### 3.3 Page context: see the entry in the document

Currently the detail panel shows a padded crop of the entry region. Two
additions — but the survey shows they have **different readiness**, so sequence
them independently rather than as one feature:

- **Transcription panel (buildable today)**: the aligned OCR lines for that page
  from `*_aligned.json`, with the entry's lines highlighted — lets a reader
  verify the structured record against the transcription, and makes the OCR text
  itself searchable alongside the fields. `*_aligned.json` is present for Tulsa,
  the Green Book family, the afro guide, London booksellers, and more — this half
  can ship immediately.
- **Page view with entry boxes (needs a prior stage)**: click "see in context" →
  full page image with *all* entry boxes overlaid, each clickable, optionally
  OpenSeadragon for pan/zoom. This reads `*_entry_annotations.json` /
  `*_box_annotations.json` from `pipeline/iiif/export_entry_boxes.py` — and **no
  such file exists anywhere in `output/` yet**, so this half is gated on running
  that stage first. Detect the file and light the feature up only when present.

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

Reality check on how often this activates: in tracked `output/`, **only
`woods_directory` has populated `lat`/`lon`** (3,179 / 3,324 rows geocoded, mostly
address-level) — it's the single clean demo for this tab. The Green Book and
Tulsa geocoding lives in the gitignored `collections/` tree, so geocoding is an
opt-in, comparatively rare state, not something most published explorers will
have. Worth building, but sequence it accordingly (it stays in the later phase).

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
  address" — the `flag_*`/`drift_*` half moved to **§1.7 as a review-mode
  authoring aid** (not a published feature); the remaining Tier-4 part is only
  the derived "missing X" filters over ordinary columns, and those too are most
  useful in review mode.
- **Cross-tab view**: category x state heatmap, click a cell to filter.
- **Saved views**: a small bar of build-time-generated starter views (top facet
  values) plus a "copy link to this view" button.
- **`<noscript>` fallback** and a print stylesheet for the current result set.
- **Keyboard**: `/` focuses search, `j`/`k` move rows, `Esc` closes detail.

---

## Suggested sequencing

| Phase | Contents | Rough effort |
|---|---|---|
| 1 | §1.1 site emitter, §1.2 rights/citation, §1.3 about page (3 model roles), §1.4 metadata, §1.6 viz toggles, §1.7 QA facets (`--qa`) | 1–2 days |
| 2 | §2.1 virtualization, §2.2 render perf, §2.3 payload (**core, not optional** — see the on-disk file sizes) | 1–1.5 days |
| 3 | §2.4 search index, §2.5 range filters | 1–2 days |
| 4 | §3.1 all-volumes (incl. series grouping), §3.4 permalinks, §1.5 corrections | 1–1.5 days |
| 5 | §3.3 transcription panel (data ready), §3.2 entity history, §3.5 map tab | 2 days |
| 6 | §3.3 page-view boxes (after `export_entry_boxes.py`), Tier 4 | ongoing |

§1.7 lands in Phase 1 because it's the same cheap surfacing work as the rest of
the tier — though it's an authoring aid gated behind `--qa` and off by default,
not a reader-facing publish feature. §2.3 is marked core: the 48/41/39 MB
explorers already on disk make payload reduction a Phase-2 requirement, not a
nicety — and the published green-books explorer (§2.3 reference note) shows the
external-data target working at ~114k rows. Phases 1 and 2 are independent and
together deliver most of the "ready to publish" outcome.

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
- **Payload ceiling is nearer than it looks.** The first draft framed this as a
  "several-hundred-thousand-row" future problem; the corpus already produces a
  48 MB file at 63,590 rows. Even dictionary-encoded, the honest supported
  ceiling for a single inline page is closer to **~100k rows / ~25 MB**; beyond
  that the answer is `--data-mode external` with paged shards + a prebuilt search
  index, or handing off to Datasette. State the ceiling in the published README
  and degrade to external mode explicitly rather than silently.
- **Scope creep into a CMS.** The explorer should stay a read-only publication
  artifact. Corrections go out to GitHub issues (§1.5); they never come back
  into the HTML. Likewise, QA/validation flags stay behind `--qa` (§1.7) — the
  published artifact is reader-facing, not an extraction console.
- **GitHub Pages limits** (100 MB/file, 1 GB/site, 100 GB/month bandwidth) are
  worth documenting in the published README, since IIIF thumbnails are pulled
  from the source institution rather than the site itself — which is also worth
  saying out loud, because it means a published explorer depends on the source
  institution's image server staying up.

## File / function map

| Change | Where |
|---|---|
| Rights/attribution extraction (v3 `rights`/`requiredStatement` → v2 `attribution`/`license` → `metadata` fallback, HTML-sanitized) | `explore_entries.py:_extract_item_meta` |
| Provenance extraction (OCR + NER models from state, filename fallback; run mode from `last_run_args`) | new `explore_entries.py:_extract_provenance` |
| Persist the prompt-generation model (so all 3 roles are surface-able) | `pipeline/generate_prompts.py` (provenance header in `*_prompt.md`, or `prompt_model` in `state.py`) |
| Per-viz toggles (`const VIZ = {…}` + `--viz`/`--no-viz` build flag; density strip off by default) | `explore_entries.html` `renderFillRate`/`renderCharts` + `explore_entries.py:build_html` |
| QA / validation facets (detect `flag_*`/`drift_*`, "Data quality" group; **`--qa` only**, condensed caveat otherwise) | `explore_entries.py:_classify_fields` + `explore_entries.html` facet sidebar |
| Field classification (`range` type, labels sidecar) | `explore_entries.py:_classify_fields`, `:255 _ID_FIELDS`, `:258 _FACET_MAX_CARDINALITY` |
| Search index construction | new `explore_entries.py:_build_search_index` |
| Site emitter | new `pipeline/publish_site.py` + `cli/main.py` subcommand + `pipeline/stages.py` StageDef |
| Virtualized table, perf fixes | `explore_entries.html:436 renderTableBody`, `:516 renderTableHead`, `:378 getFiltered` |
| Payload duplication | `explore_entries.py:build_html` |
| Labels/value maps | `explore_entries.html:254-267` → data-driven |
| Map tab | port `pipeline/geo/map_entries.py` rendering into the template, gated on `lat`/`lon` |
| Page context / transcription | `explore_entries.html:showDetail`, reading `*_aligned.json` + `*_entry_annotations.json` |

## Appendix: example collections / test corpus

Concrete inputs (all under `output/`) for validating each feature as it is
built. "Positive" is the collection that should light the feature up;
"Negative / edge" should either *not* activate it or stress the detection logic.

| Feature | Positive example (rows) | Negative / edge case |
|---|---|---|
| §1.2 rights — v3 structured | `tulsa_1921` — CC0 `rights` + `requiredStatement` | `woods_directory_73644404` — LOC v3 but no rights → "rights unknown" path |
| §1.2 rights — v2 | `greenbook:88`, `london-1841-booksellers` — ContentDM `attribution`/`license` (list w/ empty strings) | — |
| §1.2 rights — metadata fallback | `afro-american-travel-guide-1954` — NYPL v3, provider only as `metadata` HTML | — |
| §1.3 provenance / multi-model | `green_books_and_related/the_green_book_9ea5d5b0/*` — same vols under `gemini-3-flash`, `3.1-flash-lite`, `2.5-flash` + `_text_only` | `woods_directory` — no `ner_prompt.md`, no state file |
| §1.7 QA facets (`--qa` only) | combined Green Book (`flag_*`); `the_negro_motorist_green_book_2016298176` (`flag_*` + `drift_*`) | default/published build — group excluded; `london-1841-booksellers` — no flag cols at all |
| §2.1 virtualization / ceiling | `…/green_book_entries_all.csv` (63,590); `tulsa_1921` (69,843) | `goldsborough_papers_page226` (18) — never truncates |
| §2.3 payload benchmark | pre-built explorers: Green Book **48 MB**, travelguide **41 MB**, `tulsa_1921` **39 MB** | `afro-american-travel-guide-1956` (864 KB) |
| §2.5 range filter | `woods_directory` `volume_year=1911`; combined Green Book years 1937–1966 | `london-1841-booksellers` `page` = "p. 646 (scan 674)" — must *not* parse as range |
| §3.1 all-volumes / series grouping | `green_books_and_related/` (7 series, ~50 vols); `travel_guides_all.csv` (cross-series union, 42,111) | single-volume dirs — selector stays hidden |
| §3.2 entity history | combined Green Book — 4,800 names recurring across ≥5 years | `lain_healy_…1897` (`surname`/`given_name` split) — composite-key normalization stress |
| §3.3 transcription panel | `tulsa_1921`, `greenbook:88`, afro guide — `*_aligned.json` present | any dir lacking `*_aligned.json` |
| §3.3 page-view boxes | *(none on disk — run `export_entry_boxes.py` first)* | every collection today → feature stays dark |
| §3.5 map tab | `woods_directory` — 3,179 / 3,324 geocoded | everything else in tracked `output/` |
| labels sidecar | `is_advertiser` (greenbook), `is_business`… (tulsa), `is_recommended` (Green Book) | none carry `is_new_firm` (the current hardcode) |

Diverse-schema regression inputs for the generic-ness tests (§Tests):
`hodges_new_york_city_nurses_directory` (professional directory,
`region`/`hospital`/`qualifications`), `lain_healy_s_brooklyn_directory`
(`surname`/`given_name` split), `broadway_in_1851` (guidebook prose),
`the_new_york_directory_for_1786` (earliest, sparse columns).
