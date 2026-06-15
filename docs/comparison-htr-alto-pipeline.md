# Comparison: UVA Law HTR-ALTO-Pipeline vs. directory-pipeline

> Report created 2026-06-12. Source project:
> [uvalawlibrary/HTR-ALTO-Pipeline](https://github.com/uvalawlibrary/HTR-ALTO-Pipeline)
> (MIT). Reviewed: README, `htr_alto` package (`cli.py`, `fetch.py`,
> `segment.py`, `transcribe.py`, `wordboxes.py`, `alto.py`), the word-box
> algorithm, and `docs/iiif-content-search.md`.

## What the other project is

HTR-ALTO-Pipeline solves a narrower problem than ours: turning **handwritten**
archival pages into **searchable full-text manuscripts** with word-level
coordinates. It does no structured entity extraction — no NER, no CSV schema, no
geocoding. Its reason for existing is to get VLM-quality transcription *and*
reliable bounding boxes into a standard search stack
(ALTO → Solr → IIIF Content Search → Mirador/Universal Viewer).

Much of it is therefore apples-to-oranges with our directory work. But the
**OCR-text-to-coordinate mechanics overlap exactly** with our Surya + Gemini +
Needleman-Wunsch alignment stage, and that is where they made a smarter
architectural choice. That is the headline finding.

Their five stages:

1. **Segmentation** — Kraken BLLA identifies line polygons.
2. **Transcription** — VLM reads individual line crops (one crop per request).
3. **Word estimation** — proportional character-based splitting yields word boxes.
4. **Serialization** — ALTO v3 XML.
5. **Integration** — Solr + IIIF Content Search highlighting in viewers.

Backends: Gemini (default), Anthropic Claude, OpenAI, plus local models via
LM Studio / Ollama / vLLM. Production-validated on ~2,400 pages of mid-20th-c.
handwritten correspondence.

---

## The one thing worth seriously considering adopting

### Line-crop transcription instead of full-page OCR + alignment

Our pipeline does:

> `--surya-ocr` (line bboxes) → `--gemini-ocr` (full-page text) →
> `--align-ocr` (Needleman-Wunsch to map text back onto boxes, second pass for
> missed columns).

Theirs does:

> Kraken BLLA segments lines → **crop each line image → send one crop per
> request to the VLM → the returned text *is* that line, no alignment needed.**

Rationale, quoted from `transcribe.py`:

> *"one crop per request... position is already known from segmentation, the
> model only reads."*

The key insight: **by feeding the model one pre-segmented line at a time, they
eliminate the alignment problem entirely.** No Needleman-Wunsch, no anchor
heuristics, no second pass, no `--review-alignment` repair UI — because text and
box are bound together by construction. Our `align_ocr.py` and the entire Flask
review UI exist to repair a class of failure that *cannot occur* in their design.

**Trade-offs we'd take on:**

- **Cost/latency:** one API call per line instead of per page. A directory page
  can have 100+ entries. Their mitigations: per-page caching + low concurrency.
- **Loss of cross-line context:** the model can't use neighbors to disambiguate.
  Less critical for printed directories than for prose, but column
  headers/continuations could suffer.
- **Fights our `--flex` economics:** Flex's 50% discount is most valuable on
  big page-sized requests; thousands of tiny line requests is a different cost
  profile.

**Recommendation:** not a wholesale replacement — our printed multi-column
layouts suit full-page OCR, and we've already invested in alignment. But a strong
candidate for a **fallback / hard-page mode**: when alignment confidence is low
or `--review-alignment` flags a page, re-run *those pages only* in line-crop
mode. Alignment-free guarantee exactly where the current approach is weakest.

---

## Other things they do better

| Their feature | Why it's superior | Adapt? |
|---|---|---|
| **ALTO v3 XML output** | Industry-standard OCR coordinate format. Drops into Solr, Mirador, Universal Viewer, IIIF Content Search with zero glue. Our `canvas_fragment` URIs are good but bespoke. | **Yes, as an export option.** We already have Surya line bboxes — everything ALTO needs. |
| **IIIF Content Search recipe** (`docs/iiif-content-search.md`) | Documented path: ALTO → Solr (`text_ocr` field type) → shim service emitting IIIF Search 1.0 `AnnotationList` → in-viewer highlighting. | **Reference, not copy.** Blueprint for in-viewer (Clover/Mirador) full-text highlighting; pairs with our Clover/UMD work. |
| **Multi-backend abstraction** (Gemini/Claude/OpenAI + **local** via LM Studio/Ollama/vLLM) | Clean `Transcriber` class, `_call_*` methods behind one `transcribe_crop()`. Local models = zero-API-cost bulk runs. | **Partially.** A **local-VLM backend** could cut cost on bulk runs — same goal as `--flex`/Batch from another angle. |
| **Editorial presets** ("normalized" vs. "diplomatic") + composable custom instructions with **sentinel tokens** (`[illegible]`, `[blank]`) | Named transcription policies; sentinel tokens are a stable machine contract surviving prompt edits. | **Yes, the sentinel-token contract.** Document `[illegible]`/`[blank]` literals so `fix_entries.py` reasons about gaps deterministically. |
| **Per-page transcript caching** | *"transcripts cache per page, so re-runs only pay for unfinished pages."* Page-granular resume. | **Maybe.** Our `pipeline_state.json` caches at *stage* granularity; per-page resume saves money on crashed mid-volume runs. |
| **Documented concurrency lesson** | Default workers = 2 because *"commercial APIs throttle sustained parallel vision traffic by stalling, not by returning 429."* | **Reference.** Useful if/when we parallelize Gemini OCR. |
| **Word-box estimation from text** (`wordboxes.py`) | Splits line bbox proportional to per-word char count + fixed 0.8-unit gaps; solves Sayre's paradox by deriving geometry from text. Boxes *"land on the correct word essentially always"* for search. | **Niche.** Dependency-free word-level highlighting if we ever want it in the explorer. |

---

## Where our pipeline is clearly ahead

- **Structured extraction (NER → typed CSV schema).** Their output is plain
  text. Our entries / categories / per-row `canvas_fragment` provenance have no
  analog there.
- **Geocoding + Leaflet maps.** No equivalent.
- **Multi-source ingestion** (LoC / IA / generic IIIF / CONTENTdm / pre-built
  CSV). They take IIIF manifests or a local image dir only.
- **Calibrate-once-run-many** (`--select-pages`, `--generate-prompts`). More
  powerful for heterogeneous sources than their fixed editorial presets.
- **Post-processing** (`fix_entries.py`, `combine_volumes.py`, category
  inference, explorer build). A whole layer they lack.
- **Surya line detection** is arguably more robust for printed multi-column
  directories than Kraken BLLA (tuned for handwritten baselines).

---

## Recommended actions (priority order)

1. **Line-crop "hard-page" fallback mode** for low-confidence / review-flagged
   pages. Highest value: attacks our most fragile stage (NW alignment) with their
   alignment-free design, scoped to only the pages that need it.
   — ⚙️ *Prototyped, then narrowed* (see "Revised recommendation for action #1"):
   `pipeline/run_linecrop_ocr.py` exists and works, but the empirical tests showed
   a model upgrade fixes most merges first; line-crop is reserved for the residual
   niche. Not wired as a default stage.
2. **ALTO v3 export option.** We already have line text + bboxes; serializing to
   ALTO unlocks standard search infrastructure for free.
   — ✅ *Implemented 2026-06-12.* `pipeline/export_alto.py` → `*.alto.xml`; wired as
   the `--export-alto` stage (registry, `main.py`, dashboard `iiif` group,
   CLAUDE.md). Word-level `String` boxes via proportional splitting; `--line-strings`
   for line-level.
3. **Explicit sentinel tokens** (`[illegible]`/`[blank]`) in OCR/NER prompts as a
   documented machine contract.
   — ✅ *Implemented 2026-06-12.* Contract defined in `prompts/README.md`;
   referenced from `ocr_prompt.md` + `ner_prompt.md`; `analysis/fix_entries.py`
   exposes `SENTINEL_TOKENS` and folds them into hallucination-detection so they're
   never mistaken for data. (`run_linecrop_ocr.py` already emitted them.)
4. **Local-VLM backend** (Ollama/vLLM) for cost-free bulk runs, complementing
   `--flex`/Batch.
   — ⬜ *Not started.*
5. **Operational notes** — per-page caching for resume, workers=2
   vision-throttling wisdom — when we parallelize.
   — ✅ *Adopted 2026-06-12.* Per-page resume already existed (OCR skips existing
   `.txt`); the workers throttling guidance is now in `run_gemini_ocr.py`'s
   `--workers` help (stall → lower to 2, don't raise).

---

## Prototype status (branch `htr-alto-experiments`, 2026-06-12)

> **Merge status (2026-06-14):** the ALTO export, the sentinel-token contract,
> the column-merge detector, and the `align_ocr` refactor were merged to `main`.
> **`pipeline/run_linecrop_ocr.py` and its tests are *not* on `main`** — the
> line-crop fallback remains on the `htr-alto-experiments` branch as an
> experiment (see "Revised recommendation for action #1": a model upgrade beats
> it for most cases, so it was deliberately not promoted).

Action #1 (line-crop fallback) is prototyped:

- **`pipeline/align_ocr.py`** — extracted the canvas/fragment-resolution logic
  (info.json natural dims, square-placeholder guard, split `_left`/`_right`
  offset handling) into a shared `resolve_fragment_fn(image_path, img_w, img_h)`.
  `align_image` now calls it. Verified behavior-preserving: regenerating all 100
  aligned JSONs for the lain_healy 1897 set with vs. without the refactor (same
  current algorithm) produced **0/100 differing files**.
- **`pipeline/run_linecrop_ocr.py`** (new) — crops each Surya line and sends it
  to Gemini individually, emitting the same JSON shape as the NW aligner
  (`confidence: "crop"`, `method: "linecrop"`, plus `surya_text` for eval). No
  Needleman-Wunsch. Default selection is flagged pages only
  (`needs_review` / `possible_column_merge`); `--page`, `--all`, `--replace`,
  `--dry-run`, `--flex` supported. Default workers=2 per the throttling note.
- **Verified** on real data: dry-run correctly auto-detected the model and
  selected exactly the one flagged page (312 lines — a vivid illustration of the
  per-line cost); a real run on a 9-line page transcribed every line correctly,
  each bound to its Surya bbox with correctly scaled canvas fragments. Full test
  suite: 146 passed.

**Not yet done** (deliberately, pending review): registering a `pipeline
linecrop` subcommand in `stages.py`, the sentinel-token contract on the page-OCR
prompts (the line prompt already emits `[blank]`/`[illegible]`), and the
`"crop"` tier is not yet documented in the align_ocr "Confidence tiers" header
or recognized by the `--review-alignment` UI.

### Cost implications (measured 2026-06-12)

Measured on a real 9-line page (lain_healy 1897, page 0021) using
`gemini-3.1-flash-lite-preview`, full-page OCR vs. per-line crops:

| Path | Calls | Input tokens | Output tokens |
|---|---|---|---|
| Full page (1 call) | 1 | 1,167 | 8 |
| Line crops | 9 | 10,322 | 73 |
| **Ratio** | **9×** | **8.8×** | — |

The driver: **each line crop cost ~1,150 input tokens — nearly as much as the
entire full page (1,167)**. Gemini bills a per-image token floor (a ~1400×90px
line strip still spans two tiles and re-pays the system prompt), so a one-line
image is not meaningfully cheaper than a full-page image. Therefore:

* **Full-page cost is ~flat** in line count — a page is a fixed image → fixed
  tiles, ~1–3k input tokens regardless of text density.
* **Line-crop cost scales linearly** with line count (≈ N × one-page cost).

So the multiplier **grows with page density** — worst on exactly the dense pages
the fallback most wants to fix:

| Page | Lines | Full-page (in tok) | Line-crop (in tok) | Multiplier |
|---|---|---|---|---|
| Measured (sparse) | 9 | 1,167 | 10,322 | **8.8×** |
| Typical directory page | ~50 | ~1,500 | ~57,000 | **~38×** |
| A 312-line flagged page | 312 | ~3,000 | ~360,000 | **~100×+** |

Output tokens are negligible either way (same text is produced). Flex (~50% off,
already the default) applies equally to both — it halves absolute cost but does
**not** change the ratio.

**The counterweight: absolute cost is tiny.** At flash-lite's ~$0.10/1M input
tokens, even the 312-line worst case is ≈ 3–4¢ (under 2¢ with flex). A
flagged-only pass over ~20 dense pages ≈ 3–4M tokens ≈ ~$0.30 (~$0.15 flex). The
multiplier is alarming; the dollars are not — *because it is scoped to flagged
pages*. As a wholesale replacement it would be wasteful (8–100× the whole OCR
bill); as a targeted fallback it is rounding error.

Two operational consequences:

1. **Run line-crop *without* `--flex`.** Flex's 1–15 min latency is *per
   request*, and line-crop multiplies requests 50–300× per page. On a handful of
   flagged pages the standard tier finishes in seconds; flex could stall for an
   hour. Flex suits big page-sized requests, not swarms of tiny ones — so the
   script defaults to the standard tier (do not pass `--flex` here).
2. **If the multiplier ever needs softening:** batch K lines per request (one
   multi-line strip, one transcription per line). That amortizes the per-image
   floor across K lines but partially reintroduces the ordering ambiguity the
   approach exists to eliminate — a separate experiment, not the default.

### Empirical test on a real flagged page — line-crop is narrower than hoped

Tested on `tulsa_1921` page `0009_p15020coll12:2447_left`, flagged
`possible_column_merge` (Surya 172 boxes vs Gemini 66 matched lines). **Result:
line-crop made the page worse, and the page did not actually need fixing.** Three
findings, all of which generalize:

1. **The flag was a false positive.** The existing NW alignment was already good
   (66 coherent single-entry lines, 0 unmatched, 1/66 lines even resembling a
   genuine merge). The `possible_column_merge` heuristic fires on the
   Surya:Gemini line-count *ratio*, but that ratio was inflated by Surya
   **over-segmentation** (8 overlapping box-pairs + many 10–19px fragments), not
   by Gemini reading across columns. The flag cannot distinguish the two.
2. **Line-crop inherits Surya's segmentation faults.** Because Surya's boxes here
   overlap vertically by ~half a line, each crop captured parts of the lines
   above and below — so the model returned multi-line, duplicated, hyphenated
   fragments instead of clean cells. Line-crop is only as good as the boxes it
   crops; it fixes *text→box assignment*, not bad boxes.
3. **Resolution gates crop quality.** Tulsa downloads are low-res (median width
   1024px, some 872px) → ~19px line crops → poor per-crop OCR. Contrast the
   lain_healy 1897 smoke test (1704px-wide page, clean tight boxes): there every
   line transcribed perfectly.

**Revised targeting guidance.** The preconditions for line-crop to *help* are:
(a) a genuine text→box alignment failure, (b) clean tight single-line Surya
boxes, and (c) adequate resolution (≳1500px page width). `possible_column_merge`
satisfies none of these reliably and should **not** be the trigger. Better
signals for a *genuine* column-merge: aligned lines that actually contain two
street addresses (`tools/detect_column_merges.py`), or a high `unmatched_gemini`
rate (Gemini text that found no box). The script's current default selection
(`needs_review` OR `possible_column_merge`) is therefore too broad — see the
"not yet done" list.

### Second test — a *genuine* high-res merge, and the result that reframes everything

`tools/detect_column_merges.py` (new — finds aligned lines carrying two street
addresses, gated on ≥1500px download width) surfaced a clean candidate:
`green_books_and_related/…/0052_5213564`, a 2048px Green Book bar/venue page
where **23 of 50 aligned lines** were two concatenated entries
("Royal—1073 Fulton St. Carver—980 Prospect Ave."). A real column merge, on a
high-res scan, on the project's centerpiece material. Line-crop handled it
correctly — 127 clean single-entry lines.

But the same page had been OCR'd with **two** models, and that is the finding
that matters:

| Path | Calls | Result |
|---|---|---|
| `gemini-2.0-flash`, full page | 1 | 50 lines, **23 merged** ✗ |
| `gemini-3-flash-preview`, full page | 1 | 125 lines, **0 merged** ✓ |
| `gemini-3.1-flash-lite`, line-crop | 127 | 127 lines, 0 merged ✓ |

**The column merge was a weaker-model artifact.** The newer full-page model read
the two columns correctly in a *single* call; line-crop reached the same clean
result at ~127× the calls. On printed directories, **upgrading the OCR model is
the cheaper, simpler fix for column merges than line-cropping** — same outcome,
1× cost vs. 40–127×.

#### How general is that? — 1,527 dual-model pages, no API cost

`green_books_and_related` happens to have every page OCR'd with *both*
`gemini-2.0-flash` and `gemini-3-flash-preview`, so the "does a better model fix
it?" question can be answered directly from existing data. Running
`detect_column_merges.py`'s merge heuristic over both alignments for all 1,527
shared pages:

* **The classic merge case is fixed completely.** Every page where 2.0-flash
  merged ≥3 lines → **0** merged lines under 3-flash-preview (28→0, 26→0, 18→0,
  11→0, …). 100% of high-severity pages cleared, and it is genuine linearization,
  not data loss (0052: 50→125 lines, all clean single entries).
* **A small hard tail resists even the best model.** Total merged lines fell only
  160 → 135 (**16%** net), and a few pages wobbled ±1–2. The residual pages are
  **two-column lists of bare addresses with no names**
  (`413 S. Oregon St.   218½ Mesa St.` / `901 Jones St.   1011 Dart St.`), a
  layout neither model linearizes — and where the heuristic is noisiest. The 16%
  net figure is dominated by this long tail, *not* by the severe pages (which all
  cleared).

So: **the dominant, project-relevant merge case (name→address entries across
columns) is reliably fixed by a model upgrade at 1× cost.** The genuine niche for
line-crop shrinks to the small minority of dense *nameless* two-column address
tables on adequate-resolution scans — the only class where even the best
full-page model still merges.

### Revised recommendation for action #1

Line-crop's genuine niche is narrower than the report first claimed: pages where
**even the best full-page model cannot read the layout**, *and* Surya boxes are
clean, *and* resolution is adequate. For printed directories that combination is
rare — a model upgrade resolves most column merges first. That niche (text no
full-page VLM can read line-grouped correctly) is essentially **degraded
handwriting**, which is exactly HTR-ALTO's manuscript domain and only at the
edge of this project's printed-material focus. So the practical order of
operations is:

1. **First, try a stronger full-page OCR model** on merge-flagged pages
   (`--ocr-model gemini-3-flash-preview` or better). Cheapest, usually
   sufficient.
2. **Only then** consider line-crop, and only on adequate-resolution pages with
   clean Surya boxes that *still* merge under the best model.

The line-crop prototype stays as a validated tool for that residual case (and as
the natural engine if the project ever takes on handwritten material), but it
should **not** be wired in as the default response to `possible_column_merge`.
`tools/detect_column_merges.py` is the more useful immediate artifact: it finds
the genuinely merged pages so they can be re-OCR'd with a better model.

#### Detector modes (address-specific vs. content-agnostic)

The default `--mode address` is high-precision but only fires where entries
carry street addresses (city/business directories, Green Books, phone books).
For arbitrary volumes — copyright ledgers, passenger-list indexes, name/date
rosters — there is `--mode generic`: on a page that Surya geometry shows is
**multi-column**, it flags lines whose character *and* word counts are a strong
outlier above the page median (≈ two entries fused), regardless of content. The
multi-column gate keeps single-column prose (where long lines are normal) from
tripping it. `--mode both` unions the two. Verified: on a passenger-list index
with no addresses, `address` finds 0 while `generic` catches merges like
`Mary Margaret, 1(…) … Nancy Mumm, 2(…)`; on flower-dictionary prose, `generic`
correctly finds nothing. `run_linecrop_ocr.py`'s default page selection reuses
the same merge heuristic.
