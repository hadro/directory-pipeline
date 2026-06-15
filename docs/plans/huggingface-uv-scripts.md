# Strategy: Hugging Face uv-scripts + Open Models for the Directory Pipeline

**Status:** Planned — Phase 0 decision record; implementation deferred to a future session.

> **What this is:** a strategy/recommendations doc (no code yet) for replacing the
> Gemini OCR/NER steps with local open models, and for adjacent uv-scripts opportunities.
> It surveys models, gives an explicit cost comparison, and lays out a phased
> implementation path. Implementation is deferred to a future session (see roadmap).
> Created 2026-06-15.

---

## Context: what "uv scripts for Hugging Face" actually is

The HF community (notably `uv-scripts/*` on the Hub and `davanstrien/uv-scripts-for-ai`
on GitHub) publishes **single-file Python scripts with PEP 723 inline dependency
metadata**. You run one with `uv run <url> <input> <output>` — uv reads the inline
`# /// script` block, builds an isolated env, downloads the right Python, and runs it.
No clone, no virtualenv, no `pip install`. The *same* script runs on managed cloud GPU
via `hf jobs uv run --flavor l4x1 ...`.

This matters for this repo for two independent reasons:
1. **The models behind those scripts** (30+ open OCR VLMs, GLiNER for NER, embeddings,
   etc.) are exactly the local replacements we're evaluating.
2. **The packaging pattern itself** — self-contained, run-from-URL, "built for humans
   and agents" — aligns with what `pipeline/api.py` and the `colab/` notebooks already
   try to do (a curated, portable surface).

---

## Honest framing: why local models, really

The pipeline's default is `gemini-3.1-flash-lite`. At flash-lite tier, **per-page API
cost is already tiny** — well under a cent for a resized page image plus a short JSON
response. So, to be straight: **cost is not the main reason to go local.** A full
1,000-page collection on flash-lite is on the order of a coffee, not a budget line.

The real, defensible reasons to add a local backend:

- **No rate limits on bulk runs.** The current code carries 429/503 retry logic,
  temperature-escalation, and flex-tier latency handling precisely because the API
  throttles. Local inference has none of that — you saturate your own GPU and stop.
- **Reproducibility.** A pinned open checkpoint gives byte-stable behavior; a hosted
  model can change or be deprecated under you (the code already has a `FALLBACK_MODEL`
  for exactly this fragility).
- **Privacy / provenance.** Page images stay on your machine — meaningful for some
  library/archive partners and for sensitive collections.
- **Offline / no external dependency.** Runs in an air-gapped or grant-funded compute
  context without an API account.
- **Research value.** Systematically benchmarking open OCR/NER models against Gemini on
  *your* historical material is itself a contribution — and the repo already has the
  scaffolding (`compare_ocr.py`, `compare_extraction.py`, `collections/greenbook/model_eval.py`).
- **Cost/speed *do* dominate at scale.** This is corpus-size-dependent: at the directory
  pipeline's typical scale (thousands of pages, tens of thousands of entries) the API bill
  is trivial. But Mattingly's "3.6M names" project found frontier models *too slow, too
  expensive, and inaccurate* at that volume, and a fine-tuned small model won outright.
  So the local case strengthens sharply as a collection series grows.

Bottom line: pitch this as **control + reproducibility + a benchmarking capability** at
typical scale, shading into a **real cost/speed/accuracy win** once a collection type
recurs or runs into the millions of records. See "Explicit cost" below for a worked
500-page comparison (short version: ~$1–7 either way — a wash at one volume).

---

## Recommended hardware

**Primary: run locally on the same GPU you already use for Surya.** Surya already
requires GPU/Apple-Silicon (`uv sync --extra gpu`), so the heavy-ML path exists. Adding
a vision-language OCR model and an NER model is the natural extension of that pattern —
same install story, same `requires="surya"`-style optional-dependency gating in
`pipeline/stages.py`, same `colab/` fallback for people without a local GPU.

**Secondary/optional: HF Jobs (`hf jobs uv run --flavor l4x1`) as a cloud-GPU backend**
for users with no local GPU, billed per GPU-minute. This mirrors the existing
"use Colab for the GPU stages" guidance and gives a no-hardware on-ramp. Treat it as an
alternative execution target, not the default.

**Rough capability guide:**
- A100 / H100 / RTX 4090 (≥16–24 GB): comfortable for 7B VLMs, fast.
- RTX 3090 / 4080 (20 GB): fine, slower.
- Apple M-series: feasible for 4–7B (Qwen3.5, NuExtract3 has an MLX build), slower than NVIDIA.
- CPU-only: not recommended for the VLMs; GLiNER-style NER is tolerable on CPU.

---

## Part A — Replacing Gemini OCR

**Where it plugs in.** `pipeline/run_gemini_ocr.py` is the only thing that needs a
sibling. Everything downstream is already model-agnostic: it writes
`{stem}_{model_slug}.txt`, and `align_ocr.py` / `extract_entries.py` discover OCR output
purely by filename + `pipeline_state.json`. A new `run_local_ocr.py` that emits the same
`.txt` filename slots in with **zero downstream changes**. Surya bbox detection and the
Needleman–Wunsch alignment are already local and stay untouched.

**Candidate open OCR models (from `uv-scripts/ocr`, 30+ available):**

| Model | Base / type | Strengths | Watch-outs | Fit for historical directories |
|---|---|---|---|---|
| **Chandra** (`datalab-to/chandra`, `chandra-ocr-2`) | Qwen 3.5, fine-tuned by **Datalab** | Full-page VLM decode; tables/forms/**handwriting**/layout preserved; multilingual; tested on a 1913 handwritten letter; HTML/MD/JSON out | Weights are modified **OpenRAIL-M** (free for research/personal/<$2M orgs; no competing with Datalab's API) — code is Apache-2.0 | **Top pick.** Same vendor as the **Surya** you already run; built for exactly this material |
| **RolmOCR** (`reducto/RolmOCR`) | Qwen2.5-VL-7B | Fast, general, ~16K ctx, plain-text out | Not tuned for ornate type | Strong fast baseline — speed + breadth |
| **Qwen3-VL / Qwen 3.5** (`Qwen/Qwen3-VL-*`) | newest Qwen VLM | OCR in 32 langs; documented gains on degraded scans, blur/tilt, rare chars, **ancient scripts**, long-doc structure; 256K ctx | 7B+ VRAM; newer = pin a revision | Strong contender; can also do NER (one model, both roles) |
| **olmOCR / olmOCR2** (`allenai/olmOCR-*`) | 7B VLM | Higher quality, good on mixed print+handwriting | Slower, bigger VRAM | Good **quality fallback** |
| **Nanonets-OCR-s / OCR2-3B** | 3B specialized | Excellent on structured docs, markdown/LaTeX out | Weaker on handwriting | Good if pages are clean tabular print |
| **dots.ocr** (`rednote-hilab/dots.ocr`) | specialized | Strong layout/structure | Heavier | Worth a bake-off pass |
| **PaddleOCR-VL** | VLM+classifier | Multilingual, balanced | — | Useful if non-English volumes appear |
| **TrOCR** | text-only transformer | Tiny, very fast | No layout awareness, short ctx | CPU fallback only |

**Recommendation:** lead with **Chandra** — it's purpose-built for tables/handwriting/
historical layout, fine-tuned on Qwen 3.5, and crucially comes from **Datalab, the same
team behind the Surya bbox model the pipeline already depends on**, so it fits the stack and
trust boundary better than any other option. Run **RolmOCR** as the fast baseline and
**Qwen3.5** as the do-everything contender in the same first round, with **olmOCR2** as
quality escalation (mirroring the `DEFAULT_OCR_MODEL` → `FALLBACK_MODEL` pattern). Operational
bonus: Chandra, Qwen3.5, and NuExtract3 (Part B) can each do OCR *and* extraction, so a
single loaded model could collapse today's two Gemini calls into one — pay the VRAM/load
cost once. **License note:** Chandra's weights are modified OpenRAIL-M (fine for
research/non-commercial/cultural-heritage use, restricted for API-competing commercial
use) — confirm it fits the green-books distribution context; the Qwen/olmOCR/Nanonets
options are more permissive if that's a blocker. Reuse the existing quality gates in
`run_gemini_ocr.py` (repetition-loop detection, dot-leader runaway, blank-page skip via
`utils/image_utils.is_blank_page`) verbatim — they're model-agnostic text heuristics.

**Reality check on alignment:** the load-bearing risk is whether an open model's line
breaks and reading order align cleanly to Surya bboxes via `align_ocr.py`. That's the
first thing to validate (see roadmap Phase 1), not OCR character accuracy in isolation.

---

## Part B — Replacing Gemini NER / entry extraction

`extract_entries.py` sends aligned text (± page image) to Gemini and expects a JSON
`{"entries": [...]}` with a city/state/category context state-machine carried across
pages. Several open approaches, usefully different (option 5 added after reviewing
Mattingly's "3.6M names" write-up — see "Prior art" below):

1. **Local VLM + the existing JSON prompt (closest behavior).**
   Run **Qwen3-VL / Qwen 3.5** (or whichever OCR VLM you settled on) with the current
   `ner_prompt.md` and the same multimodal text+image input. Reuse the existing
   `_recover_partial_json()` salvage and per-page context persistence. This is the most
   faithful drop-in: same prompt, same flexible inferred-CSV schema, same context
   propagation. Expect "most of the way to Gemini," weaker on subtle category inference.

2. **NuExtract3 — purpose-built structured extraction (likely the best NER fit; now shipped).**
   `numind/NuExtract3` (4B VLM fine-tuned on **Qwen3.5-4B**, **Apache-2.0**, 131K context,
   safetensors + GGUF + MLX) is trained *specifically* for template-driven extraction: hand
   it a **JSON template** + the input and it fills it. It's **multimodal** and notably
   **unifies extraction *and* image→Markdown OCR in one model** — beats Qwen3.5-9B on
   NuMind's internal 600-doc benchmark. It supersedes the older NuExtract 2.0 (newer,
   smaller, cleaner license, and it can also serve the OCR step). Fits:
   - The template maps directly onto the **inferred-CSV schema** — derive it from
     `ner_prompt.md`'s fields (establishment_name, raw_address, city, state, category, …).
     Because it's *trained* to emit the template, **JSON is reliable for this model** — the
     delimited-format lesson (option 5) applies to *prompted generalists and your own
     fine-tune*, not to NuExtract3.
   - **Template type system (actionable schema design).** NuExtract3 templates declare field
     types via leaf values: `verbatim-string` (extract exactly — use for `establishment_name`,
     `raw_address`, `phone`), `string` (allow light normalization — `city`, `state`, `date`),
     **arrays** for repeating fields, and **enums** for fixed choices. The enum support maps
     directly onto the **controlled category vocabulary** — pass the Green Book / directory
     category list as an enum instead of relying on `fix_entries.py --infer-categories`
     keyword rules. The template is the NuExtract3 analog of `ner_prompt.md`, and "different
     series → different template" is exactly the *calibrate-once-run-many* pattern.
   - **Junk-page filtering for free.** An `image_type` enum (`entry_page`/`heading`/`ad`/
     `blank`/`other`) lets the model self-tag dividers/blanks/ads — augmenting the existing
     blank-page and sparse-page guards (`is_blank_page`, MIN_OCR_CHARS).
   - **Turnkey HF Jobs path (yes, runnable via HF Jobs, officially).** There's an official
     `nuextract3.py` UV script in `uv-scripts/ocr`; van Strien runs it as one command
     (`hf jobs uv run --flavor a100-large --image vllm/vllm-openai:latest … nuextract3.py
     my-cards my-records --template schema.json`): point it at a dataset of images + a schema,
     get a dataset of structured records back. For the pipeline this is near-zero integration
     — push page images (or per-entry bbox crops, see below) as a Hub dataset, run the recipe.
   - **Important nuance:** NuExtract3 does great *per-record field extraction*, but it has no
     notion of the **cross-page city/state/category context state-machine** (carry-forward,
     section resets). Keep the pipeline's existing context logic wrapping it — feed prior
     context in, persist it out, exactly as `extract_entries.py` does today. It replaces the
     *call*, not the orchestration. At 4B it's light enough to run alongside an OCR VLM, or
     to be the single model doing both OCR and extraction.
   - **Accuracy expectation (honest).** van Strien reports NuExtract3 zero-shot at ~100% title
     / ~94% publisher off a single page on open-book metadata (vs ~77–86% for general VLMs),
     but with real headroom on *handwritten/annotated* material and demos left unreviewed.
     Treat zero-shot as a strong first pass, not a final answer — which matches the roadmap.

3. **GLiNER for zero-shot span extraction (lightweight, no API, even CPU-tolerable).**
   `uv-scripts` ships GLiNER entity-extraction recipes. GLiNER does zero-shot NER from a
   label list (e.g. `establishment_name`, `street_address`, `city`, `category`) with no
   fine-tuning. It won't produce nested JSON or run the cross-page state-machine on its
   own, but it's fast, deterministic, and a strong building block — and it maps directly
   onto the **`fix_entries.py --infer-categories`** step, which today uses hand-written
   keyword rules (`_INFER_RULES`). GLiNER could augment or replace those rules.

4. **(Advanced) Structured generation via vLLM + Outlines** to *guarantee* schema-valid
   JSON, which would let you delete the partial-JSON recovery path entirely. Largely
   *unnecessary if you use NuExtract3* (it's trained to emit the template), but the right
   tool if you constrain a general VLM instead. van Strien's card workflow shows the vLLM
   side of this is already a turnkey recipe, not a research project.

5. **Fine-tune a small Qwen 3.5 model on a delimited (not JSON) target — the path for
   scale.** Mattingly parsed **3.6M historical names** by abandoning frontier models
   (too slow, too expensive, *inaccurate* at that volume) and fine-tuning a small Qwen 3.5
   model to **96% accuracy** — and his headline lesson was *"it wasn't just model size, it
   was the format."* Two takeaways that change this plan:
   - **Fine-tuning is a primary track for large/repeated series, not a fallback.** The
     pipeline is unusually well-placed to do it cheaply: run Gemini (or NuExtract) on a
     sample, hand-correct via the existing review UIs, and that labeled set *is* your
     fine-tune data. The Qwen 3.5 *small* family (≈0.8B–9B, on-device sized) is the target.
     Worth it when a collection type recurs (Green Book volumes, multi-year directories);
     overkill for a one-off volume.
   - **Prefer a flat delimited row format over JSON for the model's output.** JSON is
     token-heavy and a top failure mode for small models — it's *why* the code needs
     `_recover_partial_json()`. A pipe-delimited row per entry (aligned to the CSV columns)
     is cheaper, faster, and more reliably parsed, and the final artifact is already CSV,
     so JSON is an unnecessary intermediate. Use **pipes, not commas**, so commas inside
     names/addresses don't break parsing (the standard name/address-parsing convention).
     This applies to options 1–2 as well: consider asking even the prompted models for
     delimited rows and dropping the JSON round-trip.

**Recommendation:** for one-off / first-time collections, make **(2) NuExtract3** the lead
extraction call (closest match to the task, official HF Jobs recipe, can also OCR),
prototype **(1) Qwen 3.5** for parity with today's prompt-driven behavior, and use
**(3) GLiNER** as the fast lightweight option and category-inference upgrade. For a
collection type you'll run repeatedly or at large volume, plan toward **(5) a fine-tuned
small Qwen 3.5** — the proven high-volume path. For NuExtract3 keep the trained **JSON
template** (it's reliable); for *prompted generalists and your own fine-tune*, **switch the
target to a pipe-delimited row** to cut tokens and retire the partial-JSON crutch. New stages
(`run_local_ner.py`) register in `pipeline/stages.py` exactly like the Gemini one and
write the same `entries_{model_slug}.csv`, so `explore_entries.py`, `fix_entries.py`,
and `combine_volumes.py` need no changes.

**New architectural option this unlocks — "each entry is a card."** van Strien feeds whole
*card images* to NuExtract3. The pipeline already produces **per-line/entry bounding boxes
from Surya**. Cropping each entry's bbox into a small image and treating it as a one-record
"card" turns the hardest NER problem (segmenting a dense multi-column page into discrete
entries) into something Surya already solves, and hands NuExtract3 an easy single-record
extraction. This is a genuinely different data path from today's page-level OCR→NER, worth a
Phase-1 spike: push entry crops as a Hub dataset, run `nuextract3.py` with a directory schema,
get one structured row per entry back — with `canvas_fragment` derivable from the crop's
source bbox.

---

## Part C — Broader opportunities

Beyond the OCR/NER swap, the uv-scripts ecosystem opens adjacent extensions that fit
what the pipeline already does:

1. **Publish results as Hub datasets.** uv-scripts include **IIIF-tile→dataset** and
   image-URL→dataset recipes. The pipeline already walks IIIF manifests
   (`utils/iiif_utils.py`) and produces per-volume CSVs + `manifest.json`. Packaging a
   collection (images + `green_book_entries_all.csv` + `image_to_volume.json`) as a Hub
   dataset would give the **green-books** companion repo a second, citable distribution
   channel and make the corpus reusable by others.

2. **Semantic exploration with embeddings + Atlas.** uv-scripts embed a dataset and
   render an interactive map. Run over `establishment_name` + `category` + `notes` to
   cluster businesses, surface near-duplicates across volumes, and visualize a Green Book
   by establishment type. This complements (doesn't replace) the existing HTML explorer
   and the geographic map.

3. **A real OCR/NER benchmark harness.** uv-scripts let you run several models into the
   *same* dataset, each writing `inference_info`. That's a clean way to extend
   `compare_ocr.py` / `compare_extraction.py` / `collections/greenbook/model_eval.py`
   into a systematic open-vs-Gemini bake-off on a labeled gold set — turning "can we
   replace Gemini?" into a measured, repeatable answer (CER for OCR; field-level
   precision/recall for NER).

4. **Package leaf stages as uv scripts.** The leaf scripts already run as
   `python -m pipeline.<stage>` and there's a curated `pipeline/api.py`. Adding
   PEP 723 headers to a few of them (or thin wrappers) would make selected stages
   runnable from a URL, "agent-friendly," and shareable — the same ethos as
   `colab/library-cookbook.ipynb`, extended to the command line.

5. **HF Jobs as a zero-hardware GPU on-ramp** for `--surya-ocr` and the new local stages,
   for collaborators without a local GPU — a cloud sibling to the Colab guidance.

---

## Explicit cost: a 500-page city directory (HF Jobs vs Gemini)

**The honest headline: at 500 pages it's a wash — a few dollars either way.** Cost is not
what decides this at directory scale; it becomes decisive only at the millions-of-records
scale (Mattingly) or when you already own the GPU (then local is ~$0 marginal).

**Assumptions (city-directory pages are *dense* — adjust freely):**
- ~150–250 entries/page. **OCR** per page ≈ 1.7K input tokens (page image + prompt) + ~4K
  output. **NER** per page ≈ 6–7K input (OCR text + prompt + image) + ~6K output JSON.
- **Gemini 3.1 Flash-Lite** (the pipeline default tier): **$0.10/M input, $0.40/M output**;
  **flex/batch ≈ 50% off** (~$0.05 / $0.20).
- **HF Jobs GPU** (per-minute billing): T4 ~$0.40–0.50/hr, L4 ~$0.80/hr, L40S ~$1.80/hr,
  **a100-large (80GB) ~$4/hr** (van Strien's card recipe uses `a100-large`).
- **Throughput:** a 4B VLM (NuExtract3) on A100+vLLM ≈ 3–5 s/page incl. ~6K output tokens →
  ~500 pages in ~30–50 min (plus ~5 min model load). Two models (Chandra OCR + NuExtract3
  NER) ≈ double the GPU time.

| Path | What runs | 500-page cost | Notes |
|---|---|---|---|
| **Gemini, standard** | OCR + multimodal NER | **~$2.50** | ~$0.0049/page; no infra |
| **Gemini, flex/batch** | OCR + NER, ~50% off | **~$1.20** | default `--flex`; 1–15 min latency |
| **HF Jobs, NuExtract3 only** (OCR+NER in one 4B model) | one a100-large job | **~$2–3.50** | ~30–50 min × $4/hr; official `nuextract3.py` recipe |
| **HF Jobs, two-model** (Chandra OCR + NuExtract3 NER) | a100-large | **~$4–7** | two passes; best quality |
| **HF Jobs, cheaper GPU** (L4/L40S, slower) | L4 ~$0.80 / L40S ~$1.80/hr | **~$1.60–3** | longer wall-clock, lower $ for a small job |
| **Your local GPU** (already runs Surya) | local | **~$0 marginal** | + electricity; no per-page bill ever |

**Real-world anchor:** van Strien re-OCR'd **453,000 BPL cards for $39** (~$0.00009/card,
OCR only). A directory page ≈ ~150 cards' worth of text, so large-scale OCR is genuinely
cheap on either backend — the spend that matters is the NER/structured-extraction output
tokens (JSON is the costly part on Gemini, which is a second reason to prefer a delimited
target or a model trained to emit compact schema).

**Takeaways:**
- For **one 500-page volume**, pick on *quality, control, and privacy*, not price — Gemini
  flex (~$1.20) and HF Jobs (~$2–7) are the same order of magnitude.
- **Local GPU wins decisively only across many volumes** (a multi-year directory series, or
  re-runs while tuning prompts/schemas) where the per-page API bill recurs and local is free.
- **HF Jobs' value is "no local GPU + turnkey recipe," not raw savings** — it's roughly
  Gemini-priced for one volume, and you pay only while the job runs.
- Numbers scale ~linearly: a **5,000-page** series is ~$12–25 (Gemini) vs ~$0 local — that's
  where the case flips.

---

## Prior art / external validation (Mattingly, "3.6M historical names")

A directly analogous project from the Smithsonian's postdoc for historical-document
analysis is worth treating as a reference design — it's essentially a more opinionated
sibling of this pipeline (normalize → assemble prompt → constrained inference → strict
validation → export). Its findings shape the recommendations above:
- Frontier/API models were *too slow, too expensive, and inaccurate* at millions-of-records
  scale → a **fine-tuned small Qwen 3.5** reached **96% accuracy**.
- *"It wasn't just model size — it was the format."* The output format (flat/delimited,
  not JSON) was as load-bearing as the model choice.
This both **validates the local-model direction and the Qwen 3.5 pick**, and **adds**
the fine-tuning track (Part B option 5) and the JSON→delimited format change.

**Second reference design — van Strien, "catalogue card images → structured JSON"
(May 2026).** A GLAM/library practitioner doing the *exact* shape of this problem (record
image → schema-shaped JSON a catalogue can ingest). It's where the NuExtract3 specifics
above come from, and its stated realistic path is essentially this plan's roadmap:
*zero-shot first pass → curator reviews a sample → iterate the schema → fine-tune a small
model where it pays off.* Two framings worth borrowing: (1) "clean markdown is searchable;
it isn't *ingestible*" — for the green-books/library audience the **structured CSV, not the
OCR text, is the real deliverable**, which raises the priority of the NER step; (2)
catalogue cards (like city directories) are a problem *every* institution shares, so it's
worth **building something reusable** (a shared schema + recipe), not a one-off — directly
supporting the "publish as Hub dataset" and uv-script-packaging ideas in Part C.

---

## Why the pipeline is well-suited to this

- **File-based, model-agnostic contract.** Stages communicate only through files named
  `_{model_slug}.{ext}`; downstream discovery uses `discover_ocr_slug()` +
  `pipeline_state.json`. New backends just need to honor the filename contract.
- **Single stage registry.** `pipeline/stages.py` (`StageDef`, `model_mode="fan_out"`,
  `requires=...`) means a new stage is a few lines plus a script, with optional-dependency
  gating already modeled by Surya.
- **Centralized model constants.** `utils/models.py` (`DEFAULT_OCR_MODEL`,
  `DEFAULT_NER_MODEL`, `model_slug()`) is the one place to register new defaults.
- **Surya is the proof.** A local neural model (`run_surya_ocr.py`) already lives inside
  the pipeline behind `--extra gpu`. Local OCR/NER follows the exact same shape.

---

## Phased roadmap (for a future implementation session)

**Phase 0 — Decide & scope (this doc).** Pick primary models — OCR: **Chandra** lead
(RolmOCR baseline, Qwen3.5 contender, olmOCR2 escalation); NER: **NuExtract3** lead, Qwen 3.5
for prompt-parity, GLiNER as the light option — and confirm local-GPU primary / HF-Jobs
fallback. Note that NuExtract3 (official `nuextract3.py` recipe) and TRL fine-tuning both
run on HF Jobs with no local GPU, so the cloud path is viable end-to-end.

**Phase 1 — Validate before integrating (highest-value, ~2–3 hrs).** Standalone scripts,
no pipeline wiring. On ~10 Green Book + ~5 Tulsa pages already in `output/`:
- **OCR:** run **Chandra** and RolmOCR (and Qwen3.5); feed each one's text into existing
  `align_ocr.py` against Surya bboxes. **Gate:** alignment success ≥ ~90% and clean reading
  order. (This, not raw CER, is the real risk.) Chandra is the one to beat for handwriting/
  layout; note the OpenRAIL-M license fit early.
- **NER:** run **NuExtract3** with a template derived from `ner_prompt.md`'s fields (use
  `verbatim-string` for names/addresses, an **enum** for category, `image_type` to tag junk),
  *and* a general-VLM pass (Qwen3.5) with the existing prompt; eyeball field accuracy and
  context propagation. Verify the wrapper feeds/persists cross-page context around NuExtract3.
- **"Entry-as-card" spike:** crop a handful of Surya entry bboxes to small images, run them
  through `nuextract3.py` (locally or on HF Jobs), and compare one-row-per-crop output vs the
  page-level OCR→NER path. This tests whether Surya segmentation + per-entry extraction beats
  whole-page extraction on dense directory pages.
- **Format A/B test (prompted/fine-tune paths only):** have the *prompted* models emit a
  **pipe-delimited row per entry** vs JSON; compare parse-failure rate and token count.
- Compare all of the above against the existing Gemini `.txt` / `entries_*.csv` baseline.

**Phase 2 — Wire in as opt-in stages.** `pipeline/run_local_ocr.py` + `run_local_ner.py`,
modeled on the Gemini scripts; register in `stages.py` (`--local-ocr`, `--local-extract`,
`requires` gating); add defaults to `utils/models.py`; add deps to the `gpu` extra (or a
new `local-ml` extra) in `pyproject.toml`. Reuse quality gates, JSON recovery, context
persistence, and parallelism as-is.

**Phase 3 — Benchmark + document.** Extend the `compare_*` tooling into an open-vs-Gemini
report; write a "Local models" section in `CLAUDE.md` and a costs/tradeoffs note.

**Phase 4 — Fine-tune for a recurring collection (the scale path).** Once a collection
type is stable, bootstrap a labeled set from Gemini/NuExtract output + the existing review
UIs, fine-tune a small Qwen 3.5 on a **pipe-delimited** target, and register it as just
another `entries_{model_slug}.csv`-producing backend. This is where the Mattingly result
says the real accuracy/cost/speed wins live.

**Phase 5 — Optional reach.** Hub-dataset publishing, embeddings/Atlas explorer, GLiNER
category inference in `fix_entries.py`, vLLM+Outlines constrained NER, uv-script packaging
of leaf stages.

---

## Risks & caveats (stated honestly)

- **Cost savings are modest at flash-lite *at typical scale*.** Lead with
  control/reproducibility/benchmarking; the dollar/speed case only becomes decisive at
  recurring or millions-of-records scale (Mattingly).
- **JSON is a liability for small models.** Prefer a pipe-delimited extraction target;
  don't carry the JSON round-trip (and its recovery code) into the small-model path.
- **Fine-tuning needs labeled data.** It's the scale win, but budget for bootstrapping a
  training set from Gemini/NuExtract output + the review UIs first.
- **Alignment, not accuracy, is the integration risk.** Open VLM line-breaking may not map
  to Surya bboxes as cleanly as the Gemini prompt was tuned to; validate in Phase 1.
- **Category inference & cross-page context** are where Gemini is strong; open models may
  regress here. GLiNER + the existing `_INFER_RULES` can backstop.
- **VRAM/throughput**: 7B VLMs are heavier than Surya; on a 20 GB card you may run OCR and
  NER sequentially, not concurrently.
- **Model-ID drift**: open checkpoints get renamed/revised — pin revisions, and lean on the
  existing `FALLBACK_MODEL` idea for resilience.

---

## Status / next steps

This doc is the Phase 0 decision record. Implementation is deliberately deferred to a
future session per the phased roadmap above — start with the Phase 1 standalone validation
(no pipeline wiring) before adding any stages.

**Sources:** HF `uv-scripts/ocr` and `uv-scripts/vllm` dataset cards; `davanstrien/uv-scripts-for-ai`
(GitHub README); HF blog "Supercharge your OCR Pipelines with Open Models"; van Strien,
"Efficient batch inference for LLMs with vLLM + UV Scripts on HF Jobs";
`numind/NuExtract3` model card (4B, Apache-2.0, Qwen3.5-based); `datalab-to/chandra` +
`chandra-ocr-2` (Datalab; OpenRAIL-M weights, Apache-2.0 code); `QwenLM/Qwen3-VL` repo +
"Qwen3-VL Technical Report" (arXiv 2511.21631); W.J.B. Mattingly, "Parsing 3.6 Million
Historical Names with Small Models" (wjbmattingly.com); Daniel van Strien, "How to turn
catalogue card images into structured JSON with a 4B open model" (May 2026) + the
`uv-scripts/ocr` `nuextract3.py` recipe and BPL/NLS card datasets. Pricing: Gemini
Flash-Lite rates via pricepertoken.com (gemini-3.1-flash-lite-preview), tokenmix.ai, and
Google's API pricing; HF Jobs/Spaces GPU hourly rates via Hugging Face hardware pricing.
