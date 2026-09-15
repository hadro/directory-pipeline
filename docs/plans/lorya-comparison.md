# Review: what directory-pipeline should borrow from UNDP Serbia's Lorya

**Status:** Recommendations — no code written. Tiered by value; nothing here is committed to.

> **What this is:** a comparative review of [`UNDP-Serbia/lorya_`](https://github.com/UNDP-Serbia/lorya_)
> against this repo, identifying concrete techniques, architecture, and tooling worth
> adopting — plus an explicit list of places where this pipeline is already ahead and
> should *not* converge. Created 2026-09-15.
>
> **Reviewed at:** lorya `c94a8be` (2026-07-13, v1.3.0), shallow clone of `main`.
> All `lorya:` citations below are paths in that repo at that commit.

---

## TL;DR

| # | Recommendation | Tier | Touches |
|---|---|---|---|
| 1 | YOLOv8 + DocLayNet for layout/section detection | 1 | `detect_sections` (planned), `detect_columns.py` |
| 2 | Crop-to-region, then OCR | 1 | `run_gemini_ocr.py`, `align_ocr.py` |
| 3 | LiteLLM as the model-backend abstraction | 1 | `utils/gemini.py`, HF plan |
| 4 | Canonical JSON result envelope on stdout | 1 | every stage, `app.py` |
| 5 | Error-translation layer for terminal failures | 1 | `utils/gemini.py`, `app.py` |
| 6 | Append-only run log (`runs.jsonl`) for provenance | 1 | `pipeline/state.py` |
| 7 | Explicit per-stage `--revert` | 1 | `stages.py`, all stages |
| 8 | Post-OCR correction as its own stage, with metrics | 2 | new stage, `fix_entries.py` |
| 9 | Image preprocessing stage (binarize/deskew) | 2 | new stage, `utils/image_utils.py` |
| 10 | OpenSeadragon + Annotorious for the review UI | 2 | `review_alignment.py` |
| 11 | CHANGELOG, devcontainer, `ruff format` | 2 | repo hygiene |

**Suggested first three:** (1) YOLO/DocLayNet spike as the `detect_sections` producer,
(4)+(6) the JSON envelope and run log together, (3) evaluate LiteLLM *inside* the HF plan
rather than writing per-model backends.

---

## Context: what Lorya is, and why the architecture doesn't transfer

Lorya is a full-stack web platform for document processing: React 19 + NestJS 11 +
PostgreSQL + TypeORM, with Python AI scripts invoked as subprocesses through a conda
environment. It targets interactive, multi-user, human-in-the-loop correction of
layout regions and OCR text on (mostly) Serbian Cyrillic newspapers and magazines
from the 1890s–1930s. Turborepo monorepo, Yarn 4 workspaces.

This pipeline is a single-user batch CLI over IIIF sources, with file-on-disk state and
zero server infrastructure. So the *architecture* is not transferable — we do not want
NestJS, Postgres, TypeORM migrations, JWT auth, or conda. What is transferable is a set
of specific techniques, several data-model ideas, and a handful of process conventions.

A useful framing: Lorya is **stronger on interaction, provenance, and model
pluggability**; this repo is **stronger on batch orchestration, IIIF fidelity,
reliability, and testing**. The recommendations below are almost entirely in the first
three categories.

---

# Tier 1 — genuinely worth doing

## 1. YOLOv8 + DocLayNet for layout and section detection

**Lorya:** `lorya:scripts/ai/app/layout/run_yolo.py`, with a checked-in
`scripts/ai/app/layout/models/yolo/yolov8m-doclaynet.pt` (~50 MB) loaded at module
import (`run_yolo.py:12-13`). Eleven DocLayNet classes (`run_yolo.py:21-34`):

> Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture,
> Section-header, Table, Text, Title

Each detection is emitted with a confidence score, an `xyxy` bounding box, and a
`labelType` of `"text"` / `"non-text"` derived from a class allowlist
(`run_yolo.py:35,43`).

**Why it matters here.** `docs/plans/section-detection-plan.md` states that the
consumer side of section detection is fully wired (`utils/section_utils.py` →
`load_sections`, `section_for_page`, `prompt_for_page`, `is_section_boundary`) and
that **only the producer is missing**. DocLayNet-class layout detection is a credible
producer signal:

- `Section-header` / `Title` detections mark candidate section boundaries;
- `List-item` density distinguishes alphabetical listing pages from prose front matter;
- `Table` detections flag classified / street-directory sections;
- `Page-header` / `Page-footer` give running heads, which are a strong section signal
  in city directories specifically.

It is also a better signal than the current column detector.
`pipeline/detect_columns.py` uses vertical pixel-projection profiles (documented in its
module docstring, following Bell et al. 2020 / directoreadr). That method degrades
exactly where directory pages are hardest: a headline spanning columns, an inset
illustration, or a heavy binding shadow all flatten the gutter dips, which is why the
stage carries an explicit `low` confidence band. Clustering detected text-block
x-ranges sidesteps the whole failure mode.

Finally it fits the stated direction in `docs/plans/huggingface-uv-scripts.md`: local,
free, no API call, and far smaller and faster than the Chandra / Qwen-VL options under
consideration there.

**Caveat, stated plainly.** DocLayNet's training corpus is modern — financial reports,
scientific papers, patents, manuals, laws, government tenders. Historical directory
pages are out of distribution, and a pretrained `yolov8m-doclaynet` checkpoint should be
treated as a hypothesis, not a solution. Two mitigations: Lorya applies it to 1890s–1930s
newsprint and evidently gets usable output, which is weak but real evidence of transfer;
and the plan doc already prescribes heuristic-first with a model at the margins, so YOLO
would slot in as the margin rather than the backbone.

**Shape of the work.** Mirror `detect_columns.py` exactly as the plan doc instructs: an
opt-in stage that analyzes every page, writes a report artifact, touches no images,
declares itself in `pipeline/stages.py` with a `--force` guard. The YOLO dependency
belongs in the existing `[gpu]` extra alongside Surya, with the
`requires=` / `install_hint=` gating that `StageDef` already supports.

## 2. Crop-to-region, then OCR

**Lorya:** `lorya:scripts/ai/app/layout/crop_segments.py` takes the YOLO segment JSON,
validates each box against image bounds, crops with OpenCV, and writes one JPEG per
segment — then OCR runs per crop rather than per page.

**Today here:** `pipeline/run_gemini_ocr.py` sends whole page images to Gemini, and
`pipeline/align_ocr.py` then reconciles the returned text against Surya line boxes with
Needleman-Wunsch over the full page.

**What cropping buys:**

- **Fidelity on dense multi-column pages.** A crop has one column's worth of context,
  so the model is not choosing a reading order.
- **Reading order becomes ours.** Currently it is whatever the model inferred.
- **Per-region prompts.** The per-section prompt routing in `utils/section_utils.py`
  (`prompt_for_page`) already exists; per-region is the natural refinement.
- **Failure isolation.** One bad region no longer truncates a whole page — which is
  precisely the failure mode `PRO_THINKING_BUDGET` in `utils/gemini.py:42` was
  introduced to contain (MAX_TOKENS at 210 of 257 lines on a dense 1841 trades page).
- **Cheaper calls**, since each image is smaller.
- **Easier alignment.** Gemini text arrives already scoped to a known bbox, so the NW
  alignment works in a small coordinate space with far fewer candidate mismatches.

**Caveat:** this only works as well as the layout detection in (1), and it adds a
failure mode of its own — a region the detector misses is text that never gets OCR'd.
A whole-page fallback pass, or a coverage check (detected region area vs. ink area) is
needed before this could replace the current path rather than sit beside it.

## 3. LiteLLM as the model-backend abstraction

**Lorya:** `lorya:scripts/ai/app/llm/run_litellm.py` — a single ~130-line script fronting
any LiteLLM-supported provider. Config is a JSON file passed by path
(`lorya:apps/backend/src/llm/llm-config.ts:4-13`): `model`, `apiKey`, `defaultPrompt`,
optional `apiBase`, `parameters`, `outputFormatPrompt`. The model string carries its
provider prefix (`gemini/gemini-2.5-flash`, `openai/gpt-4o`, `ollama/...`).

**Why it matters here.** `docs/plans/huggingface-uv-scripts.md` proposes per-model
backends for Chandra, NuExtract3, Qwen3.5, GLiNER. LiteLLM would make *hosted* local
models (served via Ollama or vLLM, which is how you'd run them in production anyway)
a **config change rather than new code**. That is a materially cheaper path to the same
goal for the OCR and NER call sites. It does not help for GLiNER or any model you'd call
in-process rather than over an endpoint — those still need real backends.

**Two things to verify before committing**, both of which this repo depends on today:

1. **Flex service tier.** `utils/gemini.py:58` (`flex_http_options`) sends
   `service_tier` through `HttpOptions.extra_body` — the reason `pyproject.toml`
   pins `google-genai>=1.21` with a long explanatory comment. Flex is on by default
   across `pipeline run`/`guided`/`ocr`/`extract`, so losing it is a ~2× cost
   regression, not a nicety.
2. **Thinking budget.** `utils/gemini.py:45` (`thinking_config_for`) sets
   `thinking_budget=0` for flash models and a 4096 cap for pro-tier. The measured
   justification is in the `PRO_THINKING_BUDGET` docstring. LiteLLM's abstraction over
   provider-specific reasoning params would need to preserve this.

**Recommended shape:** keep `google-genai` as the default path; add LiteLLM behind an
optional extra (`uv sync --extra local`, same pattern as `[gpu]` and `[geo]`) for local
and alternative backends. Not a replacement — an additional backend.

**Worth stealing regardless of LiteLLM: the split prompt.**
`lorya:apps/backend/src/llm/llm-config.ts:103` (`buildFullLlmPrompt`) composes the final
prompt at runtime as **task prompt + output-format prompt**, kept as two separately
editable fields, with defaults per task in
`lorya:apps/backend/src/llm/llm-output-format.ts`. This repo's `generate_prompt.py`
emits a monolithic `ner_prompt.md` mixing task instructions and output schema. Splitting
them means a user (or a per-collection override) can rewrite the task half without
risking the JSON contract — directly relevant to the "calibrate once, run many" pattern,
where prompts get hand-edited and reused across volumes in a series.

Also note `sanitizeLlmConfig` (`llm-config.ts:116`) strips the API key before the config
is ever returned to the UI. Not needed here — env-var auth via `get_client()` is already
the safer pattern — but worth remembering if `app.py` ever surfaces model config.

## 4. Canonical JSON result envelope on stdout

**Lorya:** every Python entry point prints exactly one JSON object to stdout with a
uniform status block. From `lorya:scripts/ai/app/ocr/run_tesseract.py:77-91`:

```json
{
  "imageId": "...", "inputPath": "...",
  "status": { "success": true, "messageCode": "PROCESSING_SUCCESS", "messageText": "..." },
  "lang": "srp", "script": "cyrillic",
  "lines": [ { "line_id": 1, "words": [ { "word_id": 1, "word_text": "...", "word_confidence": 92.4 } ] } ],
  "statistics": { "avg_word_confidence": 87.3 }
}
```

Failures use the same envelope with `success: false` and a non-zero exit
(`lorya:scripts/ai/app/llm/run_litellm.py:120-131`). `_to_canonical`
(`run_litellm.py:78`) exists specifically to force a *different* backend's output into
that same shape — which is what makes the backends interchangeable at all.

**Why it matters here.** `app.py` currently **regex-scrapes subprocess output** because
stages print prose:

- `app.py:44` — `_SERVER_URL_RE = re.compile(r"http://127\.0\.0\.1:\d+\S*")`, used at `app.py:583`
- `app.py:46` — `_OUTPUT_SLUG_RE = re.compile(r"\boutput/([^/\s]+)")`, used at `app.py:608`

A `--json` flag on each stage emitting a stable envelope (`status`, `outputs`, `metrics`,
`errors`) would delete both regexes and give the dashboard real structured status instead
of string-matching log lines. It pairs directly with (6): the envelope is what gets
appended to the run log.

**Also worth copying:** `lorya:apps/backend/src/ai/script-execution.error.ts` — a
`ScriptExecutionError(message, stdout, stderr, exitCode)` that preserves all four fields
rather than flattening a subprocess failure into a string.

**One anti-lesson.** `lorya:apps/backend/src/llm/llm-script-error.utils.ts:1-33`
(`parseScriptStdoutJson`) has a three-stage fallback — try the whole stdout, then scan
lines backwards for one starting with `{`, then find the last `{` — because LiteLLM
prints banner lines to stdout. That is a workaround for a library polluting the payload
channel. The actual rule is **payload to stdout, everything else to stderr**, which
`utils/gemini.py`'s `log=` callable already defaults to
(`log or (lambda msg: print(msg, file=sys.stderr))`). Keep that discipline and the
fallback is never needed.

## 5. Error-translation layer for terminal failures

**Lorya:** `lorya:apps/backend/src/llm/llm-script-error.utils.ts:41` (`formatLlmScriptError`)
maps raw provider errors to actionable operator text:

| Detected | Message given to the user |
|---|---|
| `429` / rate limit / quota / `resource_exhausted` | rate limit or quota exceeded — check provider plan (`:60`) |
| provider + not provided/required | include the provider prefix, e.g. `gemini/gemini-2.5-flash` (`:67`) |
| `404` / `model_not_found` | model not found — verify the name in config (`:72`) |
| `no module named litellm` | LiteLLM not installed in the Python env (`:78`) |
| `api key` / `401` / unauthorized | authentication failed — check the key (`:83`) |

**Here:** `utils/gemini.py` already classifies the same error families —
`_is_rate_limit` (`:65`), `_is_unavailable` (`:70`), `_is_internal` (`:75`) — but only to
decide **whether to retry**. Once `generate_with_retry` exhausts its budget the raw
exception propagates, and what the operator sees is a `google.genai` traceback.

**Proposal:** a small `explain_gemini_error(exc) -> str` in `utils/gemini.py` reusing the
existing predicates, plus auth/model-not-found/quota cases. Every stage gets better
failure output for free, `app.py` gets something displayable, and it costs ~30 lines.

## 6. Append-only run log for real provenance

This is the largest structural gap, and the recommendation with the longest tail of
downstream value.

**Lorya** records two levels. Per operation,
`lorya:apps/backend/src/activity/activity.entity.ts`:

| Field | Note |
|---|---|
| `fileId`, `userId` | who and what |
| `category` | `MODEL_RUN` vs `MANUAL_OPERATION` (`enums/activity-category.enum.ts:2-3`) |
| `operation` | 25-value enum incl. every manual edit and revert |
| `status` | `IN_PROGRESS` / `SUCCESS` / `FAILURE` |
| `modelType`, `modelId` | which model produced it |
| `startedAt`, `finishedAt`, `durationMs` | `:85-93` |
| `exitCode`, `errorMessage` | `:97-101` |
| `metadata` | JSON, e.g. the rotation angle or crop rect (`:104`) |
| `modelRunId` | links to the aggregate run |

Per run, `lorya:apps/backend/src/model-run/model-run.entity.ts`: a monotonic `runId`
per model (`:36`, uniquely indexed at `:12`), `selectionCount` (`:40`),
`executionStatus`, `resultStatus` (`:57`), `aggregateConfidence` (`:61`),
`durationMs` (`:73`).

**Here:** `pipeline/state.py` writes a single `pipeline_state.json` per slug with
`stages_completed` and one `last_run_args` string. The docstring is explicit that it
captures *what* a stage used, not *how* it went. Consequences:

- Only the **most recent** write survives — `write_state` merges and overwrites (`:88-105`).
- **Per-page timing, cost, and failures are lost entirely.** A run where 40 of 400 pages
  failed and a run where all 400 succeeded leave identical state.
- **Nothing records manual edits** except the `"manual"` confidence marker written into
  `*_aligned.json` by the review UI.

> **Incidental finding.** `CLAUDE.md` states that "both `main.py` and the OCR/align/extract
> leaf scripts write it". The align part is not true at present: `pipeline/align_ocr.py:65`
> imports only `get_ocr_model` and never calls `record_stage`, so a direct
> `python pipeline/align_ocr.py …` invocation leaves `stages_completed` without an
> `align_ocr` entry. Only `main.py:1248`, `pipeline/run_gemini_ocr.py:624`, and
> `pipeline/extract_entries.py:1519` record stages. Either the call is missing or the
> docs are — worth resolving before building anything on top of stage state.

**Proposal, fitted to the file-based design:** an append-only `runs.jsonl` alongside
`pipeline_state.json` in each slug directory — one line per stage execution:

```json
{"stage": "gemini_ocr", "model": "gemini-3.1-flash-lite", "started": "...", "finished": "...",
 "pages_ok": 382, "pages_failed": 18, "pages_skipped_blank": 4, "argv": "...",
 "metrics": {"input_tokens": 0, "output_tokens": 0}, "error": null}
```

`pipeline_state.json` keeps its current role as the current-state summary that downstream
scripts read — no breaking change to `get_ocr_model` / `get_ner_model` / `find_state_dir`,
and `record_stage` gains an append alongside its existing merge. What it unlocks:

- **Cost and time reporting per volume**, feeding `docs/costs.md` with measured rather
  than estimated numbers.
- **Failure triage** — which pages failed, when, with what error.
- **A real history view** for the `app.py` dashboard.
- **Model comparison** without the gitignored, local-only
  `collections/greenbook/model_eval.py`.

**The `MODEL_RUN` vs `MANUAL_OPERATION` distinction matters here specifically.** Green
Book output is published downstream through `hadro.github.io/green-books`. Being able to
state which entries were human-corrected versus model-generated is a provenance claim
worth being able to make in the published CSV — and right now the information exists only
as an undifferentiated `"manual"` string inside intermediate alignment JSON that never
reaches the output schema.

## 7. Explicit per-stage revert

**Lorya:** `lorya:apps/backend/src/activity/enums/activity-operation.enum.ts:20,26-30`
defines `SEGMENT_REVERT`, `LAYOUT_IDENTIFICATION_REVERT`, `IMAGE_ENHANCEMENT_REVERT`,
`OCR_REVERT`, `POST_OCR_REVERT`, `FILE_RESET`, with service implementations at
`lorya:apps/backend/src/segment-management/segment-management.service.ts:278,329`.

**Here:** `--force` recomputes, but there is no "undo this stage for this volume and
restore the previous artifact". The most recent commit on `main` is
**`8412dbb Preserve manual alignments across a --force re-align`** — the absence is
already being worked around, one stage at a time, at the point where a destructive
recompute meets hand-corrected data.

**Proposal:** a per-stage `--revert` that removes the stage's outputs and drops its entry
from `stages_completed`, with manual-edit-bearing artifacts (`*_aligned.json` with
`"manual"` confidence) either preserved or explicitly requiring `--revert --hard`. This
generalizes `8412dbb` instead of repeating it per stage.

---

# Tier 2 — worth considering

## 8. Post-OCR correction as its own stage, with metrics

**Lorya** models post-OCR correction as a first-class stage with its own model, its own
prompt, its own output-format contract
(`lorya:apps/backend/src/llm/llm-output-format.ts`), and its own statistics.

**Here** correction is split across two places, neither of which improves the OCR *text*:
the NER prompt cleans up implicitly during extraction, and `analysis/fix_entries.py`
normalizes CSV *fields* after extraction. For abbreviation-dense directory lines a cheap
pass over the OCR text before NER — regex and dictionary first, LLM optional — is a
plausible quality win, and unlike the current arrangement it would be measurable in
isolation.

**Take the shape, not the code.** `lorya:scripts/ai/app/post_ocr/run_postocr.py` is a
stub: the parsing body is `# TODO regex` (`:28`) and the statistics are hardcoded
literals (`:32-33`, `"cer": 20.2, "wer": 10.4`). It defines an interface, not an
implementation.

**The metrics half is the better idea.** Lorya carries `avg_word_confidence` in every OCR
envelope (`run_tesseract.py:70-74`), `cer`/`wer` in post-OCR, and `aggregateConfidence`
per run. This repo captures Surya per-line confidence but drops it after alignment, and
has no accuracy metric at all in tracked code — the only one lives in
`collections/greenbook/model_eval.py`, which is gitignored and absent from a fresh clone.
Promoting a small `metrics` block into the stage envelope (4) and the run log (6) would
make model comparison a tracked, reproducible capability rather than a local script.

## 9. Image preprocessing stage

**Lorya:** `lorya:scripts/ai/app/image_enhancement/adaptive_thresholding.py` (Gaussian
adaptive threshold, `blockSize=21`, `C=10`), plus
`lorya:scripts/ai/app/image_processing/` for rotate, crop, brightness, contrast,
sharpness, and a combined `adjust_all.py`. Originals are preserved and the file is
flagged `imageModified` (`lorya:apps/backend/src/ai/ai.service.ts`, `markImageModified`).

**Here:** `utils/image_utils.py` has exactly one page-level heuristic — `is_blank_page`
(`:9`), well-reasoned and conservative, but purely a skip check. Nothing improves an
image before OCR.

For microform-sourced volumes, binarization and deskew before Surya/Gemini is a
well-established win in the HTR/OCR literature. An optional `--enhance` stage writing
`*_enhanced.jpg` **alongside untouched originals**, declarative in `stages.py`, with
downstream stages preferring the enhanced variant when present, would fit the existing
architecture without disturbing it.

**A loose thread worth pulling while here.** This repo already detects microform sources
— `_is_microform` at `sources/loc_collection_csv.py:163` and
`sources/ia_collection_csv.py:128` — but the signal is derived from *catalog metadata* at
ingest and, as far as I can tell, never reaches the OCR stages. Microform scans are
exactly the population an enhancement pass should target, so plumbing that flag through
to `pipeline_state.json` would give the stage a principled default (enhance microform,
leave native scans alone) instead of an all-or-nothing switch.

**Caveat:** aggressive binarization destroys information that a multimodal model can
otherwise use. Gemini OCR on a grayscale scan may well beat Gemini OCR on an
over-thresholded bitonal one, even where Surya prefers the latter. This should be
per-stage opt-in and measured, not applied globally.

## 10. OpenSeadragon + Annotorious for the review UI

**Lorya:** `lorya:shared/ui/src/components/ImageAnnotator/ImageAnnotator.tsx` wraps
Annotorious v3 (`@annotorious/react ^3.0.17`) over OpenSeadragon 5
(`lorya:shared/ui/package.json:21,36`), with zoom/pan and fullscreen hooks (`:938,947`),
label overlays, colour-by-`labelType`, and conversion helpers between a simple
`{bounds, label}` shape and W3C-style annotations — `RECTANGLE` selector (`:102`),
bodies with `purpose: "tagging"` (`:94`).

**Here:** `pipeline/review_alignment.py` is hand-rolled Flask using
`render_template_string` with custom box drawing, and `pipeline/visualize_alignment.py`
burns boxes into static PNGs.

**The fit is unusually good**, because this repo is already IIIF-native in ways Lorya is
not. We already emit W3C Annotation Pages (`pipeline/iiif/export_annotations.py`) and
canvas fragments with `#xywh=`. OpenSeadragon consumes IIIF image services directly, so
the review UI would not need to serve local images at all; Annotorious speaks W3C
annotations natively, so the round trip is `*_aligned.json` → annotations → edit → write
back. On 4000px scans that is a categorically better experience than absolutely
positioned divs over an `<img>`.

**Caveat:** Lorya gets this through React + Vite + Yarn workspaces. This repo ships
zero-build, self-contained HTML, and that property is worth protecting — the explorer
output is meant to be handed to people. The version to pursue is CDN `<script>` tags
inside the existing `render_template_string` page, not a frontend build pipeline.

## 11. Repo hygiene: changelog, devcontainer, formatting

**CHANGELOG.** `lorya:CHANGELOG.md` follows Keep a Changelog, mirrored as a
"Latest release" section at the top of `lorya:README.md`. This repo is at
`version = "0.2.0"` in `pyproject.toml` with no tags and no changelog — and its output is
consumed by a second repo (`hadro/green-books`), so "what changed since I last ran this"
is currently unanswerable by anyone including us. Cheap, and the value compounds.

**Devcontainer.** `lorya:.devcontainer/Dockerfile` pins Ubuntu 24.04, Node 22.18.0,
Yarn 4.12.0, and an Anaconda install with architecture detection. Lower urgency here —
`uv` already provides most of the reproducibility, and Lorya needs *three*
`environment.yml` variants (`environment.yml`, `environment.cross-platform.yml`,
`environment.linux.yml`) precisely because conda env exports carry build strings. **Do not
copy the conda part; the uv setup here is strictly better.** A thin devcontainer pinning
Python 3.11 + uv + system libs would still help Codespaces and cloud sessions.

**Formatting and commits.** Lorya runs Prettier + ESLint + Husky + lint-staged +
commitlint (conventional commits) — `lorya:package.json`. This repo deliberately runs
ruff with `select = ["F", "E9"]` and no style rules (`pyproject.toml:50-52`).
**That decision should stand.** The narrowly transferable piece is `ruff format --check`
in CI — deterministic formatting without style *lint* rules — and only if wanted.
Conventional commit messages would make a generated CHANGELOG nearly free, which is the
pairing that makes the item above cheap.

> Incidental finding while checking this: `pyproject.toml:51` points at
> `docs/refactor-plan.md` for the rationale, but that file does not exist in the repo.
> Either it was never committed or it was removed — worth either restoring it or moving
> the rationale inline, since it is currently a dangling citation for a deliberate
> design decision. (`docs/key-design-decisions.md` may be the intended home.)

---

# Where this pipeline is ahead — do not converge

Listed because a comparison that only flows one way is not a review.

| Area | Here | Lorya |
|---|---|---|
| **ALTO export** | `pipeline/export_alto.py` — ALTO v3, proper namespace, real per-word boxes from aligned Surya data, `--line-strings` option | `lorya:scripts/ai/app/export/export_to_alto.py` — bare `ET.Element("alto")` with no namespace or `MeasurementUnit` (`:27`); word widths estimated as `max(20, len(word) * 10)` (`:86`); line heights by dividing block height evenly |
| **Retry / backoff** | `utils/gemini.py:82` — independent backoff per error class (429/503/500) with separate attempt budgets and measured rationale | Flat 120 s timeout (`lorya:apps/backend/src/llm/llm.constants.ts`), no retry |
| **Stage definition** | `pipeline/stages.py` — one declarative registry driving `main.py` argv, `app.py` buttons, and gating; drift guarded by `tests/test_app_arg_defs.py` | Same flags hand-written in the Python script, the NestJS service, the DTO, and the React form — four places, no drift test |
| **Environment** | `uv` + lockfile + optional extras | conda, with three `environment.yml` variants to work cross-platform |
| **Testing** | 22 test files; CI runs lint, pytest, CLI help smoke tests, and a dry-run subcommand matrix against a stub volume (`.github/workflows/ci.yml`) | A handful of `.spec.ts` files; one Python test (`test_run_litellm.py`) |
| **IIIF** | Native throughout — manifest walking, canvas fragments, Annotation Pages, ranges, ALTO for Content Search | No IIIF; filesystem + Postgres paths |

The honest summary: Lorya's strengths are the things a multi-user web application forces
you to get right (provenance, revert, pluggable backends, structured inter-process
contracts). Its weaknesses are the things a batch pipeline forces you to get right
(reliability under rate limits, reproducible environments, format fidelity, test
coverage). The recommendations above are an attempt to take the former without giving up
the latter.

---

## Appendix: reference index

**Lorya** (`UNDP-Serbia/lorya_` @ `c94a8be`):

- `scripts/ai/app/layout/run_yolo.py` — DocLayNet YOLOv8 layout detection
- `scripts/ai/app/layout/crop_segments.py` — segment cropping
- `scripts/ai/app/llm/run_litellm.py` — provider-agnostic LLM runner, canonical envelope
- `scripts/ai/app/ocr/run_tesseract.py` — OCR envelope with per-word confidence
- `scripts/ai/app/post_ocr/run_postocr.py` — post-OCR stage interface (stub)
- `scripts/ai/app/image_enhancement/adaptive_thresholding.py` — binarization
- `scripts/ai/app/export/export_to_alto.py` — ALTO export
- `apps/backend/src/llm/llm-config.ts` — LLM config schema, split prompt composition
- `apps/backend/src/llm/llm-output-format.ts` — per-task output-format prompts
- `apps/backend/src/llm/llm-script-error.utils.ts` — error translation
- `apps/backend/src/ai/ai.service.ts` — Python subprocess invocation
- `apps/backend/src/ai/script-execution.error.ts` — structured subprocess error
- `apps/backend/src/activity/activity.entity.ts` + `enums/` — per-operation provenance
- `apps/backend/src/model-run/model-run.entity.ts` — per-run aggregates
- `shared/ui/src/components/ImageAnnotator/` — Annotorious + OpenSeadragon
- `CHANGELOG.md`, `.devcontainer/Dockerfile`, `package.json`, `turbo.json`

**This repo:**

- `docs/plans/section-detection-plan.md` — the plan item (1) and (2) serve
- `docs/plans/huggingface-uv-scripts.md` — the plan item (3) revises
- `pipeline/stages.py` — declarative stage registry any new stage must join
- `pipeline/state.py` — current state model that (6) extends
- `pipeline/detect_columns.py` — projection-profile column detection (1) would improve
- `pipeline/run_gemini_ocr.py`, `pipeline/align_ocr.py` — the OCR path (2) restructures
- `pipeline/review_alignment.py`, `pipeline/visualize_alignment.py` — what (10) replaces
- `pipeline/iiif/export_annotations.py` — existing W3C annotation output (10) reuses
- `pipeline/export_alto.py` — ALTO v3 export
- `utils/gemini.py` — client, retry, flex, thinking config; touched by (3) and (5)
- `utils/section_utils.py` — section consumer contract (1) must not break
- `utils/image_utils.py` — `is_blank_page`; extended by (9)
- `app.py:44,46` — the stdout regexes (4) removes
- `analysis/fix_entries.py` — post-extraction normalization (8) sits upstream of
