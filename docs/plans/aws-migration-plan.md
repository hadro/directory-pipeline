# Plan: Run directory-pipeline entirely on AWS

**Status:** Planned — decisions captured; implementation deferred.

## Context

The directory-pipeline currently depends on two Google services: **Gemini** (multimodal OCR,
NER extraction, and prompt generation — all via the `google-genai` SDK) and **Google Maps**
(geocoding). Gemini has no AWS-hosted offering, so an upcoming AWS-exclusive project cannot run
the pipeline as-is. This plan ports the pipeline to run entirely on AWS infrastructure, replacing
Gemini with **Amazon Bedrock (Claude)** and Google Maps with **Amazon Location Service**, while
preserving the pipeline's resumable, filesystem-staged, ~21-stage architecture.

Decisions captured from the user:
- **Model tier:** Tiered — Claude Sonnet as the default OCR/NER workhorse, escalate to Claude Opus
  on the fallback path (mirrors the existing `DEFAULT_*_MODEL` + `FALLBACK_MODEL` split).
- **AWS-only scope:** Geocode via Amazon Location (drop Google Maps); serve the Flask UIs +
  static explorer/map on AWS; **keep** IIIF source fetches off-AWS (LoC / Internet Archive / IIIF
  originals inherently live elsewhere — only processing moves to AWS).
- **Scale goal:** tens–hundreds of volumes in parallel → target **AWS Batch + Step Functions**
  (not SageMaker) for the scalable topology; start on a single GPU EC2/container for validation.

The intended outcome: the same CLI and stage outputs, with zero non-AWS LLM/geo calls, authenticated
by IAM (no API keys), backed by S3, scalable to hundreds of parallel volumes.

> Note on model IDs below: the exact Bedrock model / inference-profile IDs (region prefix such as
> `us.`, and version suffix) must be confirmed per-region against the Bedrock console when
> implementing. The IDs here are illustrative of the *tier* choice, not literal strings to paste.

---

## Strategy: a thin provider adapter, not a rewrite

Every Gemini call funnels through `get_client()` / `generate_with_retry()` in `utils/gemini.py`
(re-exported by `pipeline/api.py`). Call sites depend on only four things from `google-genai`:
1. those two functions (unchanged signatures),
2. a response object exposing `.text` and `.candidates[0].finish_reason`,
3. config/content builders: `GenerateContentConfig(...)`, `Part.from_bytes/from_text`,
   `ThinkingConfig`, `MediaResolution`, `FinishReason`,
4. the `model_slug()` filename contract in `utils/models.py`.

So the abstraction boundary is a **provider adapter inside one module**. Reimplement those helpers
and types over `boto3` `bedrock-runtime` (Converse API), keeping the same names/signatures. Call
sites then change **only their import lines** — their retry ladders, quality checks, JSON-recovery,
and context logic are untouched.

---

## Part A — Code changes

### A1. `utils/gemini.py` → `utils/llm.py` (the provider layer; highest-value change)
Rename to `utils/llm.py`; leave `utils/gemini.py` as a one-release re-export shim so `colab/` and
external notebooks keep working.

- **`get_client(required=True)`** → returns `boto3.client("bedrock-runtime", region_name=...)`.
  Drop the `GEMINI_API_KEY` check entirely (Bedrock uses IAM). Keep `load_dotenv()` only for
  optional `AWS_REGION` / model-override env vars. On missing credentials, catch
  `botocore.exceptions.NoCredentialsError` and emit the same fatal-message style.
- **Local shim types (same names so call sites don't change):** define `GenerateContentConfig`
  (dataclass holding `system_instruction`, `temperature`, `max_output_tokens`, `media_resolution`,
  `thinking_config`, `http_options`), `Part` (`from_bytes`/`from_text` → normalized internal blocks),
  `ThinkingConfig`, `MediaResolution` (enum incl. `MEDIA_RESOLUTION_HIGH`), `FinishReason` (enum incl.
  `RECITATION`, `STOP`, `MAX_TOKENS`). ~40 lines total.
- **`generate_with_retry(...)`** — same signature. Internally:
  - Translate `contents` → Converse `messages`: `Part.from_bytes(image/jpeg)` →
    `{"image":{"format":"jpeg","source":{"bytes":...}}}`; text / bare `str` → `{"text":...}`.
  - Translate `config` → `system=[{"text":...}]` + `inferenceConfig={temperature, maxTokens}`.
    `thinking_config=ThinkingConfig(thinking_budget=0)` → `additionalModelRequestFields={"thinking":
    {"type":"disabled"}}`. Append a "return only the transcription / only the JSON, no commentary"
    clause to OCR/NER system prompts to suppress any leaked reasoning (they mostly say this already).
  - Add a **prompt-cache breakpoint** on the (large, volume-stable) system prompt — near-free cost win.
  - Call `client.converse(modelId=model, ...)`.
  - **Retry mapping** (replaces the `"429"`/`"503"` string matching): catch
    `botocore.exceptions.ClientError`; `ThrottlingException`/`TooManyRequestsException` → the existing
    429 backoff loop; `ModelTimeoutException`/`ServiceUnavailableException`/`InternalServerException`/
    `ModelNotReadyException` → the existing 503 loop; else re-raise. Keep both independent
    doubling-delay loops exactly (behavior the pipeline was tuned around).
  - Return a small **response adapter** exposing `.text` (joined text blocks) and `.candidates =
    [_Candidate(finish_reason=...)]`, mapping Converse `stopReason` → `FinishReason`. **Key nuance:**
    Bedrock has no `RECITATION`; map `content_filtered → RECITATION` (closest analog) and treat
    empty text + `end_turn` as genuinely blank (no retry) — this keeps the OCR retry ladder intact.
  - If dense-page NER (`max_output_tokens=65536`) times out, switch that path to `converse_stream`
    and accumulate **inside** `generate_with_retry` so call sites don't change.
- **`flex_http_options(service_tier)`** — keep the function (call sites unchanged) but return `None`
  for the synchronous path. The cheap/async tier is handled at orchestration level (A6), since
  Bedrock Batch Inference is an S3-JSONL job API, not a per-request flag.

Route the two direct-call bypasses through `generate_with_retry` too: `generate_prompt._call_gemini`
(`pipeline/generate_prompt.py:246`) and `pipeline/compare_ocr.py:100`.

### A2. `utils/models.py` — model IDs + slug/filename contract (subtle, resumability-critical)
- Tiered IDs (confirm exact Bedrock IDs per region):
  - `DEFAULT_OCR_MODEL` = Claude **Sonnet**
  - `DEFAULT_NER_MODEL` = Claude **Sonnet**
  - `DEFAULT_PROMPT_MODEL` = Claude **Opus** (prompt generation is a once-per-collection-type calibration step where quality matters most and volume is tiny, so the cost premium is negligible)
  - `FALLBACK_MODEL` = Claude **Opus** (escalation when OCR output-quality checks fail)
- `model_slug()` does `model.replace("/", "_")`; Bedrock IDs contain dots, not slashes, so slugs
  embed cleanly in filenames — **but the discovery regexes hard-code `gemini-`** and will silently
  fail to match. Generalize them to produce/parse a short prefix-free slug (e.g. `claude-sonnet`):
  - `utils/models.py:_TXT_RX` (hard-codes `gemini-`)
  - `pipeline/extract_entries.py:_detect_aligned_slug` (two regexes, `gemini[-\w.]+`)
  - `pipeline/geo/geocode_entries.py` auto-detect uses `entries_(.+)\.csv` — already generic, OK.
- **Keep the old `gemini-*` regex branch alongside the new `claude-*` branch permanently** in
  `discover_ocr_slug` / `_detect_aligned_slug` so pre-migration `output/{slug}/` trees still resolve.
- `state.py` stores whatever model string is passed — no structural change.

### A3. Call sites — import-line changes only
- `pipeline/run_gemini_ocr.py` — imports now from `utils.llm`. Recitation branch works via the
  `content_filtered → RECITATION` mapping. `--high-res`/`MEDIA_RESOLUTION_HIGH` becomes "don't
  downscale" (advisory; Claude has native high-res vision). Quality apparatus untouched. Keep the
  file/stage name `gemini_ocr`/`--gemini-ocr` for now to avoid churning `stages.py`/`main.py`/`app.py`/docs.
- `pipeline/extract_entries.py` — imports only. Keep `_resize_image_bytes` (make `_MULTIMODAL_MAX_PX`
  a tunable constant) and the `_recover_partial_json` truncation safety net.
- `pipeline/generate_prompt.py` — `_call_gemini`/`_load_images` use `utils.llm`; route through
  `generate_with_retry` (gains free throttling backoff). Meta-prompts unchanged.
- `analysis/review_entries.py`, `analysis/compare_extraction.py`, `pipeline/compare_ocr.py` — imports only.
- `pipeline/api.py` — repoint re-exports to `utils.llm`; update docstring ("uses AWS credentials from
  the IAM role" instead of "reads GEMINI_API_KEY").

### A4. `--flex` semantics (orchestration: `cli/main.py`, `pipeline/stages.py`, `app.py`)
`--flex` meant "Gemini flex tier." Repurpose to "cheap/batch tier."
- **v1 (recommended):** make `--flex` a documented no-op (synchronous Converse only); note it's
  reserved for Bedrock Batch Inference. Lowest risk; matches the existing "Batch not yet implemented" note.
- **Scalable option:** add `pipeline/batch_runner.py` that, for the **NER stage only** (single call
  per page, truncation-recoverable), collects requests into a Bedrock Batch Inference JSONL on S3,
  submits `create_model_invocation_job`, polls, and writes outputs back. Reserve OCR for synchronous
  calls (the OCR retry ladder is inherently per-image/synchronous).

### A5. `pipeline/geo/geocode_entries.py` — Google Maps → Amazon Location
Keep `geocode_rows`, the `geocache.json` cache, dedup, and the address→city fallback levels intact.
- Replace `_geocode_google` with `_geocode_location_service`: `boto3.client("location")` (or
  `geo-places`), calling `search_place_index_for_text` against a pre-provisioned Place Index (Esri/HERE),
  or `geo-places:geocode` if standardizing on Places v2. Round lat/lon to 7 decimals; cache negatives.
- Replace the `GOOGLE_MAPS_API_KEY` check with detection of a configured Place Index name
  (`AWS_LOCATION_PLACE_INDEX` env var) + IAM creds.
- Per the user's "entirely AWS" choice: **drop the Nominatim/OSM fallback** — make city-level a second
  Amazon Location query instead. Rename `--google-delay` → `--location-delay`; keep the sleep + backoff.

### A6. `pyproject.toml`
Remove `google-genai`; add `boto3>=1.34`. Keep `python-dotenv` (region/override config). In the `geo`
extra, drop `geopy` (Nominatim no longer used). The `gpu` extra (`surya-ocr`, `transformers`) is
**unchanged** — Surya stays as the bbox detector.

### A7. Secrets / IAM
Delete `GEMINI_API_KEY` and `GOOGLE_MAPS_API_KEY` from code and `.env.template`. Bedrock + Location
authenticate via the instance/job IAM role (boto3 picks it up with zero code). Pipeline role policy:
`bedrock:InvokeModel` (+ `InvokeModelWithResponseStream` if NER streaming; `CreateModelInvocationJob`/
`GetModelInvocationJob` if Batch tier built), `geo:SearchPlaceIndexForText` (or `geo-places:Geocode`),
`s3:GetObject/PutObject/ListBucket` on the artifact bucket, `secretsmanager:GetSecretValue` only if a
residual private-IIIF credential survives. `load_dotenv()` becomes dev-only.

**Critical files:** `utils/gemini.py`→`utils/llm.py`, `utils/models.py`, `pipeline/extract_entries.py`,
`pipeline/run_gemini_ocr.py`, `pipeline/geo/geocode_entries.py` (plus supporting:
`pipeline/generate_prompt.py`, `pipeline/compare_ocr.py`, `analysis/compare_extraction.py`,
`analysis/review_entries.py`, `pipeline/api.py`, `pipeline/stages.py`, `cli/main.py`, `app.py`,
`pyproject.toml`, new `Dockerfile`, new `pipeline/storage_sync.py`).

---

## Part B — Infrastructure topology (two phases)

### Phase 1 — Single GPU EC2 + S3-backed artifacts (validate the migration)
A `g5.xlarge` (A10G, 24 GB) running the existing CLI verbatim — matches the one-orchestrator,
subprocess-stage, local-filesystem, resumable design with the least change.
- **Storage — `pipeline/storage_sync.py` (do NOT use s3fs).** Stages rely on POSIX semantics
  (sibling-file lookups, mtime-based cache invalidation in `extract_entries._load_cached_result`,
  `rglob`); s3fs would silently break cache-invalidation and resumability. Instead: EBS scratch holds
  `output/`; a thin wrapper runs `aws s3 sync` to pull `s3://bucket/output/{slug}/` before a run and
  push after each stage. `main.py`/`state.py` untouched. S3 is durable backing store + source of truth.
- **`Dockerfile` (none exists today):** CUDA base (`nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04`) so
  Surya's GPU path works; `uv sync --all-extras`. Same image runs orchestrator + all stages. Surya
  weights baked in or pulled from S3 on first run.
- **UIs:** Flask reviewers (`app.py` 5001, `pipeline/review_alignment.py`, `select_pages.py`) bind
  `0.0.0.0` behind an **ALB → EC2** with Cognito/OIDC auth (or SSM port-forward for internal use). The
  self-contained explorer + Leaflet map HTML → **S3 + CloudFront** (read-only, cached).

### Phase 2 — AWS Batch (GPU) + Step Functions (scale to 100s of volumes in parallel)
Chosen over SageMaker because the workload is heterogeneous-compute batch ETL (not training): Batch
maps 1:1 to the subprocess-per-stage model, isolates GPU cost to Surya, and supports Spot.
- **Step Functions** state machine mirrors the `pipeline/stages.py` DAG; a `Map` state fans out
  volumes. Each state runs `python -m pipeline.{stage}` as a container task.
- **AWS Batch compute envs:** a **GPU queue (g5, Spot)** for `surya_detect`/`surya_ocr` only; a **CPU
  queue** for `align_ocr` (CPU-bound Needleman-Wunsch), `download_images`, `detect_columns`, and the
  Bedrock-calling stages (`gemini_ocr`/`extract_entries` — Bedrock does the work; the container just
  calls). Surya is resumable, so Spot interruptions are cheap.
- **Storage:** reuse the same `storage_sync.py` — each Batch job syncs its `output/{slug}/` slice
  from/to S3 at start/end. (This is exactly why the sync-wrapper beats rewriting I/O: one wrapper
  serves both phases.)
- **Bedrock Batch Inference** plugs in here as the `--flex`/cheap tier for NER (a dedicated Step
  Functions branch submits the JSONL job and waits).

---

## Cost estimate — one 500-page city directory

Rates (per million tokens): **Sonnet $3 in / $15 out**, **Opus $5 in / $25 out**. Bedrock matches
Anthropic's published per-token rates (AWS sets them per region — confirm in the Bedrock pricing
console). **Batch inference = 50% off**; **prompt caching** serves cache hits at ~0.1× input.

> These are planning numbers. Per-page token counts vary a lot with page density (a sparse name
> column vs. a dense classified-ads page). Calibrate with `count_tokens` on a few real pages or a
> 10-page pilot before committing a budget. LLM tokens dominate the total — GPU/storage/geo are rounding error.

**Per-page token assumptions (defaults; tune after a pilot):**

| Stage | Input (image + system prompt + context) | Output |
|------|------|------|
| Bedrock OCR (multimodal, Sonnet) | ~4,500 tok (≈3,000 image + ~1,500 OCR prompt) | ~2,500 tok transcript |
| Bedrock NER (text-only, Sonnet) | ~4,200 tok (~2,500 transcript + ~1,500 NER prompt + context) | ~4,000 tok JSON entries |

**Per-page cost (Sonnet, standard pricing):**
- OCR: 4,500×$3/M + 2,500×$15/M = $0.0135 + $0.0375 = **$0.051**
- NER: 4,200×$3/M + 4,000×$15/M = $0.0126 + $0.060 = **$0.073**
- **≈ $0.124 / page → ~$62 for 500 pages** (LLM only).

**Full single-volume run (Phase 1, single g5 box):**

| Component | Standard pricing | Optimized (batch NER + prompt caching + Spot GPU) |
|---|---|---|
| Bedrock OCR + NER (Sonnet) | ~$62 | ~$33 (NER batched −50%; system-prompt cache-hit trims input) |
| Opus fallback escalations (~5% of pages re-run) | ~$2 | ~$2 |
| Surya GPU + orchestration (g5.xlarge, few hrs) | ~$5 (on-demand) | ~$1–2 (Spot) |
| Amazon Location geocoding (optional, if addresses) | ~$3 (≈5–10k lookups @ ~$0.50/1k, dedup-cached) | ~$3 |
| S3 storage + transfer (~0.5–1 GB) | <$1 | <$1 |
| **Total per 500-page volume** | **≈ $70** | **≈ $40** |

**Sensitivities:**
- **Max-quality (Opus as the primary OCR/NER model, not tiered):** Opus is ~1.67× Sonnet per token → LLM ≈ $103, total **≈ $110/volume**. The tiered Sonnet-default choice is the main cost lever.
- **Batch latency tradeoff:** batching NER trades the −50% for hours of turnaround; OCR stays synchronous (the retry ladder needs it).
- **At scale (Phase 2, hundreds of volumes):** cost scales ~linearly in Bedrock tokens; Spot GPU + Bedrock Batch on NER bring the per-volume figure toward the ~$40 column. 100 volumes ≈ **$4,000** optimized, ~$7,000 standard — overwhelmingly Bedrock spend, so quotas (TPM/RPM) and the model tier matter more than instance choice.

---

## Part C — Trade-offs & risks
- **OCR quality parity (top risk):** `run_gemini_ocr.py`'s quality apparatus was tuned against Gemini
  failure modes, esp. the `RECITATION` finish reason Claude doesn't expose. The quality checks run on
  *output text* (repetition / dot-leader / long-line), so they still fire regardless of provider; only
  the empty-text recitation retry trigger is provider-dependent (`content_filtered → RECITATION` is an
  analog, not identical). Run a parity eval (Part D) and re-tune `_RETRY_TEMPERATURES`/thresholds if needed.
- **Bedrock Batch latency:** async (minutes–hours, S3 JSONL); cannot drive the synchronous per-image
  OCR ladder. Restrict to NER. Communicate that `--flex` no longer means a 1–15 min per-request tier.
- **Slug/filename migration:** new `claude-*` filenames won't auto-discover old `gemini-*` trees unless
  both regex branches are retained — so retain them permanently.
- **Thinking-disabled verbosity:** Opus/Sonnet may leak reasoning with thinking off; the "plain text /
  valid JSON only" prompt clauses guard this — verify in the eval.
- **Cost:** Sonnet is pricier per token than gemini-flash-lite. Mitigate with prompt caching on the
  stable system prompt (Converse cache point) and reserving Opus for the fallback tier only.
- **Region/quota:** Bedrock model access must be enabled per-region; per-model TPM/RPM quotas now
  exercise the 429 backoff against `ThrottlingException` — confirm quotas cover the `--workers`
  concurrency (default 4–8 in `run_gemini_ocr.py`).

---

## Part D — Verification (end-to-end)
1. **Unit (mocked boto3):** in `tests/test_api.py`, assert `generate_with_retry` builds the right
   Converse payload from `Part`/`GenerateContentConfig`, maps `ThrottlingException`→429 backoff and
   `ServiceUnavailableException`→503 backoff, and that the response adapter exposes `.text` and
   `.candidates[0].finish_reason` with `content_filtered → RECITATION`. Assert `model_slug`/
   `discover_ocr_slug` round-trip new IDs and still find legacy `gemini-*` files.
2. **Single-page smoke:** run `run_gemini_ocr.py` + `extract_entries.py` on one known page against
   real Bedrock; confirm `.txt` and `_entries.json` with sane content and correct new slug filenames.
3. **Quality parity:** reuse `analysis/compare_ocr.py` / `analysis/compare_extraction.py` (Bedrock-backed)
   on held-out pages that previously hit the recitation/dot-leader/repetition paths; compare to
   archived Gemini outputs; re-tune thresholds on regression.
4. **Resumability:** run a small volume, kill mid-stage, restart; confirm `pipeline_state.json` +
   S3-synced `output/` resume and mtime cache-skip still works.
5. **Geocoding:** run `geocode_entries.py`; confirm Amazon Location address-level hits, city fallback,
   and `geocache.json` reuse (second run = zero network calls).
6. **Infra integration:** deploy the Docker image to the g5 instance with the IAM role (no API keys in
   env); confirm Bedrock/Location succeed via the role, Flask reviewers reachable via ALB, explorer/map
   served from CloudFront. Then validate one `Map` fan-out of a few volumes through Step Functions + Batch.

---

## Suggested implementation order
1. A1 provider layer (`utils/llm.py`) + shim types + retry mapping + response adapter; unit tests (D1).
2. A2 model IDs + slug regex generalization (keep legacy branch).
3. A3 call-site import swaps; single-page smoke (D2).
4. A5 Amazon Location geocoder.
5. A6/A7 deps + IAM + remove API keys.
6. Quality parity eval (D3); re-tune.
7. `Dockerfile` + `storage_sync.py`; Phase 1 EC2 deploy (D4–D6).
8. Phase 2 Batch + Step Functions for parallel scale.
