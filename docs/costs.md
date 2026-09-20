# Estimated costs

Two cost categories: **API charges** (variable; applies on any platform) and **platform costs** (compute infrastructure).

---

## Gemini API

`--gemini-ocr` and `--extract-entries` both call the Gemini API.

**Do not estimate this — measure it.** Every Gemini response carries
`usage_metadata`; `generate_with_retry()` records it and the OCR and NER stages
persist the totals to `token_usage` in each volume's `pipeline_state.json`:

```bash
python tools/report_cost.py output/                  # every volume
python tools/report_cost.py output/my-volume/ --by-reason
python tools/report_cost.py output/ --pages 1900     # project onto a corpus
```

The figures below are for sizing a run before you start. Rates live in
`utils/models.py` (`MODEL_RATES`, checked `RATES_CHECKED`) so a model bump and its
price move together — the previous arrangement left this page quoting the old
models' rates for three months.

| model | input / 1M | output / 1M |
|---|---|---|
| `gemini-3.1-flash-lite` (OCR + NER default) | $0.25 | $1.50 |
| `gemini-3.5-flash-lite` | $0.30 | $2.50 |
| `gemini-3.7-flash` (prompt generation) | $0.75 | $3.75 |
| `gemini-3.8-flash` (escalation) | $0.75 | $3.75 |
| `gemini-3.5-flash` | $1.50 | $9.00 |

Checked 2026-09-18. The 3.7/3.8 rates are promotional through 2026-12-31 and
double on 2027-01-01. **Thinking tokens bill at the output rate**, and every
non-lite 3.x flash model has no thinking-off mode (see `thinking_config_for()` in
`utils/gemini.py`), so escalated pages cost more than their input/output counts
suggest.

**Flex and Batch are both 50% off.** Flex is on by default for `--gemini-ocr` and
`--extract-entries`; pass `--no-flex` when you need real-time throughput.

### Measured, not estimated

Three pages of the 1869 Rowell directory, `gemini-3.1-flash-lite`, flex, on
2026-09-18:

| | |
|---|---|
| per page | **$0.00183** |
| calls per page | 1.33 |
| projected, 1,900 pages | **$3.48** |

That 1.33 is the part an estimate cannot see. A page blocked for RECITATION and
retried at a higher temperature is billed twice, each attempt paying full input
tokens. `--by-reason` breaks the bill down by what provoked each call:

```
reason                    calls      output          cost   share
initial                       3       3,592  $     0.0035   63.7%
temp:recitation               1       2,299  $     0.0020   36.3%
```

The two failure classes cost very differently. A RECITATION block is refused
before generation — prompt tokens only, no output, ~$0.0006 a call. A runaway
*emits*, up to whatever ceiling it is given: at the 24,576 default that is ~$0.06,
roughly 11x a clean page. That asymmetry is why retries are capped against the
page's own demonstrated length (`_retry_cap` in `run_gemini_ocr.py`) and why loop
failures skip the temperature rungs entirely.

Caveat on the numbers above: three pages, one volume, one run. Treat them as an
order of magnitude and run `report_cost.py` on your own output.

NER has not yet been measured this way. Historically it ran roughly 3x the OCR
cost per page (approx. 10,000 input / 2,000 output tokens against OCR's 2,000 /
1,800), so budget accordingly until a real run records it.

`--generate-prompts` makes 2 Gemini calls with 4–8 sample images each — a one-time per-volume cost of roughly **$0.02–$0.10 total**.

**Rough collection estimates** (June 2026, `gemini-3.1-flash-lite`; the OCR column
now reads low against the $3.48 measured above, which carries a real retry rate):

| Collection | Pages | Prompt gen | OCR | NER | Total (standard) | Total (`--flex`) |
|---|---|---|---|---|---|---|
| One Green Book volume | 80 pages | $0.05 (one-time) | $0.16 | $0.44 | $0.60 | $0.30 |
| Full Green Books corpus (23 volumes) | 1,900 pages | $0.05/volume | $3.80 | $10.50 | $14 | $7 |
| Large city directory (500+ pages) | 500 pages | $0.05 (one-time) | $1.00 | $2.75 | $3.75 | $1.88 |

`--flex` cuts OCR and NER costs by approximately 50% (prompt generation excluded — too few calls to matter). Observed cost for NER-only across all 23 Green Book volumes with `--flex` and `gemini-3.1-flash-lite`: **$5**.

**`--mode multimodal` for `--extract-entries`:** adds approx. 258 input tokens per page (one image tile at ≤768 px). At `gemini-3.1-flash-lite` standard rates that is roughly **+$0.00006/page** — effectively negligible. The increase is more noticeable with higher-capability models (e.g. `gemini-3.1-pro-preview` at $2.00/1M input: approx. +$0.0005/page). Multimodal is particularly effective for materials with mid-page geographic or section heading changes, where the visual layout context significantly improves extraction accuracy. See the [README](../README.md#multimodal-extraction) for when to enable it.

**Optional higher-capability models for `--extract-entries`:**

| Model | Input | Output | Per page (NER only) | 1,900 pages | 1,900 pages + `--flex` |
|---|---|---|---|---|---|
| `gemini-3.1-flash-lite` (default) | $0.25/1M | $1.50/1M | $0.0055 | $10.50 | $5.25 |
| `gemini-3-flash-preview` | $0.50/1M | $3.00/1M | $0.011 | $21 | $10.50 |
| `gemini-3.1-pro-preview` | $2.00/1M | $12.00/1M | $0.044 | $84 | $42 |

**Free tier:** The Gemini API free tier (no billing required) covers both models at no charge, subject to rate limits of 15 requests/minute and approx. 1,500 requests/day. A single 100-page volume (approx. 200 API calls total) fits comfortably within a single day's free quota, though the 15 RPM cap means the API stages take 15–20 minutes rather than a few minutes. For a full multi-volume corpus you will either need billing enabled or spread the run across several days.

---

## Google Maps Geocoding (optional)

The `--geocode` stage uses Nominatim (free, city-level accuracy) by default. Setting `GOOGLE_MAPS_API_KEY` enables address-level geocoding at roughly $0.005/request. Google Maps includes a $200/month free credit, which covers approx. 40,000 geocoding requests — more than the entire Green Books corpus.

---

## Platform costs

The stages that use significant compute are **Surya OCR** (`--surya-ocr`, `--surya-detect`, `--review-alignment`). Gemini API stages are network-bound and run equally fast everywhere.

> **The table below is unverified for dense directory material and was wrong by
> roughly 30x on the one volume anyone has timed.** The 1869 Rowell directory
> (373 pages, M-series Mac, `--batch-size 4`) took about **6 hours of active
> compute at ~1 page/minute**, measured from output file timestamps, against the
> 5–8 minutes per 200 pages below. The cause is not established. Line density is
> a plausible contributor but not sufficient on its own: those pages carry a
> median of 143 Surya lines against 80 for a Green Book page, under 2x.
>
> Until someone re-times it, plan Surya as an overnight job for anything
> line-dense. It resumes cleanly — pages with an existing `*_surya.json` are
> skipped — so an interrupted run costs nothing but the wall-clock already spent.

| Platform | Cost | Surya OCR (200 pages) | Notes |
|---|---|---|---|
| **Mac (M-series, 16 GB+)** | $0 (electricity) | 5–8 min (MPS, `--batch-size 4`) | Good for development and single-volume runs |
| **Mac (8 GB)** | $0 | 10–15 min (MPS, `--batch-size 1–2`) | Works; reduce batch size if OOM errors occur |
| **Google Colab (free T4)** | $0 | 2–3 min (CUDA, `--batch-size 8`) | Sessions expire; T4 not always available; `--review-alignment` requires a tunnel (e.g. ngrok) |
| **Google Colab Pro** | $10/month | 1–2 min (T4/L4, `--batch-size 8`) | Reliable GPU access, longer sessions |
| **Google Colab Pro+** | $50/month | <1 min (A100, `--batch-size 16`) | Background execution; best for large multi-volume runs |

The pipeline is designed so the compute-heavy steps (Surya OCR, interactive alignment review) can run on a GPU machine while everything else (downloading, Gemini OCR, entry extraction, geocoding, map generation) runs fine on a laptop.

A ready-to-run Colab notebook covering Surya OCR, alignment, and review is in [`colab/ocr-align-review.ipynb`](../colab/ocr-align-review.ipynb).
