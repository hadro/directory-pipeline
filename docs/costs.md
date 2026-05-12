# Estimated costs

Two cost categories: **API charges** (variable; applies on any platform) and **platform costs** (compute infrastructure).

---

## Gemini API

`--gemini-ocr` and `--extract-entries` both call the Gemini API. Pricing as of May 2026 (verify current rates at [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)):

| Stage | Model (default) | Input (standard) | Output (standard) |
|---|---|---|---|
| `--gemini-ocr` | `gemini-2.5-flash` | $0.30 / 1M tokens | $2.50 / 1M tokens |
| `--gemini-ocr` (budget option) | `gemini-3.1-flash-lite` | $0.25 / 1M tokens | $1.50 / 1M tokens |
| `--extract-entries` | `gemini-3.1-flash-lite` | $0.25 / 1M tokens | $1.50 / 1M tokens |
| fallback (dense pages) | `gemini-2.5-flash` | $0.30 / 1M tokens | $2.50 / 1M tokens |

**Flex inference (`--flex`):** Add `--flex` to `--gemini-ocr` or `--extract-entries` for approximately **50% off** the standard rates above, with 1–15 minute latency per request and best-effort availability. Flex works well for bulk runs where real-time throughput is not needed. Combine with `gemini-3.1-flash-lite` for the lowest possible cost:

```bash
python main.py URL --gemini-ocr --ocr-model gemini-3.1-flash-lite --flex
```

A Green Book page generates roughly 2,000 input tokens and 1,000 output tokens for OCR (`gemini-2.5-flash`, approx. $0.0031/page standard, $0.0016/page with `--flex`), and another approx. 10,000 input / 2,000 output tokens for entry extraction (`gemini-3.1-flash-lite`, approx. $0.0055/page standard, $0.0028/page with `--flex`) — about **$0.0086 per page** combined at standard rates, or **$0.0043/page** with `--flex`. Dense pages that exceed the output token limit automatically retry with `gemini-2.5-flash`, but this affects fewer than 5% of pages in practice.

`--generate-prompts` makes 2 Gemini calls with 4–8 sample images each — a one-time per-volume cost of roughly **$0.02–$0.10 total**.

**Rough collection estimates:**

| Collection | Pages | Prompt gen | OCR | NER | Total (standard) | Total (`--flex`) |
|---|---|---|---|---|---|---|
| One Green Book volume | 80 pages | $0.05 (one-time) | $0.25 | $0.45 | $0.70 | $0.35 |
| Full Green Books corpus (23 volumes) | 1,900 pages | $0.05/volume | $5.90 | $10.50 | $16 | $8 |
| Large city directory (500+ pages) | 500 pages | $0.05 (one-time) | $1.55 | $2.75 | $4.30 | $2.15 |

`--flex` cuts OCR and NER costs by approximately 50% (prompt generation excluded — too few calls to matter). Observed cost for NER-only across all 23 Green Book volumes with `--flex` and `gemini-3.1-flash-lite`: **$5**.

**`--mode multimodal` for `--extract-entries`:** adds approx. 258 input tokens per page (one image tile at ≤768 px). At `gemini-3.1-flash-lite` standard rates that is roughly **+$0.0001/page** — effectively negligible. The increase is more noticeable with higher-capability models (e.g. `gemini-2.5-flash` at $0.30/1M input: approx. +$0.00008/page). Multimodal is particularly effective for materials with mid-page geographic or section heading changes, where the visual layout context significantly improves extraction accuracy. See the [README](../README.md#multimodal-extraction) for when to enable it.

**Optional higher-capability models for `--extract-entries`:**

| Model | Input | Output | Per page (NER only) | 1,900 pages | 1,900 pages + `--flex` |
|---|---|---|---|---|---|
| `gemini-3.1-flash-lite` (default) | $0.25/1M | $1.50/1M | $0.0055 | $10.50 | $5.25 |
| `gemini-3-flash-preview` | $0.50/1M | $3.00/1M | $0.011 | $21 | $10.50 |
| `gemini-3.1-pro-preview` | $2.00/1M | $12.00/1M | $0.045 | $85 | $42.50 |

**Free tier:** The Gemini API free tier (no billing required) covers both models at no charge, subject to rate limits of 15 requests/minute and approx. 1,500 requests/day. A single 100-page volume (approx. 200 API calls total) fits comfortably within a single day's free quota, though the 15 RPM cap means the API stages take 15–20 minutes rather than a few minutes. For a full multi-volume corpus you will either need billing enabled or spread the run across several days.

**Chandra OCR** (`--chandra-ocr`) uses a local 5B model — no API key or API cost. Requires GPU.

---

## Google Maps Geocoding (optional)

The `--geocode` stage uses Nominatim (free, city-level accuracy) by default. Setting `GOOGLE_MAPS_API_KEY` enables address-level geocoding at roughly $0.005/request. Google Maps includes a $200/month free credit, which covers approx. 40,000 geocoding requests — more than the entire Green Books corpus.

---

## Platform costs

The stages that use significant compute are **Surya OCR** (`--surya-ocr`, `--surya-detect`, `--review-alignment`) and **Chandra OCR** (`--chandra-ocr`). Gemini API stages are network-bound and run equally fast everywhere.

| Platform | Cost | Surya OCR (200 pages) | Notes |
|---|---|---|---|
| **Mac (M-series, 16 GB+)** | $0 (electricity) | 5–8 min (MPS, `--batch-size 4`) | Good for development and single-volume runs |
| **Mac (8 GB)** | $0 | 10–15 min (MPS, `--batch-size 1–2`) | Works; reduce batch size if OOM errors occur |
| **Google Colab (free T4)** | $0 | 2–3 min (CUDA, `--batch-size 8`) | Sessions expire; T4 not always available; `--review-alignment` requires a tunnel (e.g. ngrok) |
| **Google Colab Pro** | $10/month | 1–2 min (T4/L4, `--batch-size 8`) | Reliable GPU access, longer sessions |
| **Google Colab Pro+** | $50/month | <1 min (A100, `--batch-size 16`) | Background execution; best for large multi-volume runs |

The pipeline is designed so the compute-heavy steps (Surya OCR, interactive alignment review) can run on a GPU machine while everything else (downloading, Gemini OCR, entry extraction, geocoding, map generation) runs fine on a laptop.

A ready-to-run Colab notebook covering Surya OCR, alignment, and review is in [`colab/ocr-align-review.ipynb`](../colab/ocr-align-review.ipynb).
