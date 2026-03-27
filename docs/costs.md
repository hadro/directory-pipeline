# Estimated costs

Two cost categories: **API charges** (variable; applies on any platform) and **platform costs** (compute infrastructure).

---

## Gemini API

`--gemini-ocr` and `--extract-entries` both call the Gemini API. Pricing as of early 2026 (verify current rates at [ai.google.dev/pricing](https://ai.google.dev/pricing)):

| Stage | Model (default) | Input | Output |
|---|---|---|---|
| `--gemini-ocr` | `gemini-2.0-flash` | $0.10 / 1M tokens | $0.40 / 1M tokens |
| `--extract-entries` | `gemini-3.1-flash-lite-preview` | $0.25 / 1M tokens | $1.50 / 1M tokens |
| fallback (dense pages) | `gemini-2.5-flash` | $0.30 / 1M tokens | $2.50 / 1M tokens |

A Green Book page generates roughly 2,000 input tokens and 1,000 output tokens for OCR (`gemini-2.0-flash`, ~$0.0006/page), and another ~10,000 input / 2,000 output tokens for entry extraction (`gemini-3.1-flash-lite-preview`, ~$0.0055/page) — about **$0.006 per page** combined. Dense pages that exceed the output token limit automatically retry with `gemini-2.5-flash`, but this affects fewer than 5% of pages in practice.

`--generate-prompts` makes 2 Gemini calls with 4–8 sample images each — a one-time per-volume cost of roughly **$0.01–$0.05 total**.

**Rough collection estimates:**

| Collection | Pages | Prompt generation | OCR | NER | Total |
|---|---|---|---|---|---|
| One Green Book volume | ~100 pages | ~$0.02 (one-time) | ~$0.06 | ~$0.55 | ~$0.61 |
| Full Green Books corpus (14 volumes) | ~1,400 pages | ~$0.02/volume | ~$0.84 | ~$7.70 | ~$8.54 |
| Large city directory (500+ pages) | 500 pages | ~$0.02 (one-time) | ~$0.30 | ~$2.75 | ~$3.05 |

**Free tier:** The Gemini API free tier (no billing required) covers both models at no charge, subject to rate limits of 15 requests/minute and ~1,500 requests/day for `gemini-2.0-flash`. A single 100-page volume (~200 API calls total) fits comfortably within a single day's free quota, though the 15 RPM cap means the API stages take ~15–20 minutes rather than a few minutes. For a full multi-volume corpus you will either need billing enabled or spread the run across several days.

**Chandra OCR** (`--chandra-ocr`) uses a local 5B model — no API key or API cost. Requires GPU.

---

## Google Maps Geocoding (optional)

The `--geocode` stage uses Nominatim (free, city-level accuracy) by default. Setting `GOOGLE_MAPS_API_KEY` enables address-level geocoding at roughly $0.005/request. Google Maps includes a $200/month free credit, which covers ~40,000 geocoding requests — more than the entire Green Books corpus.

---

## Platform costs

The stages that use significant compute are **Surya OCR** (`--surya-ocr`, `--surya-detect`, `--review-alignment`) and **Chandra OCR** (`--chandra-ocr`). Gemini API stages are network-bound and run equally fast everywhere.

| Platform | Cost | Surya OCR (200 pages) | Notes |
|---|---|---|---|
| **Mac (M-series, 16 GB+)** | $0 (electricity) | ~5–8 min (MPS, `--batch-size 4`) | Good for development and single-volume runs |
| **Mac (8 GB)** | $0 | ~10–15 min (MPS, `--batch-size 1–2`) | Works; reduce batch size if OOM errors occur |
| **Google Colab (free T4)** | $0 | ~2–3 min (CUDA, `--batch-size 8`) | Sessions expire; T4 not always available; `--review-alignment` requires a tunnel (e.g. ngrok) |
| **Google Colab Pro** | ~$10/month | ~1–2 min (T4/L4, `--batch-size 8`) | Reliable GPU access, longer sessions |
| **Google Colab Pro+** | ~$50/month | <1 min (A100, `--batch-size 16`) | Background execution; best for large multi-volume runs |

The pipeline is designed so the compute-heavy steps (Surya OCR, interactive alignment review) can run on a GPU machine while everything else (downloading, Gemini OCR, entry extraction, geocoding, map generation) runs fine on a laptop.

A ready-to-run Colab notebook covering Surya OCR, alignment, and review is in [`colab/ocr-align-review.ipynb`](../colab/ocr-align-review.ipynb).
