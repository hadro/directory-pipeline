# Local OCR backend (`--local-ocr`) — design + validation plan

Created 2026-06-16. Operationalizes the OCR half of `docs/plans/huggingface-uv-scripts.md`
(Phase 0 decision record). **Status: implemented as an opt-in stage; Phase-1 alignment
validation on real pages is still required locally before relying on it for a large batch.**

## Context

The goal is to run OCR on **dozens of city directories** without paying per Gemini call,
on a **MacBook Air M2 (16 GB)**, happy to run **overnight for several nights** — cost and
"runs locally" matter, throughput does not. The OCR must capably capture **multi-column
city-directory layouts** so it aligns cleanly in the existing `align_ocr.py` step.

### Recommendation

**Keep Surya for geometry; swap only the text source (Gemini → local). No single best
engine — Chandra primary, Apple Vision as a fast baseline/fallback.**

The pipeline already splits OCR into *Surya = per-line bounding boxes* and *text engine =
words*, merged by Needleman–Wunsch in `align_ocr.py`. `align_ocr.py` / `extract_entries.py`
discover OCR purely by the `{stem}_{slug}.txt` filename contract, so a new engine that emits
that filename is a drop-in with zero downstream changes.

- **Chandra OCR 2 (primary):** 5B Qwen3.5-based layout-aware VLM (Datalab — same vendor as
  Surya), MLX 8-bit build `mlx-community/chandra-ocr-2-8bit-mlx` (~5–6 GB resident, comfortable
  on 16 GB). As an instruction-following VLM it honors the existing OCR prompt's "left column
  top-to-bottom, then right column; one printed line per output line." Slow on an M2 Air — fine
  for overnight.
- **Apple Vision via `ocrmac` (baseline/fallback):** native macOS, near-instant, near-zero
  memory, free, but traditional OCR (weaker on degraded scans) that returns observations with
  **no semantic column reading order** — reconstructed spatially from its bboxes.
- **Surya's own `*_surya.txt`** is a third free baseline already produced — worth comparing.

**The load-bearing risk is reading-order/alignment, not raw character accuracy** — which is
exactly what Phase-1 validation checks.

## What was implemented

| File | Change |
|------|--------|
| `pipeline/run_local_ocr.py` | **New** stage. Pluggable `--engine {chandra,vision}`; reuses `run_gemini_ocr` helpers (`_load_scope`, `_find_prompt`, `_output_issue`, `_clean_output`, `_OCR_FAILED_PLACEHOLDER`, `_DITTO_INSTRUCTION`) and `is_blank_page`; same skip/empty/placeholder write logic and `{stem}_{slug}.txt` contract. Chandra: cached `mlx-vlm` model load + plain-text prompt guard + conservative `_to_plain_lines` markdown/HTML stripper. Vision: `ocrmac` → Y-flip to top-left pixel bboxes → `sort_by_reading_order` (imported from `align_ocr.py`) → newline-joined text. |
| `utils/models.py` | `DEFAULT_LOCAL_OCR_MODEL`, `LOCAL_OCR_SLUGS = {"chandra":"chandra-ocr-2","vision":"apple-vision"}`; `_TXT_RX` generalized from gemini-only to `(?:gemini\|chandra\|apple)-…` (still excludes `_surya.txt` / `included_pages.txt`). |
| `pipeline/stages.py` | `StageDef("local_ocr", …, "--local-ocr", requires="mlx_vlm", install_hint="uv sync --extra local-ocr")`, placed between `gemini_ocr` and `compare_ocr`. |
| `main.py` | `--local-ocr` stage flag; `--ocr-engine {chandra,vision}` option; records `ocr_model = LOCAL_OCR_SLUGS[engine]` in `pipeline_state.json` even when `--ocr-model` is unset. |
| `pyproject.toml` | `local-ocr` extra (`mlx-vlm`, `ocrmac` — Apple-Silicon/macOS-only); added to `all`. |
| `tests/` | Updated `test_models.py` (`_TXT_RX` now accepts local backends) and `test_stages_registry.py` (`local_ocr` requires `mlx_vlm`). |

Slug is fixed per engine (not derived from the checkpoint) so filenames stay clean and
downstream discovery is unaffected by quantization suffixes.

## Phase 1 — validate alignment quality on real pages (do this locally first)

Run on Brooklyn city-directory pages plus the existing Green Book / Tulsa baselines — dirs
that already have Surya JSON **and** a Gemini baseline to compare against:

- `output/lain_healy_s_brooklyn_directory_for_the_1897BPL/` — dense two-column alphabetical
  (the make-or-break reading-order test).
- `output/hearnes_brooklyn_city_directory_for_hearnesbrooklync1852/` — earlier/flatter
  (register + ads) contrast.
- a few Green Book + Tulsa pages already on disk (the original validation set).

```bash
uv sync --extra local-ocr
# pick ~10 pages spanning a dense two-column body page, a column-break/heading page, and an ad page
python -m pipeline.run_local_ocr output/<slug>/ --engine chandra        # → *_chandra-ocr-2.txt
python -m pipeline.align_ocr     output/<slug>/ --model chandra-ocr-2 --force
# eyeball *_chandra-ocr-2_aligned.json vs the Gemini *_aligned.json:
#   compare unmatched-line counts and possible_column_merge / needs_review flags;
#   optionally --visualize to draw boxes. Gate: alignment ≥ ~90% with clean column order.
python -m pipeline.run_local_ocr output/<slug>/ --engine vision          # → *_apple-vision.txt (proves Y-flip + reading-order reconstruction)
python -m pipeline.align_ocr     output/<slug>/ --model apple-vision --force
```

Tune the Chandra prompt guard + `_to_plain_lines`, and **pin the validated `mlx-vlm` /
`ocrmac` versions** in `pyproject.toml` based on what actually worked.

## End-to-end (after Phase 1 passes)

```bash
python main.py output/<slug>/ --local-ocr --ocr-engine chandra
python main.py output/<slug>/ --align-ocr --extract-entries
# confirm pipeline_state.json records ocr_model: chandra-ocr-2,
# *_chandra-ocr-2_aligned.json + entries_chandra-ocr-2.csv exist,
# and discover_ocr_slug() finds the slug with the state file removed.
```

## Risks

- **Reading order > accuracy.** Vision: the Y-flip convention and the 2-column heuristic are
  the failure points (validate visually). Chandra: markdown/HTML leakage and column-collapse
  that doesn't match Surya's per-line bboxes (watch `possible_column_merge`).
- **Chandra plain-text steering** may need prompt iteration; `_to_plain_lines` is deliberately
  conservative so dot-leaders/abbreviations in addresses aren't corrupted.
- **mlx-vlm API churn / HF repo drift** — `run_local_ocr._chandra_ocr` calls `generate`
  defensively across kwarg renames, but pin the exact version + model revision once validated.
- **16 GB is comfortable** for the 8-bit Chandra quant; a base **8 GB** Air is marginal
  (swap/OOM) and should prefer Apple Vision.
- Preflight gates on `mlx_vlm` only; a `--engine vision`-only user still installs the full
  extra (acceptable for v1). Both deps are macOS-only, so the extra is kept out of default/CI.
