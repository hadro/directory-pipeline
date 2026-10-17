# directory-pipeline docs

Documentation index. **Reference** docs describe the system as it is and are kept
current. **Plans** are point-in-time roadmaps that may be unbuilt or partially built —
check each plan's `Status:` line before relying on it.

## Reference
- [pipeline-stages.md](pipeline-stages.md) — full per-stage reference: every flag, artifact, and the [naming contract](pipeline-stages.md#artifacts-and-naming-conventions).
- [usage-examples.md](usage-examples.md) — worked end-to-end command examples by source (LoC / IA / IIIF).
- [key-design-decisions.md](key-design-decisions.md) — why the pipeline is built the way it is.
- [costs.md](costs.md) — per-stage API and platform cost breakdown.

## Background
- [prior-work.md](prior-work.md) — related research and annotated citations.
- [comparison-htr-alto-pipeline.md](comparison-htr-alto-pipeline.md) — feature comparison vs. UVA Law's HTR-ALTO-Pipeline.

## Plans
Point-in-time roadmaps. Implementation is deferred per each file's `Status:` line.

- [plans/section-detection-plan.md](plans/section-detection-plan.md) — auto-draft `sections.txt` for multi-section directories so per-section NER prompts route correctly. *Status: Planned.*
- [plans/huggingface-uv-scripts.md](plans/huggingface-uv-scripts.md) — replace the Gemini OCR/NER steps with local open models via Hugging Face uv-scripts. *Status: Planned.*
- [plans/aws-migration-plan.md](plans/aws-migration-plan.md) — port the pipeline to run entirely on AWS (Bedrock + Location Service). *Status: Planned.*
- [plans/degenerate-page-reocr.md](plans/degenerate-page-reocr.md) — re-OCR the 21 Green Book pages with runaway Gemini output, which needs per-page model selection. *Status: Deferred.*
- [plans/gutter-hoist-fix.md](plans/gutter-hoist-fix.md) — stop `plan_columns()` hoisting long mid-page entries as page-spanning headers. *Status: Deferred; not started.*
- [plans/explorer-improvements.md](plans/explorer-improvements.md) — turn the generated explorer into a deployable data publication. *Status: Proposed.*
- [plans/lorya-comparison.md](plans/lorya-comparison.md) — what to borrow from UNDP Serbia's Lorya. *Status: Recommendations.*
- [plans/jev-evaluation.md](plans/jev-evaluation.md) — whether TypeSafe AI's Jev fits the routing/gating layer (it cannot do OCR or NER). *Status: Evaluation.*
- [plans/release-0.3.md](plans/release-0.3.md) — the 0.3 landing plan and pre-release review. *Status: In progress.*

## Evidence
Measurements behind specific thresholds and decisions, each with its own README, scripts
and results. Most need the gitignored `output/` data to reproduce.

- [evidence/2026-09-02-ocr-token-burn/](evidence/2026-09-02-ocr-token-burn/) — the degenerate-OCR detector thresholds and the retry ladder.
- [evidence/2026-09-04-column-selection-criterion/](evidence/2026-09-04-column-selection-criterion/) — choosing between column and whole-page alignment. Read its 2026-10-17 erratum.
- [evidence/2026-09-22-gutter-hoist/](evidence/2026-09-22-gutter-hoist/) — mid-page lines hoisted as headers, and what it costs.
- [evidence/2026-10-16-blank-page/](evidence/2026-10-16-blank-page/) — the blank-page check that skipped pages with text, and its fix.
- [evidence/2026-10-17-column-alignment-panel/](evidence/2026-10-17-column-alignment-panel/) — per-column alignment on city directories, scored against hand-verified gold.

## Assets
- `screenshots/` — images embedded in the README and docs.
