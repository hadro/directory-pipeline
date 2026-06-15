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

## Assets
- `screenshots/` — images embedded in the README and docs.
