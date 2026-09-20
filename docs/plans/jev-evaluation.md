# Evaluation: is TypeSafe AI's Jev useful to this pipeline?

**Status:** Evaluation — no code written, nothing committed to. One proposed spike at the end.

> **What this is:** an assessment of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
> TypeSafe AI's "System One" model (launched 2026-09-15), against both this repo and the
> sibling [`city-directory-extraction`](../../../city-directory-extraction) repo. Created 2026-09-20.
>
> **Sourcing caveat:** `typesafe.ai` and `docs.typesafe.ai` were unreachable from the session
> that wrote this (blocked by an egress proxy), so the capability summary below is assembled
> from search results and third-party write-ups — Pydantic AI's provider docs, Cloudflare
> Workers AI's model page, LangChain's harness post, and community references. **Verify the
> primitives and limits against the official docs before writing any code against them.**
> One detail is reported inconsistently across sources: whether *constrained* string
> extraction (email, phone) is a first-class primitive or a Pydantic-side convenience.

---

## TL;DR

**Jev cannot do the thing this pipeline is about.** Directory NER emits verbatim strings
(`name`, `address`, `occupation_role`, `employer`) copied and transformed out of the source
line. Jev does not generate text and rejects unconstrained string fields before it sends a
request. It is not a replacement for `extract_entries.py`, not a replacement for the
fine-tuned Qwen in the sibling repo, and not a substitute for anything in
`docs/plans/huggingface-uv-scripts.md`.

**It is a candidate for the routing and gating layer around extraction** — the places that
are currently hand-tuned regex, an alphabetical-order heuristic, or nothing at all.

| # | Opportunity | Repo | Tier | Replaces / fills |
|---|---|---|---|---|
| 1 | Non-entry (prose / ad / heading) rejection before NER | extraction | 1 | nothing — documented open problem |
| 2 | `detect_sections` Phase 5 boundary adjudication | pipeline | 1 | the planned Gemini `flash-lite` call |
| 3 | Category inference residue | pipeline | 2 | `_INFER_RULES` fallthrough |
| 4 | Gold-panel semantic QA tier | extraction | 3 | adds a tier to `validate_gold.py` |
| 5 | Style-profile / publisher backfill over the 449-volume catalog | extraction | 3 | per-volume Gemini calls |

**Recommended first step:** #1 only, as a shadow run against `1906BPL` — measured, not wired.
See [The one spike worth running](#the-one-spike-worth-running).

---

## What Jev is

TypeSafe AI launched Jev on 2026-09-15 with $40M led by DCVC, built by Diogo Almeida
(co-author of RLHF / InstructGPT at OpenAI). It is the first of what they call **System One
models**: a model that makes fast typed decisions and *does not generate text at all*.

- **Non-autoregressive.** No token-by-token decoding. You send unstructured *state* plus typed
  *questions*; it answers all of them in a single parallel pass.
- **Three primitives:**
  - `Noul` — P(statement is true) ∈ [0,1]. The probability *is* the answer; no separate
    confidence field.
  - `Choice` — one of up to 255 predefined options. Returns a probability for *every* option
    plus an overall confidence.
  - `Score` — position on an ordered 2–10 level rubric. Returns a continuous score (it can
    fall between levels), the per-level distribution, and a confidence.
- **Calibrated**, via a training method they call Reinforcement Learning for Calibrated
  Decisions (RLCD). The probabilities are meant to be trusted as probabilities.
- **Claimed performance:** 40–200× faster than frontier LLMs on comparable tasks;
  $0.042 / M input tokens with output free (~400× cheaper on classification).

### Limits that matter here

- **Text only.** No image input. Non-text has to be pre-processed into text or structured
  fields first. (This pipeline already does that — OCR is upstream of every use below.)
- **~32k tokens** for state plus the longest question; ~64k total per request.
- **No free-form strings**, no nested models, no optional fields, no tool calling, no
  streaming, no arbitrary numbers. The Pydantic provider rejects these client-side.
- **No state accumulation between questions in one request.** If question B depends on
  A's answer, that is a second request.
- **Accuracy degrades as state grows with irrelevant content.** Unrelated detail acts as a
  distractor, and a large state makes it hard to tell which part produced a wrong answer.
- **Schema-constrained ≠ semantically correct.** TypeSafe says this themselves. "Cannot
  hallucinate" means it cannot emit an out-of-schema value. It can pick a valid wrong one.

Their own framing: *"Code calculates. Jev judges. Reasoning models reason and generate."*

---

## Where it does not fit

Worth stating explicitly, because the launch framing ("structured extraction", "never
hallucinates") reads like it is aimed at exactly this problem. It is not.

| Component | Why not |
|---|---|
| `pipeline/extract_entries.py` | Emits `name`, `address`, `category`, `phone` — unconstrained strings lifted from the page. Generation. |
| Qwen fine-tune (sibling repo) | 6 of its 8 fields are verbatim strings. The two weakest fields in the 4B release candidate — `name` and `address` — are exactly the string-copy fields. Jev is not a competitor to it. |
| `pipeline/run_gemini_ocr.py` | OCR is generation. |
| `pipeline/generate_prompt.py`, `analysis/review_entries.py` | Both produce prose. |
| `detect_columns.py`, `detect_spreads.py`, `select_pages.py` | Pixel-level and image-based. Jev is text-only. |
| `docs/plans/huggingface-uv-scripts.md` backends | That plan is about replacing *generation* (Chandra OCR, NuExtract3/Qwen3.5 NER) with local models. Jev is orthogonal — it cannot serve either role. |

Note the one real overlap with an existing plan: `huggingface-uv-scripts.md` floats an
`image_type` **enum** classification via NuExtract3, and `section-detection-plan.md` Phase 5
generalizes that to a `section_type` enum. *Those* are Jev-shaped. The OCR and NER halves
of the HF plan are not.

---

## Where it does fit

### 1. Non-entry rejection before NER — Tier 1, best fit in either repo

**Lives in:** `city-directory-extraction/data_prep/alpha_run_filter.py`,
`data_prep/detect_listing_bounds.py`.

The docstring in `alpha_run_filter.py` makes the case better than this document can:

> The model never refuses, so every non-entry that survives filtering becomes a fake person
> in the output. That makes page-type detection a CORRECTNESS problem, not a cost problem.
> And no line-level rule can catch it: "courts of law or equity in" is lowercase, ASCII,
> normal height, normal width, sitting on a common left margin. It passes every shape test
> because it looks exactly like an entry.

And `detect_listing_bounds.py` documents, with measurements, that the alphabetical-order
signal does not solve interior ads:

> Measured on 1906BPL: genuine listing leaf 13 carries a sort key on 34% of its lines; ad
> leaf 819 carries one on 38%. The ad page scores HIGHER. So no threshold on modal share
> separates them at leaf granularity.

Both tools are explicit that this residue is unsolved and that `--interior drop` buys
precision by discarding ~172 genuine listing leaves.

**The Jev shape.** A `Noul` per line: *"Is this line an entry in an alphabetical residential
listing, as opposed to prose, display advertising, or a section heading?"* State is the line
plus a small window of neighbours. This is a semantic judgment, which is precisely what no
shape rule and no ordering rule can make — and it is the one place in either repo where the
current answer is "we know this is broken and we do not have a fix."

**Why the economics work.** A directory line is ~15–25 tokens. At $0.042/M input with free
output, gating every line of a 1,254-leaf volume costs cents; the full 449-volume catalog
stays in the low tens of dollars. Compare `docs/costs.md` for the Gemini equivalent. Latency
matters too: 40–200× faster than an LLM call means this can run as a synchronous filter
rather than as a batch stage.

**How to wire it.** As a *gate in front of* the Qwen extractor, not as a replacement for the
heuristics. Keep the alphabetical-run signal as the cheap prior; send only what it flags
AMBIGUOUS, plus interior pages inside a letter block, to Jev. Threshold on the probability —
do not take the boolean.

### 2. `detect_sections` Phase 5 adjudication — Tier 1

**Lives in:** `docs/plans/section-detection-plan.md:274-282`.

Phase 5 already specifies this shape: *"For pages with low classification margin or near a
candidate boundary only, escalate to a `section_type` enum classification"*, with "one Gemini
`flash-lite` call per ambiguous page" as the cheap-now backend.

That is a `Choice` over `{alphabetical, street, business, frontmatter, ads}` — five options,
well inside the 255 limit. Jev is a better fit than the planned Gemini call on two counts:

- **It returns a probability per label.** Phase 2's smoothing step (`MIN_RUN`,
  `SMOOTH_WINDOW`) wants confidences to smooth over, not hard labels. A Gemini enum call
  gives you a string; Jev gives you the distribution the smoother actually wants.
- **It respects the plan's own network seam.** `section-detection-plan.md:307` requires
  Phases 0–4 to run with no API key, with Phase 5 the only networked part. Jev slots into
  exactly that gated position, as a third interchangeable backend alongside Gemini and the
  local NuExtract3 option.

**Caution.** A full directory page of OCR is several thousand tokens. That fits in 32k, but
the "large state acts as a distractor" warning bites hardest on exactly this use — a page of
dense name listings contains a great deal of text irrelevant to *what kind of page it is*.
Feed line-shape summaries, or the first N lines plus a density/heading feature dump, rather
than the raw page. The Phase 0 feature extractor already computes most of that.

### 3. Category inference residue — Tier 2

**Lives in:** `analysis/fix_entries.py:303` (`_INFER_RULES`), `:377` (`_MARKER_RULES`).

~40 hand-ordered regexes mapping business names to Green Book categories, with documented
ordering fragility — YWCA must be tested before YMCA so the shared `MCA`/`WCA` shapes cannot
cross-match; bare `resort` must follow `summer resort`; first match wins throughout.

The target vocabulary is `CANONICAL_CATEGORIES` — closed and small. That is a `Choice`.

**How to wire it.** Keep the regexes as the fast path. Route only the residue — entries whose
category is `""`, `"General"`, or the `Hotels - Motels - Tourist Homes - Restaurants`
catch-all, *and* which no rule matched — to Jev. The calibrated probability preserves the
current policy stated in `CLAUDE.md` ("does not force-assign a default for non-matching
entries"): you threshold on a real number instead of on regex coverage, and genuinely
ambiguous entries stay empty rather than being forced into a bucket.

Note that `section-detection-plan.md:193` observes these same rules are shared with section
classification, so an improvement here serves both.

### 4. Gold-panel semantic QA — Tier 3

**Lives in:** `city-directory-extraction/data_prep/validate_gold.py`.

Its WARNING tier is described as "likely a transcription/splitting mistake worth a human
glance" but is implemented structurally. A `Noul` per gold line — *"does this record
faithfully transcribe this printed line?"* — adds a semantic tier over a structural checker.

Volume is 1,583 lines, so cost is irrelevant. The value is catching label errors that
structural checks cannot, and that project has been bitten twice: the NYU derived-label
problem (three fields synthesized by regex, not transcribed) and the stale-contract Gemini
bar. Both were measurement artifacts that survived structural validation.

### 5. Style-profile and publisher backfill — Tier 3

**Lives in:** `city-directory-extraction/data_prep/style_profiles`,
`backfill_publisher.py`, `backfill_contributor.py`, `master_directories.csv` (449 volumes).

Assigning a volume to a publisher×era profile from its title and metadata is a `Choice` over
a closed profile set. Cheaper and more auditable than a per-volume Gemini call, and the
calibration tells you *which* of the 449 need human eyes rather than requiring a full pass.

`is_business` — the only boolean in the 8-field schema — is worth a mention and a dismissal:
the 4B model already scores 0.97–0.98 F1 on it. Not worth replacing. It is, however, the one
field where a calibrated second opinion during gold QA would be meaningful, since the fine-
tune's own output carries no usable confidence.

---

## Risks

**Maturity.** Five days old at the time of writing, with early access via a waitlist.
Nothing here belongs in an unattended path (`pipeline run`, `postprocess`) until it has a
track record. Every proposal above is opt-in and gated, matching how
`section-detection-plan.md` already treats its own model-backed phase.

**The "cannot hallucinate" claim, read carefully.** This repo's sibling has a hard-won
methodological lesson — recorded at length in its README — that a gain which vanishes under
re-measurement was a measurement artifact, not a result. "Mathematically cannot hallucinate"
invites precisely that error. It means Jev cannot return an out-of-schema value. It says
nothing about whether the value is right. For opportunity #1 the distinction is the whole
game: a confidently-wrong "yes, this is an entry" on a page of law-office prose reproduces
the exact failure `alpha_run_filter.py` exists to prevent, and it would arrive wearing a
type-safe wrapper.

**Vendor lock at the routing layer.** `docs/plans/lorya-comparison.md` recommends LiteLLM as
a backend abstraction so model choice stays swappable. Jev's API shape (state + typed
questions, no chat completion endpoint) does not fit that abstraction — it would be a
separate client. Integrations exist for Pydantic AI, BAML, LangChain, OpenRouter and
Cloudflare Workers AI, so it is not a closed door, but it is a second abstraction to
maintain. Weigh that against a Gemini `flash-lite` enum call, which needs no new dependency
and reuses `utils/gemini.py`'s retry helpers.

**Feature coupling.** Opportunities 1 and 2 both want good per-page/per-line features to keep
the state small. Those features are Phase 0–1 of the section-detection plan and do not exist
yet. Doing the deterministic work first is a prerequisite, not a detour — and if the
deterministic features separate the classes cleanly (the Phase 0 question), the case for
Jev at the margins gets *smaller*, not larger. That is the right order to find out in.

---

## The one spike worth running

Everything above is speculative until one number exists. The cheapest way to get it:

**Shadow-run a `Noul` gate on `1906BPL`, line by line, and compare against the
alphabetical-run labels from `alpha_run_filter.py`.**

Why this one:

- The data is already cached, already line-segmented (`data/1906BPL_lines.jsonl`), and
  already has heuristic labels to compare against.
- The interesting cases are pre-identified: leaf 13 vs leaf 819, the three largest
  letter-block gaps (M→N 88 leaves, S→T 58, B→C 39), and everything the tool prints as
  AMBIGUOUS. Those are a small enough set to eyeball by hand.
- It is read-only and costs cents.
- **It has a real decision attached.** If Jev separates leaf 13 from leaf 819 — which no
  threshold on modal share does — that is a capability the repo does not currently have, on
  a problem its own docs class as correctness rather than cost. If it does not, the
  evaluation is over and this document is the record of why.

Measure precision and recall against the ~172 genuine listing leaves that `--interior drop`
currently sacrifices. The question is not "is Jev accurate" but "does it recover leaves the
precision cut throws away, without admitting prose."

Do **not** wire it into `stages.py`, `fix_entries.py`, or the extractor before that number
exists.

---

## Sources

Primary sources were unreachable from the authoring session; these are the pages the summary
above was assembled from. Re-verify against `docs.typesafe.ai` before implementing.

- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — the announcement (not directly readable from here)
- [System One concepts](https://docs.typesafe.ai/concepts/system-one)
- [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) — documented failure modes
- [Pydantic AI: TypeSafe (Jev) provider](https://pydantic.dev/docs/ai/models/typesafe/) — the client-side type restrictions
- [Cloudflare Workers AI: Jev](https://developers.cloudflare.com/ai/models/typesafe/jev/)
- [LangChain: building a harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev)
- [Jev Patterns: Choice, Score, Noul](https://jevpatterns.com/guides/choice-score-noul)
- [MarkTechPost launch coverage](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/)

---

## Working log

_(Append one entry per working session.)_

- 2026-09-20 — Evaluation written. No code. Next: the `1906BPL` shadow run described above,
  best done locally where `data/1906BPL_lines.jsonl` and the frontmatter cache are on disk.
