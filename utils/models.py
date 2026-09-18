"""Single source of truth for Gemini model defaults and model-slug filename logic.

Every stage that calls Gemini or parses model slugs out of output filenames
imports from here. Filename conventions (see docs/pipeline-stages.md):

    {page}_{imageid}_{model-slug}.txt           Gemini OCR text
    {page}_{imageid}_{model-slug}_aligned.json  aligned OCR
    entries_{model-slug}.csv                    extracted entries

The slug is ``model_slug(model)`` — the model name with "/" replaced by "_".
"""

import re
from collections import Counter
from pathlib import Path

# Default model for Gemini OCR (--gemini-ocr) and downstream filename defaults.
#
# Held at 3.1-flash-lite deliberately. 3.5-flash-lite was tried as the default on
# 2026-09-18 and reverted: it prices at $0.30/$2.50 against 3.1's $0.25/$1.50, so
# a measured page went from $0.0032 to $0.0051 — a 59% rise — for no quality gain
# that had been demonstrated. Raise it only with a measurement that justifies it.
DEFAULT_OCR_MODEL = "gemini-3.1-flash-lite"

# Default model for NER entry extraction (--extract-entries).
DEFAULT_NER_MODEL = "gemini-3.1-flash-lite"

# More capable model for one-off generation/review tasks
# (--generate-prompts, analysis/review_entries.py).
DEFAULT_PROMPT_MODEL = "gemini-3.7-flash"

# Escalation target for a page the primary model cannot transcribe cleanly
# (used by run_gemini_ocr.py and extract_entries.py).
#
# Not an output-budget escape hatch, whatever this comment used to say: every
# candidate model has the same 65,536-token output limit, and the escalated call
# reuses the same --max-output-tokens cap as the attempt that failed. What it
# buys is capability on layouts the lite tier mishandles.
#
# 3.8-flash over 3.5-flash on price: they share that output limit, but 3.5-flash
# is $1.50/$9.00 against 3.8-flash's $0.75/$3.75. The version numbers are not a
# price ordering. Escalation is already the costliest rung in the retry ladder —
# it also has no thinking-off mode, so it bills thought tokens at the output rate
# (see thinking_config_for in utils/gemini.py) — and 3.5-flash would roughly
# double it.
FALLBACK_MODEL = "gemini-3.8-flash"


# ---------------------------------------------------------------------------
# Pricing
#
# USD per 1,000,000 tokens at the standard tier, as (input, output).
#
# Source: ai.google.dev/gemini-api/docs/pricing, checked on RATES_CHECKED.
# Keep this beside the model constants: the last bump left docs/costs.md quoting
# the previous models' rates for three months, because the price lived in a
# document instead of next to the thing it prices.
#
# Only rates actually read off the pricing page belong here. A model that is
# missing is reported as unpriced by tools/report_cost.py, which is the right
# failure — an invented rate is worse than a gap.
#
# Thinking tokens bill at the OUTPUT rate, so a model with no "thinking off"
# mode (every non-lite 3.x flash — see thinking_config_for in utils/gemini.py)
# costs more per call than its input/output counts alone suggest.
RATES_CHECKED = "2026-09-18"

MODEL_RATES: "dict[str, tuple[float, float]]" = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    # 3.7/3.8 are promotional through 2026-12-31 and double on 2027-01-01
    # (to 1.50 / 7.50). Re-check this table in January.
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
}

# Flex and Batch are both half the standard rate.
TIER_MULTIPLIERS: "dict[str, float]" = {
    "standard": 1.0,
    "flex": 0.5,
    "batch": 0.5,
}


def price_usage(model: str, counts: dict, tier: str = "standard") -> "float | None":
    """Cost in USD for one {calls,input,output,thoughts,cached} record.

    Returns None when *model* has no entry in MODEL_RATES, so callers can say
    "unpriced" rather than report a number built on a guess.
    """
    rate = MODEL_RATES.get(model)
    if rate is None:
        return None
    rate_in, rate_out = rate
    billed_out = (counts.get("output", 0) or 0) + (counts.get("thoughts", 0) or 0)
    gross = (counts.get("input", 0) or 0) / 1e6 * rate_in + billed_out / 1e6 * rate_out
    return gross * TIER_MULTIPLIERS.get(tier, 1.0)


def model_slug(model: str) -> str:
    """Filesystem-safe slug for a model name, used in all output filenames."""
    return model.replace("/", "_")


# Aligned-JSON names are structurally unambiguous, so the model match is
# prefix-agnostic: the slug is the last "_"-delimited segment before
# "_aligned.json". (Slugs containing "_" — from models with "/" in the name —
# would be truncated; no such model is in use.)
_ALIGNED_RX = re.compile(r"_([^_]+)_aligned\.json$")

# Bare .txt names are ambiguous (*_surya.txt, included_pages.txt, …), so the
# model match requires the "gemini-" prefix.
_TXT_RX = re.compile(r"_(gemini-[^_]+)\.txt$")


def discover_ocr_slug(output_root: Path, require_txt: bool = False) -> str | None:
    """Scan *output_root* (and its immediate subdirectories) for OCR output
    files and return the most common model slug, or None if nothing is found.

    Aligned JSON files are preferred over bare OCR .txt files. This is the
    fallback used when neither a --model flag nor pipeline_state.json
    identifies the OCR model.

    Pass *require_txt* when the caller's required input is the OCR text rather
    than the aligned JSON — the aligner, above all. Without it a directory
    holding alignments from a model whose .txt files are absent (an earlier
    run's output, or a model retired since) resolves to that model, and every
    page then reports "missing" and is skipped. That failure is silent under
    ``--quiet``: the run logs clean having done nothing. It cost a full
    re-align of 1,448 files, including the volume holding most of this
    project's hand-corrected alignments.
    """
    if not output_root.is_dir():
        return None
    dirs = [output_root] + [d for d in sorted(output_root.iterdir()) if d.is_dir()]

    def _txt_slugs() -> set[str]:
        found: set[str] = set()
        for d in dirs:
            for f in d.glob("*.txt"):
                m = _TXT_RX.search(f.name)
                if m:
                    found.add(m.group(1))
        return found

    usable = _txt_slugs() if require_txt else None
    for pattern, rx in (("*_aligned.json", _ALIGNED_RX), ("*.txt", _TXT_RX)):
        counts: Counter[str] = Counter()
        for d in dirs:
            for f in d.glob(pattern):
                m = rx.search(f.name)
                if m and (usable is None or m.group(1) in usable):
                    counts[m.group(1)] += 1
        if counts:
            return counts.most_common(1)[0][0]
    return None
