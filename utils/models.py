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
DEFAULT_OCR_MODEL = "gemini-3.1-flash-lite"

# Default model for NER entry extraction (--extract-entries).
DEFAULT_NER_MODEL = "gemini-3.1-flash-lite"

# Local (free, on-device) OCR backends — see pipeline/run_local_ocr.py.
# Default Chandra checkpoint (8-bit MLX weights for Apple Silicon).
DEFAULT_LOCAL_OCR_MODEL = "mlx-community/chandra-ocr-2-8bit-mlx"
# Stable filename/state slug per engine. The slug is fixed per engine (not derived
# from the checkpoint) so output filenames stay clean and downstream discovery is
# unaffected by quantization suffixes.
LOCAL_OCR_SLUGS = {"chandra": "chandra-ocr-2", "vision": "apple-vision"}

# More capable model for one-off generation/review tasks
# (--generate-prompts, analysis/review_entries.py).
DEFAULT_PROMPT_MODEL = "gemini-3-flash-preview"

# Escalation target when the primary model exhausts its output budget on a
# dense page (used by run_gemini_ocr.py and extract_entries.py).
FALLBACK_MODEL = "gemini-3-flash-preview"


def model_slug(model: str) -> str:
    """Filesystem-safe slug for a model name, used in all output filenames."""
    return model.replace("/", "_")


# Aligned-JSON names are structurally unambiguous, so the model match is
# prefix-agnostic: the slug is the last "_"-delimited segment before
# "_aligned.json". (Slugs containing "_" — from models with "/" in the name —
# would be truncated; no such model is in use.)
_ALIGNED_RX = re.compile(r"_([^_]+)_aligned\.json$")

# Bare .txt names are ambiguous (*_surya.txt, included_pages.txt, …), so the
# model match requires a known OCR-backend prefix: "gemini-" (Gemini), "chandra-"
# (local Chandra VLM), or "apple-" (local Apple Vision). Keeping an explicit
# allow-list — rather than a fully generic _([^_]+)\.txt$ — preserves the
# exclusion of _surya.txt / included_pages.txt etc.
_TXT_RX = re.compile(r"_((?:gemini|chandra|apple)-[^_]+)\.txt$")


def discover_ocr_slug(output_root: Path) -> str | None:
    """Scan *output_root* (and its immediate subdirectories) for OCR output
    files and return the most common model slug, or None if nothing is found.

    Aligned JSON files are preferred over bare OCR .txt files. This is the
    fallback used when neither a --model flag nor pipeline_state.json
    identifies the OCR model.
    """
    if not output_root.is_dir():
        return None
    dirs = [output_root] + [d for d in sorted(output_root.iterdir()) if d.is_dir()]
    for pattern, rx in (("*_aligned.json", _ALIGNED_RX), ("*.txt", _TXT_RX)):
        counts: Counter[str] = Counter()
        for d in dirs:
            for f in d.glob(pattern):
                m = rx.search(f.name)
                if m:
                    counts[m.group(1)] += 1
        if counts:
            return counts.most_common(1)[0][0]
    return None
