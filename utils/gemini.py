"""Shared Gemini client construction and retry/backoff logic.

Every script that calls the Gemini API uses ``get_client()`` for the standard
GEMINI_API_KEY check and ``generate_with_retry()`` for transient-error
backoff, so rate-limit handling lives in exactly one place.
"""

import os
import re
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions, ThinkingConfig


def get_client(required: bool = True) -> "genai.Client | None":
    """Return a Gemini client authenticated from the GEMINI_API_KEY env var.

    Loads .env first. When the key is missing: exits with the standard error
    message if *required*, else returns None (callers pass required=False or
    skip the call entirely for --dry-run paths).
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        if required:
            print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
            sys.exit(1)
        return None
    return genai.Client(api_key=api_key)


# Pro-tier thinking cap. Left on the API's default (dynamic) budget, a pro model
# can spend tens of thousands of thinking tokens on a dense page and hit the
# output ceiling before finishing its answer — truncating OCR mid-line, or
# cutting an entries JSON array short. Measured on a 3-column 1841 trades
# directory page (~700 reference marks, a prompt asking the model to disambiguate
# each one): the dynamic budget burned 62,915 thinking tokens and returned
# MAX_TOKENS at 210 of 257 lines, while a 4,096 cap used 3,965 and completed. An
# 8,192 cap produced byte-identical output, so 4,096 is not binding here.
PRO_THINKING_BUDGET = 4096


_GEMINI_VERSION_RX = re.compile(r"gemini-(\d+)(?:\.(\d+))?")


def _gemini_version(model: str) -> "tuple[int, int] | None":
    """Major/minor version out of a Gemini model name, or None for an alias.

    'gemini-2.5-flash-lite' → (2, 5);  'gemini-3-flash-preview' → (3, 0);
    'gemini-flash-lite-latest' → None (a moving alias — treated as newest).
    """
    m = _GEMINI_VERSION_RX.search(model)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def thinking_config_for(model: str) -> ThinkingConfig:
    """Return the ThinkingConfig to use for *model*.

    These stages transcribe and extract; they do not reason, so they want as
    little thinking as each model will allow. How you ask for that changed
    between model generations, and asking the wrong way fails in two different
    directions — one loud, one silent. Measured against the live API
    (2026-09-18), one call per cell, reading ``usage_metadata.thoughts_token_count``:

        model                    budget=0      level="minimal"
        gemini-2.5-flash-lite    0 thoughts    400
        gemini-3.1-flash-lite    0 thoughts    0 thoughts
        gemini-3.5-flash-lite    400           0 thoughts
        gemini-3.7-flash         72 thoughts   400
        gemini-3.8-flash         46 thoughts   400

    Two things to take from that. ``thinking_level="minimal"`` is the 3.x
    replacement for the ``thinking_budget=0`` sentinel, and 3.1-flash-lite is
    the only model that accepts both — 3.5-flash-lite dropped the sentinel and
    400s on it. And the non-lite 3.x flash models never had a true "off": they
    accept budget=0 without complaint and think anyway (3.8-flash returned 46,
    143 and 0 thought tokens across three identical calls), while rejecting
    "minimal" outright. Their real floor is ``thinking_level="low"``.

    So the axis is the model generation, not the tier — the old ``"-pro" in
    model`` test was right for the 2.x family and is no longer sufficient.
    Unversioned aliases ('gemini-flash-latest') are treated as newest, since
    the level vocabulary is where the API is going and a wrong guess there
    fails loudly rather than silently billing thinking tokens.

    Pro-tier models only operate in thinking mode and reject any request to
    disable it, so they keep a positive cap instead (see PRO_THINKING_BUDGET).
    """
    if "-pro" in model:
        return ThinkingConfig(thinking_budget=PRO_THINKING_BUDGET)

    version = _gemini_version(model)
    if version is not None and version < (3, 0):
        # 2.x: the budget sentinel is the only spelling, and it is honored.
        return ThinkingConfig(thinking_budget=0)

    if "-lite" in model:
        # 3.x lite: the one family that can genuinely run with thinking off.
        return ThinkingConfig(thinking_level="minimal")

    # 3.x non-lite: "minimal" is rejected; "low" is the cheapest that works.
    # Thinking is minimized here, not disabled — these calls do bill thoughts.
    return ThinkingConfig(thinking_level="low")


def flex_http_options(service_tier: "str | None") -> "HttpOptions | None":
    """HttpOptions enabling a Gemini service tier (e.g. "flex"), or None."""
    if not service_tier:
        return None
    return HttpOptions(extra_body={"service_tier": service_tier})


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def _is_unavailable(exc: Exception) -> bool:
    s = str(exc)
    return "503" in s or "UNAVAILABLE" in s or "overloaded" in s.lower()


def _is_internal(exc: Exception) -> bool:
    """Server-side 500 INTERNAL — transient, and the same request usually
    succeeds on retry (observed on multi-image prompt-generation calls)."""
    s = str(exc)
    return "500" in s or "INTERNAL" in s


def generate_with_retry(
    client: genai.Client,
    *,
    model: str,
    contents,
    config,
    label: str = "",
    log=None,
    max_retries: int = 5,
    base_delay: float = 10.0,
    max_503_retries: int = 4,
    base_503_delay: float = 30.0,
    max_500_retries: int = 4,
    base_500_delay: float = 5.0,
):
    """client.models.generate_content with exponential backoff on transient errors.

    Rate limits (429/RESOURCE_EXHAUSTED), unavailability (503/overloaded), and
    server-side faults (500/INTERNAL) back off independently: up to
    *max_retries* attempts with delays doubling from *base_delay*,
    *max_503_retries* attempts doubling from *base_503_delay*, and
    *max_500_retries* attempts doubling from *base_500_delay* respectively.
    Anything else raises immediately.

    500s get a shorter starting delay than 503s: they are per-request faults
    rather than a signal that the service is saturated, so there is no reason
    to wait out a load spike before retrying.

    *label* names the work item (e.g. an image filename) in retry log lines;
    *log* is a print-style callable (defaults to stderr print).
    """
    log = log or (lambda msg: print(msg, file=sys.stderr))
    tag = f" {label}" if label else ""
    attempt_429 = 0
    delay_429 = base_delay
    attempt_503 = 0
    delay_503 = base_503_delay
    attempt_500 = 0
    delay_500 = base_500_delay
    while True:
        try:
            return client.models.generate_content(model=model, config=config, contents=contents)
        except Exception as exc:
            if _is_rate_limit(exc) and attempt_429 < max_retries - 1:
                attempt_429 += 1
                log(f"  Rate limited — retrying{tag} in {delay_429:.0f}s (attempt {attempt_429}/{max_retries - 1})")
                time.sleep(delay_429)
                delay_429 *= 2
            elif _is_unavailable(exc) and attempt_503 < max_503_retries - 1:
                attempt_503 += 1
                log(f"  Service unavailable — retrying{tag} in {delay_503:.0f}s (attempt {attempt_503}/{max_503_retries - 1})")
                time.sleep(delay_503)
                delay_503 *= 2
            elif _is_internal(exc) and attempt_500 < max_500_retries - 1:
                attempt_500 += 1
                log(f"  Internal server error — retrying{tag} in {delay_500:.0f}s (attempt {attempt_500}/{max_500_retries - 1})")
                time.sleep(delay_500)
                delay_500 *= 2
            else:
                raise
