"""Shared Gemini client construction and retry/backoff logic.

Every script that calls the Gemini API uses ``get_client()`` for the standard
GEMINI_API_KEY check and ``generate_with_retry()`` for transient-error
backoff, so rate-limit handling lives in exactly one place.
"""

import os
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


def thinking_config_for(model: str) -> ThinkingConfig:
    """Return the ThinkingConfig to use for *model*.

    Flash/flash-lite models support thinking_budget=0 (thinking disabled), which
    is what these stages want — the work is transcription and extraction, not
    reasoning. Pro-tier models only operate in thinking mode and reject budget=0,
    so they get a positive cap instead (see PRO_THINKING_BUDGET).
    """
    if "-pro" in model:
        return ThinkingConfig(thinking_budget=PRO_THINKING_BUDGET)
    return ThinkingConfig(thinking_budget=0)


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
