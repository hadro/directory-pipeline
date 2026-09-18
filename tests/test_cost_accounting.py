"""Token accounting and pricing.

Recorded usage is what makes cost a measurement rather than an estimate, and the
thing an estimate always misses is the retry ladder: the temperature/RECITATION
rungs in run_gemini_ocr.py loop *outside* generate_with_retry, so each rung is
its own billed call, and a blocked response still carries prompt tokens. Those
cases are pinned here.
"""

import pytest

from google.genai.types import GenerateContentConfig

import utils.gemini as gemini_mod
from pipeline.state import add_usage, read_state
from utils.gemini import (
    flex_http_options,
    generate_with_retry,
    record_usage,
    reset_usage,
    usage_snapshot,
)
from utils.models import MODEL_RATES, TIER_MULTIPLIERS, price_usage


class _Usage:
    def __init__(self, prompt=0, candidates=0, thoughts=0, cached=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts
        self.cached_content_token_count = cached


class _Response:
    def __init__(self, usage=None):
        self.usage_metadata = usage


@pytest.fixture(autouse=True)
def _clean_accumulator():
    reset_usage()
    yield
    reset_usage()


# --- recording --------------------------------------------------------------

def test_records_one_response():
    record_usage("m", _Response(_Usage(prompt=100, candidates=20, thoughts=5)))
    assert usage_snapshot() == {
        "m|standard|initial": {"calls": 1, "input": 100, "output": 20, "thoughts": 5, "cached": 0}
    }


def test_accumulates_across_calls():
    for _ in range(3):
        record_usage("m", _Response(_Usage(prompt=10, candidates=2)))
    snap = usage_snapshot()["m|standard|initial"]
    assert (snap["calls"], snap["input"], snap["output"]) == (3, 30, 6)


def test_blocked_response_still_counts_its_input():
    """A RECITATION block returns prompt tokens and a None candidate count.

    Those calls were billed; dropping them is how an estimate loses the retry
    ladder's cost.
    """
    record_usage("m", _Response(_Usage(prompt=2149, candidates=None)))
    snap = usage_snapshot()["m|standard|initial"]
    assert snap == {"calls": 1, "input": 2149, "output": 0, "thoughts": 0, "cached": 0}


def test_response_without_usage_metadata_is_ignored():
    record_usage("m", _Response(None))
    assert usage_snapshot() == {}


def test_models_and_tiers_are_kept_apart():
    record_usage("a", _Response(_Usage(prompt=1)), tier="flex")
    record_usage("a", _Response(_Usage(prompt=1)), tier="standard")
    record_usage("b", _Response(_Usage(prompt=1)), tier="flex")
    assert set(usage_snapshot()) == {"a|flex|initial", "a|standard|initial", "b|flex|initial"}


def test_snapshot_is_a_copy():
    record_usage("m", _Response(_Usage(prompt=1)))
    usage_snapshot()["m|standard|initial"]["input"] = 999
    assert usage_snapshot()["m|standard|initial"]["input"] == 1


# --- the retry ladder -------------------------------------------------------

class _FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_content(self, **kwargs):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


class _Boom(Exception):
    pass


def test_each_ladder_rung_is_billed(monkeypatch):
    """Four attempts on one page bill four times, not once.

    run_gemini_ocr retries a blocked page at rising temperatures by calling
    generate_with_retry again, so the rungs arrive here as separate calls.
    """
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda *_: None)
    blocked = _Response(_Usage(prompt=2149, candidates=None))
    ok = _Response(_Usage(prompt=2149, candidates=1806))
    for response in (blocked, blocked, blocked, ok):
        generate_with_retry(
            _FakeClient([response]), model="m", contents=[], config=None,
            log=lambda _m: None,
        )
    snap = usage_snapshot()["m|standard|initial"]
    assert snap["calls"] == 4
    assert snap["input"] == 2149 * 4
    assert snap["output"] == 1806


def test_transient_retries_bill_only_the_successful_call(monkeypatch):
    """A 429 raises before any response exists, so there is nothing to record."""
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda *_: None)
    client = _FakeClient([_Boom("429 RESOURCE_EXHAUSTED"),
                          _Response(_Usage(prompt=10, candidates=5))])
    generate_with_retry(client, model="m", contents=[], config=None, log=lambda _m: None)
    assert usage_snapshot()["m|standard|initial"]["calls"] == 1


@pytest.mark.parametrize(
    "tier, expected_key",
    [("flex", "m|flex|initial"), (None, "m|standard|initial")],
)
def test_service_tier_is_read_from_the_config(monkeypatch, tier, expected_key):
    """Tier comes off the config the caller already builds, so --flex runs price
    themselves correctly without any stage having to pass it separately."""
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda *_: None)
    config = GenerateContentConfig(http_options=flex_http_options(tier))
    generate_with_retry(
        _FakeClient([_Response(_Usage(prompt=10))]),
        model="m", contents=[], config=config, log=lambda _m: None,
    )
    assert expected_key in usage_snapshot()


def test_config_of_none_is_treated_as_standard(monkeypatch):
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda *_: None)
    generate_with_retry(
        _FakeClient([_Response(_Usage(prompt=10))]),
        model="m", contents=[], config=None, log=lambda _m: None,
    )
    assert "m|standard|initial" in usage_snapshot()


# --- pricing ----------------------------------------------------------------

def test_thinking_tokens_bill_at_the_output_rate():
    """The whole reason a "thinking off" that silently no-ops costs money."""
    rate_in, rate_out = MODEL_RATES["gemini-3.8-flash"]
    counts = {"input": 1_000_000, "output": 0, "thoughts": 1_000_000}
    assert price_usage("gemini-3.8-flash", counts) == pytest.approx(rate_in + rate_out)


def test_flex_is_half_of_standard():
    counts = {"input": 1_000_000, "output": 1_000_000}
    standard = price_usage("gemini-3.5-flash-lite", counts, "standard")
    assert price_usage("gemini-3.5-flash-lite", counts, "flex") == pytest.approx(standard / 2)
    assert TIER_MULTIPLIERS["batch"] == TIER_MULTIPLIERS["flex"]


def test_unknown_model_is_unpriced_rather_than_guessed():
    """An invented rate is worse than a gap — callers must be able to say so."""
    assert price_usage("gemini-9.9-imaginary", {"input": 1_000, "output": 1_000}) is None


def test_missing_counts_default_to_zero():
    assert price_usage("gemini-3.5-flash-lite", {}) == 0.0


def test_measured_page_prices_as_observed():
    """Page 0009 measured 2,104 input / 1,806 output on 2026-09-18."""
    cost = price_usage("gemini-3.5-flash-lite",
                       {"input": 2104, "output": 1806, "thoughts": 0})
    assert cost == pytest.approx(0.00515, abs=1e-5)


# --- persistence ------------------------------------------------------------

def test_add_usage_accumulates_across_runs(tmp_path):
    add_usage(tmp_path, {"m|flex": {"calls": 1, "input": 10, "output": 2}})
    add_usage(tmp_path, {"m|flex": {"calls": 2, "input": 20, "output": 4}})
    assert read_state(tmp_path)["token_usage"]["m|flex"] == {
        "calls": 3, "input": 30, "output": 6
    }


def test_add_usage_leaves_last_run_args_alone(tmp_path):
    from pipeline.state import write_state
    write_state(tmp_path, {"slug": "v"}, args="main.py output/v --gemini-ocr")
    add_usage(tmp_path, {"m|standard": {"calls": 1, "input": 5}})
    state = read_state(tmp_path)
    assert state["last_run_args"] == "main.py output/v --gemini-ocr"
    assert state["slug"] == "v"


def test_add_usage_of_nothing_writes_nothing(tmp_path):
    add_usage(tmp_path, {})
    assert not (tmp_path / "pipeline_state.json").exists()


# --- reason tagging ---------------------------------------------------------

def test_reason_defaults_to_initial():
    record_usage("m", _Response(_Usage(prompt=1)))
    assert "m|standard|initial" in usage_snapshot()


def test_reasons_are_kept_apart():
    record_usage("m", _Response(_Usage(prompt=1)), reason="initial")
    record_usage("m", _Response(_Usage(prompt=1)), reason="temp:recitation")
    record_usage("m", _Response(_Usage(prompt=1)), reason="escalate:block_loop")
    assert set(usage_snapshot()) == {
        "m|standard|initial",
        "m|standard|temp:recitation",
        "m|standard|escalate:block_loop",
    }


def test_generate_with_retry_passes_the_reason(monkeypatch):
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda *_: None)
    generate_with_retry(
        _FakeClient([_Response(_Usage(prompt=10))]),
        model="m", contents=[], config=None, log=lambda _m: None,
        reason="simple:leader",
    )
    assert "m|standard|simple:leader" in usage_snapshot()


def test_recording_is_thread_safe():
    """The OCR stage records from a ThreadPoolExecutor; += is not atomic.

    Without the lock this loses calls, which is the one thing a cost record
    must never do.
    """
    from concurrent.futures import ThreadPoolExecutor

    def hit(_):
        record_usage("m", _Response(_Usage(prompt=1, candidates=1)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(hit, range(2000)))
    snap = usage_snapshot()["m|standard|initial"]
    assert snap["calls"] == 2000
    assert snap["input"] == 2000


# --- retry ceiling ----------------------------------------------------------

def test_retry_cap_narrows_to_the_best_candidate():
    from pipeline.run_gemini_ocr import _retry_cap

    # 400 distinct lines at ~13 tokens/line, doubled for headroom.
    page = "\n".join(f"line {i}" for i in range(400))
    cap = _retry_cap([(page, "")], 24576)
    assert cap == 400 * 13 * 2
    assert cap < 24576


def test_retry_cap_counts_distinct_lines_not_raw_ones():
    """A block_loop page runs thousands of lines with a dozen distinct ones.

    Capping on raw length would not bite on the exact runaway this targets:
    6,142 raw lines would derive a cap far above the ceiling.
    """
    from pipeline.run_gemini_ocr import _retry_cap

    runaway = "\n".join(["the same line over and over"] * 6142)
    assert _retry_cap([(runaway, "block_loop")], 24576) == 2048


def test_retry_cap_keeps_the_ceiling_when_nothing_came_back():
    """A page blocked on its first attempt has produced no evidence of its
    own length, so narrowing would risk truncating the rescue."""
    from pipeline.run_gemini_ocr import _retry_cap

    assert _retry_cap([("", "recitation")], 24576) == 24576
    assert _retry_cap([], 24576) == 24576


def test_retry_cap_never_raises_the_ceiling():
    from pipeline.run_gemini_ocr import _retry_cap

    page = "\n".join(f"line {i}" for i in range(100_000))
    assert _retry_cap([(page, "")], 24576) == 24576


def test_retry_cap_has_a_floor():
    """A two-line candidate must not cap the retry at 52 tokens.

    Below the floor this would stop being a cost guard and start truncating
    rescues on pages whose only candidates so far are degenerate.
    """
    from pipeline.run_gemini_ocr import _retry_cap

    assert _retry_cap([("a\nb\n", "")], 24576) == 2048


def test_retry_cap_respects_an_uncapped_run():
    """--max-output-tokens 0 disables the cap; retries must not reinstate one."""
    from pipeline.run_gemini_ocr import _retry_cap

    page = "\n".join(f"line {i}" for i in range(200))
    assert _retry_cap([(page, "")], 0) == 0
    assert _retry_cap([(page, "")], None) is None
