"""utils/gemini.py — transient-error classification and retry/backoff.

The three transient families (429 rate limit, 503 unavailable, 500 internal)
each get an independent attempt counter and delay schedule, so a call that
hits one family does not consume another's budget. Everything else must
raise on the first attempt — retrying a 400 just burns quota.

``time.sleep`` is patched out in every retry test; the recorded delay lists
are what assert the backoff schedule.
"""

import pytest

import utils.gemini as gemini_mod
from utils.gemini import (
    PRO_THINKING_BUDGET,
    _gemini_version,
    _is_internal,
    _is_rate_limit,
    _is_unavailable,
    flex_http_options,
    generate_with_retry,
    get_client,
    thinking_config_for,
)
from utils.models import (
    DEFAULT_NER_MODEL,
    DEFAULT_OCR_MODEL,
    DEFAULT_PROMPT_MODEL,
    FALLBACK_MODEL,
)


class Boom(Exception):
    """Stand-in for google.genai's exception types, which stringify to
    messages like "500 INTERNAL. {'error': {'code': 500, ...}}"."""


class _FakeModels:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0
        self.last_kwargs = None

    def generate_content(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.errors:
            raise Boom(self.errors.pop(0))
        return "RESPONSE"


class _FakeClient:
    """Raises each message in *errors* in turn, then returns "RESPONSE"."""

    def __init__(self, errors=()):
        self.models = _FakeModels(errors)


@pytest.fixture
def slept(monkeypatch):
    """Collect sleep durations instead of waiting; yields the list."""
    delays: list[float] = []
    monkeypatch.setattr(gemini_mod.time, "sleep", delays.append)
    return delays


def _call(client, **kwargs):
    kwargs.setdefault("log", lambda msg: None)
    return generate_with_retry(client, model="m", contents=[], config=None, **kwargs)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestClassifiers:
    def test_rate_limit_matches_code_and_status(self):
        assert _is_rate_limit(Boom("429 Too Many Requests"))
        assert _is_rate_limit(Boom("RESOURCE_EXHAUSTED"))

    def test_unavailable_matches_code_status_and_overloaded(self):
        assert _is_unavailable(Boom("503 Service Unavailable"))
        assert _is_unavailable(Boom("UNAVAILABLE"))
        assert _is_unavailable(Boom("The model is Overloaded"))

    def test_internal_matches_code_and_status(self):
        assert _is_internal(Boom("500 INTERNAL. {'error': {'code': 500}}"))
        assert _is_internal(Boom("INTERNAL"))

    def test_classifiers_reject_client_errors(self):
        exc = Boom("400 INVALID_ARGUMENT")
        assert not _is_rate_limit(exc)
        assert not _is_unavailable(exc)
        assert not _is_internal(exc)

    def test_503_is_not_classified_as_internal(self):
        # Both predicates are substring matches; "503" must not read as "500".
        assert not _is_internal(Boom("503 UNAVAILABLE"))


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

def test_success_on_first_attempt_does_not_sleep(slept):
    client = _FakeClient()
    assert _call(client) == "RESPONSE"
    assert client.models.calls == 1
    assert slept == []


def test_passes_arguments_through_to_generate_content(slept):
    client = _FakeClient()
    generate_with_retry(
        client, model="gemini-x", contents=["hi"], config="CFG", log=lambda m: None
    )
    assert client.models.last_kwargs == {
        "model": "gemini-x",
        "contents": ["hi"],
        "config": "CFG",
    }


def test_internal_error_recovers(slept):
    client = _FakeClient(["500 INTERNAL"] * 2)
    assert _call(client) == "RESPONSE"
    assert client.models.calls == 3
    assert slept == [5.0, 10.0]          # base_500_delay doubling


def test_rate_limit_recovers(slept):
    client = _FakeClient(["429 RESOURCE_EXHAUSTED"] * 2)
    assert _call(client) == "RESPONSE"
    assert slept == [10.0, 20.0]         # base_delay doubling


def test_unavailable_recovers(slept):
    client = _FakeClient(["503 UNAVAILABLE"] * 2)
    assert _call(client) == "RESPONSE"
    assert slept == [30.0, 60.0]         # base_503_delay doubling


def test_internal_starts_backoff_sooner_than_unavailable(slept):
    """A 500 is a per-request fault, not service saturation — it should not
    wait out a load spike before the first retry."""
    _call(_FakeClient(["500 INTERNAL"]))
    first_500 = slept[0]
    slept.clear()
    _call(_FakeClient(["503 UNAVAILABLE"]))
    assert first_500 < slept[0]


def test_non_transient_error_raises_immediately(slept):
    client = _FakeClient(["400 INVALID_ARGUMENT"])
    with pytest.raises(Boom, match="400"):
        _call(client)
    assert client.models.calls == 1
    assert slept == []


@pytest.mark.parametrize(
    "message, expected_calls",
    [
        ("500 INTERNAL", 4),             # max_500_retries
        ("503 UNAVAILABLE", 4),          # max_503_retries
        ("429 RESOURCE_EXHAUSTED", 5),   # max_retries
    ],
)
def test_persistent_error_exhausts_its_budget_then_raises(slept, message, expected_calls):
    client = _FakeClient([message] * 20)
    with pytest.raises(Boom):
        _call(client)
    assert client.models.calls == expected_calls


def test_counters_are_independent_across_families(slept):
    """Alternating families must not exhaust a shared budget: 6 failures here
    stay within each family's own allowance and the call still succeeds."""
    client = _FakeClient([
        "503 UNAVAILABLE", "500 INTERNAL", "429 RESOURCE_EXHAUSTED",
        "503 UNAVAILABLE", "500 INTERNAL", "429 RESOURCE_EXHAUSTED",
    ])
    assert _call(client) == "RESPONSE"
    assert client.models.calls == 7


def test_retries_are_capped_per_family_not_in_total(slept):
    """Exhausting the 500 budget raises even though 429/503 budgets are unused."""
    client = _FakeClient(["500 INTERNAL"] * 10)
    with pytest.raises(Boom):
        _call(client)
    assert client.models.calls == 4


# ---------------------------------------------------------------------------
# Retry logging
# ---------------------------------------------------------------------------

def test_log_names_the_family_and_label(slept):
    logs: list[str] = []
    _call(_FakeClient(["500 INTERNAL"]), label="0007_59098937.jpg", log=logs.append)
    assert len(logs) == 1
    assert "Internal server error" in logs[0]
    assert "0007_59098937.jpg" in logs[0]


def test_log_omits_label_when_not_given(slept):
    logs: list[str] = []
    _call(_FakeClient(["503 UNAVAILABLE"]), log=logs.append)
    assert "retrying in" in logs[0]


def test_log_defaults_to_stderr(slept, capsys):
    generate_with_retry(
        _FakeClient(["500 INTERNAL"]), model="m", contents=[], config=None
    )
    assert "Internal server error" in capsys.readouterr().err


def test_quiet_callers_can_suppress_logging(slept, capsys):
    # generate_prompt.py passes a no-op log when --quiet is set.
    _call(_FakeClient(["500 INTERNAL"]), log=lambda msg: None)
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def test_flex_http_options_none_when_no_tier():
    assert flex_http_options(None) is None
    assert flex_http_options("") is None


def test_flex_http_options_sets_service_tier():
    opts = flex_http_options("flex")
    assert opts.extra_body == {"service_tier": "flex"}


def test_get_client_optional_returns_none_without_key(monkeypatch):
    monkeypatch.setattr(gemini_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_client(required=False) is None


def test_get_client_required_exits_without_key(monkeypatch):
    monkeypatch.setattr(gemini_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        get_client()


# --- thinking configuration -------------------------------------------------
#
# The table in thinking_config_for()'s docstring is the measured contract. These
# tests pin the three cells that bite, so a model bump fails here rather than
# either 400-ing every call or silently billing thinking tokens.


def _level(cfg) -> "str | None":
    """The thinking level as a lowercase string.

    The SDK coerces the string we pass into a ThinkingLevel enum whose value is
    uppercase ("MINIMAL"), so comparing against the spelling in the API docs
    needs normalizing.
    """
    lvl = cfg.thinking_level
    return None if lvl is None else str(getattr(lvl, "value", lvl)).lower()


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gemini-2.0-flash", (2, 0)),
        ("gemini-2.5-flash-lite", (2, 5)),
        ("gemini-3-flash-preview", (3, 0)),
        ("gemini-3.5-flash-lite", (3, 5)),
        ("gemini-3.8-flash", (3, 8)),
        ("gemini-flash-lite-latest", None),
    ],
)
def test_gemini_version_parsing(model, expected):
    assert _gemini_version(model) == expected


def test_pro_models_keep_a_positive_budget():
    """Pro tiers reject any request to disable thinking, so they get a cap."""
    cfg = thinking_config_for("gemini-3.5-pro")
    assert cfg.thinking_budget == PRO_THINKING_BUDGET
    assert _level(cfg) is None


def test_2x_models_use_the_budget_sentinel():
    """thinking_budget=0 is the only spelling 2.x accepts, and it is honored."""
    cfg = thinking_config_for("gemini-2.5-flash-lite")
    assert cfg.thinking_budget == 0
    assert _level(cfg) is None


@pytest.mark.parametrize("model", ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"])
def test_3x_lite_models_use_minimal(model):
    """gemini-3.5-flash-lite 400s on thinking_budget=0 — 'minimal' replaced it.

    This is the loud failure: it took out the OCR and NER defaults entirely.
    """
    cfg = thinking_config_for(model)
    assert _level(cfg) == "minimal"
    assert cfg.thinking_budget is None


@pytest.mark.parametrize("model", ["gemini-3.7-flash", "gemini-3.8-flash"])
def test_3x_non_lite_models_use_low_not_minimal(model):
    """Non-lite 3.x flash rejects 'minimal' and ignores budget=0, thinking anyway.

    This is the silent failure: budget=0 returned no error and 46-143 thought
    tokens per call. 'low' is the cheapest setting these models honor.
    """
    cfg = thinking_config_for(model)
    assert _level(cfg) == "low"
    assert cfg.thinking_budget is None


def test_unversioned_aliases_are_treated_as_newest():
    """A moving alias gets the level vocabulary, not the retired sentinel."""
    assert _level(thinking_config_for("gemini-flash-lite-latest")) == "minimal"
    assert _level(thinking_config_for("gemini-flash-latest")) == "low"


@pytest.mark.parametrize(
    "model",
    [DEFAULT_OCR_MODEL, DEFAULT_NER_MODEL, DEFAULT_PROMPT_MODEL, FALLBACK_MODEL],
)
def test_every_configured_default_gets_a_usable_config(model):
    """Guards the bump that broke this: a new default must resolve to a config
    the API will actually accept, never to the sentinel a 3.x model rejects."""
    cfg = thinking_config_for(model)
    if cfg.thinking_budget is not None:
        assert _gemini_version(model) < (3, 0) or "-pro" in model
    else:
        assert _level(cfg) in ("minimal", "low")
