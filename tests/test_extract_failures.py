"""pipeline/extract_entries.py — a page NER failed on must not pass for success.

A page whose NER call raised, or whose response could not be parsed, writes no
``_entries.json``. Until 2026-10-21 that was all: one status line among the
run's pages, the stage recorded complete, exit code 0. Page 0078 of the 1958
Green Book was lost that way and nothing reported it. process_item now returns
the failed pages so main() can name them and exit non-zero.
"""

import json

import pytest

import pipeline.extract_entries as ee


@pytest.mark.parametrize("status", ["api_error:timed out", "error:bad json", "parse_error"])
def test_failure_statuses(status):
    assert ee._is_failure(status)


@pytest.mark.parametrize(
    "status", ["ok", "partial_recovery", "skipped", "empty", "sparse_page", "dry_run (40 lines)"]
)
def test_statuses_that_are_not_failures(status):
    """Sparse and empty pages are deliberate skips; a partial recovery kept entries."""
    assert not ee._is_failure(status)


def _volume(tmp_path, stems):
    vol = tmp_path / "vol"
    vol.mkdir()
    for stem in stems:
        (vol / f"{stem}_m_aligned.json").write_text(json.dumps({"lines": []}), encoding="utf-8")
    return vol


def test_process_item_returns_the_failed_pages(tmp_path, monkeypatch, capsys):
    vol = _volume(tmp_path, ["0077_a", "0078_b", "0079_c"])
    outcome = {
        "0077_a_m_aligned.json": ("ok", [{"name": "x"}], {"city": "IDLEWILD"}),
        "0078_b_m_aligned.json": ("api_error:timed out", [], None),
        "0079_c_m_aligned.json": ("ok", [{"name": "y"}], {"city": "PAW PAW"}),
    }
    seen_context = {}

    def fake_page(client, aligned_path, model, prompt, context, *a, **k):
        status, entries, ctx = outcome[aligned_path.name]
        seen_context[aligned_path.name] = dict(context)
        return {"entries": entries, "page_context": ctx or context, "status": status}

    monkeypatch.setattr(ee, "process_page", fake_page)

    entries, failed = ee.process_item(
        None, vol, "m", "prompt", "text", force=True, dry_run=True, quiet=True,
    )

    assert failed == ["0078_b_m_aligned.json"]
    assert [e["name"] for e in entries] == ["x", "y"]
    # A failed page hands on the context it was given.
    assert seen_context["0079_c_m_aligned.json"] == {"city": "IDLEWILD"}
    # --quiet suppresses per-page lines, but never a failure.
    err = capsys.readouterr().err
    assert "0078_b_m_aligned.json  api_error:timed out" in err
    assert "0077_a_m_aligned.json" not in err


def test_spent_credit_is_not_reported_as_a_page_failure(tmp_path, monkeypatch):
    """A 402 must reach main() and stop the run, not become one api_error per page."""
    from utils.gemini import CreditsExhausted

    def refuse(*a, **k):
        raise CreditsExhausted("402 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(ee, "_call_gemini", refuse)
    with pytest.raises(CreditsExhausted):
        ee._run_ner_call(None, "p", "m", "sys", "user", None, tmp_path / "e.txt", None)
