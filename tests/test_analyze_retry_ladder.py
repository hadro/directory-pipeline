"""Retry-ladder log analysis.

The inference this makes is subtle and got it wrong once: a page whose log ends
mid-ladder looks exactly like a page the temperature rung rescued, and counting
the first as the second inverts the result. That case is pinned here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from analyze_retry_ladder import classify, parse  # noqa: E402

MODEL = "gemini-3.1-flash-lite"


def _write(tmp_path: Path, lines: "list[str]") -> Path:
    p = tmp_path / "run.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _done(page: str, n: int = 1) -> str:
    return f"[000{n}/1] Done: {page[:-4]}_{MODEL}.txt"


# --- kind classification ----------------------------------------------------

@pytest.mark.parametrize(
    "message, expected",
    [
        ("repetition loop (98% of lines are [\" Irving…])", "repetition"),
        ("line too long (170305 chars): [\" Josephine…]", "long_line"),
        ("spaced leader runaway in line [Triado ( ( (…]", "spaced_leader"),
        ("RECITATION", "recitation"),
        ("template loop (120 consecutive lines…)", "template_loop"),
        ("block loop (6142 lines, 13 distinct)", "block_loop"),
        ("something nobody has seen before", "unknown"),
    ],
)
def test_classify_maps_messages_to_kinds(message, expected):
    """The log carries _output_issue's message, not its kind, so this parser has
    to match text — the one place that is allowed."""
    assert classify(message) == expected


# --- rescue inference -------------------------------------------------------

def test_temperature_rescue_is_reaching_done_without_the_simple_rung(tmp_path):
    page = "0001_58192510.jpg"
    log = _write(tmp_path, [
        f"  repetition loop (98% of lines are [x…]) — retrying at temperature=0.1: {page}",
        f"  temperature=0.1 still problematic (repetition loop (97%…)): {page}",
        f"  repetition loop (98% of lines are [x…]) — retrying at temperature=0.7: {page}",
        _done(page),
    ])
    rec = parse([log])[page]
    assert rec["kind"] == "repetition"
    assert rec["temp"] and rec["complete"]
    assert not rec["simple"], "never reached the simple rung, so temperature rescued it"


def test_truncated_trace_is_not_counted_as_a_rescue(tmp_path):
    """An interrupted run is not a rescue.

    Without the terminal-line check this page is indistinguishable from the one
    above, and the archived harvest logs contain several of them.
    """
    page = "0003_page.jpg"
    log = _write(tmp_path, [
        f"  line too long (6910 chars): [31-33-35…] — retrying at temperature=0.1: {page}",
    ])
    rec = parse([log])[page]
    assert rec["temp"]
    assert not rec["complete"]


def test_simple_prompt_rescue_is_recorded(tmp_path):
    page = "0002_58082929.jpg"
    log = _write(tmp_path, [
        f"  line too long (76547 chars): [x…] — retrying at temperature=0.1: {page}",
        f"  temperature=0.1 still problematic (line too long (77721 chars)): {page}",
        f"  Trying simple fallback prompt at temperature=0.7: {page}",
        f"  Simple prompt succeeded at temperature=0.7: {page}",
        _done(page),
    ])
    rec = parse([log])[page]
    assert rec["kind"] == "long_line"
    assert rec["simple"] and rec["simple_ok"] and rec["complete"]


def test_a_rescue_clears_the_pre_rewrite_exhausted_line(tmp_path):
    """Logs written before the best() rewrite print "All retries exhausted" even
    on pages the simple prompt had just rescued."""
    page = "0002_x.jpg"
    log = _write(tmp_path, [
        f"  repetition loop (90%…) — retrying at temperature=0.1: {page}",
        f"  Trying simple fallback prompt at temperature=0.3: {page}",
        f"  Simple prompt succeeded at temperature=0.3: {page}",
        f"  All retries exhausted — keeping best available output: {page}",
    ])
    rec = parse([log])[page]
    assert rec["simple_ok"]
    assert not rec["unresolved"]


def test_escalation_outcome_is_recorded(tmp_path):
    page = "0005_x.jpg"
    log = _write(tmp_path, [
        f"  repetition loop (90%…) — skipping temperature retries (ineffective for loops): {page}",
        f"  Trying simple fallback prompt at temperature=0.3: {page}",
        f"  Escalating to gemini-3.8-flash: {page}",
        f"  Model escalation: imperfect output (gemini-3.8-flash): {page}",
        f"  All retries exhausted — keeping best available output: {page}",
    ])
    rec = parse([log])[page]
    assert rec["skipped_temp"] and not rec["temp"]
    assert rec["esc"] and not rec["esc_ok"]
    assert rec["unresolved"] and rec["complete"]


def test_page_names_containing_colons_are_parsed(tmp_path):
    """LoC filenames carry colons — '0009_public:gdc:sn82007064_1869:0009.jpg'."""
    page = "0009_public:gdc:sn82007064_1869:0009.jpg"
    log = _write(tmp_path, [
        f"  RECITATION — retrying at temperature=0.1: {page}",
        _done(page),
    ])
    rec = parse([log])[page]
    assert rec["kind"] == "recitation"
    assert rec["complete"]


def test_the_first_kind_wins_when_it_drifts(tmp_path):
    """One evidence page enters as "line too long" and leaves as "spaced leader".

    The kind that provoked the first retry is what a decision about the ladder
    would branch on.
    """
    page = "0002_x.jpg"
    log = _write(tmp_path, [
        f"  line too long (76547 chars): [x…] — retrying at temperature=0.1: {page}",
        f"  temperature=0.7 still problematic (spaced leader runaway in line [T ( ( (…]): {page}",
        f"  Trying simple fallback prompt at temperature=0.3: {page}",
        _done(page),
    ])
    assert parse([log])[page]["kind"] == "long_line"


def test_interleaved_pages_are_kept_apart(tmp_path):
    """Workers write concurrently, so a page's lines are not contiguous."""
    a, b = "0001_a.jpg", "0002_b.jpg"
    log = _write(tmp_path, [
        f"  repetition loop (90%…) — retrying at temperature=0.1: {a}",
        f"  line too long (500 chars): [x…] — retrying at temperature=0.1: {b}",
        f"  Trying simple fallback prompt at temperature=0.3: {b}",
        f"  Simple prompt succeeded at temperature=0.3: {b}",
        _done(a),
        _done(b, 2),
    ])
    pages = parse([log])
    assert pages[a]["kind"] == "repetition" and not pages[a]["simple"]
    assert pages[b]["kind"] == "long_line" and pages[b]["simple_ok"]
