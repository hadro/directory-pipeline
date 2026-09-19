"""pipeline/run_gemini_ocr.py — output-quality detection and the retry ladder.

The detector and the ladder are where OCR money and OCR data loss both live, and
neither had coverage before. The fixtures here are shaped after real pages:

  * degenerate ones from docs/evidence/2026-09-02-ocr-token-burn/ (Trow 1907
    leaves 0456/0906/0454, Polk 1933),
  * legitimate ones from output/ that an earlier proposal misclassified —
    Green Book hotel rate tables, Lain's Brooklyn street sections, brewers'
    statistical tables, the Douglass ledger.

The false-positive tests are the important half: a wrongly flagged page does not
just cost API calls, it sends a clean transcription through a ladder that can
replace it with a worse one.
"""

import pytest

from pipeline.run_gemini_ocr import (
    _BLOCK_LOOP_MAX_DISTINCT,
    _LOOP_KINDS,
    _REPETITION_THRESHOLD,
    _TEMPLATE_MIN_CONSEC_RUN,
    _distinct_line_count,
    _output_issue,
    process_image,
)


def kind(text: str) -> str:
    return _output_issue(text)[0]


NAMES = ["Smith", "Brown", "Callahan", "Devlin", "Esposito", "Farrell", "Gorman",
         "Hanrahan", "Ivers", "Jankowski", "Kowalski", "Lindqvist", "Moretti",
         "Novak", "O'Rourke", "Petrakis", "Quinlan", "Rosenblum", "Stavros", "Tobin"]


def directory_lines(n: int) -> "list[str]":
    """Ordinary directory entries — the non-numeric text varies, as on a real page."""
    trades = ["carpenter", "grocer", "clerk", "mason", "teamster", "printer"]
    return [
        f"{NAMES[i % len(NAMES)]} {chr(65 + i % 26)}, {trades[i % len(trades)]}, h {100 + i} Main st"
        for i in range(n)
    ]


# --- clean pages must stay clean -------------------------------------------

def test_empty_and_short_text_is_clean():
    assert kind("") == ""
    assert kind("a page\nwith three\nshort lines") == ""


def test_ordinary_directory_page_is_clean():
    assert kind("\n".join(directory_lines(60))) == ""


def test_hotel_rate_table_is_not_a_loop():
    """Green Book rate tables: 60% of lines normalise to '#.#-#.#'.

    A page-wide digit-normalised ratio flags this at 0.60; the consecutive-run
    rule sees a longest run of 2 and leaves it alone.
    """
    rows = []
    for i, name in enumerate(["Abbey", "Algonquin", "Allerton", "America", "Astor"] * 8):
        rows += [f"{name}, {100 + i} W. {40 + i} St.", f"$ {8 + i % 5}.50-1{i % 5}.50", f"$1{i % 4}.50-16.50"]
    assert kind("\n".join(rows)) == ""


def test_street_house_number_section_is_not_a_loop():
    """Lain's Brooklyn: lines differing only by digits, legitimately."""
    rows = []
    for n in range(0, 60):
        rows += [f"{1 + 2 * n} {2 + 2 * n} Nassau av", f"{3 + 2 * n} * Dobbin"]
    assert kind("\n".join(rows)) == ""


def test_statistical_table_is_not_a_loop():
    """Brewers' Guide: '1886.... 2,256,193 $1,231,388' — all-numeric rows.

    Shaped after the real page, which interleaves state headings between runs of
    figures; its longest consecutive templated run measures 16.
    """
    rows = []
    for state in ["ALABAMA", "ARKANSAS", "CALIFORNIA", "COLORADO", "DELAWARE"]:
        rows.append(state)
        rows += [f"{1880 + n}.................... {n},256,193 ${n},231,388" for n in range(12)]
        rows.append("Total for the State.")
    assert kind("\n".join(rows)) == ""


def test_column_of_bare_page_numbers_is_not_a_loop():
    """A classified index whose number column OCRs as its own run of lines.

    Every line normalises to the same 1-char template '#', so the run length is
    long — the normalised-template minimum length is what keeps this clean.
    """
    assert kind("\n".join(str(100 + n) for n in range(80))) == ""


# --- degenerate pages must be caught ---------------------------------------

def test_exact_repetition_loop_is_caught():
    text = "\n".join(["a real line here"] * 30 + [f"line {n}" for n in range(10)])
    assert kind(text) == "repetition"


def test_template_counting_loop_is_caught():
    """Trow 1907 leaf 0456: 'John, 148 W 128th' incrementing for 89 lines.

    Every line is distinct, so the exact-line check sees ~0%.
    """
    text = "\n".join(["BROWN"] + [f"John, {148 + 4 * n} W 128th" for n in range(60)])
    assert kind(text) == "template_loop"
    assert _output_issue(text)[0] not in ("repetition",)


def test_template_loop_needs_a_consecutive_run():
    """Just under the run threshold stays clean — the separation the rule relies on."""
    run = _TEMPLATE_MIN_CONSEC_RUN - 2
    text = "\n".join([f"John, {148 + 4 * n} W 128th" for n in range(run)] + directory_lines(60))
    assert kind(text) == ""


def test_template_loop_needs_to_dominate_the_page():
    """A long templated run inside an otherwise varied page is not a loop.

    Real numeric tables reach long runs without being degenerate; what marks a
    runaway is that the run is most of the page.
    """
    run = _TEMPLATE_MIN_CONSEC_RUN + 20
    text = "\n".join([f"John, {148 + 4 * n} W 128th" for n in range(run)] + directory_lines(200))
    assert kind(text) == ""


def test_block_loop_is_caught():
    """Trow 1907 leaf 0454: a 3-line obituary block repeated for 6,142 lines.

    The top single line sits under the 40% repetition threshold, and no line
    repeats back to back, so only the distinct-ratio rule sees it.
    """
    block = ["BROWN-In loving memory of my dear brother", "William, who died Oct. 24, 1905.", "Brother."]
    text = "\n".join(block * 400)
    top_share = 400 / 1200
    assert top_share < _REPETITION_THRESHOLD  # invisible to the exact check
    assert kind(text) == "block_loop"


def test_leader_runaway_is_caught_and_classified():
    text = "Smith Hotel" + "." * 60 + "123 Main St\n" + "\n".join(f"line {n}" for n in range(30))
    assert kind(text) == "leader"


def test_overlong_line_is_caught():
    text = "x" * 600 + "\n" + "\n".join(f"line {n}" for n in range(30))
    assert kind(text) == "long_line"


def test_loop_kinds_cover_the_three_loop_detectors():
    assert _LOOP_KINDS == {"repetition", "template_loop", "block_loop"}


def test_block_loop_threshold_is_a_distinct_ratio():
    """A long page with varied content stays clean even at high line counts."""
    text = "\n".join(directory_lines(2000))
    assert _distinct_line_count(text) / 2000 > _BLOCK_LOOP_MAX_DISTINCT
    assert kind(text) == ""


def test_distinct_line_count_ignores_blanks_and_duplicates():
    assert _distinct_line_count("a\n\nb\na\n  \n") == 2


# --- the retry ladder -------------------------------------------------------

class _Cand:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason
        self.safety_ratings = None


class _Resp:
    def __init__(self, text, finish_reason=None):
        self.text = text
        self.candidates = [_Cand(finish_reason)] if finish_reason is not None else []


class _Recorder:
    """Fake Gemini client returning scripted texts and recording each config."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []
        self.models = self

    def generate_content(self, *, model, config, contents):
        self.calls.append({"model": model, "config": config})
        item = self.texts.pop(0) if self.texts else ""
        return _Resp(*item) if isinstance(item, tuple) else _Resp(item)


@pytest.fixture
def page(tmp_path, monkeypatch):
    """A real file on disk plus a stubbed blank-page check."""
    import pipeline.run_gemini_ocr as mod

    img = tmp_path / "0001_page.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    monkeypatch.setattr(mod, "is_blank_page", lambda p: False)
    return img


def _run(client, page, **kw):
    return process_image(client, page, "gemini-3.1-flash-lite", "SYSTEM PROMPT", **kw)


def test_every_call_carries_the_service_tier_and_token_cap(page):
    """Regression guard for the retry ladder dropping flex — it billed every
    retry at standard rate on exactly the pages that retry most."""
    loop = "\n".join(["the same line over and over"] * 40)
    client = _Recorder([loop] * 8)

    _run(client, page, service_tier="flex", max_output_tokens=24576,
         fallback_model="gemini-3-flash-preview")

    assert len(client.calls) > 1, "expected the ladder to make retry calls"
    for call in client.calls:
        assert call["config"].http_options is not None, "a call lost the flex tier"
        assert call["config"].max_output_tokens, "a call lost the token cap entirely"
        assert call["config"].max_output_tokens <= 24576, "a call raised the token cap"

    # The first attempt gets the full budget; retries are narrowed to what the
    # page has shown it needs (see _retry_cap). That narrowing is the point —
    # a runaway retried at the full ceiling emits ~11x a clean page's output.
    assert client.calls[0]["config"].max_output_tokens == 24576
    assert any(c["config"].max_output_tokens < 24576 for c in client.calls[1:]), \
        "retries should be capped below the initial ceiling"


def test_repetition_loop_skips_the_temperature_rungs(page):
    """Loops go straight to the simple prompt: across the harvest logs no loop
    page was ever rescued by a temperature retry, while the simple prompt
    rescued 3 of the 6 pages that reached it."""
    loop = "\n".join(["the same line over and over"] * 40)
    clean = "\n".join(directory_lines(40))
    client = _Recorder([loop, clean])

    status, _ = _run(client, page, service_tier="flex", max_output_tokens=1000)

    assert status == "ok"
    assert len(client.calls) == 2, "a loop should not walk the temperature rungs"
    temps = [c["config"].temperature for c in client.calls]
    assert temps == [0.0, 0.3], f"expected simple-prompt rung next, got {temps}"


def test_recitation_style_failure_keeps_the_temperature_rungs(page):
    """A leader runaway is resampleable, so it must still try temperatures."""
    leader = "Smith Hotel" + "." * 60 + "123 Main\n" + "\n".join(f"l {n}" for n in range(30))
    clean = "\n".join(directory_lines(40))
    client = _Recorder([leader, clean])

    _run(client, page, service_tier=None, max_output_tokens=None)

    assert [c["config"].temperature for c in client.calls] == [0.0, 0.1]


def test_best_candidate_wins_over_the_last_one(page):
    """The ladder used to keep whatever came last. A good first response must
    survive a sequence of worse retries."""
    good = "\n".join(directory_lines(240)) + "\n" + "." * 60
    tiny = "\n".join(["junk line repeated"] * 40)
    client = _Recorder([good, tiny, tiny, tiny, tiny, tiny, tiny])

    _run(client, page, service_tier=None, max_output_tokens=None, fallback_model=None)

    written = (page.parent / "0001_page_gemini-3.1-flash-lite.txt").read_text()
    assert directory_lines(240)[200] in written, "the 240-line response was discarded"
    assert len(written.splitlines()) > 200


def test_total_failure_writes_no_file_and_reports_failed(page):
    """No .txt means a re-run retries the page. The old placeholder was written
    as a real page, counted 'ok', and skipped forever afterwards."""
    from google.genai.types import FinishReason

    client = _Recorder([("", FinishReason.RECITATION)] + [""] * 8)

    status, ok = _run(client, page, service_tier=None, max_output_tokens=None,
                      fallback_model="gemini-3-flash-preview")

    assert (status, ok) == ("failed", False)
    assert not (page.parent / "0001_page_gemini-3.1-flash-lite.txt").exists()
    assert len(client.calls) > 1, "RECITATION should still walk the ladder"


def test_bad_output_with_no_better_candidate_is_still_kept(page):
    """When nothing recovers, the best available text is written rather than
    discarded — a flawed page beats no page."""
    loop = "\n".join(["the same line over and over"] * 40)
    client = _Recorder([loop] + [""] * 8)

    status, _ = _run(client, page, service_tier=None, max_output_tokens=None,
                     fallback_model="gemini-3-flash-preview")

    assert status == "ok"
    assert (page.parent / "0001_page_gemini-3.1-flash-lite.txt").read_text().strip()


def test_empty_first_response_writes_nothing_and_reports_empty(page):
    """A zero-byte .txt was counted as a processed page, then deleted and
    re-requested on every later run — paying for the page again each time."""
    client = _Recorder([""])

    status, ok = _run(client, page, service_tier=None, max_output_tokens=None)

    assert status == "empty"
    assert not (page.parent / "0001_page_gemini-3.1-flash-lite.txt").exists()
    assert len(client.calls) == 1, "an empty response must not start the ladder"


def test_a_wholly_blocked_page_still_reaches_the_simple_prompt(page):
    """RECITATION on every attempt must not skip the rungs written for it.

    A blocked response carries no text, so candidates hold only empty strings and
    best() returns ("", ""). Guarding the simple-prompt and escalation rungs on
    `if kind:` therefore read false and failed the page having tried nothing but
    temperature — which is how two of eight 1869 Rowell validation pages failed.
    """
    from google.genai.types import FinishReason

    # Blocked on the initial call and all three temperatures, then the simple
    # prompt gets through — the sequence two Rowell pages actually produced.
    client = _Recorder(
        [("", FinishReason.RECITATION), "", "", "", "A clean transcription of the page."]
    )

    status, _ = _run(client, page, service_tier="flex", max_output_tokens=24576,
                     fallback_model="gemini-3.8-flash")

    assert status != "failed", "the simple-prompt rung should have rescued this page"
    prompts = [c["config"].system_instruction for c in client.calls]
    assert any("Transcribe all text on this page exactly" in (p or "") for p in prompts), \
        "expected the simple fallback prompt to be tried"
