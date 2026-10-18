#!/usr/bin/env python3
"""Run Gemini-powered OCR on images, saving plain-text output alongside each image.

For each .jpg found in the images directory, sends the image to the Gemini API and
saves a .txt file alongside it. Already-processed images (where .txt already exists)
are skipped, so the script is safe to re-run.

The system prompt is read from ocr_prompt.md in the same directory as this script.

Requires a Gemini API key in the GEMINI_API_KEY environment variable.

Usage
-----
    python run_gemini_ocr.py output/travelguide
    python run_gemini_ocr.py output/greenbooks --workers 8
    python run_gemini_ocr.py output/travelguide --model gemini-2.5-flash
    python run_gemini_ocr.py output/travelguide --quiet
    python run_gemini_ocr.py output/travelguide --flex                      # ~50% cheaper, 1-15 min/req
    python run_gemini_ocr.py output/travelguide --model gemini-3.1-flash-lite --flex
"""

import argparse
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()
from google.genai.types import FinishReason, GenerateContentConfig, MediaResolution, Part, ThinkingConfig

from utils.gemini import (
    drain_usage,
    flex_http_options,
    generate_with_retry,
    get_client,
    thinking_config_for,
)
from utils.image_utils import image_is_complete, is_blank_page
from pipeline.download_images import redownload_page
from pipeline.state import add_usage, find_state_dir, record_stage
from utils.models import DEFAULT_OCR_MODEL, FALLBACK_MODEL, model_slug

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "ocr_prompt.md"

# Bare-minimum prompt used as a last resort before model escalation.
# Deliberately short — complex prompts can themselves trigger recitation on hard pages.
_SIMPLE_FALLBACK_PROMPT = (
    "Transcribe all text on this page exactly as it appears. "
    "Return plain text only. No commentary, no formatting."
)

# Keywords in the OCR prompt that indicate handwriting → high resolution recommended
_HANDWRITING_KEYWORDS = ("handwrit", "manuscript", "cursive")

# Default per-call output ceiling. The OCR stage previously set none at all, so a
# page that started looping ran to the model's own limit — the expensive failure
# this cap exists to bound. Sized from the corpus: the largest legitimate page
# measured is ~29KB of abbreviation-dense directory text (~8-12k tokens), so this
# is roughly 2x headroom. Thinking tokens share this budget, so a "-pro" model
# also spends PRO_THINKING_BUDGET from it (see utils/gemini.py).
DEFAULT_MAX_OUTPUT_TOKENS = 24576

# Persist token usage every this many billed pages, rather than once at the end.
# A 365-page run takes over two hours on the flex tier, and writing only on
# completion meant an interrupted or crashed run recorded nothing at all — the
# spend still happened, the evidence of it did not. It also left `report_cost.py`
# answering "no token usage recorded yet" for the whole of a long run, which is
# exactly when you want to look. Each flush drains the accumulator, so flushes
# add up rather than double-count.
USAGE_FLUSH_EVERY = 25


def _needs_high_res(prompt_text: str) -> bool:
    lower = prompt_text.lower()
    return any(kw in lower for kw in _HANDWRITING_KEYWORDS)

_DITTO_INSTRUCTION = (
    "\nDitto marks: when you see small raised comma-pairs (printed as '' or 〃, "
    "often misread as 66 or as a quotation mark) used to repeat a value from the "
    "row above, expand them in place — write out the full repeated value rather "
    "than transcribing the mark itself."
)

_NO_LEADER_INSTRUCTION = (
    "\n\nCRITICAL: Do NOT reproduce dot leaders (sequences of dots, or '. . .' "
    "patterns used as visual spacing between a name and an address). Replace them "
    "with a single space or omit them entirely. For example, transcribe "
    "\"Smith Hotel. . . . . . . . . . 123 Main St\" as \"Smith Hotel 123 Main St\"."
)

_print_lock = threading.Lock()


def _load_scope(output_root: Path) -> "set[str] | None":
    """Return the set of filenames to process, or None (= process all).

    Reads included_pages.txt from output_root or output_root.parent.
    Returns None when no file is found (backward-compatible: process everything).
    """
    for d in (output_root.resolve(), output_root.resolve().parent):
        p = d / "included_pages.txt"
        if p.exists():
            lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.startswith("#")]
            if lines:
                return set(lines)
    return None


def _find_prompt(output_root: Path, fallback: Path) -> Path:
    """Return a volume-specific ocr_prompt.md if one exists alongside the images,
    otherwise return the global fallback.

    Search order:
    1. output_root itself (item directory passed directly)
    2. output_root.parent (slug-level directory, when called with item subdir)
    3. Any immediate subdirectory of output_root (item subdir when called with
       slug-level directory — covers the case where --select-pages and
       --generate-prompts wrote ocr_prompt.md into the item subdir but the
       OCR stage is invoked with the parent slug directory)
    """
    root = output_root.resolve()
    for candidate_dir in (root, root.parent):
        p = candidate_dir / "ocr_prompt.md"
        if p.exists():
            return p
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir():
            p = subdir / "ocr_prompt.md"
            if p.exists():
                return p
    return fallback


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


# Output quality thresholds
_REPETITION_THRESHOLD = 0.40   # fraction of lines that are identical → runaway loop
_REPETITION_MIN_REPEATS = 20   # minimum line count before repetition check applies
_REPETITION_MIN_LINE_LEN = 4   # ignore lines shorter than this when counting

# Template loop — the same line modulo its digits, repeated back to back. Catches
# counting runaways ("John, 148 W 128th" / "John, 152 W 128th" / …) where every
# line is technically distinct, so the exact-line check above sees ~0%.
#
# Measured over 11,866 real OCR pages plus the 2026-09-02 harvest evidence set
# (docs/evidence/2026-09-02-ocr-token-burn/): degenerate pages run 89–2,788
# consecutive templated lines, while legitimate numeric pages — hotel rate
# tables, brewers' statistics, street/house-number sections, manuscript ledgers —
# top out at 16. A page-wide ratio cannot separate them (those pages reach 47–60%
# normalised and would be flagged), but a consecutive run separates them cleanly.
_TEMPLATE_MIN_CONSEC_RUN = 50  # consecutive digit-normalised identical lines → loop
_TEMPLATE_MIN_NORM_LEN = 4     # ignore templates shorter than this AFTER normalising,
                               # so a column of bare page numbers ("#") is not a loop
_TEMPLATE_MIN_PAGE_SHARE = 0.5  # the run must also dominate the page. A degenerate
                               # page is nearly all loop (measured 61-99% across every
                               # known case); a numeric table that happens to run long
                               # still carries headings, totals and section breaks, so
                               # its longest run covers far less of the page.

# Block loop — a short cycle of lines repeated for the whole page, which neither
# the exact-line nor the template check sees (the top single line can sit under
# 40% while the page carries only a dozen distinct lines). Trow leaf 0454 is the
# reference case: 6,142 lines, 13 distinct, an obituary block repeated ~2,045x.
_BLOCK_LOOP_MIN_LINES = 200    # only applies to pages long enough to be suspicious
_BLOCK_LOOP_MAX_DISTINCT = 0.35  # distinct/total below this → cycling on a block

# Issue kinds returned by _output_issue(). Grouped so the retry ladder can branch
# on the failure class instead of substring-matching a human-readable message.
_LEADER_KINDS = frozenset({"leader", "spaced_leader", "long_line"})
_LOOP_KINDS = frozenset({"repetition", "template_loop", "block_loop"})
_LEADER_RUNAWAY_MIN_RUN = 50   # consecutive identical non-word chars → dot-leader runaway
_LEADER_SPACED_MIN_RUN = 50   # spaced pattern like ". . . . ." repeated this many times
_LEADER_MAX_LINE_LEN = 500    # any line longer than this is a runaway (catches spaced variants)

# Temperatures to try in order when output quality checks fail
_RETRY_TEMPERATURES = [0.1, 0.3, 0.7]

# Output ceiling for a retry, derived from what the page has already produced.
#
# The first attempt needs the full --max-output-tokens budget: a legitimately
# dense page is indistinguishable from a runaway until it arrives. A retry is
# different — the page has shown how much text it actually holds.
#
# This exists because the two failure classes cost wildly different amounts. A
# RECITATION block is refused before generation: prompt tokens only, no output,
# ~$0.0006 a call. A runaway *emits*, to whatever ceiling it is given — $0.06 at
# the 24,576 default, ~11x a clean page, and a leader page walking the whole
# ladder can reach $0.47.
#
# Two measured constants, not guesses:
#   * ~13 output tokens per line of directory text (5,904 output tokens across
#     453 lines on the 1869 Rowell probe pages, 2026-09-18). An earlier draft of
#     this used 1.3 and would have truncated every rescue.
#   * distinct lines, not raw lines. A block_loop page runs 6,142 lines with 13
#     distinct (Trow leaf 0454), so a raw-line cap would not bite on the exact
#     runaway it targets. _distinct_line_count is also what best() ranks by, so
#     the cap and the selection agree on what "long" means.
_RETRY_CAP_TOKENS_PER_LINE = 13
_RETRY_CAP_MULTIPLE = 2
# Floor: roughly 150 lines of headroom. Below this the cap could truncate a
# rescue on a page whose only candidates so far are degenerate, which would turn
# a cost guard into a quality regression.
_RETRY_CAP_FLOOR = 2048


def _retry_cap(candidates: "list[tuple[str, str]]", ceiling: "int | None") -> "int | None":
    """Output ceiling for the next retry, from the best candidate so far.

    Returns *ceiling* unchanged when nothing usable has come back yet (a page
    blocked on its first attempt has produced no evidence of its own length),
    when the run is uncapped, or when the derived cap would not be lower.
    """
    if not ceiling:
        return ceiling
    longest = max((_distinct_line_count(t) for t, _ in candidates if t), default=0)
    if not longest:
        return ceiling
    cap = longest * _RETRY_CAP_TOKENS_PER_LINE * _RETRY_CAP_MULTIPLE
    return min(ceiling, max(cap, _RETRY_CAP_FLOOR))

# Regex: _LEADER_RUNAWAY_MIN_RUN (50)+ consecutive identical non-word, non-space characters
_LEADER_RE = re.compile(r"([^\w\s])\1{%d,}" % (_LEADER_RUNAWAY_MIN_RUN - 1))
# Regex: spaced dot-leader like ". . . . ." — non-word char + space repeated 50+ times
_LEADER_SPACED_RE = re.compile(r"([^\w\s] ){%d,}" % _LEADER_SPACED_MIN_RUN)
# Regex: collapse any run of 10+ dots (with optional spaces) to a canonical 5-dot leader
_LEADER_COLLAPSE_RE = re.compile(r"(\.\s*){10,}")
# Regex: a run of digits, collapsed to "#" so a counting loop reduces to one template
_DIGIT_RUN_RE = re.compile(r"\d+")


def _clean_output(text: str) -> str:
    """Apply post-processing cleanup to OCR text before saving.

    Collapses any run of 10+ dots (with optional spaces) down to a canonical
    5-dot leader, preserving any content that follows (e.g. 'r.' or 'r. m.').
    This handles pages where dot-leader runaways survive all temperature retries.
    """
    return _LEADER_COLLAPSE_RE.sub(".....", text)


def _output_issue(text: str) -> "tuple[str, str]":
    """Classify *text* as degenerate model output.

    Returns ``(kind, message)`` — ``("", "")`` when the text looks fine. *kind* is
    one of the values in _LEADER_KINDS or _LOOP_KINDS and is what callers should
    branch on; *message* is for logging only and embeds page content, so it must
    never be pattern-matched (a page repeating the word "leader" would otherwise
    read as a dot-leader failure).

    Failure modes, in order of the checks:
      1. Dot-leader runaway — 50+ consecutive identical non-word chars, the spaced
         ". . . ." variant, or any line over 500 chars.
      2. Repetition loop    — one identical line is >= 40% of all lines.
      3. Template loop      — the same line modulo digits, repeated back to back.
      4. Block loop         — a long page carrying very few distinct lines.
    """
    if not text:
        return "", ""

    # Dot-leader runaway — consecutive (.....) or spaced (. . . .) or just very long lines
    for line in text.splitlines():
        if _LEADER_RE.search(line):
            return "leader", f"leader runaway in line [{line[:50]}…]"
        if _LEADER_SPACED_RE.search(line):
            return "spaced_leader", f"spaced leader runaway in line [{line[:50]}…]"
        if len(line) > _LEADER_MAX_LINE_LEN:
            return "long_line", f"line too long ({len(line)} chars): [{line[:50]}…]"

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < _REPETITION_MIN_REPEATS:
        return "", ""

    # Repetition loop — one line dominates the page
    counts = Counter(l for l in lines if len(l) >= _REPETITION_MIN_LINE_LEN)
    if counts:
        top_line, top_count = counts.most_common(1)[0]
        ratio = top_count / len(lines)
        if ratio >= _REPETITION_THRESHOLD:
            return "repetition", f"repetition loop ({ratio:.0%} of lines are [{top_line[:40]}…])"

    # Template loop — consecutive run of the same line once digits are collapsed
    run_template, run_len = "", 0
    for line in lines:
        template = _DIGIT_RUN_RE.sub("#", line)
        if len(template) < _TEMPLATE_MIN_NORM_LEN:
            run_template, run_len = "", 0
            continue
        if template == run_template:
            run_len += 1
            if (run_len >= _TEMPLATE_MIN_CONSEC_RUN
                    and run_len / len(lines) >= _TEMPLATE_MIN_PAGE_SHARE):
                return "template_loop", (
                    f"template loop ({run_len} consecutive lines, "
                    f"{run_len / len(lines):.0%} of the page, matching [{template[:40]}…])"
                )
        else:
            run_template, run_len = template, 1

    # Block loop — the page cycles a handful of lines for its whole length
    if len(lines) >= _BLOCK_LOOP_MIN_LINES:
        distinct = len(set(lines)) / len(lines)
        if distinct < _BLOCK_LOOP_MAX_DISTINCT:
            return "block_loop", (
                f"block loop ({len(set(lines))} distinct lines in {len(lines)}, "
                f"{distinct:.1%})"
            )

    return "", ""


def _distinct_line_count(text: str) -> int:
    """Number of distinct non-empty lines — the ranking key for imperfect output.

    Raw length is the wrong measure when choosing among failed candidates: both
    failure modes the ladder exists to catch (repetition loops and leader
    runaways) make output *longer*, so preferring length systematically selects
    the most degenerate candidate. Distinct lines cannot be inflated by a loop.
    """
    return len({l.strip() for l in text.splitlines() if l.strip()})


def _thinking_config_for(model: str) -> ThinkingConfig:
    """Thinking budget for *model* — see utils.gemini.thinking_config_for."""
    return thinking_config_for(model)


def _call_gemini(
    client: genai.Client,
    img_bytes: bytes,
    image_name: str,
    model: str,
    system_prompt: str,
    media_resolution: "MediaResolution | None",
    temperature: float,
    *,
    service_tier: "str | None",
    max_output_tokens: "int | None",
    reason: str = "initial",
):
    """Call Gemini with 429-retry and 503-retry logic. Returns the response object.

    *service_tier* and *max_output_tokens* are keyword-only and have no defaults
    on purpose: every call on the retry ladder must carry the same billing tier
    and output ceiling as the first one. Defaulting them made a forgotten
    argument silently drop the run off the flex tier — a doubled bill on exactly
    the pages that retry most.
    """
    return generate_with_retry(
        client,
        model=model,
        config=GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            media_resolution=media_resolution,
            max_output_tokens=max_output_tokens,
            thinking_config=_thinking_config_for(model),
            http_options=flex_http_options(service_tier),
        ),
        contents=[Part.from_bytes(data=img_bytes, mime_type="image/jpeg")],
        reason=reason,
        label=image_name,
        log=_log,
    )


# Versions before 6b165df wrote this placeholder into the .txt when every attempt
# failed. It is not a transcription, but a non-empty file used to count as done,
# so the page was skipped on every later run while align and extract read the
# placeholder as page text. 17 were still in output/ on 2026-10-17.
_FAILURE_PLACEHOLDER = "[OCR FAILED"


def _is_failure_placeholder(txt_path: Path) -> bool:
    """True if *txt_path* holds a failure placeholder rather than page text."""
    if txt_path.stat().st_size > 512:   # the placeholders are one short line
        return False
    try:
        head = txt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return head.lstrip().startswith(_FAILURE_PLACEHOLDER)


def process_image(
    client: genai.Client,
    image_path: Path,
    model: str,
    system_prompt: str,
    media_resolution: "MediaResolution | None" = None,
    service_tier: str | None = None,
    fallback_model: str | None = None,
    skip_blank: bool = True,
    max_output_tokens: "int | None" = None,
) -> tuple[str, bool | None]:
    """
    OCR one image via Gemini. Returns (status, success) where status is one of
    'skipped', 'blank', 'corrupt', 'empty', 'ok', 'failed'.
    """
    txt_path = image_path.parent / f"{image_path.stem}_{model_slug(model)}.txt"
    if txt_path.exists():
        if txt_path.stat().st_size > 0 and not _is_failure_placeholder(txt_path):
            return "skipped", None
        # Empty, or a placeholder from a run that failed every attempt: either
        # way the page was never transcribed. Delete and retry.
        _log(f"  Re-running (no transcription in output file): {txt_path.name}")
        txt_path.unlink()

    # An interrupted download leaves an empty or truncated image. The blank
    # check cannot read one, and the API would be paid to read half a page, so
    # fetch it again first. If that fails the page is 'corrupt', which, like
    # 'failed', keeps the stage from being recorded complete.
    if not image_is_complete(image_path):
        _log(f"  Corrupt image — downloading again: {image_path.name}")
        if not redownload_page(image_path):
            return "corrupt", None

    # Blank pages are hallucination magnets — the model invents the text the
    # prompt describes instead of returning nothing. Detect and skip them
    # before spending an API call. No .txt is written, so downstream stages
    # (align, extract) never see the page; re-runs just re-check (cheap).
    #
    # Because a skip is silent, the check is tuned to return False when in
    # doubt. Until 2026-10-16 it measured a 512 px thumbnail against a 0.5x
    # background cutoff and called full pages of text blank — small type on
    # toned paper, grey-printed Green Books, light microfilm. See is_blank_page
    # and docs/evidence/2026-10-16-blank-page/. --no-blank-skip remains for runs
    # that must OCR every page regardless.
    if skip_blank and is_blank_page(image_path):
        return "blank", None

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    def call(prompt: str, temperature: float, call_model: str = model,
             reason: str = "initial", cap: "int | None" = None):
        """One API call carrying this page's tier and output ceiling.

        *cap* narrows max_output_tokens for this call only — see _retry_cap.
        """
        return _call_gemini(
            client, img_bytes, image_path.name, call_model, prompt,
            media_resolution, temperature=temperature,
            service_tier=service_tier,
            max_output_tokens=max_output_tokens if cap is None else cap,
            reason=reason,
        )

    response = call(system_prompt, 0.0)

    text = response.text or ""

    # Determine whether this output needs a retry.
    # Empty text: retry only on RECITATION (STOP with no text = genuinely blank page).
    # Non-empty text: retry on repetition loop or dot-leader runaway.
    if not text:
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.finish_reason == FinishReason.RECITATION:
            needs_retry, kind, reason = True, "recitation", "RECITATION"
        else:
            needs_retry, kind, reason = False, "", ""
            if candidate:
                _log(f"  finish_reason: {candidate.finish_reason}  [{image_path.name}]")
                if candidate.safety_ratings:
                    _log(f"  safety_ratings: {candidate.safety_ratings}  [{image_path.name}]")
            # Write nothing rather than a zero-byte .txt. The empty file was
            # reported as a processed page, and the resume check deletes and
            # re-requests it on every later run — paying for the page again each
            # time with no record that it had already come back empty.
            return "empty", None
    else:
        # Non-empty but cut off at the output ceiling: the tail of the page is
        # missing. Retrying at a different temperature cannot recover it, so warn
        # loudly rather than silently writing a short page and reporting success.
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.finish_reason == FinishReason.MAX_TOKENS:
            _log(
                f"  WARNING: truncated at the output token limit (MAX_TOKENS) — "
                f"transcription is incomplete ({len(text.splitlines())} lines): {image_path.name}"
            )
        kind, reason = _output_issue(text)
        needs_retry = bool(kind)

    if needs_retry:
        is_leader_issue = kind in _LEADER_KINDS
        is_loop_issue = kind in _LOOP_KINDS
        retry_prompt = (system_prompt + _NO_LEADER_INSTRUCTION) if is_leader_issue else system_prompt

        # Every attempt is kept as a candidate and the best one is chosen at the
        # end. The ladder used to assign last-wins, so a page whose first call
        # returned 240 good lines could be saved as a 12-line temperature-0.7
        # response — the variable was named best_text but never compared anything.
        candidates: "list[tuple[str, str]]" = [(text, kind)]

        def best() -> "tuple[str, str]":
            """Pick the strongest candidate: clean beats flawed, then most distinct lines."""
            usable = [(t, k) for t, k in candidates if t]
            if not usable:
                return "", ""
            clean = [(t, k) for t, k in usable if not k]
            pool = clean or usable
            return max(pool, key=lambda c: _distinct_line_count(c[0]))

        # Temperature retries. Skipped for degenerate loops: across the 10 harvest
        # logs in docs/evidence/2026-09-02-ocr-token-burn/, 14 pages entered this
        # rung and no loop page was ever rescued by it, while the simple prompt
        # below rescued 3 of the 6 that reached it. Recitation and leader failures
        # do respond to resampling, so they keep the full ladder.
        if is_loop_issue:
            _log(f"  {reason} — skipping temperature retries (ineffective for loops): {image_path.name}")
        for temp in ([] if is_loop_issue else _RETRY_TEMPERATURES):
            _log(f"  {reason} — retrying at temperature={temp}: {image_path.name}")
            retry_text = call(retry_prompt, temp, reason=f"temp:{kind}",
                              cap=_retry_cap(candidates, max_output_tokens)).text or ""
            retry_kind, retry_reason = _output_issue(retry_text)
            if retry_text:
                candidates.append((retry_text, retry_kind))
                if not retry_kind:
                    break  # clean output — accept and stop retrying
                _log(f"  temperature={temp} still problematic ({retry_reason}): {image_path.name}")

        text, kind = best()
        if kind and is_leader_issue:
            cleaned = _clean_output(text)
            candidates.append((cleaned, _output_issue(cleaned)[0]))
            text, kind = best()

        # Strategy 1: simple prompt — complex prompts can themselves trigger
        # recitation on visually dense pages, and empirically this is the rung
        # that actually rescues repetition loops.
        #
        # Guarded on "no usable output yet", not on "best candidate is flawed".
        # A page blocked for RECITATION on every attempt has only empty
        # candidates, so best() returns ("", "") and a `if kind:` guard reads
        # false — skipping this rung and the escalation below, and failing the
        # page having tried nothing but temperature. That is the exact case this
        # rung was written for. Found on the 1869 Rowell select lists, where two
        # of eight validation pages failed this way.
        if kind or not text:
            for temp in (0.3, 0.7):
                _log(f"  Trying simple fallback prompt at temperature={temp}: {image_path.name}")
                simple_text = call(_SIMPLE_FALLBACK_PROMPT, temp, reason=f"simple:{kind}",
                                   cap=_retry_cap(candidates, max_output_tokens)).text or ""
                if simple_text:
                    simple_kind, _ = _output_issue(simple_text)
                    candidates.append((simple_text, simple_kind))
                    if not simple_kind:
                        _log(f"  Simple prompt succeeded at temperature={temp}: {image_path.name}")
                        break
            text, kind = best()

        # Strategy 2: model escalation — a larger model handles layout complexity
        # that causes recitation in smaller ones. Same guard as above: a wholly
        # blocked page must reach here too.
        if kind or not text:
            esc_model = fallback_model
            if esc_model and esc_model != model:
                _log(f"  Escalating to {esc_model}: {image_path.name}")
                esc_text = call(_SIMPLE_FALLBACK_PROMPT, 0.0, esc_model,
                                reason=f"escalate:{kind}",
                                cap=_retry_cap(candidates, max_output_tokens)).text or ""
                if esc_text:
                    esc_kind, _ = _output_issue(esc_text)
                    candidates.append((esc_text, esc_kind))
                    if not esc_kind:
                        _log(f"  Model escalation succeeded ({esc_model}): {image_path.name}")
                    else:
                        _log(f"  Model escalation: imperfect output ({esc_model}): {image_path.name}")
                text, kind = best()

        if not text:
            _log(f"  All retries exhausted — no output recovered: {image_path.name}")
            return "failed", False
        if kind:
            _log(f"  All retries exhausted — keeping best available output: {image_path.name}")

    txt_path.write_text(text, encoding="utf-8")
    return "ok", True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini OCR on downloaded NYPL images, producing .txt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Root images directory to process (e.g. output/travelguide)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_OCR_MODEL,
        help=f"Gemini model name (default: {DEFAULT_OCR_MODEL})",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Number of parallel API requests (default: 4). Vision APIs tend to "
            "throttle sustained high parallelism by stalling rather than returning "
            "429, so if runs hang, lower this (2 is a safe floor) rather than "
            "raising it. Re-runs are cheap: completed pages (existing .txt) are "
            "skipped, so an interrupted run resumes where it left off."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        help=(
            "Path to a custom OCR system prompt (default: prompts/ocr_prompt.md). "
            "Use generate_prompt.py to produce a volume-specific prompt."
        ),
    )
    parser.add_argument(
        "--expand-dittos",
        dest="expand_dittos",
        action="store_true",
        help=(
            "Expand ditto marks in place rather than transcribing them literally. "
            "Ditto marks (printed as '' or 〃) are common in tabular historical "
            "documents and are often misread as 66. When this flag is set, the model "
            "is instructed to write out the repeated value on each row instead."
        ),
    )
    parser.add_argument(
        "--high-res",
        dest="high_res",
        action="store_true",
        help=(
            "Send images at high media resolution (more detail, higher token cost). "
            "Auto-enabled when the OCR prompt mentions handwriting or manuscripts."
        ),
    )
    parser.add_argument(
        "--flex",
        action="store_true",
        help=(
            "Use Flex inference (service_tier='flex'): ~50%% cheaper, "
            "1–15 min latency per request, best-effort availability."
        ),
    )
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL,
        metavar="MODEL",
        help=(
            f"Model to escalate to when the primary model exhausts all retries "
            f"with no usable output (default: {FALLBACK_MODEL}). "
            "Pass an empty string to disable escalation."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        metavar="N",
        help=(
            f"Cap on output tokens per OCR call (default: {DEFAULT_MAX_OUTPUT_TOKENS}). "
            "Without a cap a degenerating page runs to the model ceiling — one "
            "observed run emitted 32,758 lines before stopping at MAX_TOKENS. The "
            "largest legitimate page measured across this corpus is ~29KB, roughly "
            "8-12k tokens, so the default leaves about 2x headroom. Truncation is "
            "reported as a WARNING and the partial page is still written, so raise "
            "this if dense pages start coming back short. Pass 0 for no cap."
        ),
    )
    parser.add_argument(
        "--sections",
        metavar="PATH",
        help=(
            "Path to sections.txt marking section boundaries. "
            "When provided, each page is OCR'd with the prompt for its section "
            "(ocr_prompt_{label}.md) falling back to ocr_prompt.md if not found."
        ),
    )
    parser.add_argument(
        "--no-blank-skip",
        dest="no_blank_skip",
        action="store_true",
        help=(
            "Do not skip pages the blank-page heuristic flags. A skipped page "
            "writes no .txt and silently drops out of align and extract. The "
            "heuristic is tuned to keep any page with measurable ink, but use "
            "this when every page must be OCR'd regardless — evaluation runs, "
            "or a volume whose content is a single faint line per page."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = parser.parse_args()

    # Images root (needed for prompt auto-discovery below)
    output_root = Path(args.output_dir)

    # System prompt — explicit flag > volume-specific > global default
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
    else:
        prompt_path = _find_prompt(output_root, PROMPT_FILE)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8")
    if args.expand_dittos:
        system_prompt = system_prompt.rstrip() + _DITTO_INSTRUCTION
    if not args.quiet and prompt_path != PROMPT_FILE:
        print(f"Using volume prompt: {prompt_path}", file=sys.stderr)
    if not args.quiet and args.expand_dittos:
        print("Ditto mark expansion: enabled", file=sys.stderr)

    # Media resolution: explicit flag > auto-detect from prompt > None (API default)
    if args.high_res:
        media_resolution = MediaResolution.MEDIA_RESOLUTION_HIGH
    elif _needs_high_res(system_prompt):
        media_resolution = MediaResolution.MEDIA_RESOLUTION_HIGH
        if not args.quiet:
            print("High media resolution: auto-enabled (handwriting detected in prompt)", file=sys.stderr)
    else:
        media_resolution = None
    if not output_root.exists():
        print(f"Error: directory not found: {output_root}", file=sys.stderr)
        sys.exit(1)

    all_jpgs = sorted(output_root.rglob("*.jpg"))
    images = []
    for p in all_jpgs:
        # Skip visualization output files
        if p.stem.endswith("_viz"):
            continue
        # Always include split output files
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        # Skip the original if split files exist (OCR the splits instead)
        left = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        images.append(p)
    if not images:
        print(f"No .jpg files found under {output_root}", file=sys.stderr)
        sys.exit(0)

    # Apply scope filter (included_pages.txt) if present
    scope = _load_scope(output_root)
    if scope is not None:
        before = len(images)
        images = [p for p in images if p.name in scope]
        if not args.quiet:
            print(
                f"Scope filter: {len(images)} of {before} pages included"
                f" (from included_pages.txt)",
                file=sys.stderr,
            )

    service_tier = "flex" if args.flex else None

    # --- Per-page prompt mapping (sections mode) ----------------------------
    img_prompts: dict[Path, str] = {}
    if args.sections:
        from utils.section_utils import load_sections, prompt_for_page

        sections_path = Path(args.sections)
        if not sections_path.exists():
            print(f"Error: sections file not found: {sections_path}", file=sys.stderr)
            sys.exit(1)
        all_names = [p.name for p in images]
        try:
            sections = load_sections(sections_path, all_names)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Resolve slug_dir — where section-specific prompt files live
        root = output_root.resolve()
        slug_dir = root
        for candidate in (root, root.parent):
            if (candidate / "ocr_prompt.md").exists() or any(candidate.glob("ocr_prompt_*.md")):
                slug_dir = candidate
                break

        prompt_cache: dict[str, str] = {}
        for img in images:
            ppath = prompt_for_page(img.name, sections, all_names, slug_dir, "ocr_prompt")
            key = str(ppath)
            if key not in prompt_cache:
                if ppath.exists():
                    text = ppath.read_text(encoding="utf-8")
                else:
                    text = system_prompt  # fall through to generic
                if args.expand_dittos:
                    text = text.rstrip() + _DITTO_INSTRUCTION
                prompt_cache[key] = text
            img_prompts[img] = prompt_cache[key]

        if not args.quiet:
            unique_prompts = len(set(img_prompts.values()))
            print(
                f"Sections mode: {len(sections)} section(s), {unique_prompts} distinct prompt(s)",
                file=sys.stderr,
            )
    else:
        img_prompts = {img: system_prompt for img in images}

    total = len(images)
    if not args.quiet:
        tier_note = " [flex inference]" if service_tier else ""
        print(
            f"Processing {total} image(s) with {args.workers} worker(s) using {args.model}{tier_note}…",
            file=sys.stderr,
        )

    client = get_client()
    counts = {"ok": 0, "skipped": 0, "blank": 0, "corrupt": 0, "empty": 0, "failed": 0}
    completed = 0

    fallback_model = args.fallback_model.strip() or None
    if fallback_model and not args.quiet:
        print(f"Fallback model (on exhaustion): {fallback_model}", file=sys.stderr)

    state_dir = find_state_dir(output_root)
    pending_pages = 0

    def _flush_usage() -> None:
        """Persist usage accrued since the last flush. Safe to call repeatedly."""
        nonlocal pending_pages
        drained = drain_usage()
        if drained or pending_pages:
            add_usage(state_dir, drained, pages=pending_pages)
            pending_pages = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_image, client, img, args.model, img_prompts[img], media_resolution, service_tier, fallback_model, not args.no_blank_skip, args.max_output_tokens or None): img
                for img in images
            }
            for future in as_completed(futures):
                image_path = futures[future]
                completed += 1
                try:
                    status, _ = future.result()
                except Exception as exc:  # noqa: BLE001
                    status = "failed"
                    _log(f"Warning: exception processing {image_path}: {exc}")

                counts[status] += 1

                # Only pages that actually called the API are billed pages. An
                # empty response is one: it paid for its prompt and its usage is
                # already in the totals, so leaving its page out overstated the
                # cost per page.
                if status in ("ok", "failed", "empty"):
                    pending_pages += 1
                    if pending_pages >= USAGE_FLUSH_EVERY:
                        _flush_usage()

                if not args.quiet:
                    txt_name = f"{image_path.stem}_{model_slug(args.model)}.txt"
                    if status == "skipped":
                        _log(f"[{completed:04d}/{total}] Skipped (exists): {txt_name}")
                    elif status == "blank":
                        _log(f"[{completed:04d}/{total}] Blank page — no OCR: {image_path.name}")
                    elif status == "corrupt":
                        _log(f"[{completed:04d}/{total}] CORRUPT image, download failed — no OCR: {image_path.name}")
                    elif status == "empty":
                        _log(f"[{completed:04d}/{total}] Empty response — nothing written: {image_path.name}")
                    elif status == "ok":
                        _log(f"[{completed:04d}/{total}] Done: {txt_name}")
                    else:
                        _log(f"[{completed:04d}/{total}] FAILED: {image_path.name}")
    finally:
        # Runs on Ctrl-C and on an unhandled error too: the calls were billed
        # whether or not the run reached its end.
        _flush_usage()

    if not args.quiet:
        print(
            f"\nDone. {total} image(s): "
            f"{counts['ok']} processed, {counts['skipped']} skipped, "
            f"{counts['blank']} blank, {counts['corrupt']} corrupt, "
            f"{counts['empty']} empty, {counts['failed']} failed.",
            file=sys.stderr,
        )

    # Record the stage so a direct invocation updates pipeline_state.json the
    # same way an orchestrated run does (main.py is otherwise the only writer).
    # Gated on there being no failures, not merely one success: recording the
    # stage complete after 1 of 83 pages made a 1%-finished run indistinguishable
    # from a clean one, and main.py trusts this file.
    if counts["ok"] and not counts["failed"] and not counts["corrupt"]:
        record_stage(find_state_dir(output_root), "gemini_ocr",
                     updates={"ocr_model": args.model})

    if counts["failed"] or counts["corrupt"]:
        print(
            f"Error: {counts['failed']} page(s) failed and {counts['corrupt']} image(s) "
            "are corrupt and could not be downloaded again; stage not recorded as "
            "complete. Re-run to retry them — neither writes a .txt, so they are "
            "not skipped.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
