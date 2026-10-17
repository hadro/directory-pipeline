"""Advertisement delimiters on the aligned extraction path.

Gemini marks advertisement blocks with ``=== ADVERTISEMENT ===`` and the
advertisement NER prompts extract one record per delimiter pair. align_ocr
consumes those lines to keep ad text out of the NW anchor set and then drops
them — they appear in neither "lines" nor "unmatched_gemini" — so rebuilding the
page from the aligned JSON handed NER 468 advertisements with no boundaries on
the 1869 Rowell volume alone, and 11,756 delimiter lines across the four
collections that use the convention.

The rule is deliberately narrow: raw OCR text is used *only* on pages that carry
delimiters. Raw and aligned-derived text differ on 72% of pages, so switching
wholesale would change NER input for every existing collection.
"""

import re

import pytest

from pipeline.extract_entries import _page_text_for_ner, _page_text_from_aligned


def _aligned(*texts, unmatched=()):
    return {
        "lines": [{"gemini_text": t, "bbox": [0, 0, 1, 1]} for t in texts],
        "unmatched_gemini": list(unmatched),
    }


def _ocr(tmp_path, body):
    p = tmp_path / "0001_page_gemini-x.txt"
    p.write_text(body, encoding="utf-8")
    return p


AD_PAGE = """\
=== ADVERTISEMENT ===
THE MINNEAPOLIS TRIBUNE.
Daily, Tri-Weekly and Weekly.
=== END ADVERTISEMENT ===
=== ADVERTISEMENT ===
THE ST. PAUL PRESS.
=== END ADVERTISEMENT ===
"""


def test_delimiters_survive_on_an_advertisement_page(tmp_path):
    aligned = _aligned("THE MINNEAPOLIS TRIBUNE.", "Daily, Tri-Weekly and Weekly.",
                       "THE ST. PAUL PRESS.")
    out = _page_text_for_ner(aligned, _ocr(tmp_path, AD_PAGE))
    assert out.count("=== ADVERTISEMENT ===") == 2
    assert out.count("=== END ADVERTISEMENT ===") == 2
    assert "THE ST. PAUL PRESS." in out


def test_a_page_without_delimiters_is_left_exactly_as_before(tmp_path):
    """The narrow rule: no delimiters means no behaviour change at all."""
    aligned = _aligned("ABBEVILLE, Henry Co. Register; Saturdays;", "democratic; four pages;")
    plain = "ABBEVILLE, Henry Co. Register; Saturdays;\ndemocratic; four pages;\n"
    assert _page_text_for_ner(aligned, _ocr(tmp_path, plain)) == _page_text_from_aligned(aligned)


def test_no_ocr_file_falls_back_to_the_aligned_rebuild():
    aligned = _aligned("one line", "another line")
    assert _page_text_for_ner(aligned, None) == _page_text_from_aligned(aligned)


def test_an_unreadable_ocr_file_falls_back(tmp_path):
    missing = tmp_path / "gone.txt"
    aligned = _aligned("one line")
    assert _page_text_for_ner(aligned, missing) == _page_text_from_aligned(aligned)


def test_printed_order_is_preserved_on_a_delimiter_page(tmp_path):
    """The aligned rebuild appends unmatched lines after the matched ones.

    On an ad page that scrambles the very boundaries the prompt depends on.
    """
    aligned = _aligned("THE MINNEAPOLIS TRIBUNE.", "THE ST. PAUL PRESS.",
                       unmatched=["Daily, Tri-Weekly and Weekly."])
    out = _page_text_for_ner(aligned, _ocr(tmp_path, AD_PAGE)).splitlines()
    assert out.index("Daily, Tri-Weekly and Weekly.") < out.index("THE ST. PAUL PRESS.")


def test_short_display_lines_are_kept_on_a_delimiter_page(tmp_path):
    """The aligned rebuild drops unmatched fragments of 8 characters or fewer.

    Display advertising is full of them — ornament rules and one-word lines —
    and on one sampled page that cost 28 lines out of 109.
    """
    body = "=== ADVERTISEMENT ===\n---\nPUBLISHED AT\n---\n=== END ADVERTISEMENT ===\n"
    out = _page_text_for_ner(_aligned("PUBLISHED AT"), _ocr(tmp_path, body))
    assert out.count("---") == 2


@pytest.mark.parametrize("line", ["=== ADVERTISEMENT ===", "=== END ADVERTISEMENT ==="])
def test_detection_requires_a_full_line(tmp_path, line):
    """A mid-sentence '=== ' must not switch the page onto the raw path."""
    aligned = _aligned("a line")
    body = f"text mentioning {line} inline\nanother line\n"
    assert _page_text_for_ner(aligned, _ocr(tmp_path, body)) == _page_text_from_aligned(aligned)


# --- unmatched-rate outlier reporting ---------------------------------------
#
# needs_review keys on Surya confidence, so a page losing 93% of its lines to
# region blobs sails through — the detector was confident about its blobs. The
# cutoff has to be relative: median unmatched runs 0.0%-29.6% across the 50
# collections on disk, and the Green Books pair a 2.0% median with a 56% p95.

from pipeline.align_ocr import _report_unmatched_outliers


def _collect(rates):
    out = []
    _report_unmatched_outliers(rates, out.append)
    return "\n".join(out)


def test_a_clean_volume_reports_nothing():
    """Every page under the floor: reporting the best pages as the worst is noise."""
    assert _collect([(f"p{i}", 0.5) for i in range(50)]) == ""


def test_the_catastrophic_page_is_reported():
    rates = [(f"p{i}", 0.5) for i in range(50)] + [("blobbed.json", 93.4)]
    out = _collect(rates)
    assert "blobbed.json" in out and "93.4" in out
    assert "--blank-threshold" in out, "should point at the usual cause"


def test_a_high_baseline_volume_is_not_flooded():
    """Manuscript material runs high by nature — 20% is normal, not a fault."""
    rates = [(f"p{i}", 20.0) for i in range(200)]
    assert _collect(rates) == ""


def test_a_heavy_tail_reports_at_most_five_percent():
    """A multiple of the median reported 1,846 of 5,788 Green Book pages."""
    # Spread the tail: a constant-valued tail puts p95 at the maximum, so
    # nothing exceeds it and the rule correctly reports nothing.
    rates = ([(f"low{i}", 2.0) for i in range(800)]
             + [(f"high{i}", 20.0 + i * 0.35) for i in range(200)])
    out = _collect(rates)
    reported = int(re.search(r"⚠\s+(\d+) page", out).group(1))
    assert reported <= len(rates) * 0.05 + 1, f"reported {reported} of {len(rates)}"


def test_a_tiny_volume_is_skipped():
    """A percentile over five pages means nothing."""
    assert _collect([(f"p{i}", 90.0) for i in range(5)]) == ""


def test_the_list_is_capped_but_the_count_is_not():
    rates = [(f"p{i}", 1.0) for i in range(500)] + [(f"bad{i}", 80.0) for i in range(25)]
    out = _collect(rates)
    assert "and 10 more" in out
    assert out.count("80.0%") == 15
