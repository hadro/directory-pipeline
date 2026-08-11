"""Tests for analysis/fix_entries.py — the sentinel-token contract and
name-based category inference.

Sentinel tokens ([illegible]/[blank], see prompts/README.md) must be treated as
explicit gap markers: folded into the placeholder set so a page where a field
legitimately repeats a sentinel is not mistaken for hallucinated ghost text.

Category inference must only ever emit values from CANONICAL_CATEGORIES — an
off-vocabulary result silently splits the explorer's facets, and because
`_INFER_TARGET_CATEGORIES` gates on a small set of rewritable values, a bad
category is never corrected on a later pass.
"""

import pytest

from analysis.fix_entries import (
    CANONICAL_CATEGORIES,
    SENTINEL_TOKENS,
    _INFER_RULES,
    _MARKER_RULES,
    _NER_PLACEHOLDER_VALUES,
    find_hallucinated_pages,
    infer_category_from_name,
)


def _rows(proprietor: str, n: int = 5, image: str = "p1.jpg"):
    return [{"image": image, "name": f"Place {i}", "proprietor": proprietor}
            for i in range(n)]


class TestSentinelContract:
    def test_tokens_are_the_documented_pair(self):
        assert SENTINEL_TOKENS == frozenset({"[illegible]", "[blank]"})

    def test_sentinels_folded_into_placeholder_values(self):
        assert SENTINEL_TOKENS <= _NER_PLACEHOLDER_VALUES


class TestHallucinationDetectionIgnoresSentinels:
    def test_real_repeated_proprietor_is_flagged(self):
        # 5 identical real proprietors on one page → ghost-text signature.
        flagged = find_hallucinated_pages(_rows("John Q. Manager"))
        assert flagged == {0, 1, 2, 3, 4}

    def test_repeated_illegible_is_not_flagged(self):
        # The same applies to [blank]; sentinels legitimately repeat.
        assert find_hallucinated_pages(_rows("[illegible]")) == set()
        assert find_hallucinated_pages(_rows("[blank]")) == set()


class TestCategoryInferenceVocabulary:
    def test_every_rule_targets_a_canonical_category(self):
        off_vocab = {cat for _, cat in _INFER_RULES if cat not in CANONICAL_CATEGORIES}
        assert not off_vocab

    def test_canonical_categories_are_unique(self):
        assert len(CANONICAL_CATEGORIES) == len(set(CANONICAL_CATEGORIES))

    def test_unmatched_name_returns_none_rather_than_a_default(self):
        # No rule may invent a fallback — an unmatched name stays empty so a
        # later pass can still infer it.
        assert infer_category_from_name("MRS. ROSE ALLEN") is None
        assert infer_category_from_name("") is None


class TestLodgingInference:
    """The travel guides are dominated by guest houses, Ys, and resorts."""

    @pytest.mark.parametrize("name", [
        "BLUE RIBBON GUEST HOUSE",
        "The Blue Ribbon Guest House",   # inference is case-insensitive
        "ANDERSON GUEST HOME",
        "SEASIDE GUEST COTTAGE",
    ])
    def test_guest_lodging_is_a_tourist_home(self, name):
        assert infer_category_from_name(name) == "TOURIST HOMES"

    @pytest.mark.parametrize("name", ["YMCA", "Y.M.C.A.", "Y. M. C. A.", "Wabash YMCA"])
    def test_ymca_variants(self, name):
        assert infer_category_from_name(name) == "YMCA"

    @pytest.mark.parametrize("name", ["YWCA", "Y.W.C.A.", "Y. W. C. A."])
    def test_ywca_is_not_swallowed_by_the_ymca_rule(self, name):
        assert infer_category_from_name(name) == "YWCA"

    def test_bare_resort_defaults_to_vacation_resorts(self):
        assert infer_category_from_name("EVERGREEN RESORT") == "VACATION RESORTS"

    def test_seasonal_phrasing_wins_over_the_bare_word(self):
        # More specific rules must precede the bare "resort" rule.
        assert infer_category_from_name("IDLEWILD SUMMER RESORT") == "SUMMER RESORTS"

    @pytest.mark.parametrize("name, expected", [
        ("STEWART'S HOTEL", "HOTELS"),
        ("LIN-MAR MOTEL", "HOTELS"),
        ("SUNSET LODGE", "HOTELS"),
        ("JOHNSON'S RESTAURANT", "RESTAURANTS"),
        ("TOURIST HOME", "TOURIST HOMES"),
    ])
    def test_preexisting_rules_still_win_where_they_should(self, name, expected):
        assert infer_category_from_name(name) == expected


class TestTravelAgencyInference:
    @pytest.mark.parametrize("name", [
        "DONALD SMITH TRAVEL AGENCY",
        "VALLENTIN AGUIRRE TRAVEL AGENCY",
        "VACATION TRAVEL SERVICE",
        "J. BRYON TRAVEL SERVICE",
        "HARLEM TRAVEL BUREAU",
        "FRENCH-CANADA TOURS",
        "ACME TOURIST AGENCY",
    ])
    def test_travel_business_names(self, name):
        assert infer_category_from_name(name) == "TRAVEL AGENCIES"

    @pytest.mark.parametrize("prose", [
        "travelling is the AAA tour",   # bare \btravel\b would match
        "a pause in travel",
        "book a tour of the grounds",   # singular "tour" is not a business name
    ])
    def test_bare_travel_words_do_not_match(self, prose):
        assert infer_category_from_name(prose) is None


class TestNoteFieldScoping:
    """The broad keyword sweep sees names only; note fields carry ad slogans
    and proprietor lines, and a wrong category inferred from them is permanent
    (_INFER_TARGET_CATEGORIES never revisits a non-empty, non-General value)."""

    @pytest.mark.parametrize("name, notes", [
        ("SEASIDE VILLA", "FLORIDA'S MOST MODERN RESORT"),
        ("JOHNSON RESIDENCE", "5 Minutes Walk to Boardwalk, Modern Kitchen"),
        ("SMITH HOUSE", "Rooms with bath, near the grill"),
    ])
    def test_slogans_in_notes_do_not_set_a_category(self, name, notes):
        assert infer_category_from_name(name, notes) is None

    def test_name_keyword_wins_over_conflicting_note_text(self):
        # Pre-change this returned RESTAURANTS, from "restaurant" in the notes.
        assert infer_category_from_name(
            "ROYAL HOTEL", "TELEPHONE AND RADIO IN ALL ROOMS, restaurant"
        ) == "HOTELS"

    @pytest.mark.parametrize("notes", ["(Guests)", "(Guest)", "( guests )", "Mrs. Allen (Guests)"])
    def test_guest_marker_in_notes_is_a_tourist_home(self, notes):
        # Guest listings are person names no keyword rule can reach.
        assert infer_category_from_name("MRS. ROSE ALLEN", notes) == "TOURIST HOMES"

    def test_guest_marker_in_the_name_also_works(self):
        assert infer_category_from_name("W. A. GILLUM (Guests)") == "TOURIST HOMES"

    def test_compound_phrase_in_notes_is_a_genuine_signal(self):
        assert infer_category_from_name(
            "Hilton G. Hill, Inc.", "An International Travel Agency, Phone: LO 5-1234"
        ) == "TRAVEL AGENCIES"

    def test_marker_rules_target_canonical_categories(self):
        assert not {cat for _, cat in _MARKER_RULES if cat not in CANONICAL_CATEGORIES}
