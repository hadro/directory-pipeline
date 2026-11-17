"""Tests for utils/entry_binding.py — the shared entry→aligned-line matcher.

The regression tests here are the important half. Each one reproduces a real
misplacement measured in the published Green Book corpus, using the actual entry
and line text that produced it, so a future change that reintroduces the bug
fails loudly rather than quietly re-corrupting 1,200 bindings.
"""

import pytest

from utils.entry_binding import (
    Binding,
    bind_entries,
    identity_text,
    is_edge_bleed,
    order_violations,
    review_queue,
    score,
    tokens,
)


def line(text, fragment="https://ex.org/c/1/full/max/0/default.jpg#xywh=10,20,300,40"):
    return {"gemini_text": text, "canvas_fragment": fragment}


def frag(x, y, w, h):
    return f"https://ex.org/c/1/full/max/0/default.jpg#xywh={x},{y},{w},{h}"


# ---------------------------------------------------------------------------
# identity_text — which fields may supply the match
# ---------------------------------------------------------------------------

class TestIdentityFields:
    def test_takes_name_and_address(self):
        e = {"name": "Chicken Paradise", "address": "1210 U. St., N. W."}
        assert identity_text(e) == "Chicken Paradise 1210 U. St., N. W."

    def test_category_is_never_identity(self):
        """The bug that caused 489 of ~900 attributable misplacements."""
        e = {"category": "BEAUTY PARLORS", "state": "OHIO", "city": "CLEVELAND"}
        assert identity_text(e) == ""

    def test_geographic_aliases_are_not_identity(self):
        """A denylist of four names let all of these through."""
        e = {
            "location_city": "ATLANTIC CITY",
            "state_country": "NEW JERSEY",
            "state_or_province": "ONTARIO",
            "state_region": "NEW YORK STATE",
            "location_primary": "WESTCHESTER",
        }
        assert identity_text(e) == ""

    def test_detail_fields_are_not_identity(self):
        e = {"phone": "4-1184", "notes": "Open all year", "mileage": "365 mi-NYC",
             "price": "$4. up", "manager_owner": "Frank Cardozo, Prop."}
        assert identity_text(e) == ""

    @pytest.mark.parametrize("field", [
        "firm_name", "business_name", "establishment_name", "ship_name",
    ])
    def test_name_suffix_fields_qualify(self, field):
        """Per-collection schemas invent their own name field; shape admits them."""
        assert identity_text({field: "Hotel Theresa"}) == "Hotel Theresa"

    def test_name_suffix_denylist_still_wins(self):
        assert identity_text({"section_name": "HOTELS"}) == ""
        assert identity_text({"category_name": "TAVERNS"}) == ""

    def test_only_first_of_each_kind_is_taken(self):
        e = {"name": "Smith's", "firm_name": "Smith's", "address": "12 Main St"}
        assert identity_text(e) == "Smith's 12 Main St"

    def test_non_string_values_ignored(self):
        assert identity_text({"name": None, "firm_name": "Real Name"}) == "Real Name"
        assert identity_text({"name": 42, "business_name": "Real"}) == "Real"


# ---------------------------------------------------------------------------
# score — agreement strength
# ---------------------------------------------------------------------------

class TestScore:
    def test_exact_match_scores_one(self):
        assert score(tokens("Roma Gardens 257 W 116th St"),
                     tokens("Roma Gardens, 257 W. 116th St.")) == 1.0

    def test_disjoint_scores_zero(self):
        assert score(tokens("Liberty 1519 Baltic Ave"),
                     tokens("TAVERN & LIQUOR STORE")) == 0.0

    def test_single_short_shared_token_rejected(self):
        """name='Ave' must not bind to the first line containing 'Ave'."""
        assert score(tokens("Ave"), tokens("Ave-1538 Wylie Ave.")) == 0.0

    def test_single_long_shared_token_accepted(self):
        assert score(tokens("Metropolitan"), tokens("Metropolitan-")) > 0.0

    def test_two_short_shared_tokens_accepted(self):
        assert score(tokens("Ave B"), tokens("Ave B Grill")) > 0.0

    def test_empty_inputs(self):
        assert score([], tokens("anything")) == 0.0
        assert score(tokens("anything"), []) == 0.0


# ---------------------------------------------------------------------------
# Regression: the real misplacements from the published corpus
# ---------------------------------------------------------------------------

class TestCorpusRegressions:
    def test_category_does_not_steal_the_heading_box(self):
        """Page 0051_58019307: a duplicate 'Liberty' took the heading's box.

        Its name and address were already consumed by the real Liberty entry, so
        the old matcher fell through to category and exact-matched the printed
        'TAVERN & LIQUOR STORE' heading 2,000px down the page.
        """
        lines = [
            line("LIBERTY HOTEL", frag(738, 2182, 588, 57)),
            line("1519 Baltic Avenue Tel. 4-1184", frag(297, 2487, 587, 46)),
            line("TAVERN & LIQUOR STORE", frag(656, 4133, 746, 47)),
            line("WEEKES'-1700-2 Baltic Ave.", frag(250, 4181, 1561, 60)),
        ]
        entries = [
            {"name": "LIBERTY HOTEL", "address": "1519 Baltic Avenue", "category": "HOTELS"},
            {"name": "WEEKES'", "address": "1700-2 Baltic Ave.",
             "category": "TAVERN & LIQUOR STORE"},
            # the spurious duplicate
            {"name": "Liberty", "address": "1519 Baltic Ave.",
             "category": "TAVERN & LIQUOR STORE"},
        ]
        b = bind_entries(entries, lines, canvas_width=2396)
        assert lines[b[1].line_index]["gemini_text"] == "WEEKES'-1700-2 Baltic Ave."
        # The duplicate must NOT take the heading box — refusing is correct.
        assert b[2].line_index is None or "TAVERN" not in lines[b[2].line_index]["gemini_text"]
        assert all(bb.line_index != 2 for bb in b), "heading line was bound to an entry"

    @pytest.mark.parametrize("name,address,heading", [
        ("NAACP", "2416 Binney", "ORGANIZATIONS"),
        ("Roma", "3839 Finney Ave.", "TAVERNS"),
        ("Arcade", "122 E. Hargett St.", "TAILORS"),
        ("The Parker House", "627 Erie Ave.", "BEAUTY PARLORS"),
        ("Anchor Inn", "", "PENNSYLVANIA"),
    ])
    def test_entry_never_binds_to_a_bare_heading(self, name, address, heading):
        lines = [line(heading, frag(700, 100, 400, 50))]
        entries = [{"name": name, "address": address, "category": heading}]
        assert bind_entries(entries, lines, canvas_width=2400)[0].line_index is None

    def test_short_name_does_not_bind_by_substring(self):
        """name='Ave' once matched the earliest line containing that substring."""
        lines = [
            line("Avenue Bar-100 Main St.", frag(10, 10, 500, 40)),
            line("Ave-1538 Wylie Ave.", frag(10, 60, 500, 40)),
        ]
        entries = [{"name": "Ave", "address": "1538 Wylie Ave."}]
        b = bind_entries(entries, lines, canvas_width=2400)
        assert b[0].line_index == 1

    def test_wrapped_entry_binds_to_its_name_line(self):
        """Green Book entries often wrap: name on one line, address on the next."""
        lines = [
            line("Metropolitan-", frag(10, 10, 500, 40)),
            line("1200 Springwood Ave.", frag(10, 60, 500, 40)),
        ]
        entries = [{"name": "Metropolitan", "address": "1200 Springwood Ave."}]
        assert bind_entries(entries, lines, canvas_width=2400)[0].line_index is not None


# ---------------------------------------------------------------------------
# bind_entries — assignment behaviour
# ---------------------------------------------------------------------------

class TestBinding:
    def test_one_to_one_no_line_reused(self):
        lines = [line("Hotel Grand-1 A St.", frag(10, 10, 500, 40))]
        entries = [
            {"name": "Hotel Grand", "address": "1 A St."},
            {"name": "Hotel Grand", "address": "1 A St."},
        ]
        b = bind_entries(entries, lines, canvas_width=2400)
        bound = [x.line_index for x in b if x.line_index is not None]
        assert len(bound) == 1 and len(set(bound)) == 1

    def test_best_match_wins_the_contested_line(self):
        """Greedy-by-score settles the strongest claim first."""
        lines = [line("Hotel Theresa-2090 7th Ave.", frag(10, 10, 500, 40))]
        entries = [
            {"name": "Theresa", "address": ""},                       # weak
            {"name": "Hotel Theresa", "address": "2090 7th Ave."},     # exact
        ]
        b = bind_entries(entries, lines, canvas_width=2400)
        assert b[1].line_index == 0
        assert b[0].line_index is None

    def test_out_of_order_entries_still_bind(self):
        """A monotonic constraint refused these; it was measured and rejected."""
        lines = [
            line("Alpha Hotel-1 A St.", frag(10, 10, 500, 40)),
            line("Beta Hotel-2 B St.", frag(10, 60, 500, 40)),
        ]
        entries = [
            {"name": "Beta Hotel", "address": "2 B St."},
            {"name": "Alpha Hotel", "address": "1 A St."},
        ]
        b = bind_entries(entries, lines, canvas_width=2400)
        assert (b[0].line_index, b[1].line_index) == (1, 0)

    def test_unbound_entry_carries_its_nearest_miss(self):
        lines = [line("Hotel Theresa-2090 7th Ave.", frag(10, 10, 500, 40))]
        entries = [{"name": "Completely Different Place", "address": "99 Nowhere"}]
        b = bind_entries(entries, lines, canvas_width=2400)
        assert b[0].line_index is None
        assert b[0].score == 0.0

    def test_edge_bleed_lines_are_not_offered(self):
        lines = [line("Hotel Grand-1 A St.", frag(0, 100, 40, 40))]   # sliver at x=0
        entries = [{"name": "Hotel Grand", "address": "1 A St."}]
        assert bind_entries(entries, lines, canvas_width=2400)[0].line_index is None

    def test_deterministic_under_tie(self):
        lines = [line("Same Text Here", frag(10, 10, 500, 40)),
                 line("Same Text Here", frag(10, 60, 500, 40))]
        entries = [{"name": "Same Text Here"}]
        first = bind_entries(entries, lines, canvas_width=2400)[0].line_index
        for _ in range(5):
            assert bind_entries(entries, lines, canvas_width=2400)[0].line_index == first

    def test_empty_inputs(self):
        assert bind_entries([], [line("x")], canvas_width=100) == []
        b = bind_entries([{"name": "Anything"}], [], canvas_width=100)
        assert b[0].line_index is None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_order_violations_counts_backwards_jumps(self):
        assert order_violations([Binding(0, 0, 1.0), Binding(1, 1, 1.0)]) == 0
        assert order_violations([Binding(0, 5, 1.0), Binding(1, 2, 1.0)]) == 1
        assert order_violations([Binding(0, 1, 1.0), Binding(1, None, 0.0),
                                 Binding(2, 2, 1.0)]) == 0

    def test_review_queue_reports_only_refusals(self):
        lines = [line("Hotel Theresa-2090 7th Ave.", frag(10, 10, 500, 40)),
                 line("Some Other Place", frag(10, 60, 500, 40))]
        entries = [{"name": "Hotel Theresa", "address": "2090 7th Ave."},
                   {"name": "Nowhere At All", "address": "1 Nothing St"}]
        b = bind_entries(entries, lines, canvas_width=2400)
        q = review_queue(entries, lines, b)
        assert len(q) == 1
        assert q[0]["entry_index"] == 1
        assert "Nowhere At All" in q[0]["entry_text"]


class TestEdgeBleed:
    def test_narrow_strip_at_left_edge(self):
        assert is_edge_bleed(frag(2, 100, 30, 40), 2400) is True

    def test_narrow_strip_at_right_edge(self):
        assert is_edge_bleed(frag(2380, 100, 18, 40), 2400) is True

    def test_wide_line_at_edge_is_fine(self):
        assert is_edge_bleed(frag(0, 100, 2000, 40), 2400) is False

    def test_narrow_strip_mid_page_is_fine(self):
        assert is_edge_bleed(frag(1200, 100, 30, 40), 2400) is False

    def test_missing_inputs(self):
        assert is_edge_bleed("", 2400) is False
        assert is_edge_bleed(frag(0, 0, 10, 10), 0) is False
        assert is_edge_bleed("no-fragment-here", 2400) is False


# ---------------------------------------------------------------------------
# Containment — correct bindings that token-set F1 alone rejects
# ---------------------------------------------------------------------------

class TestContainment:
    """Directory lines carry rates and room codes the entry does not.

    F1 charges the entry for that extra text and pushes real matches under the
    floor. Each case here was measured as a regression against the published
    corpus before containment was added.
    """

    @pytest.mark.parametrize("name,line_text", [
        ("CARVER HOTEL",    "CARVER HOTEL 6-9562 or 2-9138"),
        ("Broadmoor Hotel", "Broadmoor Hotel-(450B) E. $5.25 up A. $1"),
        ("Highland Glen",   "Highland Glen (6) $6 [2-4-7-8/F-P-W]"),
    ])
    def test_diluted_line_still_binds(self, name, line_text):
        from utils.entry_binding import contains_entry, is_match
        et, lt = tokens(name), tokens(line_text)
        assert contains_entry(et, lt)
        assert is_match(et, lt)
        b = bind_entries([{"name": name}], [line(line_text)], canvas_width=2400)
        assert b[0].line_index == 0

    def test_dilution_can_push_a_real_match_under_the_floor(self):
        """The reason containment exists, stated as a fact about the scores.

        Enough tariff notation on the line drags F1 below DEFAULT_FLOOR even
        though every token of the name is present.
        """
        from utils.entry_binding import DEFAULT_FLOOR, contains_entry
        et = tokens("Highland Glen")
        lt = tokens("Highland Glen (6) $6 [2-4-7-8/F-P-W]")
        assert score(et, lt) < DEFAULT_FLOOR
        assert contains_entry(et, lt)

    def test_containment_does_not_readmit_heading_steal(self):
        """The category fix must survive: no shared tokens means no containment."""
        from utils.entry_binding import contains_entry
        assert not contains_entry(tokens("Liberty 1519 Baltic Ave"),
                                  tokens("TAVERN & LIQUOR STORE"))
        assert not contains_entry(tokens("Elm Court Inn"), tokens("SUMMER RESORTS"))

    def test_containment_respects_the_short_token_guard(self):
        """A lone short token is coincidence even when fully contained."""
        from utils.entry_binding import contains_entry
        assert not contains_entry(tokens("Ave"), tokens("Ave-1538 Wylie Ave."))

    def test_partial_overlap_is_not_containment(self):
        from utils.entry_binding import contains_entry
        assert not contains_entry(tokens("Hotel Theresa Harlem"),
                                  tokens("Hotel Theresa"))

    def test_stronger_claim_still_wins_the_line(self):
        """Containment ranks by F1, so it cannot pre-empt a better match."""
        lines = [line("Broadmoor Hotel-(450B) E. $5.25 up")]
        entries = [
            {"name": "Broadmoor"},                                  # contained, diluted
            {"name": "Broadmoor Hotel", "address": "(450B) E."},    # much stronger
        ]
        b = bind_entries(entries, lines, canvas_width=2400)
        assert b[1].line_index == 0
        assert b[0].line_index is None


# ---------------------------------------------------------------------------
# Wrapped entries — name on one line, address indented onto the next
# ---------------------------------------------------------------------------

class TestIdentityVariants:
    """Green Book entries wrap, and each printed line gets its own bbox.

    Scoring only "name + address" charges the entry for address tokens that
    were never on the line it belongs to. Every case here was measured as a
    regression against the published corpus.
    """

    def test_variants_are_name_plus_address_then_name(self):
        from utils.entry_binding import identity_variants
        v = identity_variants({"name": "Hotel Donnelly", "address": "1st & Broadway"})
        assert v == [tokens("Hotel Donnelly 1st & Broadway"), tokens("Hotel Donnelly")]

    def test_no_duplicate_variant_when_there_is_no_address(self):
        from utils.entry_binding import identity_variants
        assert identity_variants({"name": "Hotel Donnelly"}) == [tokens("Hotel Donnelly")]

    @pytest.mark.parametrize("name,address,line_text", [
        ("CARVER HOTEL",   "1401 E. 12th St", "CARVER HOTEL 6-9562 or 2-9138"),
        ("Hotel Donnelly", "1st & Broadway",  "Hotel Donnelly-200 rooms-E. $2.00 up"),
        ("Senator Hotel",  "7th Ave",         "Senator Hotel-58 rooms-E. $2.00 up"),
    ])
    def test_wrapped_entry_binds_via_its_name(self, name, address, line_text):
        entries = [{"name": name, "address": address}]
        b = bind_entries(entries, [line(line_text)], canvas_width=2400)
        assert b[0].line_index == 0

    def test_address_still_discriminates_between_same_named_entries(self):
        """The name-only fallback must not flatten entries sharing a name."""
        lines = [
            line("Grand Hotel-100 First St.", frag(10, 10, 500, 40)),
            line("Grand Hotel-200 Second St.", frag(10, 60, 500, 40)),
        ]
        entries = [
            {"name": "Grand Hotel", "address": "200 Second St."},
            {"name": "Grand Hotel", "address": "100 First St."},
        ]
        b = bind_entries(entries, lines, canvas_width=2400)
        assert (b[0].line_index, b[1].line_index) == (1, 0)

    def test_name_fallback_does_not_readmit_heading_steal(self):
        entries = [{"name": "Elm Court Inn", "address": "1 Main St",
                    "category": "SUMMER RESORTS"}]
        b = bind_entries(entries, [line("SUMMER RESORTS")], canvas_width=2400)
        assert b[0].line_index is None


# --- context fields named after the set, not the entry ----------------------
#
# entry_binding treats any field ending `_name` as the entry's own identity, and
# identity_text takes the FIRST populated one in dict order. The 1869 Rowell
# classified lists carried `list_name` ("A LIST OF RELIGIOUS NEWSPAPERS AND
# PERIODICALS") and its index carried `index_name`; both sort before
# `publication_name`, so every entry on a page claimed the page's own caption as
# its identity and tried to match the heading line. 2,468 of 3,034 entries went
# unplaced across three sections.
#
# That is the same heading steal `category` caused in the Green Books, reached
# through a field name nobody had thought to guard.

@pytest.mark.parametrize("context_field", [
    "list_name", "index_name", "group_name", "heading_name",
    "series_name", "volume_name", "collection_name", "table_name", "street_name",
])
def test_a_set_name_never_supplies_the_identity(context_field):
    entry = {
        context_field: "A LIST OF RELIGIOUS NEWSPAPERS AND PERIODICALS",
        "state_or_province": "NEW YORK",
        "town": "New York",
        "publication_name": "Examiner and Chronicle",
    }
    assert identity_text(entry) == "Examiner and Chronicle"


def test_the_guard_holds_when_the_context_field_sorts_first():
    """Order is what made this bite: _first_field takes the first match, and a
    dict built from a prompt lists context before the entry's own fields."""
    entry = {"list_name": "A LIST OF DAILY PAPERS", "publication_name": "Herald"}
    assert list(entry)[0] == "list_name", "guard is meaningless if order changes"
    assert identity_text(entry) == "Herald"


def test_an_entry_with_only_a_set_name_has_no_identity():
    """Refusing beats binding every entry on the page to its own heading."""
    assert identity_text({"list_name": "A LIST OF RELIGIOUS NEWSPAPERS"}) == ""


def test_a_street_page_binds_by_occupant_not_by_street():
    """Tulsa 1921's street section, page 0337 right: one street heads every entry,
    and with it as identity the binder placed 1 of 162."""
    street = "MAIN AV—NORTH"
    entries = [{"street_name": street, "house_number": "421", "occupant_name": n}
               for n in ("J V Hulcomb", "J F Phillips", "W C Walker")]
    lines = [line(t, frag(10, 20 * i, 300, 18)) for i, t in enumerate(
        ["Main Av-N TULSA CITY DIRECTORY, 1921. Main Av-N 673",
         "421 J V Hulcomb.", "421 J F Phillips.", "421 W C Walker."])]
    b = bind_entries(entries, lines, canvas_width=2400)
    assert [x.line_index for x in b] == [1, 2, 3]


class TestSplitNames:
    """Lain-Healy's Brooklyn 1897 extracts `surname` + `given_name`. `surname` has
    no `_name` suffix, so the identity was the given name alone."""

    def test_surname_leads_the_given_name(self):
        e = {"surname": "Ellison", "given_name": "Wood", "is_business": True,
             "occupation": "stationers"}
        assert identity_text(e) == "Ellison Wood"

    def test_a_surname_alone_is_the_name(self):
        assert identity_text({"surname": "Ellison", "address": "12 Fulton"}) == "Ellison 12 Fulton"

    def test_a_whole_name_field_is_not_doubled(self):
        assert identity_text({"name": "Ellison Wood", "surname": "Ellison"}) == "Ellison Wood"

    def test_same_given_name_different_surnames(self):
        lines = [line("Ellison Wood stationers", frag(10, 10, 300, 18)),
                 line("Abbott Wood clerk", frag(10, 30, 300, 18))]
        entries = [{"surname": "Abbott", "given_name": "Wood", "occupation": "clerk"},
                   {"surname": "Ellison", "given_name": "Wood", "occupation": "stationers"}]
        b = bind_entries(entries, lines, canvas_width=2400)
        assert (b[0].line_index, b[1].line_index) == (1, 0)


def test_a_genuine_name_field_still_qualifies():
    """The suffix rule must keep working — firm_name/ship_name/publication_name
    are what it exists for."""
    for field in ("publication_name", "firm_name", "ship_name", "town_name"):
        assert identity_text({field: "Examiner and Chronicle"}) == "Examiner and Chronicle"
