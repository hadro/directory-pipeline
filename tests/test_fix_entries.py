"""Tests for analysis/fix_entries.py — focused on the sentinel-token contract.

Sentinel tokens ([illegible]/[blank], see prompts/README.md) must be treated as
explicit gap markers: folded into the placeholder set so a page where a field
legitimately repeats a sentinel is not mistaken for hallucinated ghost text.
"""

from analysis.fix_entries import (
    SENTINEL_TOKENS,
    _NER_PLACEHOLDER_VALUES,
    find_hallucinated_pages,
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
