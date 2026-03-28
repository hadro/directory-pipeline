"""Tests for slug-generation and URL-normalization utilities.

All functions tested here are pure (no network calls, no filesystem access),
making these fast and reliable unit tests.

Modules under test:
  sources/nypl_utils.py   — make_slug()
  sources/loc_utils.py    — _make_loc_slug(), _resource_url_to_item_url() fallback path
  sources/ia_utils.py     — _make_ia_slug(), _extract_ia_identifier()
  utils/iiif_utils.py     — manifest_item_id() (already covered in test_iiif_utils.py
                             but slug-specific cases repeated here for context)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.nypl_utils import make_slug as nypl_make_slug
from sources.loc_utils import _make_loc_slug
from sources.ia_utils import _make_ia_slug, _extract_ia_identifier


# ---------------------------------------------------------------------------
# NYPL make_slug
# ---------------------------------------------------------------------------

class TestNyplMakeSlug:
    def test_basic_title_and_uuid(self):
        result = nypl_make_slug("Green Book", "abcd1234-5678-0000-0000-000000000000")
        assert result == "green_book_abcd1234"

    def test_uuid_truncated_to_8_chars_without_hyphens(self):
        result = nypl_make_slug("", "abcd-ef01-2345-6789-abcdef012345")
        # No title → just first 8 chars of uuid with hyphens stripped
        assert result == "abcdef01"

    def test_special_characters_sanitized(self):
        result = nypl_make_slug("New York, 1940–1950!", "aaaabbbb-0000-0000-0000-000000000000")
        assert result == "new_york_1940_1950_aaaabbbb"

    def test_unicode_lowercased(self):
        result = nypl_make_slug("Ñoño Café", "00000000-0000-0000-0000-000000000000")
        assert result == "o_o_caf_00000000"

    def test_title_truncated_at_40_chars(self):
        long_title = "A" * 80
        result = nypl_make_slug(long_title, "00000000-0000-0000-0000-000000000000")
        title_part = result.rsplit("_", 1)[0]
        assert len(title_part) <= 40

    def test_no_leading_or_trailing_underscores_in_title(self):
        result = nypl_make_slug("___Hello___", "12345678-0000-0000-0000-000000000000")
        assert not result.startswith("_")

    def test_consecutive_underscores_collapsed(self):
        result = nypl_make_slug("Hello   World", "aabbccdd-0000-0000-0000-000000000000")
        assert "__" not in result

    def test_empty_title_falls_back_to_uuid8(self):
        result = nypl_make_slug("", "deadbeef-0000-0000-0000-000000000000")
        assert result == "deadbeef"

    def test_whitespace_only_title_falls_back_to_uuid8(self):
        result = nypl_make_slug("   ", "cafebabe-0000-0000-0000-000000000000")
        assert result == "cafebabe"

    def test_slug_contains_uuid_suffix(self):
        result = nypl_make_slug("My Book", "aabbccdd-1234-5678-abcd-ef0123456789")
        assert result.endswith("aabbccdd")


# ---------------------------------------------------------------------------
# LoC _make_loc_slug
# ---------------------------------------------------------------------------

class TestLocMakeSlug:
    def test_basic_title_and_item_id(self):
        result = _make_loc_slug("The Brooklyn City Directory", "01015253")
        assert result == "the_brooklyn_city_directory_01015253"

    def test_item_id_truncated_to_12_chars(self):
        result = _make_loc_slug("", "rbc0001.2026batch96169559")
        # No title → just first 12 chars of item_id: "rbc0001.2026"
        assert result == "rbc0001.2026"

    def test_special_chars_in_title_sanitized(self):
        result = _make_loc_slug("Brooklyn: City & County Directory!", "id123")
        assert result == "brooklyn_city_county_directory_id123"

    def test_title_truncated_at_40_chars(self):
        long_title = "B" * 80
        result = _make_loc_slug(long_title, "itemid")
        title_part = result.rsplit("_", 1)[0]
        assert len(title_part) <= 40

    def test_empty_title_returns_item_id_only(self):
        result = _make_loc_slug("", "sn83030313")
        assert result == "sn83030313"

    def test_no_double_underscores(self):
        result = _make_loc_slug("Hello   World", "id456")
        assert "__" not in result


# ---------------------------------------------------------------------------
# Internet Archive _make_ia_slug
# ---------------------------------------------------------------------------

class TestIaMakeSlug:
    def test_basic_title_and_identifier(self):
        result = _make_ia_slug("Green Book 1956", "greenbook1956")
        assert result == "green_book_1956_greenbook1956"

    def test_identifier_truncated_to_20_chars(self):
        long_id = "a" * 40
        result = _make_ia_slug("", long_id)
        assert result == "a" * 20

    def test_special_chars_sanitized(self):
        result = _make_ia_slug("Title: With Punctuation!", "someid")
        assert result == "title_with_punctuation_someid"

    def test_empty_title_returns_identifier_only(self):
        result = _make_ia_slug("", "myidentifier")
        assert result == "myidentifier"

    def test_title_truncated_at_40_chars(self):
        long_title = "C" * 80
        result = _make_ia_slug(long_title, "shortid")
        title_part = result.rsplit("_", 1)[0]
        assert len(title_part) <= 40

    def test_no_double_underscores(self):
        result = _make_ia_slug("Multiple   Spaces   Here", "id789")
        assert "__" not in result


# ---------------------------------------------------------------------------
# Internet Archive _extract_ia_identifier
# ---------------------------------------------------------------------------

class TestExtractIaIdentifier:
    def test_standard_details_url(self):
        url = "https://archive.org/details/greenbook1956"
        assert _extract_ia_identifier(url) == "greenbook1956"

    def test_details_url_with_trailing_slash(self):
        url = "https://archive.org/details/greenbook1956/"
        # The regex doesn't include a trailing slash, and rstrip removes it
        assert _extract_ia_identifier(url) == "greenbook1956"

    def test_details_url_with_query_string(self):
        url = "https://archive.org/details/greenbook1956?ui-theme=dark"
        assert _extract_ia_identifier(url) == "greenbook1956"

    def test_non_archive_url_returns_none(self):
        url = "https://example.com/some/path"
        assert _extract_ia_identifier(url) is None

    def test_collection_url(self):
        url = "https://archive.org/details/americana"
        assert _extract_ia_identifier(url) == "americana"
