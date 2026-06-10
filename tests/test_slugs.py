"""Slug derivation and source handling: pure branches only — no network.

loc_slug() fetches item titles from the LoC API for /item/ URLs, so only its
/collections/ branch and fallback are tested here; the title→slug composition
is covered via _make_loc_slug directly.
"""

import main
from sources.ia_utils import _extract_ia_identifier, _make_ia_slug
from sources.loc_utils import _make_loc_slug, loc_slug
from utils.iiif_utils import manifest_item_id


# --- LoC ---

def test_loc_slug_collections_url():
    assert loc_slug("https://www.loc.gov/collections/civil-war-maps/") == "civil-war-maps"


def test_loc_slug_fallback_sanitizes():
    slug = loc_slug("https://www.loc.gov/search/?q=directories")
    assert slug == "https_www_loc_gov_search_q_directories"[:40]


def test_make_loc_slug_title_and_id():
    assert (
        _make_loc_slug("The Brooklyn city directory", "01015253")
        == "the_brooklyn_city_directory_01015253"
    )


def test_make_loc_slug_empty_title_uses_id():
    assert _make_loc_slug("", "0101525399999999") == "010152539999"  # id[:12]


# --- Internet Archive ---

def test_ia_identifier_from_details_url():
    assert _extract_ia_identifier("https://archive.org/details/ldpd_11290437_000/") == "ldpd_11290437_000"


def test_ia_identifier_ignores_query():
    assert _extract_ia_identifier("https://archive.org/details/foo?tab=about") == "foo"


def test_ia_identifier_non_ia_url():
    assert _extract_ia_identifier("https://example.org/details/foo") is None


def test_make_ia_slug_sanitizes_title():
    assert _make_ia_slug("Lain & Healy's Brooklyn Directory!", "1897BPL") == (
        "lain_healy_s_brooklyn_directory_1897BPL"
    )


def test_make_ia_slug_empty_title_uses_id_prefix():
    assert _make_ia_slug("", "a" * 30) == "a" * 20


# --- Generic IIIF ---

def test_manifest_item_id_manifest_json():
    assert manifest_item_id("https://www.loc.gov/item/01015253/manifest.json") == "01015253"


def test_manifest_item_id_skips_generic_segments():
    url = "https://dcmny.org/do/dedab5e6-8b7d/metadata/iiifmanifest/default.jsonld"
    assert manifest_item_id(url) == "dedab5e6-8b7d"


def test_manifest_item_id_plain_path():
    assert manifest_item_id("https://example.org/iiif/abc123/manifest.json") == "abc123"


# --- main.py source classification + target loading ---

def test_url_classifiers():
    assert main.is_loc_url("https://www.loc.gov/item/x/")
    assert main.is_ia_url("https://archive.org/details/x")
    assert main._is_generic_iiif_url("https://example.org/iiif/manifest.json")
    assert not main._is_generic_iiif_url("https://www.loc.gov/item/x/")
    assert not main._is_generic_iiif_url("output/some_dir")


def test_load_targets_single_url():
    assert main.load_targets("https://archive.org/details/x") == ["https://archive.org/details/x"]


def test_load_targets_list_file_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "collections.txt"
    f.write_text("# comment\n\nhttps://a\n  https://b  \n", encoding="utf-8")
    assert main.load_targets(str(f)) == ["https://a", "https://b"]


def test_load_targets_csv_passthrough(tmp_path):
    f = tmp_path / "items.csv"
    f.write_text("item_id,item_title\n1,x\n", encoding="utf-8")
    assert main.load_targets(str(f)) == [str(f)]
