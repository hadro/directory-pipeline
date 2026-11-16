"""Slug derivation and source handling: pure branches only — no network.

loc_slug() fetches item titles from the LoC API for /item/ URLs, so only its
/collections/ branch and fallback are tested here; the title→slug composition
is covered via _make_loc_slug directly.
"""

import main
import sources.loc_utils as loc_utils
from sources.ia_utils import _extract_ia_identifier, _make_ia_slug
from sources.loc_collection_csv import _extract_item_path, _manifest_url
from sources.loc_utils import _make_loc_slug, _resource_url_to_item_url, loc_slug
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


# --- LoC item paths ---
#
# A LoC item URL can name a sub-resource.  /item/{id}/manifest.json is the
# serial's catalog record and has no page images on it, so a manifest URL built
# from the item ID alone downloads nothing at all — silently, with a zero exit.

def test_item_path_plain_item():
    assert _extract_item_path("https://www.loc.gov/item/01015253/") == "01015253"


def test_item_path_strips_manifest_and_query():
    assert _extract_item_path("https://www.loc.gov/item/01015253/manifest.json") == "01015253"
    assert _extract_item_path("https://www.loc.gov/item/01015253/?fo=json") == "01015253"


def test_item_path_keeps_chronicling_america_issue():
    url = "https://www.loc.gov/item/sn83030313/1847-04-30/ed-1/manifest.json"
    assert _extract_item_path(url) == "sn83030313/1847-04-30/ed-1"


def test_item_path_keeps_serial_volume_resource():
    url = "https://www.loc.gov/item/sn82007064/resource/gdc.sn82007064_1869/manifest.json"
    assert _extract_item_path(url) == "sn82007064/resource/gdc.sn82007064_1869"


def test_item_path_non_item_url():
    assert _extract_item_path("https://www.loc.gov/collections/civil-war-maps/") is None


def test_manifest_url_preserves_sub_resource():
    url = "https://www.loc.gov/item/sn82007064/resource/gdc.sn82007064_1869/"
    assert _manifest_url(_extract_item_path(url)) == (
        "https://www.loc.gov/item/sn82007064/resource/gdc.sn82007064_1869/manifest.json"
    )


def test_resource_url_fallback_keeps_item_path(monkeypatch):
    """When the API lookup fails, an /item/…/resource/… URL keeps its path.

    The old fallback swapped /resource/ → /item/ unconditionally, inventing a
    second /item/ segment and losing the volume.
    """
    def _boom(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(loc_utils.requests, "get", _boom)
    url = "https://www.loc.gov/item/sn82007064/resource/gdc.sn82007064_1869/manifest.json"
    assert _resource_url_to_item_url(url) == (
        "https://www.loc.gov/item/sn82007064/resource/gdc.sn82007064_1869/"
    )


def test_resource_url_fallback_maps_bare_resource_to_item(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(loc_utils.requests, "get", _boom)
    assert _resource_url_to_item_url("https://www.loc.gov/resource/abc123/?sp=74") == (
        "https://www.loc.gov/item/abc123/"
    )


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


# --- _fetch_loc_title error path ---

def test_fetch_loc_title_warns_and_returns_empty_when_the_api_fails(monkeypatch, capsys):
    """A network failure must degrade to a warning, not raise.

    The except branch interpolated `item_id` after the parameter was renamed to
    `item_path`, so every failure raised NameError from inside the handler —
    turning a recoverable miss into a crash, and masking the original error.
    Nothing exercised this path, so ruff's F821 was the only thing that saw it.
    """
    def _boom(*_a, **_kw):
        raise OSError("network down")

    monkeypatch.setattr(loc_utils.requests, "get", _boom)
    assert loc_utils._fetch_loc_title("sn82007064") == ""
    assert "sn82007064" in capsys.readouterr().err


def test_fetch_loc_title_reports_the_full_sub_resource_path(monkeypatch, capsys):
    """The warning should name what was actually requested, sub-resource and all."""
    def _boom(*_a, **_kw):
        raise OSError("network down")

    path = "sn82007064/resource/gdc.sn82007064_1869"
    monkeypatch.setattr(loc_utils.requests, "get", _boom)
    assert loc_utils._fetch_loc_title(path) == ""
    assert path in capsys.readouterr().err
