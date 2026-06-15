"""app.py — _detect_output_dir resolves URLs to existing output directories."""

import json

import app


def _make_dir(root, name, source_url=None):
    d = root / name
    d.mkdir()
    if source_url:
        (d / "pipeline_state.json").write_text(
            json.dumps({"slug": name, "source_url": source_url}), encoding="utf-8"
        )
    return d


def test_uuid_match(tmp_path, monkeypatch):
    d = _make_dir(tmp_path, "greenbook_4d2f36d0-1234")
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path)
    url = "https://www.loc.gov/item/4d2f36d0-c5cb-012f-77f9-58d385a7bc34/"
    assert app._detect_output_dir(url) == d


def test_ia_identifier_match(tmp_path, monkeypatch):
    d = _make_dir(tmp_path, "the_artistic_language_of_flowers_cu31924074093760")
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path)
    url = "https://archive.org/details/cu31924074093760"
    assert app._detect_output_dir(url) == d


def test_source_url_match_from_state(tmp_path, monkeypatch):
    # No identifier in the dir name — only pipeline_state.json links them.
    url = "https://example.org/iiif/some-item/manifest.json"
    d = _make_dir(tmp_path, "opaque_slug_name", source_url=url + "/")
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path)
    assert app._detect_output_dir(url) == d  # trailing slash normalized


def test_no_match_returns_none(tmp_path, monkeypatch):
    _make_dir(tmp_path, "unrelated_volume", source_url="https://other.org/x")
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path)
    assert app._detect_output_dir("https://archive.org/details/zzz999") is None


def test_missing_output_root_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path / "does-not-exist")
    assert app._detect_output_dir("https://archive.org/details/abc") is None
