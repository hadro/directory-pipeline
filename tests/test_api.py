"""pipeline/api.py — the curated public surface stays importable and lean."""

import sys


def test_all_names_resolve():
    import pipeline.api as api

    for name in api.__all__:
        assert hasattr(api, name), f"__all__ lists {name} but it is not importable"


def test_api_import_pulls_no_optional_deps():
    # The facade must work with a core-only install: importing it may not
    # drag in the gpu/geo extras.
    import pipeline.api  # noqa: F401

    assert "surya" not in sys.modules
    assert "geopy" not in sys.modules


def test_constants_and_callables():
    import pipeline.api as api

    assert isinstance(api.DEFAULT_OCR_MODEL, str)
    for fn in (api.iter_canvases, api.image_url, api.parse_surya,
               api.process_csv, api.combine_volumes, api.generate_with_retry):
        assert callable(fn)
