"""Section grouping in the explorer.

A combined CSV is the union of every section's schema. The 1869 Rowell volume
has 54 columns, 39 of them populated on under 20% of rows, because a gazetteer
row and an advertisement row share almost nothing. Grouping by section gives
each one its own column set and its own facets.
"""

import pytest

from pipeline.explore_entries import _group_by_section, _section_field_meta


def _rows():
    return [
        {"section": "catalog", "publication_name": "Herald", "frequency": "daily",
         "town_name": "", "description": "", "image": "0009.jpg"},
        {"section": "catalog", "publication_name": "Post", "frequency": "weekly",
         "town_name": "", "description": "", "image": "0010.jpg"},
        {"section": "gazetteer", "publication_name": "", "frequency": "",
         "town_name": "Lynn", "description": "a city of 25,000 pop.", "image": "0150.jpg"},
    ]


def test_rows_are_grouped_by_section():
    g = _group_by_section(_rows())
    assert set(g) == {"catalog", "gazetteer"}
    assert g["catalog"]["count"] == 2
    assert g["gazetteer"]["count"] == 1


def test_sections_keep_the_order_of_the_book():
    """Not alphabetical: catalog precedes gazetteer because the book does."""
    assert list(_group_by_section(_rows())) == ["catalog", "gazetteer"]


def test_rows_are_not_duplicated_into_the_sections():
    """Every row carries its own `section`, so the template filters the volume's
    rows. Embedding both doubled the Rowell explorer to 42 MB for no new data."""
    for payload in _group_by_section(_rows()).values():
        assert "rows" not in payload
        assert set(payload) == {"field_meta", "count"}


def test_a_section_drops_columns_it_never_populates():
    """Rows keep every key of the combined CSV, so classifying them directly
    returns all of them and the table renders empty columns."""
    g = _group_by_section(_rows())
    cat = {f["name"] for f in g["catalog"]["field_meta"]}
    gaz = {f["name"] for f in g["gazetteer"]["field_meta"]}
    assert "publication_name" in cat and "town_name" not in cat
    assert "town_name" in gaz and "publication_name" not in gaz


def test_a_volume_with_no_section_column_is_unchanged():
    rows = [{"publication_name": "Herald"}, {"publication_name": "Post"}]
    assert _group_by_section(rows) == {}


def test_a_single_section_volume_gets_no_grouping():
    """One section is not a choice — the dropdown would have one option."""
    rows = [{"section": "catalog", "publication_name": "Herald"}]
    assert _group_by_section(rows) == {}


def test_rows_with_a_blank_section_are_skipped_not_grouped():
    rows = _rows() + [{"section": "", "publication_name": "Orphan"}]
    g = _group_by_section(rows)
    assert set(g) == {"catalog", "gazetteer"}
    assert sum(v["count"] for v in g.values()) == 3


def test_empty_input_is_safe():
    assert _group_by_section([]) == {}


def test_section_field_meta_keeps_id_fields_regardless():
    """Provenance columns are how a row is identified, even when sparse."""
    rows = [{"image": "0009.jpg", "publication_name": "Herald", "unused": ""}]
    names = {f["name"]: f for f in _section_field_meta(rows)}
    assert "unused" not in names
    assert "publication_name" in names
