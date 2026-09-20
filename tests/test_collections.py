"""Which collections count as evaluation data.

output/ mixes production volumes with source probes, five-page evidence slices,
backup directories and genre experiments. A cross-collection table once reported
_ocr_backups — 6,862 aligned files and zero images — as a collection, and a
threshold calibrated on manuscript tests would have been wrong for every printed
directory in the corpus.
"""

import pytest

from utils.collections import (
    LOCAL_REGISTRY_PATH,
    REGISTRY_PATH,
    Rule,
    classify,
    eval_collections,
    load_rules,
    summarize_exclusions,
)


def _reg(tmp_path, body):
    p = tmp_path / "registry.txt"
    p.write_text(body, encoding="utf-8")
    return p


# --- the real registry ------------------------------------------------------

@pytest.mark.parametrize("name", [
    "_ocr_backups",
    "ia_trow_1907_trowsgen",
    "nypl_polk_1917_4c08ab00",
    "loc_spooner_01015253",
    "harvest_polk1933mb_a",
    "copyright_ledgers_copyright_records_new_2022657932",
    "frederick_douglass_papers_financial_pape_mss118790088",
    "polish_declarations_of_admiration_and_fr_pldec.104",
    "william_brewster_correspondence_letters_williambrewster00bre",
    "goldsborough_papers_pages223-229",
    "passenger_lists_of_ships_coming_to_north_passengerlistsof00l",
    "joh_leonhard_frisch_beschreibung_von_all_johleonhardfrisc01f",
    "little_book_of_greenwich_village_a_handb_littlebookofgree00a",
    "e77514b9-2450-45b8-a7c1-7d486aa82c0a",
    "high_street_cook_book_ca10002203",
    "drives_about_san_jose_and_vicinity_tmp96008618",
])
def test_known_non_collections_are_excluded(name):
    keep, _ = classify(name)
    assert not keep


@pytest.mark.parametrize("name", [
    "the_artistic_language_of_flowers_cu31924074093760",
    "the_language_and_sentiment_of_flowers_languagese",
    "the_miniature_language_of_flowers_miniaturelangu",
    "flower_lore_the_teachings_of_flowers_his_cu31924",
    "flora_s_dictionary_by_mrs_e_w_wirt_mobot31753000",
])
def test_every_flower_book_is_caught(name):
    """Five different naming shapes; 'language_and_sentiment' does not contain
    'language_of_flowers', so one glob is not enough."""
    keep, category = classify(name)
    assert not keep
    assert category == "deferred", "flagged deferred, not rejected — may return to these"


@pytest.mark.parametrize("name", [
    "gazetteer_of_surface_waters_of_iowa_unk82032541",   # confirmed production
    "green_books_and_related",
    "tulsa_1921",
    "the_national_directory_of_morticians_nationaldirectornati",
    "1869_geo_p_rowell_and_co_s_american_news_sn82007064_gdc_sn82007064_1869",
])
def test_production_collections_are_kept(name):
    keep, _ = classify(name)
    assert keep


def test_an_unregistered_collection_is_kept():
    """Production is the norm. A new volume must not vanish from an evaluation
    because nobody remembered to register it."""
    keep, category = classify("some_brand_new_1889_city_directory_abc123")
    assert keep and category == "production"


# --- rule semantics ---------------------------------------------------------

def test_include_overrides_a_broader_exclude(tmp_path):
    rules = load_rules(_reg(tmp_path, "exclude ia_* probe\ninclude ia_keeper_* rescued\n"))
    assert classify("ia_keeper_vol1", rules) == (True, "rescued")
    assert classify("ia_other", rules)[0] is False


def test_comments_and_blank_lines_are_ignored(tmp_path):
    rules = load_rules(_reg(tmp_path, "# a note\n\nexclude foo_* junk   # trailing\n"))
    assert len(rules) == 1
    assert rules[0] == Rule("exclude", "foo_*", "junk")


def test_a_malformed_line_is_skipped_not_fatal(tmp_path):
    rules = load_rules(_reg(tmp_path, "nonsense\nexclude foo_* junk\nexclude\n"))
    assert [r.glob for r in rules] == ["foo_*"]


def test_a_missing_registry_keeps_everything(tmp_path):
    rules = load_rules(tmp_path / "absent.txt")
    assert rules == []
    assert classify("anything", rules) == (True, "production")


# --- directory splitting ----------------------------------------------------

def test_eval_collections_splits_and_reports(tmp_path):
    for name in ("green_books", "ia_probe", "_ocr_backups", "harvest_x"):
        (tmp_path / name).mkdir()
    rules = load_rules()
    kept, dropped = eval_collections(tmp_path, rules)
    assert [p.name for p in kept] == ["green_books"]
    assert {n for n, _ in dropped} == {"ia_probe", "_ocr_backups", "harvest_x"}


def test_exclusions_are_summarised_for_a_report_header():
    note = summarize_exclusions([("a", "source-probe"), ("b", "source-probe"), ("c", "deferred")])
    assert "3 collection(s)" in note
    assert "2 source-probe" in note and "1 deferred" in note
    assert "collection_registry" in note, "the reader must know where to change it"


def test_nothing_excluded_means_no_note():
    assert summarize_exclusions([]) == ""


# --- shared vs local layering -----------------------------------------------
#
# The shared file holds rules whose directories the pipeline itself creates, so
# they fire in a fresh clone. The local file names volumes in one working copy
# and is gitignored. Mixing them put 186 of Josh's directory names into a
# tracked file where they matched nothing for anyone else.


def test_shared_rules_only_match_pipeline_generated_names():
    """Every tracked rule must describe something the code produces.

    ia_*/nypl_*/loc_* come from sources/, harvest_* from the evidence harness,
    _* from backup tooling. A volume title in this file is a corpus judgement
    and belongs in the local one.
    """
    shared = load_rules(REGISTRY_PATH)
    assert shared, "shared registry should not be empty"
    allowed = {"_*", "ia_*", "nypl_*", "loc_*", "harvest_*"}
    assert {r.glob for r in shared} <= allowed, (
        "a corpus-specific glob leaked into the tracked registry"
    )


def test_shared_and_local_are_both_loaded():
    combined = {r.glob for r in load_rules()}
    assert {r.glob for r in load_rules(REGISTRY_PATH)} <= combined
    if LOCAL_REGISTRY_PATH.exists():
        assert {r.glob for r in load_rules(LOCAL_REGISTRY_PATH)} <= combined


def test_a_local_include_overrides_a_shared_exclude(tmp_path):
    """Carving one volume out of a shared convention rule must be possible."""
    shared = tmp_path / "shared.txt"
    shared.write_text("exclude ia_* source-probe\n", encoding="utf-8")
    local = tmp_path / "local.txt"
    local.write_text("include ia_trow_1907_* promoted\n", encoding="utf-8")
    rules = load_rules(shared) + load_rules(local)
    assert classify("ia_trow_1907_trowsgen", rules) == (True, "promoted")
    assert classify("ia_other_probe", rules)[0] is False


def test_a_working_copy_with_no_local_file_still_works(tmp_path):
    """A fresh clone has the shared file and no local one."""
    shared = tmp_path / "shared.txt"
    shared.write_text("exclude harvest_* evidence-slice\n", encoding="utf-8")
    rules = load_rules(shared) + load_rules(tmp_path / "absent.txt")
    assert classify("harvest_polk1933mb_a", rules)[0] is False
    assert classify("green_books_and_related", rules)[0] is True


def test_the_example_file_is_tracked_and_parses():
    """The local file is gitignored, so the example is the only discoverable
    record that the mechanism exists."""
    example = REGISTRY_PATH.with_name("collection_registry.local.txt.example")
    assert example.exists()
    load_rules(example)  # must not raise; every live line is commented out
