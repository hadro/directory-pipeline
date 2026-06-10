"""pipeline/stages.py — registry invariants and declarative argv construction.

The argv assertions mirror the command lines main.py produced before the
registry existed (verified byte-identical during T11); changing them is a
behavior change for every consumer of the orchestrator.
"""

import argparse

from pipeline.stages import (
    INTERACTIVE_STAGES,
    STAGE_BY_NAME,
    STAGES,
    build_declarative_args,
)


def ns(**overrides) -> argparse.Namespace:
    """A parsed-namespace stand-in with the orchestrator's defaults."""
    base = dict(
        models=[],
        ocr_model=None,
        workers=None,
        batch_size=None,
        threshold=None,
        max_columns=None,
        force=False,
        min_surya_confidence=None,
        no_text=False,
        expand_dittos=False,
        high_res=False,
        ocr_prompt=None,
        ner_prompt=None,
        mode=None,
        flex=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# --- registry invariants ---

def test_stage_names_and_flags_are_unique():
    names = [s.name for s in STAGES]
    flags = [s.flag for s in STAGES]
    assert len(names) == len(set(names))
    assert len(flags) == len(set(flags))


def test_stage_by_name_is_complete():
    assert set(STAGE_BY_NAME) == {s.name for s in STAGES}


def test_interactive_stages():
    assert INTERACTIVE_STAGES == {"select_pages", "review_alignment"}


def test_declarative_stages_all_require_images():
    assert all(s.requires_images for s in STAGES if s.declarative)


def test_module_path_mapping():
    assert STAGE_BY_NAME["align_ocr"].module == "pipeline.align_ocr"
    assert STAGE_BY_NAME["map"].module == "pipeline.geo.map_entries"
    assert STAGE_BY_NAME["loc_csv"].module == "sources.loc_collection_csv"


def test_known_stage_order_prefix():
    # Source stages must come first and explore last — downstream stages
    # depend on this fixed execution order.
    names = [s.name for s in STAGES]
    assert names[:4] == ["loc_csv", "ia_csv", "iiif_csv", "download"]
    assert names[-1] == "explore"


# --- declarative argv construction ---

def test_fan_out_no_model_single_bare_run(tmp_path):
    runs = build_declarative_args(STAGE_BY_NAME["explore"], tmp_path, ns())
    assert runs == [[str(tmp_path)]]


def test_fan_out_single_ocr_model(tmp_path):
    runs = build_declarative_args(
        STAGE_BY_NAME["geocode"], tmp_path, ns(ocr_model="gemini-x")
    )
    assert runs == [[str(tmp_path), "--model", "gemini-x"]]


def test_fan_out_multiple_models(tmp_path):
    runs = build_declarative_args(
        STAGE_BY_NAME["align_ocr"], tmp_path, ns(models=["a", "b"], force=True)
    )
    assert runs == [
        [str(tmp_path), "--model", "a", "--force"],
        [str(tmp_path), "--model", "b", "--force"],
    ]


def test_gemini_ocr_full_option_order(tmp_path):
    runs = build_declarative_args(
        STAGE_BY_NAME["gemini_ocr"],
        tmp_path,
        ns(ocr_model="m", workers=5, expand_dittos=True, high_res=True,
           ocr_prompt="p.md", flex=True),
        ctx={"sections": "/abs/sections.txt"},
    )
    assert runs == [[
        str(tmp_path), "--model", "m",
        "--workers", "5", "--expand-dittos", "--high-res",
        "--prompt-file", "p.md", "--flex",
        "--sections", "/abs/sections.txt",
    ]]


def test_extract_entries_uses_aligned_model_flag(tmp_path):
    runs = build_declarative_args(
        STAGE_BY_NAME["extract_entries"],
        tmp_path,
        ns(ocr_model="gemini-ocr-slug", ner_prompt="ner.md", mode="multimodal", flex=True),
    )
    assert runs == [[
        str(tmp_path), "--aligned-model", "gemini-ocr-slug",
        "--prompt", "ner.md", "--flex", "--mode", "multimodal",
    ]]


def test_review_alignment_takes_first_model_flat(tmp_path):
    args = build_declarative_args(
        STAGE_BY_NAME["review_alignment"], tmp_path, ns(models=["a", "b"])
    )
    assert args == [str(tmp_path), "--model", "a"]


def test_review_alignment_falls_back_to_ocr_model(tmp_path):
    args = build_declarative_args(
        STAGE_BY_NAME["review_alignment"], tmp_path, ns(ocr_model="m")
    )
    assert args == [str(tmp_path), "--model", "m"]


def test_not_none_options_emit_zero_values(tmp_path):
    # workers=0 / min_surya_confidence=0.0 are deliberate values, not "unset".
    runs = build_declarative_args(
        STAGE_BY_NAME["align_ocr"], tmp_path, ns(workers=0, min_surya_confidence=0.0)
    )
    assert runs == [[str(tmp_path), "--workers", "0", "--min-surya-confidence", "0.0"]]


def test_switch_options_absent_when_false(tmp_path):
    args = build_declarative_args(STAGE_BY_NAME["surya_detect"], tmp_path, ns())
    assert args == [str(tmp_path)]


def test_surya_ocr_batch_size(tmp_path):
    args = build_declarative_args(
        STAGE_BY_NAME["surya_ocr"], tmp_path, ns(batch_size=2)
    )
    assert args == [str(tmp_path), "--batch-size", "2"]


def test_detect_columns_option_order(tmp_path):
    args = build_declarative_args(
        STAGE_BY_NAME["detect_columns"], tmp_path,
        ns(workers=2, threshold=0.12, max_columns=3, force=True),
    )
    assert args == [
        str(tmp_path), "--workers", "2", "--threshold", "0.12",
        "--max-columns", "3", "--force",
    ]
