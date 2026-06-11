"""Single registry of pipeline stages — the one place the stage graph is defined.

Consumed by:
  - main.py     — stage order, orchestrator flag names, interactivity, and the
                  declarative argv builder for the ~dozen "plain" stages
  - app.py      — flag names + interactivity for the dashboard's stage buttons

Stages whose argv construction has real logic (source-type validation, file
auto-discovery, multi-run expansion) are declared here with ``declarative=False``
and keep explicit code in ``main.py: build_stage_args()``. A registry that is
80% declarative plus 20% explicit code beats a 100% DSL — don't force the
special cases into the table.

Option semantics (``Opt``):
  kind  "value"   → emits ``[flag, str(v)]``     |  "switch" → emits ``[flag]``
  when  "truthy"  → emit if ``bool(v)``          |  "not_none" → emit if ``v is not None``
  from_ctx        → read the value from the per-target ctx dict (computed by the
                    orchestrator, e.g. the resolved ``--sections`` path) instead
                    of the parsed argparse namespace.

Model handling (``model_mode``):
  "fan_out" → one run per model in ``--models`` (or the single ``--ocr-model``),
              passing each as ``model_flag``; one run with no model flag if
              neither was given (stage uses its own default).
  "first"   → single run, passing the first ``--models`` entry or ``--ocr-model``.
  None      → stage takes no model flag from the orchestrator.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Opt:
    attr: str
    flag: str
    kind: str = "value"      # "value" | "switch"
    when: str = "truthy"     # "truthy" | "not_none"
    from_ctx: bool = False


@dataclass(frozen=True)
class StageDef:
    name: str                # argparse dest on the orchestrator namespace
    script: str              # repo-relative script path (module = path with / → .)
    flag: str                # orchestrator CLI flag (e.g. "--align-ocr")
    interactive: bool = False    # Ctrl+C means "stage finished", not "abort"
    requires_images: bool = True # needs output/{slug}/ to exist before running
    declarative: bool = False    # argv built by build_declarative_args()
    model_mode: "str | None" = None
    model_flag: str = "--model"
    opts: tuple = ()
    requires: "str | None" = None  # importable package from an optional extra
    install_hint: str = ""         # how to install it, e.g. "uv sync --extra gpu"

    @property
    def module(self) -> str:
        """Module path for ``python -m`` launching (pipeline/align_ocr.py → pipeline.align_ocr)."""
        return self.script.removesuffix(".py").replace("/", ".")


# Stages always execute in this order, regardless of flag order on the command line.
STAGES: "list[StageDef]" = [
    StageDef("loc_csv",  "sources/loc_collection_csv.py",  "--loc-csv",  requires_images=False),
    StageDef("ia_csv",   "sources/ia_collection_csv.py",   "--ia-csv",   requires_images=False),
    StageDef("iiif_csv", "sources/iiif_manifest_csv.py",   "--iiif-csv", requires_images=False),
    StageDef("download", "pipeline/download_images.py",    "--download", requires_images=False),
    StageDef("detect_spreads", "pipeline/detect_spreads.py", "--detect-spreads"),
    StageDef("split_spreads",  "pipeline/split_spreads.py",  "--split-spreads"),
    StageDef("select_pages",   "pipeline/select_pages.py",   "--select-pages", interactive=True),
    StageDef("generate_prompts", "pipeline/generate_prompt.py", "--generate-prompts"),
    StageDef(
        "surya_detect", "pipeline/surya_detect.py", "--surya-detect",
        declarative=True,
        opts=(Opt("force", "--force", kind="switch"),),
        requires="surya", install_hint="uv sync --extra gpu",
    ),
    StageDef(
        "detect_columns", "pipeline/detect_columns.py", "--detect-columns",
        declarative=True,
        opts=(
            Opt("workers", "--workers", when="not_none"),
            Opt("threshold", "--threshold", when="not_none"),
            Opt("max_columns", "--max-columns", when="not_none"),
            Opt("force", "--force", kind="switch"),
        ),
    ),
    StageDef(
        "surya_ocr", "pipeline/run_surya_ocr.py", "--surya-ocr",
        declarative=True,
        opts=(Opt("batch_size", "--batch-size", when="not_none"),),
        requires="surya", install_hint="uv sync --extra gpu",
    ),
    StageDef(
        "gemini_ocr", "pipeline/run_gemini_ocr.py", "--gemini-ocr",
        declarative=True, model_mode="fan_out",
        opts=(
            Opt("workers", "--workers", when="not_none"),
            Opt("expand_dittos", "--expand-dittos", kind="switch"),
            Opt("high_res", "--high-res", kind="switch"),
            Opt("ocr_prompt", "--prompt-file"),
            Opt("flex", "--flex", kind="switch"),
            Opt("sections", "--sections", from_ctx=True),
        ),
    ),
    StageDef("compare_ocr", "pipeline/compare_ocr.py", "--compare-ocr"),
    StageDef(
        "align_ocr", "pipeline/align_ocr.py", "--align-ocr",
        declarative=True, model_mode="fan_out",
        opts=(
            Opt("workers", "--workers", when="not_none"),
            Opt("force", "--force", kind="switch"),
            Opt("min_surya_confidence", "--min-surya-confidence", when="not_none"),
        ),
    ),
    StageDef(
        "visualize", "pipeline/visualize_alignment.py", "--visualize",
        declarative=True, model_mode="fan_out",
        opts=(
            Opt("no_text", "--no-text", kind="switch"),
            Opt("force", "--force", kind="switch"),
        ),
    ),
    StageDef(
        "review_alignment", "pipeline/review_alignment.py", "--review-alignment",
        interactive=True, declarative=True, model_mode="first",
    ),
    StageDef(
        "extract_entries", "pipeline/extract_entries.py", "--extract-entries",
        declarative=True, model_mode="fan_out", model_flag="--aligned-model",
        opts=(
            Opt("ner_prompt", "--prompt"),
            Opt("force", "--force", kind="switch"),
            Opt("flex", "--flex", kind="switch"),
            Opt("mode", "--mode"),
            Opt("sections", "--sections", from_ctx=True),
        ),
    ),
    StageDef("geocode", "pipeline/geo/geocode_entries.py", "--geocode",
             declarative=True, model_mode="fan_out",
             requires="geopy", install_hint="uv sync --extra geo"),
    StageDef("map", "pipeline/geo/map_entries.py", "--map",
             declarative=True, model_mode="fan_out"),
    StageDef("explore", "pipeline/explore_entries.py", "--explore",
             declarative=True, model_mode="fan_out"),
]

STAGE_BY_NAME: "dict[str, StageDef]" = {s.name: s for s in STAGES}

INTERACTIVE_STAGES: "set[str]" = {s.name for s in STAGES if s.interactive}


def build_declarative_args(
    sd: StageDef,
    output_dir: Path,
    parsed,
    ctx: "dict | None" = None,
) -> "list[str] | list[list[str]]":
    """Build argv for a declarative stage from the orchestrator's parsed namespace.

    Returns a flat argv list, or a list of argv lists for fan-out stages
    (one run per requested model). Argv order follows the ``opts`` declaration
    order, preserving the historical command lines exactly.
    """
    ctx = ctx or {}

    def opt_args(o: Opt) -> "list[str]":
        v = ctx.get(o.attr) if o.from_ctx else getattr(parsed, o.attr, None)
        present = (v is not None) if o.when == "not_none" else bool(v)
        if not present:
            return []
        return [o.flag] if o.kind == "switch" else [o.flag, str(v)]

    def one_run(model: "str | None") -> "list[str]":
        a = [str(output_dir)]
        if model:
            a += [sd.model_flag, model]
        for o in sd.opts:
            a += opt_args(o)
        return a

    if sd.model_mode == "fan_out":
        models = parsed.models if parsed.models else ([parsed.ocr_model] if parsed.ocr_model else [None])
        return [one_run(m) for m in models]
    if sd.model_mode == "first":
        m = (parsed.models[0] if parsed.models else None) or parsed.ocr_model
        return one_run(m)
    return one_run(None)
