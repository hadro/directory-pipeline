"""main.py preflight — optional-extra dependency checking.

The install command must always be a single `uv sync` naming every extra the
run needs plus any already installed: uv sync is exact, so a hint naming only
one extra uninstalls the other and sends users ping-ponging between
`--extra gpu` and `--extra geo`.
"""

from main import preflight_requirements


def _spec(installed: set):
    return lambda pkg: object() if pkg in installed else None


def test_all_present_returns_no_missing():
    missing, cmd = preflight_requirements(
        {"surya_ocr", "geocode"}, find_spec=_spec({"surya", "geopy"})
    )
    assert missing == [] and cmd == ""


def test_single_missing_extra():
    missing, cmd = preflight_requirements(
        {"geocode"}, find_spec=_spec(set())
    )
    assert [sd.name for sd in missing] == ["geocode"]
    assert cmd == "uv sync --extra geo"


def test_both_needed_and_missing_yields_combined_command():
    missing, cmd = preflight_requirements(
        {"surya_ocr", "geocode"}, find_spec=_spec(set())
    )
    assert {sd.name for sd in missing} == {"surya_ocr", "geocode"}
    assert cmd == "uv sync --extra gpu --extra geo"


def test_installed_extra_is_preserved_in_command():
    # The ping-pong case: geocode missing, surya already installed.
    # The command must include gpu too, or running it would remove surya.
    missing, cmd = preflight_requirements(
        {"geocode"}, find_spec=_spec({"surya"})
    )
    assert [sd.name for sd in missing] == ["geocode"]
    assert cmd == "uv sync --extra gpu --extra geo"


def test_stage_without_requires_never_missing():
    missing, cmd = preflight_requirements(
        {"download", "gemini_ocr", "align_ocr"}, find_spec=_spec(set())
    )
    assert missing == [] and cmd == ""
