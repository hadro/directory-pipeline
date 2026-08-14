"""main.py preflight — optional-extra dependency checking.

The install command must always be a single `uv sync` naming every extra the
run needs plus any already installed: uv sync is exact, so a hint naming only
one extra uninstalls the other and sends users ping-ponging between
`--extra gpu` and `--extra geo`.
"""

from main import interpreter_arch_warning, preflight_requirements


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


# --- interpreter architecture -------------------------------------------------
# An Intel Python under Rosetta on Apple Silicon can neither install Surya
# (no macOS x86_64 torch wheel since 2.2.2) nor use the GPU if torch is forced
# in from conda-forge. Both failures are silent, so warn on exactly that pair.


def test_intel_python_on_apple_silicon_warns():
    msg = interpreter_arch_warning(machine="x86_64", hw_arm64=True)
    assert "Rosetta" in msg and "arm64" in msg


def test_native_arm64_python_is_silent():
    assert interpreter_arch_warning(machine="arm64", hw_arm64=True) == ""


def test_genuine_intel_mac_is_silent():
    # Nothing to fix: the hardware really is x86_64, so no Rosetta is involved.
    assert interpreter_arch_warning(machine="x86_64", hw_arm64=False) == ""


def test_linux_x86_64_is_silent():
    assert interpreter_arch_warning(machine="x86_64", hw_arm64=False) == ""
