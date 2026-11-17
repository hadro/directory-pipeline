"""utils/surya_version.py — the supported Surya range and the import-failure message."""

import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

import utils.surya_version as sv

ROOT = Path(__file__).resolve().parent.parent


def test_spec_matches_the_gpu_extra():
    # The message tells people what to install; it must name what uv installs.
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"]["gpu"]
    assert sv.SUPPORTED_SPEC in extra


@pytest.mark.parametrize("ver, unsupported", [
    ("0.17.1", False),
    ("0.19.9", False),
    ("0.20.0", True),   # removed surya.foundation
    ("0.24.0", True),   # Surya 2.5: no line detector
    ("1.0", True),
    ("0.24.0rc1", True),
])
def test_unsupported_from_0_20(ver, unsupported):
    assert sv._is_unsupported(ver) is unsupported


def _installed(monkeypatch, ver):
    def fake_version(name):
        assert name == "surya-ocr"
        if ver is None:
            raise PackageNotFoundError(name)
        return ver
    monkeypatch.setattr(sv, "version", fake_version)


def test_message_names_a_too_new_release_and_the_fix(monkeypatch):
    _installed(monkeypatch, "0.24.0")
    msg = sv.surya_import_error(ModuleNotFoundError("No module named 'surya.detection'"))
    assert "surya-ocr 0.24.0 is installed" in msg
    assert "uv sync --extra gpu" in msg
    assert f'pip install "{sv.SUPPORTED_SPEC}"' in msg
    assert "surya.detection" in msg  # the original error is kept


@pytest.mark.parametrize("ver", [None, "0.17.1"])
def test_message_is_the_install_hint_otherwise(monkeypatch, ver):
    # Missing, or a supported release that failed for another reason (a broken
    # torch, say): the version is not the problem, so do not blame it.
    _installed(monkeypatch, ver)
    msg = sv.surya_import_error(ImportError("boom"))
    assert msg == "Error: boom\nInstall with:  uv sync --extra gpu"


def test_run_surya_ocr_exits_with_the_message_when_surya_is_missing(tmp_path):
    try:
        import surya  # noqa: F401
        pytest.skip("surya is installed here")
    except ImportError:
        pass
    (tmp_path / "0001_page.jpg").write_bytes(b"")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "pipeline" / "run_surya_ocr.py"), str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 1
    assert "Install with:  uv sync --extra gpu" in proc.stderr
