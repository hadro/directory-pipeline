"""End-to-end --dry-run smoke tests for the orchestrator and the pipeline CLI.

Hermetic versions of the T11 fixture matrix: each test runs main.py (or
cli.main) as a subprocess against a stub volume inside a temp working
directory and asserts on the commands it would run. No network, no API key,
nothing written outside tmp_path.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN = str(REPO_ROOT / "main.py")
VOL = "output/myvol"


@pytest.fixture()
def workdir(tmp_path):
    (tmp_path / VOL).mkdir(parents=True)
    (tmp_path / VOL / "0001_img.jpg").write_bytes(b"")
    return tmp_path


def run_main(workdir, *args: str) -> str:
    res = subprocess.run(
        [sys.executable, MAIN, *args, "--dry-run"],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout + res.stderr


def test_core_chain_commands(workdir):
    out = run_main(workdir, VOL, "--gemini-ocr", "--align-ocr", "--extract-entries", "--explore")
    assert f"-m pipeline.run_gemini_ocr {VOL} --flex" in out
    assert f"-m pipeline.align_ocr {VOL}" in out
    assert f"-m pipeline.extract_entries {VOL} --flex" in out
    assert f"-m pipeline.explore_entries {VOL}" in out


def test_no_flex_and_model_passthrough(workdir):
    out = run_main(workdir, VOL, "--gemini-ocr", "--ocr-model", "gemini-x", "--workers", "2", "--no-flex")
    assert f"-m pipeline.run_gemini_ocr {VOL} --model gemini-x --workers 2" in out
    assert "--flex" not in out


def test_extract_shorthand_expands(workdir):
    out = run_main(workdir, VOL, "--extract")
    for module in ("pipeline.download_images", "pipeline.run_gemini_ocr",
                   "pipeline.extract_entries", "pipeline.explore_entries"):
        assert f"-m {module}" in out


def test_generic_iiif_url_uses_manifest_mode(workdir):
    out = run_main(workdir, "https://example.org/iiif/manifest.json", "--download")
    assert "-m pipeline.download_images --manifest https://example.org/iiif/manifest.json" in out
    assert "--resume" in out


def test_cli_ocr_subcommand_passthrough(workdir):
    res = subprocess.run(
        [sys.executable, "-m", "cli.main", "ocr", VOL, "--dry-run"],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout + res.stderr
    assert f"-m pipeline.run_surya_ocr {VOL}" in out
    assert f"-m pipeline.run_gemini_ocr {VOL} --flex" in out
    assert f"-m pipeline.align_ocr {VOL}" in out
