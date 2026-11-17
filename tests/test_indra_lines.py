"""Tests for tools/indra_lines.py — Indra line detection written in Surya's schema.

Indra's models download from Hugging Face, so these run the tool against a fake
``indra_ocr`` that returns canned pages. What is pinned: the record parses as a
Surya file, pages are chosen as run_surya_ocr chooses them, and nothing the
later stages write can reach the real volume through the mirror.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.indra_lines as T  # noqa: E402
from pipeline.align_ocr import parse_surya  # noqa: E402


@dataclass
class FakeLine:
    box: list
    text: str
    conf: float


@dataclass
class FakePage:
    lines: list
    timings: dict = field(default_factory=lambda: {"total_ms": 210.0, "det_ms": 90.0})
    blocks: list = field(default_factory=list)
    source: str = "ocr"
    note: str = ""
    rotation: int = 0
    skew: float = 0.0
    vertical: bool = False


def jpg(path: Path, size=(400, 600)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path, "JPEG")
    return path


# ---- page selection -------------------------------------------------------

def test_selection_matches_run_surya_ocr(tmp_path):
    vol = tmp_path / "vol"
    jpg(vol / "0001_a.jpg")
    jpg(vol / "0001_a_viz.jpg")                       # a render, not a page
    jpg(vol / "0002_b.jpg")
    jpg(vol / "0002_b_left.jpg")                      # split: halves replace it
    jpg(vol / "0002_b_right.jpg")
    jpg(vol / "item" / "0003_c.jpg")
    jpg(vol / "_backup" / "0004_d.jpg")               # backup dirs are not pages
    names = [p.name for p in T.select_images(vol)]
    assert names == ["0001_a.jpg", "0002_b_left.jpg", "0002_b_right.jpg", "0003_c.jpg"]


def test_selection_skips_the_mirror_inside_the_volume(tmp_path):
    vol = tmp_path / "output"
    jpg(vol / "v" / "0001_a.jpg")
    jpg(vol / "mirror" / "v" / "0001_a.jpg")
    assert [p.parent.name for p in T.select_images(vol, exclude=vol / "mirror")] == ["v"]


def test_mirror_keeps_the_path_under_output(tmp_path):
    out = tmp_path / "output"
    (out / "coll" / "vol").mkdir(parents=True)
    m = T.mirror_root_for(out / "coll" / "vol", out / "_indra_lines", out)
    assert m == out / "_indra_lines" / "coll" / "vol"
    elsewhere = tmp_path / "elsewhere" / "vol2"
    elsewhere.mkdir(parents=True)
    assert T.mirror_root_for(elsewhere, out / "_indra_lines", out) == out / "_indra_lines" / "vol2"


# ---- the record -----------------------------------------------------------

def test_record_is_a_surya_file(tmp_path):
    page = FakePage(
        lines=[
            FakeLine([10.4, 20.6, 200.2, 40.0], "SMITH John, clerk, 12 Main", 0.93),
            FakeLine([-3, 50, 450, 70], "", 0.41),           # under Indra's MIN_CONF: no text
            FakeLine([100, 80, 100, 90], "zero width", 0.9),  # degenerate: dropped
        ],
        blocks=[{"label": "List-Group", "bbox": [5, 15, 390, 75], "text": "…"}],
    )
    rec = T.page_record(page, 400, 600, "1.1.2")
    assert rec["lines"][0] == {"bbox": [10, 21, 200, 40], "text": "SMITH John, clerk, 12 Main",
                               "confidence": 0.93}
    assert rec["lines"][1]["bbox"] == [0, 50, 400, 70]   # clamped to the image
    assert rec["detector"]["degenerate_boxes_dropped"] == 1
    assert rec["detector"]["timings_ms"]["total_ms"] == 210.0
    assert rec["blocks"][0]["label"] == "List-Group"

    p = tmp_path / "0001_a_surya.json"
    p.write_text(json.dumps(rec))
    page_bbox, lines, median = parse_surya(p, min_confidence=0.35)
    assert page_bbox == (0, 0, 400, 600)
    assert [ln["text"] for ln in lines] == ["SMITH John, clerk, 12 Main", ""]
    assert median == pytest.approx(0.67)


# ---- the mirror -----------------------------------------------------------

def test_mirror_links_images_and_copies_what_stages_write(tmp_path):
    vol = tmp_path / "vol"
    img = jpg(vol / "item" / "0001_a.jpg")
    (vol / "pipeline_state.json").write_text('{"ocr_model": "gemini-3.1-flash-lite"}')
    (vol / "ner_prompt.md").write_text("prompt")
    (vol / "item" / "manifest.json").write_text("{}")
    (vol / "item" / "0001_a_gemini-3.1-flash-lite.txt").write_text("SMITH John")
    (vol / "item" / "0001_a_surya.txt").write_text("surya's text")
    (vol / "item" / "0001_a_gemini-3.1-flash-lite_entries_error.txt").write_text("raw")
    mirror = tmp_path / "mirror" / "vol"

    linked = T.mirror_inputs(img, vol, mirror)

    assert linked.is_symlink() and linked.resolve() == img.resolve()
    item = mirror / "item"
    gem = item / "0001_a_gemini-3.1-flash-lite.txt"
    assert gem.read_text() == "SMITH John" and not gem.is_symlink()
    assert not (item / "0001_a_surya.txt").exists()
    assert not (item / "0001_a_gemini-3.1-flash-lite_entries_error.txt").exists()
    assert (item / "manifest.json").exists() and not (item / "manifest.json").is_symlink()
    state = mirror / "pipeline_state.json"
    assert state.exists() and (mirror / "ner_prompt.md").exists()

    # A stage writing its state in the mirror must not touch the volume.
    state.write_text('{"stages": ["align_ocr"]}')
    assert json.loads((vol / "pipeline_state.json").read_text()) == {"ocr_model": "gemini-3.1-flash-lite"}


# ---- end to end, with a fake Indra -----------------------------------------

def fake_indra(pages_by_shape):
    mod = types.ModuleType("indra_ocr")
    mod.__version__ = "1.1.2"

    class ModelError(Exception):
        pass

    class OrtOCR:
        def __init__(self, path, threads=4):
            self.threads = threads

        def __call__(self, gray, turn=True):
            assert turn is False, "pages must be read upright"
            assert gray.dtype.name == "uint8" and gray.ndim == 2
            return pages_by_shape[gray.shape]

    mod.ModelError, mod.OrtOCR = ModelError, OrtOCR
    mod.model_dir = lambda *a, **k: "models"
    return mod


def test_main_writes_records_and_timings(tmp_path, monkeypatch):
    out = tmp_path / "output"
    vol = out / "vol"
    jpg(vol / "0001_a.jpg", (400, 600))
    jpg(vol / "0002_b.jpg", (300, 500))
    (vol / "0001_a_gemini-3.1-flash-lite.txt").write_text("SMITH John")
    pages = {
        (600, 400): FakePage([FakeLine([10, 20, 200, 40], "SMITH John", 0.95)]),
        (500, 300): FakePage([], source="error", note="RuntimeError: boom"),
    }
    monkeypatch.setitem(sys.modules, "indra_ocr", fake_indra(pages))
    monkeypatch.setattr(sys, "argv", ["indra_lines.py", str(vol), "--output-root", str(out),
                                      "--mirror-root", str(out / "_indra_lines"), "-q"])
    with pytest.raises(SystemExit) as e:
        T.main()
    assert e.value.code == 1                       # the second page failed

    m = out / "_indra_lines" / "vol"
    rec = json.loads((m / "0001_a_surya.json").read_text())
    assert rec["lines"][0]["text"] == "SMITH John"
    assert (m / "0001_a_surya.txt").read_text() == "SMITH John"
    assert not (m / "0002_b_surya.json").exists()  # a failed page writes nothing
    log = json.loads((m / "indra_run.json").read_text())
    assert set(log["pages"]) == {"0001_a.jpg"}
    assert log["pages"]["0001_a.jpg"]["indra_ms"] == 210.0
    assert not list(vol.glob("*_surya.json"))      # the volume itself is untouched


def test_image_arguments_are_grouped_by_directory(tmp_path):
    a = jpg(tmp_path / "v1" / "0001_a.jpg")
    b = jpg(tmp_path / "v1" / "0002_b.jpg")
    c = jpg(tmp_path / "v2" / "0001_c.jpg")
    targets = dict(T._targets([a, b, c, tmp_path / "v3"]))
    assert targets[tmp_path / "v1"] == [a, b]
    assert targets[tmp_path / "v2"] == [c]
    assert targets[tmp_path / "v3"] == []          # a directory: every page
