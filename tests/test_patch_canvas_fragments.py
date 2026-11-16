"""Tests for tools/patch_canvas_fragments.py — the offline fragment repair path.

This is the tool that re-binds an already-extracted volume without spending any
Gemini calls, so it is the vehicle for repairing a corpus after a re-align. Two
things it got wrong are pinned here.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.patch_canvas_fragments import patch_item_dir  # noqa: E402

CANVAS = "https://iiif.example.org/iiif/3/1234/full/!760,760/0/default.jpg"


def frag(x, y, w, h):
    return f"{CANVAS}#xywh={x},{y},{w},{h}"


def write_page(d: Path, stem: str, *, mode: str, entries: list[dict],
               lines: list[dict], aligned_slug="gemini-2.0-flash",
               ner_slug="gemini-3.1-flash-lite", canvas_width=2400):
    (d / f"{stem}_{aligned_slug}_aligned.json").write_text(json.dumps({
        "image": f"{stem}.jpg",
        "canvas_uri": CANVAS,
        "canvas_width": canvas_width,
        "canvas_height": 4000,
        "lines": lines,
    }), encoding="utf-8")
    (d / f"{stem}_{ner_slug}_entries.json").write_text(json.dumps({
        "image": f"{stem}.jpg", "mode": mode, "entries": entries,
    }), encoding="utf-8")


def read_entries(d: Path, stem: str, ner_slug="gemini-3.1-flash-lite"):
    return json.loads((d / f"{stem}_{ner_slug}_entries.json").read_text())["entries"]


class TestModeGate:
    """Two thirds of the corpus — and every Green Book volume — is `multimodal`.

    The mode records how the NER call was made, not where the boxes came from,
    so gating on it made this tool silently skip those pages and report success.
    """

    def _page(self, tmp_path, mode):
        write_page(
            tmp_path, "0001_1234", mode=mode,
            lines=[{"gemini_text": "Hotel Theresa-2090 7th Ave.",
                    "canvas_fragment": frag(100, 200, 900, 50)}],
            entries=[{"name": "Hotel Theresa", "address": "2090 7th Ave.",
                      "canvas_fragment": CANVAS}],
        )

    def test_multimodal_pages_are_patched(self, tmp_path):
        self._page(tmp_path, "multimodal")
        stats = patch_item_dir(tmp_path, "gemini-2.0-flash")
        assert stats["patched"] == 1
        assert stats["skipped"] == 0
        assert read_entries(tmp_path, "0001_1234")[0]["canvas_fragment"] == frag(100, 200, 900, 50)

    def test_text_only_pages_are_still_patched(self, tmp_path):
        self._page(tmp_path, "text-only")
        assert patch_item_dir(tmp_path, "gemini-2.0-flash")["patched"] == 1

    def test_unknown_mode_is_skipped(self, tmp_path):
        self._page(tmp_path, "something-else")
        stats = patch_item_dir(tmp_path, "gemini-2.0-flash")
        assert stats["patched"] == 0 and stats["skipped"] == 1


class TestSharedMatcher:
    """It used to keep its own copy of the matcher, with the same bugs."""

    def test_category_cannot_supply_a_box(self, tmp_path):
        write_page(
            tmp_path, "0002_1234", mode="multimodal",
            lines=[{"gemini_text": "BEAUTY PARLORS", "canvas_fragment": frag(700, 100, 400, 50)}],
            entries=[{"name": "The Parker House", "address": "627 Erie Ave.",
                      "category": "BEAUTY PARLORS", "canvas_fragment": CANVAS}],
        )
        stats = patch_item_dir(tmp_path, "gemini-2.0-flash")
        assert stats["patched"] == 0
        assert stats["unplaced"] == 1
        # the original fragment is left alone rather than overwritten with a guess
        assert read_entries(tmp_path, "0002_1234")[0]["canvas_fragment"] == CANVAS

    def test_two_entries_cannot_share_one_box(self, tmp_path):
        write_page(
            tmp_path, "0003_1234", mode="multimodal",
            lines=[{"gemini_text": "Grand Hotel-1 A St.", "canvas_fragment": frag(10, 10, 500, 40)}],
            entries=[{"name": "Grand Hotel", "address": "1 A St.", "canvas_fragment": CANVAS},
                     {"name": "Grand Hotel", "address": "1 A St.", "canvas_fragment": CANVAS}],
        )
        patch_item_dir(tmp_path, "gemini-2.0-flash")
        got = [e["canvas_fragment"] for e in read_entries(tmp_path, "0003_1234")]
        assert got.count(frag(10, 10, 500, 40)) == 1

    def test_dry_run_writes_nothing(self, tmp_path):
        write_page(
            tmp_path, "0004_1234", mode="multimodal",
            lines=[{"gemini_text": "Hotel Theresa-2090 7th Ave.",
                    "canvas_fragment": frag(100, 200, 900, 50)}],
            entries=[{"name": "Hotel Theresa", "address": "2090 7th Ave.",
                      "canvas_fragment": CANVAS}],
        )
        assert patch_item_dir(tmp_path, "gemini-2.0-flash", dry_run=True)["patched"] == 1
        assert read_entries(tmp_path, "0004_1234")[0]["canvas_fragment"] == CANVAS


class TestCounters:
    def test_no_aligned_files_returns_every_key(self, tmp_path):
        """The caller sums these; a missing key was a KeyError."""
        stats = patch_item_dir(tmp_path, "gemini-2.0-flash")
        assert set(stats) == {"patched", "skipped", "no_match", "unplaced"}
