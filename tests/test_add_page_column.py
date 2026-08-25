"""tools/add_page_column.py — deriving a `page` facet from the volume manifest.

The two things that can silently mislabel a whole volume: split-spread
filenames whose `_left`/`_right` suffix never matches a canvas, and a manifest
canvas that iter_canvases skips, which would slide a positional label lookup
permanently out of step.
"""

import json

from tools.add_page_column import build_page_labels, image_id_from_filename


def _canvas(image_id: str, label=None) -> dict:
    canvas = {
        "id": f"https://ex.org/canvas/{image_id}",
        "items": [{"items": [{"body": {
            "service": [{"id": f"https://img.ex.org/iiif/{image_id}"}],
        }}]}],
    }
    if label is not None:
        canvas["label"] = label
    return canvas


def _manifest(*canvases) -> dict:
    return {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "items": list(canvases),
    }


def _write(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# image_id_from_filename
# ---------------------------------------------------------------------------

class TestImageIdFromFilename:
    def test_strips_download_index_prefix(self):
        assert image_id_from_filename("0001_p16445coll4:27074.jpg") == "p16445coll4:27074"

    def test_strips_left_spread_suffix(self):
        # split_spreads writes {stem}_left.jpg; without stripping it, the id
        # never matches a manifest canvas and every row goes unlabelled.
        assert image_id_from_filename("0001_p16445coll4:27074_left.jpg") == "p16445coll4:27074"

    def test_strips_right_spread_suffix(self):
        assert image_id_from_filename("0001_p16445coll4:27074_right.jpg") == "p16445coll4:27074"

    def test_spread_halves_share_one_image_id(self):
        left = image_id_from_filename("0042_abc123_left.jpg")
        right = image_id_from_filename("0042_abc123_right.jpg")
        assert left == right == "abc123"

    def test_no_prefix_returns_stem(self):
        assert image_id_from_filename("abc123.jpg") == "abc123"

    def test_underscore_inside_image_id_preserved(self):
        assert image_id_from_filename("0001_coll_4_27074.jpg") == "coll_4_27074"


# ---------------------------------------------------------------------------
# build_page_labels
# ---------------------------------------------------------------------------

class TestBuildPageLabels:
    def test_label_page_number_wins_over_position(self, tmp_path):
        manifest = _manifest(
            _canvas("img1", {"en": ["Front cover"]}),
            _canvas("img2", {"en": ["Page 7"]}),
        )
        labels = build_page_labels(_write(tmp_path, manifest), None)
        assert labels["img2"] == "Page 7"

    def test_falls_back_to_canvas_position(self, tmp_path):
        manifest = _manifest(_canvas("img1"), _canvas("img2"))
        labels = build_page_labels(_write(tmp_path, manifest), None)
        assert labels == {"img1": "Page 1", "img2": "Page 2"}

    def test_printed_offset_shows_both_numbers(self, tmp_path):
        manifest = _manifest(_canvas("img1", {"en": ["Page 674"]}))
        labels = build_page_labels(_write(tmp_path, manifest), 28)
        assert labels["img1"] == "p. 646 (scan 674)"

    def test_skipped_canvas_does_not_shift_labels(self, tmp_path):
        # The middle canvas has no image service, so iter_canvases drops it.
        # A positional lookup into the raw list would hand img3 the skipped
        # canvas's "Page 2" instead of its own "Page 3".
        service_less = {
            "id": "https://ex.org/canvas/gap",
            "label": {"en": ["Page 2"]},
            "items": [{"items": [{"body": {"id": "https://ex.org/img.jpg"}}]}],
        }
        manifest = _manifest(
            _canvas("img1", {"en": ["Page 1"]}),
            service_less,
            _canvas("img3", {"en": ["Page 3"]}),
        )
        labels = build_page_labels(_write(tmp_path, manifest), None)
        assert labels == {"img1": "Page 1", "img3": "Page 3"}

    def test_v2_manifest_labels(self, tmp_path):
        manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "sequences": [{"canvases": [{
                "@id": "https://ex.org/v2/canvas/0",
                "label": "Page 5",
                "images": [{"resource": {
                    "service": {"@id": "https://img.ex.org/iiif/v2img1"},
                }}],
            }]}],
        }
        labels = build_page_labels(_write(tmp_path, manifest), None)
        assert labels == {"v2img1": "Page 5"}
