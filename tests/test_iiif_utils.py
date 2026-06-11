"""utils/iiif_utils.py — version detection, canvas iteration, image URLs,
manifest item IDs, and annotation-target rewriting.

The compact V2/V3 dicts below cover edge cases (missing service, trailing
slash, v2 maxWidth); the richer two-canvas fixtures in conftest.py
(v2_manifest / v3_manifest) cover the spec-shaped happy paths.
"""

from utils.iiif_utils import (
    image_url,
    iter_canvases,
    manifest_item_id,
    manifest_version,
    update_annotation_targets,
)

V2 = {
    "@context": "http://iiif.io/api/presentation/2/context.json",
    "sequences": [{
        "canvases": [
            {
                "@id": "https://ex.org/canvas/1",
                "width": 2048, "height": 3000,
                "images": [{"resource": {
                    "service": {"@id": "https://ex.org/iiif/img1/", "maxWidth": 1024},
                }}],
            },
            {   # no service block → skipped
                "@id": "https://ex.org/canvas/2",
                "width": 10, "height": 10,
                "images": [{"resource": {}}],
            },
        ],
    }],
}

V3 = {
    "@context": "http://iiif.io/api/presentation/3/context.json",
    "items": [{
        "id": "https://ex.org/canvas/1",
        "width": 2048, "height": 3000,
        "items": [{"items": [{"body": {
            "service": [{"id": "https://ex.org/iiif/img1"}],
        }}]}],
    }],
}


# ---------------------------------------------------------------------------
# manifest_version
# ---------------------------------------------------------------------------

def test_manifest_version_v2_context():
    assert manifest_version(V2) == 2


def test_manifest_version_v3():
    assert manifest_version(V3) == 3


def test_manifest_version_inferred_from_sequences():
    assert manifest_version({"sequences": [{}]}) == 2


def test_manifest_version_list_context():
    m = {"@context": ["http://iiif.io/api/presentation/2/context.json"]}
    assert manifest_version(m) == 2


def test_manifest_version_defaults_to_v3():
    assert manifest_version({"items": []}) == 3


# ---------------------------------------------------------------------------
# iter_canvases — v2
# ---------------------------------------------------------------------------

def test_iter_canvases_v2():
    canvases = list(iter_canvases(V2))
    assert len(canvases) == 1  # the service-less canvas is skipped
    c = canvases[0]
    assert c["canvas_id"] == "https://ex.org/canvas/1"
    assert (c["canvas_width"], c["canvas_height"]) == (2048, 3000)
    assert c["service_id"] == "https://ex.org/iiif/img1"  # trailing slash stripped
    assert c["image_id"] == "img1"
    assert c["max_width"] == 1024


class TestIterCanvasesV2:
    def test_yields_correct_count(self, v2_manifest):
        assert len(list(iter_canvases(v2_manifest))) == 2

    def test_canvas_ids(self, v2_manifest):
        canvases = list(iter_canvases(v2_manifest))
        assert canvases[0]["canvas_id"] == "https://example.org/v2/canvas/0"
        assert canvases[1]["canvas_id"] == "https://example.org/v2/canvas/1"

    def test_canvas_dimensions(self, v2_manifest):
        canvas = list(iter_canvases(v2_manifest))[0]
        assert canvas["canvas_width"] == 1800
        assert canvas["canvas_height"] == 2600

    def test_service_id_from_dict(self, v2_manifest):
        canvas = list(iter_canvases(v2_manifest))[0]
        assert canvas["service_id"] == "https://img.example.org/iiif/v2img001"

    def test_service_id_from_list(self, v2_manifest):
        canvas = list(iter_canvases(v2_manifest))[1]
        assert canvas["service_id"] == "https://img.example.org/iiif/v2img002"

    def test_image_id_is_last_path_segment(self, v2_manifest):
        canvas = list(iter_canvases(v2_manifest))[0]
        assert canvas["image_id"] == "v2img001"

    def test_canvas_without_images_skipped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "sequences": [{
                "canvases": [{
                    "@id": "https://example.org/canvas/0",
                    "width": 100,
                    "height": 100,
                    "images": [],
                }]
            }],
        }
        assert list(iter_canvases(manifest)) == []


# ---------------------------------------------------------------------------
# iter_canvases — v3
# ---------------------------------------------------------------------------

class TestIterCanvasesV3:
    def test_yields_correct_count(self, v3_manifest):
        assert len(list(iter_canvases(v3_manifest))) == 2

    def test_canvas_ids(self, v3_manifest):
        canvases = list(iter_canvases(v3_manifest))
        assert canvases[0]["canvas_id"] == "https://example.org/canvas/0"
        assert canvases[1]["canvas_id"] == "https://example.org/canvas/1"

    def test_canvas_dimensions(self, v3_manifest):
        canvas = list(iter_canvases(v3_manifest))[0]
        assert canvas["canvas_width"] == 2048
        assert canvas["canvas_height"] == 3000

    def test_service_id_from_list(self, v3_manifest):
        canvas = list(iter_canvases(v3_manifest))[0]
        assert canvas["service_id"] == "https://img.example.org/iiif/image001"

    def test_service_id_from_dict(self, v3_manifest):
        canvas = list(iter_canvases(v3_manifest))[1]
        assert canvas["service_id"] == "https://img.example.org/iiif/image002"

    def test_image_id_is_last_path_segment(self, v3_manifest):
        canvas = list(iter_canvases(v3_manifest))[0]
        assert canvas["image_id"] == "image001"

    def test_max_width_absent_is_none(self, v3_manifest):
        canvas = list(iter_canvases(v3_manifest))[0]
        assert canvas["max_width"] is None

    def test_empty_items_yields_nothing(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [],
        }
        assert list(iter_canvases(manifest)) == []

    def test_canvas_without_body_skipped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [{
                "id": "https://example.org/canvas/0",
                "type": "Canvas",
                "width": 100,
                "height": 100,
                "items": [],  # no annotation page
            }],
        }
        assert list(iter_canvases(manifest)) == []

    def test_canvas_without_service_skipped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [{
                "id": "https://example.org/canvas/0",
                "type": "Canvas",
                "width": 100,
                "height": 100,
                "items": [{"items": [{"body": {
                    "id": "https://example.org/img.jpg",  # no "service" key
                }}]}],
            }],
        }
        assert list(iter_canvases(manifest)) == []

    def test_max_width_present(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [{
                "id": "https://example.org/canvas/0",
                "type": "Canvas",
                "width": 2048,
                "height": 3000,
                "items": [{"items": [{"body": {
                    "service": {
                        "id": "https://img.example.org/iiif/img1",
                        "maxWidth": 4096,
                    },
                }}]}],
            }],
        }
        canvas = list(iter_canvases(manifest))[0]
        assert canvas["max_width"] == 4096

    def test_service_trailing_slash_stripped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [{
                "id": "https://example.org/canvas/0",
                "type": "Canvas",
                "width": 100,
                "height": 100,
                "items": [{"items": [{"body": {
                    "service": {"id": "https://img.example.org/iiif/img1/"},
                }}]}],
            }],
        }
        canvas = list(iter_canvases(manifest))[0]
        assert not canvas["service_id"].endswith("/")


# ---------------------------------------------------------------------------
# image_url
# ---------------------------------------------------------------------------

def test_image_url_fixed_width():
    assert image_url("https://ex.org/iiif/img1", 1024) == (
        "https://ex.org/iiif/img1/full/1024,/0/default.jpg"
    )


def test_image_url_native_v2_uses_full():
    assert image_url("https://ex.org/iiif/img1", 0, iiif_version=2).endswith("/full/full/0/default.jpg")


def test_image_url_native_v3_uses_max():
    assert image_url("https://ex.org/iiif/img1", 0, iiif_version=3).endswith("/full/max/0/default.jpg")


# ---------------------------------------------------------------------------
# manifest_item_id
# ---------------------------------------------------------------------------

class TestManifestItemId:
    def test_loc_manifest_url(self):
        url = "https://www.loc.gov/item/01015253/manifest.json"
        assert manifest_item_id(url) == "01015253"

    def test_bare_identifier_url(self):
        url = "https://api-collections.nypl.org/manifests/abc-def-123"
        assert manifest_item_id(url) == "abc-def-123"

    def test_strips_manifest_json_suffix(self):
        url = "https://example.org/iiif/item42/manifest.json"
        assert manifest_item_id(url) == "item42"

    def test_strips_manifest_suffix_no_extension(self):
        url = "https://example.org/iiif/item42/manifest"
        assert manifest_item_id(url) == "item42"

    def test_strips_manifest_jsonld_suffix(self):
        url = "https://example.org/iiif/item42/manifest.jsonld"
        assert manifest_item_id(url) == "item42"

    def test_ignores_query_string(self):
        url = "https://example.org/iiif/item42/manifest.json?foo=bar"
        assert manifest_item_id(url) == "item42"

    def test_ignores_fragment(self):
        url = "https://example.org/iiif/item42/manifest.json#section"
        assert manifest_item_id(url) == "item42"

    def test_skips_generic_segment_metadata(self):
        # /metadata/iiifmanifest/default.jsonld → pick the UUID before those segments
        url = "https://dcmny.org/do/dedab5e6-1234/metadata/iiifmanifest/default.jsonld"
        assert manifest_item_id(url) == "dedab5e6-1234"

    def test_uuid_style_identifier(self):
        url = "https://example.org/manifests/550e8400-e29b-41d4-a716-446655440000"
        assert manifest_item_id(url) == "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# update_annotation_targets
# ---------------------------------------------------------------------------

class TestUpdateAnnotationTargets:
    def test_plain_canvas_target_rewritten(self):
        canvas_map = {"https://example.org/old/canvas/0": "https://example.org/new/canvas/0"}
        ann_data = {"items": [{"target": "https://example.org/old/canvas/0"}]}
        result = update_annotation_targets(ann_data, canvas_map)
        assert result["items"][0]["target"] == "https://example.org/new/canvas/0"

    def test_fragment_preserved_after_rewrite(self):
        canvas_map = {"https://example.org/old/canvas/0": "https://example.org/new/canvas/0"}
        ann_data = {"items": [{"target": "https://example.org/old/canvas/0#xywh=10,20,100,50"}]}
        result = update_annotation_targets(ann_data, canvas_map)
        assert result["items"][0]["target"] == "https://example.org/new/canvas/0#xywh=10,20,100,50"

    def test_unknown_canvas_not_rewritten(self):
        canvas_map = {"https://example.org/old/canvas/0": "https://example.org/new/canvas/0"}
        original_target = "https://example.org/other/canvas/99"
        ann_data = {"items": [{"target": original_target}]}
        result = update_annotation_targets(ann_data, canvas_map)
        assert result["items"][0]["target"] == original_target

    def test_empty_items_list(self):
        ann_data = {"items": []}
        assert update_annotation_targets(ann_data, {})["items"] == []

    def test_modifies_in_place_and_returns(self):
        canvas_map = {"https://old/canvas/0": "https://new/canvas/0"}
        ann_data = {"items": [{"target": "https://old/canvas/0"}]}
        result = update_annotation_targets(ann_data, canvas_map)
        assert result is ann_data
