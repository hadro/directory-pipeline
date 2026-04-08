"""Tests for utils/iiif_utils.py — version-agnostic IIIF manifest parsing."""

import pytest

from utils.iiif_utils import (
    manifest_version,
    iter_canvases,
    image_url,
    manifest_item_id,
    update_annotation_targets,
)


# ---------------------------------------------------------------------------
# manifest_version
# ---------------------------------------------------------------------------

class TestManifestVersion:
    def test_v3_context_string(self):
        manifest = {"@context": "http://iiif.io/api/presentation/3/context.json"}
        assert manifest_version(manifest) == 3

    def test_v2_context_string(self):
        manifest = {"@context": "http://iiif.io/api/presentation/2/context.json"}
        assert manifest_version(manifest) == 2

    def test_v2_detected_by_sequences_key(self):
        # No @context but has a non-empty "sequences" and no "items" → v2
        # (empty sequences list is falsy and does not trigger the heuristic)
        manifest = {"sequences": [{"canvases": []}]}
        assert manifest_version(manifest) == 2

    def test_v3_default_when_no_context(self):
        manifest = {"items": []}
        assert manifest_version(manifest) == 3

    def test_v2_context_in_list(self):
        manifest = {
            "@context": [
                "http://www.w3.org/ns/anno.jsonld",
                "http://iiif.io/api/presentation/2/context.json",
            ]
        }
        assert manifest_version(manifest) == 2


# ---------------------------------------------------------------------------
# iter_canvases — v3
# ---------------------------------------------------------------------------

class TestIterCanvasesV3:
    def test_yields_correct_count(self, v3_manifest):
        canvases = list(iter_canvases(v3_manifest))
        assert len(canvases) == 2

    def test_canvas_ids(self, v3_manifest):
        canvases = list(iter_canvases(v3_manifest))
        assert canvases[0]["canvas_id"] == "https://example.org/canvas/0"
        assert canvases[1]["canvas_id"] == "https://example.org/canvas/1"

    def test_canvas_dimensions(self, v3_manifest):
        canvas = list(iter_canvases(v3_manifest))[0]
        assert canvas["canvas_width"] == 2048
        assert canvas["canvas_height"] == 3000

    def test_service_id_from_list(self, v3_manifest):
        # First canvas has service as a list
        canvas = list(iter_canvases(v3_manifest))[0]
        assert canvas["service_id"] == "https://img.example.org/iiif/image001"

    def test_service_id_from_dict(self, v3_manifest):
        # Second canvas has service as a dict
        canvas = list(iter_canvases(v3_manifest))[1]
        assert canvas["service_id"] == "https://img.example.org/iiif/image002"

    def test_image_id_is_last_path_segment(self, v3_manifest):
        canvas = list(iter_canvases(v3_manifest))[0]
        assert canvas["image_id"] == "image001"

    def test_empty_items_yields_nothing(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [],
        }
        assert list(iter_canvases(manifest)) == []

    def test_canvas_without_body_skipped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [
                {
                    "id": "https://example.org/canvas/0",
                    "type": "Canvas",
                    "width": 100,
                    "height": 100,
                    "items": [],  # no annotation page
                }
            ],
        }
        assert list(iter_canvases(manifest)) == []

    def test_canvas_without_service_skipped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [
                {
                    "id": "https://example.org/canvas/0",
                    "type": "Canvas",
                    "width": 100,
                    "height": 100,
                    "items": [
                        {
                            "items": [
                                {
                                    "body": {
                                        "id": "https://example.org/img.jpg",
                                        # no "service" key
                                    }
                                }
                            ]
                        }
                    ],
                }
            ],
        }
        assert list(iter_canvases(manifest)) == []

    def test_max_width_present(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [
                {
                    "id": "https://example.org/canvas/0",
                    "type": "Canvas",
                    "width": 2048,
                    "height": 3000,
                    "items": [
                        {
                            "items": [
                                {
                                    "body": {
                                        "service": {
                                            "id": "https://img.example.org/iiif/img1",
                                            "maxWidth": 4096,
                                        }
                                    }
                                }
                            ]
                        }
                    ],
                }
            ],
        }
        canvas = list(iter_canvases(manifest))[0]
        assert canvas["max_width"] == 4096

    def test_max_width_absent_is_none(self, v3_manifest):
        canvas = list(iter_canvases(v3_manifest))[0]
        assert canvas["max_width"] is None

    def test_service_trailing_slash_stripped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/3/context.json",
            "items": [
                {
                    "id": "https://example.org/canvas/0",
                    "type": "Canvas",
                    "width": 100,
                    "height": 100,
                    "items": [
                        {
                            "items": [
                                {
                                    "body": {
                                        "service": {
                                            "id": "https://img.example.org/iiif/img1/",
                                        }
                                    }
                                }
                            ]
                        }
                    ],
                }
            ],
        }
        canvas = list(iter_canvases(manifest))[0]
        assert not canvas["service_id"].endswith("/")


# ---------------------------------------------------------------------------
# iter_canvases — v2
# ---------------------------------------------------------------------------

class TestIterCanvasesV2:
    def test_yields_correct_count(self, v2_manifest):
        canvases = list(iter_canvases(v2_manifest))
        assert len(canvases) == 2

    def test_canvas_ids(self, v2_manifest):
        canvases = list(iter_canvases(v2_manifest))
        assert canvases[0]["canvas_id"] == "https://example.org/v2/canvas/0"
        assert canvases[1]["canvas_id"] == "https://example.org/v2/canvas/1"

    def test_canvas_dimensions(self, v2_manifest):
        canvas = list(iter_canvases(v2_manifest))[0]
        assert canvas["canvas_width"] == 1800
        assert canvas["canvas_height"] == 2600

    def test_service_id_from_dict(self, v2_manifest):
        # First canvas: service is a plain dict with @id
        canvas = list(iter_canvases(v2_manifest))[0]
        assert canvas["service_id"] == "https://img.example.org/iiif/v2img001"

    def test_service_id_from_list(self, v2_manifest):
        # Second canvas: service is a list
        canvas = list(iter_canvases(v2_manifest))[1]
        assert canvas["service_id"] == "https://img.example.org/iiif/v2img002"

    def test_image_id_is_last_path_segment(self, v2_manifest):
        canvas = list(iter_canvases(v2_manifest))[0]
        assert canvas["image_id"] == "v2img001"

    def test_canvas_without_images_skipped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "sequences": [
                {
                    "canvases": [
                        {
                            "@id": "https://example.org/canvas/0",
                            "width": 100,
                            "height": 100,
                            "images": [],  # empty
                        }
                    ]
                }
            ],
        }
        assert list(iter_canvases(manifest)) == []

    def test_canvas_without_service_skipped(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "sequences": [
                {
                    "canvases": [
                        {
                            "@id": "https://example.org/canvas/0",
                            "width": 100,
                            "height": 100,
                            "images": [
                                {
                                    "resource": {
                                        "@id": "https://example.org/img.jpg",
                                        # no "service"
                                    }
                                }
                            ],
                        }
                    ]
                }
            ],
        }
        assert list(iter_canvases(manifest)) == []


# ---------------------------------------------------------------------------
# image_url
# ---------------------------------------------------------------------------

class TestImageUrl:
    def test_specific_width(self):
        url = image_url("https://img.example.org/iiif/img1", 800)
        assert url == "https://img.example.org/iiif/img1/full/800,/0/default.jpg"

    def test_full_resolution(self):
        url = image_url("https://img.example.org/iiif/img1", 0)
        assert url == "https://img.example.org/iiif/img1/full/full/0/default.jpg"

    def test_url_format_components(self):
        url = image_url("https://img.example.org/iiif/img1", 1200)
        parts = url.split("/")
        # Pattern: {service}/full/{size}/0/default.jpg
        assert "full" in parts
        assert "1200," in parts
        assert parts[-1] == "default.jpg"
        assert parts[-2] == "0"


# ---------------------------------------------------------------------------
# manifest_item_id
# ---------------------------------------------------------------------------

class TestManifestItemId:
    def test_loc_manifest_url(self):
        url = "https://www.loc.gov/item/01015253/manifest.json"
        assert manifest_item_id(url) == "01015253"

    def test_nypl_api_url(self):
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
        canvas_map = {
            "https://example.org/old/canvas/0": "https://example.org/new/canvas/0"
        }
        ann_data = {
            "items": [
                {"target": "https://example.org/old/canvas/0"}
            ]
        }
        result = update_annotation_targets(ann_data, canvas_map)
        assert result["items"][0]["target"] == "https://example.org/new/canvas/0"

    def test_fragment_preserved_after_rewrite(self):
        canvas_map = {
            "https://example.org/old/canvas/0": "https://example.org/new/canvas/0"
        }
        ann_data = {
            "items": [
                {"target": "https://example.org/old/canvas/0#xywh=10,20,100,50"}
            ]
        }
        result = update_annotation_targets(ann_data, canvas_map)
        assert result["items"][0]["target"] == "https://example.org/new/canvas/0#xywh=10,20,100,50"

    def test_unknown_canvas_not_rewritten(self):
        canvas_map = {
            "https://example.org/old/canvas/0": "https://example.org/new/canvas/0"
        }
        original_target = "https://example.org/other/canvas/99"
        ann_data = {
            "items": [{"target": original_target}]
        }
        result = update_annotation_targets(ann_data, canvas_map)
        assert result["items"][0]["target"] == original_target

    def test_empty_items_list(self):
        ann_data = {"items": []}
        result = update_annotation_targets(ann_data, {})
        assert result["items"] == []

    def test_modifies_in_place_and_returns(self):
        canvas_map = {"https://old/canvas/0": "https://new/canvas/0"}
        ann_data = {"items": [{"target": "https://old/canvas/0"}]}
        result = update_annotation_targets(ann_data, canvas_map)
        assert result is ann_data
