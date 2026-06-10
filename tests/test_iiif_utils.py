"""utils/iiif_utils.py — version detection, canvas iteration, image URLs."""

from utils.iiif_utils import image_url, iter_canvases, manifest_version


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


def test_manifest_version_v2_context():
    assert manifest_version(V2) == 2


def test_manifest_version_v3():
    assert manifest_version(V3) == 3


def test_manifest_version_inferred_from_sequences():
    assert manifest_version({"sequences": [{}]}) == 2


def test_manifest_version_list_context():
    m = {"@context": ["http://iiif.io/api/presentation/2/context.json"]}
    assert manifest_version(m) == 2


def test_iter_canvases_v2():
    canvases = list(iter_canvases(V2))
    assert len(canvases) == 1  # the service-less canvas is skipped
    c = canvases[0]
    assert c["canvas_id"] == "https://ex.org/canvas/1"
    assert (c["canvas_width"], c["canvas_height"]) == (2048, 3000)
    assert c["service_id"] == "https://ex.org/iiif/img1"  # trailing slash stripped
    assert c["image_id"] == "img1"
    assert c["max_width"] == 1024


def test_iter_canvases_v3_service_list():
    canvases = list(iter_canvases(V3))
    assert len(canvases) == 1
    assert canvases[0]["service_id"] == "https://ex.org/iiif/img1"
    assert canvases[0]["max_width"] is None


def test_image_url_fixed_width():
    assert image_url("https://ex.org/iiif/img1", 1024) == (
        "https://ex.org/iiif/img1/full/1024,/0/default.jpg"
    )


def test_image_url_native_v2_uses_full():
    assert image_url("https://ex.org/iiif/img1", 0, iiif_version=2).endswith("/full/full/0/default.jpg")


def test_image_url_native_v3_uses_max():
    assert image_url("https://ex.org/iiif/img1", 0, iiif_version=3).endswith("/full/max/0/default.jpg")
