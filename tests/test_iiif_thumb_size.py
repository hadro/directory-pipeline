"""Never ask an image service to upscale a crop.

A region requested at or above its own pixel count is served from a coarse
pyramid level and compressed far harder. Measured on tile.loc.gov, one 737x145
crop of the 1869 Rowell directory:

    pct: region, size 800,   ->  3,203 bytes    25 KB/Mpx   unreadable
    pct: region, size 737,   ->  2,917 bytes    27 KB/Mpx   unreadable
    pct: region, size 600,   -> 25,339 bytes   350 KB/Mpx
    pct: region, size full   -> 39,679 bytes   363 KB/Mpx

Note the second row: with pct: addressing even the *native* width fails, because
the percentage rounds the region down to 736 and 737 is then an upscale. Asking
for /full/ is the only request that always returns the region's own pixels.
"""

from pipeline.geo.map_entries import _iiif_region_url

CANVAS = "https://tile.loc.gov/image-services/iiif/public:gdc:x"
SVC_MAP = {CANVAS: (CANVAS, 1600, 2514, "")}


def _url(xywh, *, thumb_width=220, pad=20):
    return _iiif_region_url(f"{CANVAS}#xywh={xywh}", SVC_MAP,
                            thumb_width=thumb_width, pad=pad)


def test_a_region_narrower_than_the_thumb_asks_for_full():
    """The upscale case — this is what produced 3 KB images in the explorer."""
    url = _url("100,100,40,20", thumb_width=220)
    assert "/full/0/default.jpg" in url
    assert "/220,/" not in url


def test_a_region_wider_than_the_thumb_keeps_the_explicit_width():
    """A genuine downscale is fine and keeps the payload small."""
    url = _url("100,100,900,40", thumb_width=220)
    assert "/220,/" in url


def test_the_boundary_is_treated_as_an_upscale():
    """Equal width still trips it: pct: rounds the region down."""
    # pad=0 so the region is exactly 220 wide
    url = _url("100,100,220,40", thumb_width=220, pad=0)
    assert "/full/" in url


def test_the_region_is_still_addressed_by_percentage():
    """pct: keeps the crop correct whatever resolution the server holds."""
    url = _url("160,251,320,100", pad=0)
    assert "/pct:10.0000,9.9841,20.0000,3.9777/" in url


def test_a_malformed_fragment_yields_no_url():
    assert _iiif_region_url("no-fragment-here", SVC_MAP) == ""
    assert _url("1,2,0,0") == ""



# --- Image API version ------------------------------------------------------

NYPL = "https://iiif.nypl.org/iiif/3/56855255"
V3_MAP = {NYPL: (NYPL, 1600, 2514, "", 3)}


def test_a_v3_service_is_asked_for_max_not_full():
    """iiif.nypl.org speaks Image API 3 and answers 400 "Invalid size" to /full/,
    which broke every narrow Green Book thumbnail."""
    url = _iiif_region_url(f"{NYPL}#xywh=100,100,40,20", V3_MAP, thumb_width=220, pad=20)
    assert "/max/0/default.jpg" in url
    assert "/full/" not in url


def test_a_v3_downscale_keeps_the_explicit_width():
    url = _iiif_region_url(f"{NYPL}#xywh=100,100,900,40", V3_MAP, thumb_width=220, pad=20)
    assert "/220,/" in url


def test_the_explorer_canvas_map_carries_the_service_version(tmp_path):
    import json
    from pipeline.explore_entries import _build_canvas_thumb_map
    (tmp_path / "vol").mkdir()
    (tmp_path / "vol" / "manifest.json").write_text(json.dumps({
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "items": [{"id": "https://ex.org/canvas/1", "width": 10, "height": 10,
                   "items": [{"items": [{"body": {"service": [
                       {"id": NYPL, "type": "ImageService3"}]}}]}]}],
    }), encoding="utf-8")
    assert _build_canvas_thumb_map(tmp_path)["https://ex.org/canvas/1"][4] == 3

# --- context lines ----------------------------------------------------------

from pipeline.explore_entries import build_html


def _template(**kw):
    return build_html([{"image": "0009.jpg"}], [{"name": "image", "type": "id"}],
                      {}, "t", **kw)


def test_context_lines_defaults_to_three():
    assert "const CONTEXT_LINES = 3;" in _template()


def test_context_lines_is_configurable():
    assert "const CONTEXT_LINES = 7;" in _template(context_lines=7)


def test_context_lines_cannot_be_zero_or_negative():
    """A zero would crop to the line's own bbox with no surrounding text, and a
    negative would invert the region."""
    assert "const CONTEXT_LINES = 1;" in _template(context_lines=0)
    assert "const CONTEXT_LINES = 1;" in _template(context_lines=-4)


def test_the_explorer_picks_the_keyword_by_version():
    assert '(svc[4] || 2) >= 3 ? "max" : "full"' in _template()
