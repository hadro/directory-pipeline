"""pipeline/download_images.py — corrupt and interrupted downloads.

On 2026-10-16, 30 pages in output/ were corrupt: empty files, bodies cut at a
chunk boundary, IA responses stopping at 108,620 bytes. download_file skipped any
file that existed, so none was ever fetched again and every later stage failed on
them. Each came back whole on a single re-request.
"""

import io
import json

import pytest
from PIL import Image

import pipeline.download_images as dl


def _jpeg(width=40, height=60) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), 200).save(buf, "JPEG")
    return buf.getvalue()


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


class _Session:
    """Serves queued bodies in order and records the URLs requested."""

    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.urls = []

    def get(self, url, **kw):
        self.urls.append(url)
        return _Resp(self.bodies.pop(0))


def _no_partial_files(directory):
    return not list(directory.glob("*.part"))


def test_complete_file_is_skipped_without_a_request(tmp_path):
    dest = tmp_path / "0001_a.jpg"
    dest.write_bytes(_jpeg())
    session = _Session()
    assert dl.download_file(session, "https://x/img", dest, delay=0) is False
    assert session.urls == []


@pytest.mark.parametrize("damage", [lambda b: b[: len(b) // 2], lambda b: b""],
                         ids=["truncated", "empty"])
def test_corrupt_file_is_downloaded_again(tmp_path, damage):
    dest = tmp_path / "0001_a.jpg"
    dest.write_bytes(damage(_jpeg()))
    session = _Session(_jpeg())
    assert dl.download_file(session, "https://x/img", dest, delay=0) is True
    assert dl.image_is_complete(dest)
    assert len(session.urls) == 1


def test_incomplete_body_is_fetched_again(tmp_path):
    dest = tmp_path / "0001_a.jpg"
    body = _jpeg()
    session = _Session(body[:100], body)
    assert dl.download_file(session, "https://x/img", dest, delay=0) is True
    assert dl.image_is_complete(dest)
    assert len(session.urls) == 2
    assert _no_partial_files(tmp_path)


def test_persistently_incomplete_body_raises_and_writes_nothing(tmp_path):
    # An interrupted stream used to be written straight to dest, which is how
    # the truncated files got there in the first place.
    dest = tmp_path / "0001_a.jpg"
    session = _Session(*[_jpeg()[:100]] * dl.DOWNLOAD_ATTEMPTS)
    with pytest.raises(OSError):
        dl.download_file(session, "https://x/img", dest, delay=0)
    assert not dest.exists()
    assert _no_partial_files(tmp_path)


@pytest.fixture
def volume(tmp_path, v3_manifest, monkeypatch):
    """A downloaded volume: cached manifest, a good page 1, a corrupt page 2."""
    (tmp_path / "manifest.json").write_text(json.dumps(v3_manifest))
    ids = [row[0] for row in dl.iter_canvas_images(v3_manifest, 0)]
    (tmp_path / f"0001_{ids[0]}.jpg").write_bytes(_jpeg(width=1200))
    corrupt = tmp_path / f"0002_{ids[1]}.jpg"
    corrupt.write_bytes(_jpeg(width=1200)[:500])
    session = _Session(_jpeg(width=1200))
    monkeypatch.setattr(dl, "new_session", lambda: session)
    return corrupt, session


def test_redownload_page_restores_a_corrupt_page_at_its_siblings_width(volume):
    # The volume was fetched at 1200 px, not the 2048 default, and nothing
    # recorded that. A replacement at 2048 would not match its neighbours.
    corrupt, session = volume
    assert dl.redownload_page(corrupt, delay=0) is True
    assert dl.image_is_complete(corrupt)
    assert len(session.urls) == 1
    assert "/full/1200,/" in session.urls[0]


def test_resumed_download_repairs_a_corrupt_page_at_its_siblings_width(volume, v3_manifest):
    # Same rule as the OCR stages: a repair matches the volume, so this run's
    # --width (the 2048 default here) applies only to pages not yet downloaded.
    corrupt, session = volume
    canvases = dl.iter_canvas_images(v3_manifest, 2048)
    downloaded, skipped = dl._download_item_images(
        canvases, corrupt.parent, _Session(), 2048, delay=0, quiet=True)
    assert (downloaded, skipped) == (1, 1)
    assert dl.image_is_complete(corrupt)
    assert "/full/1200,/" in session.urls[0]


def test_redownload_page_will_not_guess_at_an_unmatched_name(volume):
    corrupt, session = volume
    renamed = corrupt.with_name("0002_something_else.jpg")
    corrupt.rename(renamed)
    assert dl.redownload_page(renamed, delay=0) is False
    assert session.urls == []


def test_redownload_page_needs_the_cached_manifest(volume):
    corrupt, session = volume
    (corrupt.parent / "manifest.json").unlink()
    assert dl.redownload_page(corrupt, delay=0) is False
    assert session.urls == []


# --- IIIF Image API version ---------------------------------------------------

LOC_SVC = "https://tile.loc.gov/image-services/iiif/public:gdcmassbookdig:x:x_0011"


def test_native_size_uses_the_services_version_not_the_manifests():
    """A Presentation 3 manifest can point at an Image API 2 server. Native-size
    requests used the manifest's version and asked tile.loc.gov for "max",
    which it answers with a 400."""
    manifest = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "items": [{"id": "https://ex.org/c1", "width": 100, "height": 100,
                   "items": [{"items": [{"body": {"service": [
                       {"id": LOC_SVC, "type": "ImageService2"}]}}]}]}],
    }
    [(_, url, _, _)] = dl.iter_canvas_images(manifest, 0)
    assert url == f"{LOC_SVC}/full/full/0/default.jpg"


def test_the_fallback_loc_manifest_declares_an_image_api_2_service(monkeypatch):
    """When loc.gov blocks the real manifest one is built from the item JSON. It
    used to label tile.loc.gov's service ImageService3, which steered every
    size keyword to "max"."""
    from utils import iiif_utils

    class _Item:
        def json(self):
            return {"item": {"title": "A directory"},
                    "resources": [{"files": [[{"info": f"{LOC_SVC}/info.json",
                                               "width": 1600, "height": 2514}]]}]}

    monkeypatch.setattr(dl, "get_with_retry", lambda *a, **k: _Item())
    manifest = dl._build_loc_manifest(object(), "https://www.loc.gov/item/12345/manifest.json")
    [canvas] = list(iiif_utils.iter_canvases(manifest))
    assert canvas["service_version"] == 2
    body = manifest["items"][0]["items"][0]["items"][0]["body"]
    assert body["id"].endswith("/full/full/0/default.jpg")
