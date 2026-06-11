import sys
from pathlib import Path

import pytest

# main.py lives at the repo root and is not an installed package — make it
# importable as `import main` regardless of where pytest is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def v3_manifest():
    """Minimal valid IIIF Presentation v3 manifest with two canvases.

    Canvas 0 carries its image service as a list, canvas 1 as a dict — both
    shapes appear in the wild and iter_canvases must handle each.
    """
    return {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": "https://example.org/manifest.json",
        "type": "Manifest",
        "items": [
            {
                "id": "https://example.org/canvas/0",
                "type": "Canvas",
                "width": 2048,
                "height": 3000,
                "items": [
                    {
                        "id": "https://example.org/canvas/0/page/0",
                        "type": "AnnotationPage",
                        "items": [
                            {
                                "id": "https://example.org/canvas/0/page/0/annotation/0",
                                "type": "Annotation",
                                "motivation": "painting",
                                "body": {
                                    "id": "https://img.example.org/iiif/image001/full/full/0/default.jpg",
                                    "type": "Image",
                                    "service": [
                                        {
                                            "id": "https://img.example.org/iiif/image001",
                                            "type": "ImageService3",
                                        }
                                    ],
                                },
                                "target": "https://example.org/canvas/0",
                            }
                        ],
                    }
                ],
            },
            {
                "id": "https://example.org/canvas/1",
                "type": "Canvas",
                "width": 2048,
                "height": 3000,
                "items": [
                    {
                        "id": "https://example.org/canvas/1/page/0",
                        "type": "AnnotationPage",
                        "items": [
                            {
                                "id": "https://example.org/canvas/1/page/0/annotation/0",
                                "type": "Annotation",
                                "motivation": "painting",
                                "body": {
                                    "id": "https://img.example.org/iiif/image002/full/full/0/default.jpg",
                                    "type": "Image",
                                    "service": {
                                        "id": "https://img.example.org/iiif/image002",
                                        "type": "ImageService3",
                                    },
                                },
                                "target": "https://example.org/canvas/1",
                            }
                        ],
                    }
                ],
            },
        ],
    }


@pytest.fixture
def v2_manifest():
    """Minimal valid IIIF Presentation v2 manifest with two canvases.

    Canvas 0 carries its image service as a dict, canvas 1 as a list —
    mirroring the v3 fixture's shape coverage.
    """
    return {
        "@context": "http://iiif.io/api/presentation/2/context.json",
        "@id": "https://example.org/v2/manifest.json",
        "@type": "sc:Manifest",
        "sequences": [
            {
                "@type": "sc:Sequence",
                "canvases": [
                    {
                        "@id": "https://example.org/v2/canvas/0",
                        "@type": "sc:Canvas",
                        "width": 1800,
                        "height": 2600,
                        "images": [
                            {
                                "@type": "oa:Annotation",
                                "motivation": "sc:painting",
                                "resource": {
                                    "@id": "https://img.example.org/iiif/v2img001/full/full/0/default.jpg",
                                    "@type": "dctypes:Image",
                                    "service": {
                                        "@context": "http://iiif.io/api/image/2/context.json",
                                        "@id": "https://img.example.org/iiif/v2img001",
                                        "profile": "http://iiif.io/api/image/2/level2.json",
                                    },
                                },
                            }
                        ],
                    },
                    {
                        "@id": "https://example.org/v2/canvas/1",
                        "@type": "sc:Canvas",
                        "width": 1800,
                        "height": 2600,
                        "images": [
                            {
                                "@type": "oa:Annotation",
                                "motivation": "sc:painting",
                                "resource": {
                                    "@id": "https://img.example.org/iiif/v2img002/full/full/0/default.jpg",
                                    "@type": "dctypes:Image",
                                    "service": [
                                        {
                                            "@context": "http://iiif.io/api/image/2/context.json",
                                            "@id": "https://img.example.org/iiif/v2img002",
                                            "profile": "http://iiif.io/api/image/2/level2.json",
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                ],
            }
        ],
    }
