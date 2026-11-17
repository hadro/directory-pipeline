"""Which Surya releases the pipeline can use, and what to say when it has another.

The pipeline needs Surya's line detector: DetectionPredictor, plus a
RecognitionPredictor built over FoundationPredictor, which together return a box,
text and confidence per printed line. Alignment, entry binding, ALTO export and the
review UI are all built on those lines.

- 0.20 removed ``surya.foundation``.
- 0.24 (Surya 2.5, 2026-11-13) removed the line detector altogether. It is a
  layout VLM that returns block-level HTML: a directory column comes back as one
  or a few blocks.

``uv.lock`` is not tracked, so the cap in pyproject.toml's ``gpu`` extra is what
keeps a fresh ``uv sync --extra gpu`` on a usable release. Before the cap, that
resolved to 0.20.0 and then 0.24.0, and ``--surya-ocr`` failed on import.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version

SUPPORTED_SPEC = "surya-ocr>=0.17.1,<0.20"
_FIRST_UNSUPPORTED = (0, 20)


def installed_surya_version() -> "str | None":
    """The installed surya-ocr version string, or None if it is not installed."""
    try:
        return version("surya-ocr")
    except PackageNotFoundError:
        return None


def _is_unsupported(ver: str) -> bool:
    parts = [int(n) for n in re.findall(r"\d+", ver)[:2]]
    return len(parts) == 2 and tuple(parts) >= _FIRST_UNSUPPORTED


def surya_import_error(exc: ImportError) -> str:
    """The message for a failed Surya import.

    When Surya is installed but too new, "No module named 'surya.foundation'"
    reads like a broken install, and reinstalling does not help unless the
    reinstall honours the cap. Name the cause, and both ways to a supported
    release: this checkout's ``uv sync`` re-resolves under the cap, and a
    notebook kernel or other pip environment needs the spec spelled out.
    """
    installed = installed_surya_version()
    if installed and _is_unsupported(installed):
        return (
            f"Error: {exc}\n"
            f"surya-ocr {installed} is installed. This pipeline needs Surya's line "
            "detector, which 0.20 and later removed (0.24 replaced it with a layout "
            "model that returns blocks, not lines). Install a supported release:\n"
            "  uv sync --extra gpu        (this checkout caps the version)\n"
            f'  pip install "{SUPPORTED_SPEC}"   (outside uv, e.g. a notebook kernel)'
        )
    return f"Error: {exc}\nInstall with:  uv sync --extra gpu"
