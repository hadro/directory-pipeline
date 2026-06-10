# tools/

One-off repair and triage utilities — **not pipeline stages**. None of these are
invoked by `main.py` or the `pipeline` CLI; run them directly when the situation
they fix arises.

| Script | When to use it |
|---|---|
| `rescale_canvas_fragments.py` | Aligned JSON was written against a placeholder (square) canvas size because an info.json fetch failed mid-run — rescales `canvas_fragment` coordinates in place. `align_ocr.py` prints a warning naming this script when it detects the condition. |
| `patch_canvas_fragments.py` | Retroactively copy bounding-box fragments from aligned JSON into an already-extracted entries CSV (when extraction ran before alignment). |
| `review_ocr.py` | Generate an HTML triage report flagging pages whose OCR line counts deviate from their neighbors — quick visual scan for OCR problems. |

Run from the repo root with the project venv (e.g. `uv run python tools/review_ocr.py output/my_vol/item/`).
