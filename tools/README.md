# tools/

One-off repair and triage utilities — **not pipeline stages**. None of these are
invoked by `main.py` or the `pipeline` CLI; run them directly when the situation
they fix arises.

| Script | When to use it |
|---|---|
| `rescale_canvas_fragments.py` | Aligned JSON was written against a placeholder (square) canvas size because an info.json fetch failed mid-run — rescales `canvas_fragment` coordinates in place. `align_ocr.py` prints a warning naming this script when it detects the condition. |
| `patch_canvas_fragments.py` | Retroactively copy bounding-box fragments from aligned JSON into an already-extracted entries CSV (when extraction ran before alignment). |
| `review_ocr.py` | Generate an HTML triage report flagging pages whose OCR line counts deviate from their neighbors — quick visual scan for OCR problems. |
| `slice_manifest.py` | Only one section of a huge volume is worth extracting — writes a synthetic `output/{slug}/manifest.json` holding just the canvases you want, so `--download` fetches those pages instead of all 890. Select by position (`--from`/`--to`) or by canvas-URI substring (`--from-id`/`--to-id`); `--list` dumps every canvas with its id and label. Canvas `@id`s are preserved, so `canvas_fragment` values still resolve against the source repository. Feed the result straight to `main.py` as the source. |

Run from the repo root with the project venv (e.g. `uv run python tools/review_ocr.py output/my_vol/item/`).
