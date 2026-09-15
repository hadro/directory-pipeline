# tools/

One-off repair and triage utilities — **not pipeline stages**. None of these are
invoked by `main.py` or the `pipeline` CLI; run them directly when the situation
they fix arises.

| Script | When to use it |
|---|---|
| `rescale_canvas_fragments.py` | Aligned JSON was written against a placeholder (square) canvas size because an info.json fetch failed mid-run — rescales `canvas_fragment` coordinates in place. `align_ocr.py` prints a warning naming this script when it detects the condition. |
| `patch_canvas_fragments.py` | Retroactively copy bounding-box fragments from aligned JSON into an already-extracted entries CSV (when extraction ran before alignment). Shares its matcher with `extract_entries` via `utils/entry_binding.py` — it used to keep a second, divergent copy that carried the same category-steal bug. Reports `unplaced=` for entries no line matched strongly enough; those keep whatever fragment they had rather than being overwritten with a guess. |
| `audit_entry_bindings.py` | Answer the question coverage metrics cannot: is each published entry's box on that entry's line? Compares every CSV row's `canvas_fragment` against the text of the aligned line it points at — no ground truth needed. `--a-b` also reports what the current binder would change, which makes it the before/after checkpoint for any alignment or binding change. Use `--resolve coverage` after a re-align, `--resolve overlap` (default) when auditing already-shipped fragments. |
| `collate_review_queue.py` | Collate the entries `extract_entries` could not place (its `unplaced_entries`) across a volume or collection into one list, ranked by how close the nearest miss came. `nearest_score` is the triage signal: just under the floor usually means a real line the binder wouldn't claim; 0.00 usually means no line exists. |
| `review_ocr.py` | Generate an HTML triage report flagging pages whose OCR line counts deviate from their neighbors — quick visual scan for OCR problems. |
| `slice_manifest.py` | Only one section of a huge volume is worth extracting — writes a synthetic `output/{slug}/manifest.json` holding just the canvases you want, so `--download` fetches those pages instead of all 890. Select by position (`--from`/`--to`) or by canvas-URI substring (`--from-id`/`--to-id`); `--list` dumps every canvas with its id and label. Canvas `@id`s are preserved, so `canvas_fragment` values still resolve against the source repository. Feed the result straight to `main.py` as the source. |

Run from the repo root with the project venv (e.g. `uv run python tools/review_ocr.py output/my_vol/item/`).
