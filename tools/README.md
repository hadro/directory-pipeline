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
| `report_cost.py` | What a volume's Gemini calls actually cost, priced from the token usage the OCR and NER stages record in `pipeline_state.json` (measured, not estimated: retries and blocked calls are in it). `--by-model` aggregates across volumes; `--by-reason` splits the bill by what provoked each call (initial, temperature retry, simple prompt, escalation, and extraction's `ner` tags). Pointed at a root, it skips test collections unless `--all-collections`. |
| `analyze_retry_ladder.py` | Which rungs of the OCR retry ladder actually rescue a page, read from run logs (including the archived ones in `docs/evidence/`). The other half of `report_cost.py`: what each rung achieved rather than what it cost. |
| `compare_alignments.py` | Score two sets of aligned JSON against each other without ground truth (`--old`, `--new`, `--surya`). Reports geometric metrics (reading-order inversions, backward jumps) and agreement between each line's Gemini text and what Surya read in its box. The text score is partly circular, so trust the geometric ones first. |
| `detect_column_merges.py` | Find pages where Gemini read across two columns and merged their rows into one line. `align_ocr`'s `possible_column_merge` flag over-reports this, because Surya over-segmenting a page trips it too. `--mode address`, `generic` or `both`. |
| `add_page_column.py` | Add a readable `page` column to entries CSVs from the manifest's canvas labels (or scan position), which the explorer then offers as a facet. `--printed-offset N` shows the printed page beside the scan position. |
| `slice_manifest.py` | Only one section of a huge volume is worth extracting — writes a synthetic `output/{slug}/manifest.json` holding just the canvases you want, so `--download` fetches those pages instead of all 890. Select by position (`--from`/`--to`) or by canvas-URI substring (`--from-id`/`--to-id`); `--list` dumps every canvas with its id and label. Canvas `@id`s are preserved, so `canvas_fragment` values still resolve against the source repository. Feed the result straight to `main.py` as the source. |

Run from the repo root with the project venv (e.g. `uv run python tools/review_ocr.py output/my_vol/item/`).
