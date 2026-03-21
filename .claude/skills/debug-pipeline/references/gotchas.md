# Known failure modes and gotchas

## 1. `--aligned-model` mismatch

**Symptom:** `canvas_fragment` has no `#xywh=` even though `*_aligned.json` files exist.

**Cause:** OCR was run with one model (e.g. `gemini-2.0-flash`), producing files named
`*_gemini-2.0-flash_aligned.json`. Extraction was called without `--aligned-model`, so
it looks for `*_{ner_model}_aligned.json` and finds nothing, silently falling back to
text-only mode.

**Detection:** `entries_*.json` files have `"mode": "text-only"`. The aligned JSON
filenames contain a different model slug than the entries CSV filename.

**Fix:**
```bash
python main.py output/{slug} --extract-entries \
  --aligned-model gemini-2.0-flash \
  --force
```

---

## 2. Scope filter extension mismatch

**Symptom:** Same as above — text-only fallback despite aligned files existing.

**Cause:** `included_pages.txt` (written by `--select-pages`) stores image filenames
ending in `.jpg`. The scope filter in `extract_entries.py` strips the last extension
to get the stem (e.g. `foo.jp2.jpg` → stem `foo.jp2`). But aligned JSON filenames
use `.jp2` as their stem (e.g. `foo_gemini-2.0-flash_aligned.json`). When the scope
filter compares `foo.jp2` (from `included_pages.txt`) against aligned file stems, it
may fail to match if the `.jp2` is in one place but not the other.

**Detection:** `len(passing_aligned_files) == 0` even though aligned files exist.
Compare stem format in `included_pages.txt` against `*_aligned.json` filename stems.

**Fix:** Delete `included_pages.txt` to process all pages, or edit it to match the
exact stem format used in the aligned JSON filenames.

---

## 3. Cached text-only entries blocking re-run

**Symptom:** Re-running `--extract-entries` with `--aligned-model` still produces
no `#xywh=` fragments.

**Cause:** `extract_entries.py` skips pages whose `*_entries.json` output file already
exists (line 464–471 in `pipeline/extract_entries.py`). If a prior run produced
text-only entries, those cached files block the fresh aligned run.

**Detection:** `entries_*.json` files exist and have `"mode": "text-only"`.

**Fix:** Add `--force` to the extraction command to overwrite cached files.

---

## 4. `line_text` empty → wrong fallback field used for bbox matching

**Symptom:** Most entries on certain pages (e.g. ship name indexes, bibliography
sections) have page-level thumbnails instead of entry-level crops, even after
alignment runs correctly.

**Cause:** The NER extraction didn't populate `line_text`. The bbox-matching fallback
tries field values in dict order — for schema types where a generic field like
`section = "INDEX OF SHIP NAMES"` comes first, it fails to match any aligned line
and falls back to the full canvas URI. The `ship_name` or other distinctive field
that would have matched comes later in the dict.

**Status:** Fixed in `pipeline/extract_entries.py` — the fallback now tries all
non-trivial string field values in order until one matches.

**If still occurring:** Re-run `--extract-entries --force` with the latest code.

---

## 5. Explorer shows no thumbnails (`--output-dir` not passed)

**Symptom:** The data explorer HTML opens but shows no images.

**Cause:** `explore_entries.py` needs to know where the images and `manifest.json`
are. When called on a CSV that lives in a different directory from the images,
`--output-dir` must be passed explicitly.

**Fix:**
```bash
python pipeline/explore_entries.py output/{slug}/entries_{model}.csv \
  --output-dir output/{slug}/
```

---

## 6. `--select-pages` scope tab not used

**Symptom:** Extraction processes frontmatter, title pages, index sections, ads —
producing many junk entries.

**Cause:** The `--select-pages` UI has two tabs: Sample (for prompt calibration) and
Scope (for filtering which pages get processed). If the Scope tab was never used,
all pages including frontmatter are processed.

**Fix:** Re-run `--select-pages`, use the Scope tab to exclude frontmatter/back-matter,
save, then re-run `--extract-entries --force`.

---

## 7. Prompt reuse not applied to subsequent volumes

**Symptom:** Entry quality drops on a second volume in the same series because the
wrong (generic) NER prompt was used.

**Cause:** `extract_entries.py` auto-detects a nearby `ner_prompt.md` but only if it
exists in the same directory. For a new volume with a different output slug, no prompt
is found and the global default is used.

**Fix:** Always pass `--ner-prompt output/{first-slug}/ner_prompt.md` when processing
additional volumes in the same series.
