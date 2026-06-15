# prompts/

Generic fallback prompts plus examples of generated, collection-specific ones.

## Runtime fallbacks (used when no volume-specific prompt exists)

- `ocr_prompt.md` — default OCR system prompt for `--gemini-ocr`
- `ner_prompt.md` — default NER system prompt for `--extract-entries`, and the
  structural template that `--generate-prompts` adapts (`--ner-template`)

Both stages prefer a volume-specific prompt and only fall back to these files.
Lookup order: `ocr_prompt.md` / `ner_prompt.md` in the volume output directory,
then its parent, then any immediate item subdirectory — then the files here.
Run `--generate-prompts` (or `pipeline calibrate`) once per collection type to
create tailored prompts in the volume directory; pass `--ocr-prompt` /
`--ner-prompt` to reuse them across volumes in the same series.

## Sentinel tokens (machine contract)

Two literal tokens are reserved across every prompt so downstream code can tell
*intentionally empty/unreadable* apart from *merely absent*. Treat them as exact
literals — never translate, paraphrase, or wrap them.

| Token | Meaning | Emitted by |
|-------|---------|-----------|
| `[blank]` | The region genuinely contains no text. | OCR for an empty region (chiefly per-line crop OCR, where an empty crop is real signal). |
| `[illegible]` | Text is present but cannot be read at all, after a best-effort attempt. | OCR for a span no reading can recover; NER when a source value is present but unreadable. |

Rules of the contract:

- **OCR** still transcribes a best reading for merely *degraded* text — reserve
  `[illegible]` for spans that are genuinely impossible to read, and `[blank]`
  for empty regions.
- **NER** preserves a `[illegible]`/`[blank]` token that appears in the source
  rather than inventing a value; a field that is simply *not present* stays
  empty (it is **not** filled with a sentinel).
- **Post-processing** (`analysis/fix_entries.py`) recognizes these literals as
  explicit gap markers (`SENTINEL_TOKENS`) so they are excluded from
  hallucination-repetition detection and never confused with real data.

## examples/

Prompts that `--generate-prompts` produced for the Green Book collection — kept
as reference output, not loaded by any stage.
