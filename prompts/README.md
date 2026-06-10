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

## examples/

Prompts that `--generate-prompts` produced for the Green Book collection — kept
as reference output, not loaded by any stage.
