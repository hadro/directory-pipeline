#!/bin/bash
# harvest_slice.sh <manifest_url> <slug> <from_canvas> <to_canvas>
# Slice a IIIF manifest to a canvas range -> download -> Gemini OCR -> name-pool NER.
# Calls the leaf scripts directly so --no-blank-skip can be passed (main.py does not
# forward it, and the blank heuristic misfires on light-ink dense directory pages).
set -uo pipefail
PIPE=/Users/joshhadro/github/directory-pipeline
PY=$PIPE/.venv/bin/python
MANIFEST="$1"; SLUG="$2"; FROM="$3"; TO="$4"
cd "$PIPE"

echo "=== [$SLUG] slice canvases $FROM-$TO ==="
$PY tools/slice_manifest.py "$MANIFEST" --from "$FROM" --to "$TO" --slug "$SLUG" || exit 1

WANT=$(( TO - FROM + 1 ))
# IA's image server times out intermittently; retry until the slice is complete.
for attempt in 1 2 3 4; do
  HAVE=$(ls "output/$SLUG"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
  [ "$HAVE" -ge "$WANT" ] && break
  echo "=== [$SLUG] download attempt $attempt (have $HAVE/$WANT) ==="
  $PY pipeline/download_images.py --manifest "output/$SLUG/manifest.json" \
      -o "output/$SLUG" --width 1800 --resume
done
HAVE=$(ls "output/$SLUG"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "=== [$SLUG] downloaded $HAVE/$WANT pages ==="
[ "$HAVE" -eq 0 ] && { echo "[$SLUG] FATAL: no pages downloaded"; exit 1; }

echo "=== [$SLUG] Gemini OCR ==="
$PY pipeline/run_gemini_ocr.py "output/$SLUG" --workers 4 --expand-dittos --no-blank-skip

echo "=== [$SLUG] extract name-pool entries ==="
$PY pipeline/extract_entries.py "output/$SLUG" --prompt "$PIPE/prompts/harvest/ner_prompt.md"

echo "=== [$SLUG] REPORT ==="
OCR_OK=0; OCR_BAD=0
for t in "output/$SLUG"/*_gemini*.txt; do
  [ -f "$t" ] || continue
  if grep -q "OCR FAILED" "$t" || [ "$(wc -c < "$t")" -lt 400 ]; then OCR_BAD=$((OCR_BAD+1)); else OCR_OK=$((OCR_OK+1)); fi
done
echo "SLUG=$SLUG PAGES=$HAVE OCR_OK=$OCR_OK OCR_FAILED=$OCR_BAD"
for f in "output/$SLUG"/entries_*.csv; do
  [ -f "$f" ] || continue
  echo "CSV=$f ROWS=$(( $(wc -l < "$f") - 1 ))"
  echo "--- header ---"; head -1 "$f"
  echo "--- 6 sample rows ---"; sed -n '2,7p' "$f"
done
