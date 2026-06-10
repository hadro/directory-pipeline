#!/bin/zsh
# Dry-run regression harness — captures `main.py --dry-run` output for a fixed
# matrix of invocations so a refactor can be verified byte-identical:
#
#   tests/dryrun_harness.sh /tmp/before     # on the baseline commit
#   ... make changes ...
#   tests/dryrun_harness.sh /tmp/after
#   diff -r /tmp/before /tmp/after          # must be empty
#
# Hermetic equivalents of the highest-value cases also run in CI as pytest
# tests (tests/test_dryrun_cli.py); this script remains useful for full-matrix
# byte-identical comparisons against a real local volume.
#
# Requires a local volume at $VOL (any output dir with images + a {slug}.csv
# works — adjust VOL below). Network-free except fixture 14: LoC *item* URLs
# fetch the item title from the LoC API to derive the slug (collections URLs
# and generic IIIF URLs are pure string parsing).
set -u
OUT="$1"; mkdir -p "$OUT"
VOL=output/lain_healy_s_brooklyn_directory_for_the_1897BPL
P="$(dirname "$0")/../.venv/bin/python"
run() { n="$1"; shift; "$P" main.py "$@" --dry-run > "$OUT/$n.txt" 2>&1; }

run 01_layout      $VOL/ --surya-detect --detect-columns --detect-spreads --split-spreads
run 02_core_chain  $VOL/ --surya-ocr --gemini-ocr --align-ocr --extract-entries --geocode --map --explore --visualize
run 03_extract     $VOL/ --extract
run 04_guided      $VOL/ --guided
run 05_compare     $VOL/ --compare-ocr --models gemini-2.0-flash gemini-3.1-flash-lite --workers 3 --skip-empty-rerun --high-res
run 06_gemini_opts $VOL/ --gemini-ocr --ocr-model gemini-2.0-flash --workers 5 --expand-dittos --high-res --ocr-prompt prompts/ocr_prompt.md --no-flex
run 07_align_multi $VOL/ --align-ocr --models gemini-2.0-flash gemini-3.1-flash-lite --force --min-surya-confidence 0.35
run 08_extract_mm  $VOL/ --extract-entries --ner-prompt prompts/ner_prompt.md --mode multimodal --force
run 09_calibrate   $VOL/ --select-pages --generate-prompts --no-open
run 10_columns     $VOL/ --detect-columns --threshold 0.12 --max-columns 3 --workers 2 --force
run 11_surya_batch $VOL/ --surya-ocr --batch-size 2
run 12_viz_opts    $VOL/ --visualize --no-text --force --ocr-model gemini-2.0-flash
run 13_review      $VOL/ --review-alignment
run 14_loc_url     https://www.loc.gov/item/01015253/ --loc-csv --download --width 1024
run 15_iiif_url    https://example.org/iiif/manifest.json --download
run 16_sections    $VOL/ --gemini-ocr --extract-entries --sections prompts/ner_prompt.md
run 17_slug        $VOL/ --explore --slug brooklyn_custom
run 18_csv_src     $VOL/$(basename $VOL).csv --download
echo "captured to $OUT"
