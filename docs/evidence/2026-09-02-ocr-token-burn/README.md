# Evidence — Gemini OCR token burn and undetected degenerate pages

**Date:** 2026-09-02
**Source run:** an 83-page NYC city-directory name harvest (Trow 1907, Polk 1917/1931/1933,
Longworth 1820, Boyd 1885), driven by `harvest_slice.sh` in this folder.
**Consumer:** `city-directory-extraction` — see its `docs/HANDOFF.md`, section
"⛔ THE NAME HARVEST DOES NOT CLOSE THE COVERAGE GAP".

Four issues, each with the artifact that demonstrates it. Everything here is reproducible;
`check_repetition.py` runs standalone with no dependencies.

---

## Issue 1 — `run_gemini_ocr.py` sets no `max_output_tokens`

**Claim.** `_call_gemini()`'s `GenerateContentConfig` (`pipeline/run_gemini_ocr.py`, ~line 206)
omits `max_output_tokens`, so a degenerating page runs to the model ceiling (~65k tokens).

**This is an omission, not a design choice** — the other two call sites already cap:

```
pipeline/extract_entries.py:227      max_output_tokens=65536
analysis/review_entries.py:86        max_output_tokens=8192
pipeline/run_gemini_ocr.py           (absent)
```

**Evidence.** `logs/log_job1.txt`, `logs/log_job3.txt`:

```
WARNING: truncated at the output token limit (MAX_TOKENS) — transcription is
incomplete (32758 lines): …trowsgeneraldir1907p1trow_0302.jp2.jpg
repetition loop (100% of lines are [[blank]…]) — retrying at temperature=0.1
```

32,758 lines of `[blank]` at `FinishReason.MAX_TOKENS` ≈ 65k output tokens for one useless call.

**Sizing.** Largest *legitimate* page in this 83-page corpus is ~29KB ≈ 7k tokens. A 16384 cap
gives >2× headroom and cuts every wasted attempt ~75%. Make it a flag, not a hardcode — the
code already warns-and-accepts on `MAX_TOKENS`, so too tight a cap truncates quietly.

---

## Issue 2 — the retry ladder runs in full even when it cannot help

**Claim.** Every failure walks 3 temperatures + 2 simple-prompt attempts + 1 model escalation —
up to **6 extra calls**, each able to burn the full (uncapped) output budget.

**Evidence.** `logs/log_job2.txt`, one page:

```
repetition loop (93% of lines are [" Morris (M & M) h360 Beckman av…]) — retrying at temperature=0.7
Trying simple fallback prompt at temperature=0.3
Trying simple fallback prompt at temperature=0.7
Escalating to gemini-3-flash-preview
```

Across the run, **no repetition-loop page was rescued by a temperature retry** — every one walked
the whole ladder. The re-run captured in this folder shows the same shape: for leaf 0452,
temperatures 0.1/0.3/0.7 all returned repetition loops and only the simple prompt helped.

**Suggested change.** Branch on failure type: for `repetition loop`, go from one retry straight to
model escalation (6 calls → 2). Keep the full ladder for `RECITATION`, where varying the prompt
genuinely changes the outcome.

**Related.** The `keep best available output` path writes known-bad transcripts to disk —
`pages/polk1933mb_58192514_ocr_accepted-repetition.txt` is 144KB of which 2,788 lines are one
repeated string. It was correctly *detected* at generation time and saved anyway.

---

## Issue 3 — the repetition check misses template repetition (this one also corrupts data)

**Claim.** `_output_issue()` counts *identical* lines. A degenerating model often repeats a
template with a varying number, so every line differs and the check passes.

**Evidence.** `check_repetition.py` implements the current check (constants copied verbatim from
`run_gemini_ocr.py:126-128`) alongside a variant that normalises digit runs to `#` first:

```
$ python3 check_repetition.py pages/*_ocr*.txt

file                                     lines  current  proposed  verdict
polk1933mb_58192514_…accepted-repetition  3501     80%      80%    both flag
trow1907p1_leaf0452_ocr.txt                  4      0%       0%    clean
trow1907p1_leaf0454_ocr.txt               6142     33%      33%    clean   <- see gap below
trow1907p1_leaf0456_ocr.txt                 97      1%      92%    CAUGHT ONLY BY PROPOSED
    dominant: 'John, # W #th'
trow1907p1_leaf0906_ocr.txt                453      0%      99%    CAUGHT ONLY BY PROPOSED
    dominant: '# W #th St'
```

**False-positive rate: zero.** Swept over all 83 harvested pages, the proposed check flags nothing
that the current one does not, except genuinely degenerate pages:

```
$ python3 check_repetition.py ../../../output/harvest_*/*_gemini*.txt
flagged only by the proposed check: 0   not flagged by either: 78
```

**Downstream damage this caused.** These pages emitted a fake surname `Edgey` 627 times and 89
phantom `Brown John` rows into the harvest's name pool, and inflated `Brown` to 1,349. They had to
be found and removed by hand.

**A second, separate gap: the 0.40 threshold is too high.** Leaf 0454's saved transcript is 180KB,
8,188 lines, with 2,046 repeats of `"William, who died Oct. 24, 1905."` — and scores **33%**, so
neither check flags it. Lowering `_REPETITION_THRESHOLD` to **0.30** catches it, plus one more real
failure (`0001_58192510`, 39%, whose transcript is 3,064/3,898 lines of two repeated templates),
**with zero false positives on this corpus** — no clean page scores above 2%. The gap between
signal and noise is wide, so 0.30 is not a tight fit.

---

## Issue 4 — the `[blank]` sentinel is a repetition attractor

**Claim.** `prompts/ocr_prompt.md:13` invites a per-region `[blank]`, handing a degenerating model
a short, safe, repeatable line to loop on. The 32,758-line runs in Issue 1 are made **entirely of
that token**.

**Do not delete it** — it is a documented machine contract:

```
prompts/README.md:26                 the sentinel table
analysis/fix_entries.py:589          SENTINEL_TOKENS = frozenset({"[illegible]", "[blank]"})
tests/test_fix_entries.py:34,48-49   asserts both tokens and that they legitimately repeat
docs/pipeline-stages.md:84           the OCR→extract contract
```

**Suggested change:** re-scope from region level to page level — "if the entire page contains no
text, output `[blank]` once and nothing else; never repeat it line after line." Same for
`[illegible]`. Keeps the contract, removes the attractor.

---

## Cost context

Measured for the 83-page run: ~483k input tokens, ~1.7M output tokens, of which **more than half
went to transcripts that were then discarded**. At flash-lite rates that is roughly $0.75 total,
~$0.40 of it wasted. The absolute number is small; the *ratio* is what scales.

Note `--flex` (~50% cheaper, 1–15 min latency, default-on in `main.py`) was disabled for this run
via `--no-flex` to keep wall-clock sane. `harvest_slice.sh` shows the exact invocation.

---

## Contents

```
README.md                 this file
check_repetition.py       runnable current-vs-proposed detector comparison; no dependencies
harvest_slice.sh          the exact driver used (slice → download → OCR → NER)
pages/
  trow1907p1_leaf0452.jpg / _ocr.txt    degenerate: "- 122 W 23d st" template
  trow1907p1_leaf0454.jpg / _ocr.txt    degenerate, 180KB, scores 33% — under the threshold
  trow1907p1_leaf0456.jpg / _ocr.txt    degenerate: current 1% vs proposed 92%
  trow1907p1_leaf0906.jpg / _ocr.txt    degenerate: current 0% vs proposed 99%
  polk1933mb_58192514_ocr_accepted-repetition.txt
                                        detected at 80% and saved anyway ("keep best available")
logs/
  log_job*.txt, log_tjob*.txt, log_trow_*.txt
                                        raw stdout of the harvest runs: MAX_TOKENS warnings,
                                        repetition-loop detections, the full retry ladder
```

**Provenance caveat, stated plainly.** The four `trow1907p1_leaf*` transcripts were **regenerated
on 2026-09-02** by re-fetching the same IIIF canvases and re-running
`run_gemini_ocr.py --workers 4 --expand-dittos --no-blank-skip` — the originals were deleted while
cleaning the harvest before this evidence folder was assembled. The degeneration reproduced, which
is itself useful: it is a property of these pages, not a one-off. The saved transcripts differ in
detail from the originals because the retry ladder is nondeterministic and keeps whichever attempt
it judges best. The `polk1933mb_*` file and every log are original, untouched artifacts.

To regenerate from scratch:

```bash
PY=.venv/bin/python
M="https://iiif.archive.org/iiif/trowsgeneraldir1907p1trow/manifest.json"
$PY tools/slice_manifest.py "$M" --from 450 --to 456 --slug repro_d
$PY pipeline/download_images.py --manifest output/repro_d/manifest.json -o output/repro_d --width 1800 --resume
$PY pipeline/run_gemini_ocr.py output/repro_d --workers 4 --expand-dittos --no-blank-skip
```

Leaves 0452, 0454 and 0456 are in that range; leaf 0906 is in canvases 900-903.
