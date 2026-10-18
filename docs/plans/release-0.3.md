# Release 0.3: landing plan and pre-release review

**Created 2026-10-17.** **Status:** Ready for the PRs. Decisions are made, and Phases 0
and 2 are done locally, on branch `release-0.3`:
- every blocker and fix-before-tagging item;
- the docs pass and the version bump;
- an end-to-end smoke run, and fixes for what it found.

`release-0.3` is 49 commits ahead of local `main`, which is 13 ahead of `origin/main`. A clean
checkout passes CI's steps: ruff, 670 tests passing with 16 skipped (the ones that need the
local registry), and the full dry-run matrix.

Nothing is pushed yet. Remaining: Phase 0's pushes, then Phases 1, 3, 4 and 5.

This replaces PR #10's framing as "base for the 0.3 release". Most of #10 is already on `main` as
cherry-picks, and the rest has been rebased into `alignment-column-criterion`.

## Where things stand

v0.2.0 was tagged 2026-06-11. Since then the work has spread across five places.

| Stream | Where | Commits | State |
|---|---|---|---|
| ALTO export, column-merge detector, `slice_manifest.py`, prompt contracts, leaf-script state, manual-alignment preservation, … | `origin/main` | 47 (+5 merges) | pushed, CI green 2026-09-15 |
| OCR retry ladder and degenerate detection, entry binder, Gemini default bumps, measured cost accounting, LoC fixes | local `main` | 13 | **not pushed**. 489 tests pass, ruff clean |
| Per-column alignment, running heads, shared column detection, inversion-aware selection | `alignment-column-criterion` | 5 | sits on local `main`'s tip. Supersedes PR #10 |
| Collection registry, ad-delimiter NER, `section` column and explorer switch, thumbnail sizing, Surya `--blank-threshold`, unmatched-rate outliers | `ocr-retry-hardening` (`aee65e7..8740c4a`) | 11 | never reviewed; see the review below |
| Blank-page fix, interrupted-download repair; gutter-hoist evidence and plan | working tree of `ocr-retry-hardening` | — | uncommitted |

`ocr-retry-hardening` is the old linear trunk. Its other 25 commits are copies of what is now
on `main` and `alignment-column-criterion`. Only the 11 commits above, plus the working tree,
are unique to it.

**The whole stack lands linearly**, in this order: `main`, then `alignment-column-criterion`,
then the 11, then the working tree, then the fixes.

It was replayed in a scratch worktree on 2026-10-17:

- There was one trivial conflict, in `tools/report_cost.py` (see Phase 2).
- The result equals `ocr-retry-hardening` plus the `main`-only fixes it lacks: manual-alignment
  preservation, the LoC `item_id` crash fix, and the Lorya review. Nothing is lost.
- With the working tree applied on top, a clean checkout runs **629 of 645 tests green, with 4
  ruff errors**. CI would fail on both. B1 to B4 below fix them.

### Side branches

| Branch | What | Recommendation |
|---|---|---|
| `origin/claude/jev-directory-pipeline-ljnhjt` | Jev evaluation (docs, 1 commit) | merge, like the Lorya review |
| `claude/data-explorer-improvements-b546iu` | explorer-improvements plan (docs, 2 commits) | merge |
| `claude/local-ocr-city-directories-ey5741` | `--local-ocr` backend using Chandra MLX or Apple Vision (8 commits, 7 unpushed) | **0.4**; see the note below this table |
| `htr-alto-experiments` | experimental line-crop OCR fallback (1 commit, local only) | tag `archive/…`, push the tag, delete the branch |
| `origin/claude/load-external-csv-sMJ70` | `--external-csv` for explorer and map (March, before 0.2) | tag and delete; the explorer has since been rewritten |
| `origin/claude/analyze-test-coverage-zOZ5Y` | PR #4, closed | delete; superseded by the test suite |

Why local OCR waits for 0.4:

- It is 46 commits behind `main`.
- It touches `main.py`, `stages.py` and `models.py`, all of which have changed heavily since.
- It declares the `gpu` and `local-ocr` extras as conflicting. That disables
  `uv sync --all-extras`, which the README and CLAUDE.md give as the "everything" command.

Push it now as a backup.

Delete these after the tag; they are merged or superseded:

- Local: `city_directories`, `claude/epic-cannon`, `claude/hopeful-clarke`,
  `claude/recursing-chebyshev-fb174c`, `rosetta-preflight-and-env-docs`, `t11-stage-registry`,
  `fix/loc-serial-volume-manifest` (landed as `3660708`), and `alignment-running-heads` (PR #10).
- On `origin`, already merged: `claude/directory-pipeline-aws-plan-ccbt3a`,
  `claude/directory-pipeline-review-bynegl`, `claude/uv-hugging-face-directory-pipeline-0wm8ei`,
  `claude/hopeful-clarke`, `city_directories`.

Three worktrees are stale, all detached on merged commits:

- `.claude/worktrees/epic-cannon`
- `.claude/worktrees/hopeful-clarke`
- one inside `output/green_books_and_related/the_green_book_9ea5d5b0/.claude/worktrees/`

### Open PRs

**#10** (`alignment-running-heads`): close it as superseded by the release PR.

**#5** (Fondren-Rice, PDF input) is an outside contribution. It has been open since 2026-05-13
with no reply and now conflicts. It cannot merge as written:

- It targets the pre-0.2 `main.py` stage list, not `pipeline/stages.py`.
- `--dpi` is parsed, but `DEFAULT_DPI` is what gets passed to `convert_from_path`.
- Pages are named `1.jpg`, not `NNNN_<id>.jpg`, which the page-ordering globs expect.
- It writes no manifest, so rows get no `canvas_fragment`.
- The docs say `pipeline/pdf_to_images.py`, but the file sits at the repo root.
- `pdf2image` is not declared as a dependency.
- A Windows `poppler_path` is left commented in.

Reply before tagging either way. A reasonable 0.4 item that would credit the contributor: a PDF
source that writes a synthetic `manifest.json`, the way `slice_manifest.py` does.

## Pre-release review

Scope:

- Read line by line: the 11 commits, the working tree, and `alignment-column-criterion`.
- Run (tests and ruff) and spot-checked: local `main`'s 13 commits.

Findings are ordered by severity.

### Blockers: CI goes red, or a shipped feature breaks

**B1. Sixteen tests depend on a gitignored file.** Two parametrized tests in
`tests/test_collections.py` assert judgements that live in
`analysis/collection_registry.local.txt`:

- `test_known_non_collections_are_excluded` (11 cases)
- `test_every_flower_book_is_caught` (5 cases)

They pass on this machine and fail in any fresh clone, CI included. Either point them at a
`tmp_path` registry (or the `.example` file), or `skipif` the local file is absent. The rest
of that test file already tests layering with `tmp_path`.

**B2. Explorer and map thumbnails break on IIIF Image API 3 services, the Green Books among
them.** Commit `cfafcaf` requests size `full` for any crop narrower than the display width
(`pipeline/explore_entries.html:793`, `pipeline/geo/map_entries.py:165`). Checked live on
2026-10-17:

| service | `full` | `max` |
|---|---|---|
| `iiif.nypl.org/iiif/3/` (Cantaloupe, v3) | **400** "Invalid size" | 200 |
| `tile.loc.gov` (v2) | 200 | **400** |

No single keyword works for both. The repo already records this split:
`utils/iiif_utils.image_url()` (line 187) says v3 needs `max` and v2 needs `full`, and names
Internet Archive as a v3 server.

The fix:

- Carry each service's Image API version in the canvas-to-service map.
- Choose `max` or `full` the same way `image_url()` does.
- Add a test with an NYPL v3 service.

**B3. `tools/analyze_retry_ladder.py` never excludes a collection.** The shadowing happens like
this:

1. It imports `utils.collections.classify` (line 61).
2. It then defines its own `classify(message)` (line 95), which replaces the import.
3. The filter at line 244, `classify(part)[0]`, now gets a kind string such as `"unknown"`.
4. It takes the first letter of that string, which is always truthy, so nothing is excluded.

`--all-collections` therefore changes nothing, and nothing reports it. No test covers this
path. Alias the import (`classify as classify_collection`) and add a test.

**B4. ruff fails.** The four errors:

- unused import, `tests/test_ad_delimiters.py:16`
- unused imports, `tests/test_explorer_sections.py:9` and `:94`
- B3's F811

The branch's other two errors are already fixed on `main` and disappear in the replay: F821
(`item_id` in `sources/loc_utils.py`) and F541 (`tools/report_cost.py`).

### Fix before tagging: cheap now, a schema or UX break once released

**F1. The new `section` column collides with NER schemas.** `extract_entries.py:1168` and
`:1238` call `entry.setdefault("section", page_section)`.

Several prompts already use a `section` field:

- `prompts/examples/ner_prompt_london_booksellers.md` and `prompts/harvest/ner_prompt.md`
- eight prompts in local `output/`. Tulsa 1921 and 1922 use it with the same meaning
  ("Alphabetical List of Names", "Street and Avenue Directory"); others use it for unrelated
  things.

Two consequences:

- `setdefault` keeps any value the NER returned, including the `""` these prompts emit when
  unsure. So the structural section never fills in on exactly the multi-section city
  directories `sections.txt` exists for.
- The explorer's section switch keys on `section`. Any volume whose prompt has that field gets a
  selector built from NER free text, whether or not it has a `sections.txt`.

Decide the contract before 0.3 writes it into published CSVs. Two options:

- A reserved name for the structural value (for example `volume_section`) that the explorer
  keys on.
- "Fill when empty" semantics, if the two meanings are meant to merge.

Renaming after release is a CSV schema change for `green-books`.

**F2. The `build_aligned_lines()` docstring describes the rejected rule.**
`pipeline/align_ocr.py:982` says "the one leaving less Gemini text unboxed wins". That is the
coverage rule the 2026-09-04 evidence folder exists to stop someone restoring. The code
actually calls `_alignment_cost()`. A one-sentence fix.

**F3. Two leaf-only flags that the release depends on.** `main.py` and `pipeline` forward
neither:

- `--no-blank-skip` (`run_gemini_ocr.py`)
- `--blank-threshold` (`run_surya_ocr.py`)

The new align outlier report (`align_ocr.py:958`) tells users to "try --blank-threshold 0.5 on
those pages". But `run_surya_ocr.py` has no page selection and no `--force`; its only flags are
`--batch-size`, `--blank-threshold` and `--quiet`. Following the hint today means:

1. Delete those pages' `_surya.json` files by hand.
2. Re-run the leaf script with the flag.
3. Re-align.

Two ways to fix it:

- Forward both flags the way `--max-output-tokens` is forwarded: an argument in `main.py`, an
  `Opt(...)` in `pipeline/stages.py`, and an argument in `cli/main.py`.
- Have the report print the actual recipe.

**F4. Align records its stage on partial failure.** `align_ocr.py:1646` gates on `counts["ok"]`
alone. OCR now gates on no failures and no corrupt pages, for the reason given in its own
comment: a 1%-finished run must not look like a clean one, because `main.py` trusts the state
file. Apply the same one-line rule here.

**F5. Old `[OCR FAILED]` placeholders are treated as finished pages.** Before `6b165df`, a page
that failed every retry got a `.txt` containing `[OCR FAILED: page could not be transcribed
after all retries]`. `6b165df` stops writing them, but the skip check in `process_image()`
only re-runs empty files. So existing placeholders are never retried, and align and extract
read the placeholder as page text.

Found 2026-10-17: 17 placeholders in `output/`, in 8 collections:

- 6 in `the_negro_motorist_green_book_2016298176`
- 3 in Miller's *New York As It Is*
- 2 in `tulsa_1922`
- 6 in panel probe directories. Two of these, with a `-20261016` suffix, were written by
  `historical-ocr-eval`.

Fix: treat a file starting `[OCR FAILED` like an empty one (delete it and retry).

### Say it in the release notes

**Per-column alignment on city directories.** The evidence folder
(`docs/evidence/2026-09-04-column-selection-criterion/`) says Tulsa makes more bad picks than
the rule it replaced. **That figure is an artifact; correct the README before release.** Its
harness has two faults:

- Its "whole-page" candidate calls `build_aligned_lines(column_groups=None)`. That branch
  ignores `fallback_order` and aligns over column-major order, which production never does.
- It infers the pick from equal line counts.

Re-run on 2026-10-17, `compare_criteria.py` still prints "inversion-aware 20/22" for 100 Tulsa
pages. But on those same pages the selection step ran once, and all 22 "whole-page picks" are
alignments that differ while binding the same number of lines.

To fix the harness, take the whole-page candidate from production's revert path
(`_build_line_aligned_lines(fallback_order, …)`) and compare the outputs themselves. Tulsa
1921 under the 0.3 code: a fresh alignment matches the tree on disk on 149 of 150 sampled
pages, and the column path completed on 1 of 59 multi-column pages.

The `green-books` audit of 2026-09-05 recommends shipping the inversion-aware selection as a code
fix.

A placement pilot on 2026-10-17 narrowed this.

- Ground truth: `city-directory-extraction`'s hand-gold panel, 1,766 lines on 39 pages,
  1786–1933. Each gold line's Surya box was recovered by the editor's column-major order.
- Judge: did the aligner put the verified text on that box? No candidate can move either one.

Results:

- **The per-column path engaged on only 4 of 19 multi-column pages with usable OCR.**
  - 9 two-column pages reverted via Guard 1, with 11–23% of their lines hoisted as headers.
    That is the gutter-hoist bug, seen on a second corpus.
  - 6 pages, five of them the dense four-to-six-column ones, reverted because a column's start
    could not be found in the Gemini text.
- **Placement was 96.4% correct whole-page against 96.1% for the shipped rule**, rising to
  99.0% against 98.8% without the two pages with region-level Surya.
- **Where the path ran** (217 lines), the shipped rule chose it every time. It cost 4 lines.
- **Errors cluster on dense pages that never reached the column path:** Polk 1917 0/48,
  Queens 1933 20/31, Polk 1925 11/14.

The pilot's limits:

- The gold stops at column 2.
- Tulsa's eval set is out, because its boxes come from this aligner.

Reading of the result: on city directories the per-column path rarely completes. On the panel
and on Tulsa there is no measurable gain where it ran, and it mostly did not run. It says
nothing about columns 3+. The levers are the gutter-hoist fix and column-start location, both
0.4. An opt-out flag is not needed for 0.3.

The pilot is now `docs/evidence/2026-10-17-column-alignment-panel/`.

**Gutter hoist** (`docs/plans/gutter-hoist-fix.md`): about 450 lines are mis-bound on 1,676
pages, and Guard 1 reverts 960 more. Deferred, with a plan.

**The blank-page fix changes which pages get OCR'd.** The old check silently skipped 1,142
pages that had ink. Volumes OCR'd between 2026-06-11 and this release can have real pages with
no `.txt`. Re-running `--gemini-ocr` sends only those pages, because pages with a `.txt` are
skipped as existing; then re-run align and extract.

**Models.** The defaults are:

- `gemini-3.1-flash-lite` for OCR and NER
- `gemini-3.7-flash` for prompt generation
- `gemini-3.8-flash` as the fallback

`gemini-2.0-flash` is retired and returns 404.

### Nits

Two are fixed. The two still open are carried to 0.4:

- Still open: `report_cost.find_volumes()` keeps a volume when *any* path component names a kept
  collection; what it means is `relative_to(root).parts[0]`. This is latent: nothing leaks on
  the current `output/`, because the backups hold no `pipeline_state.json`.
- Still open: when `redownload_page()` cannot find a manifest beside a corrupt page,
  `_download_item_images()` skips the page instead of falling back to a plain fetch at
  `--width`. Only reachable with a local manifest plus `--output-dir`. Since `45b1896` the
  page counts as missing and fails the stage, so it is no longer silent.
- Fixed: the explorer grouped sections on stripped labels but filtered with an unstripped `===`.
- Fixed: `run_surya_ocr.py` logged a hardcoded `(default 0.35)`.

### Docs and housekeeping

All done on `release-0.3`. The list is kept as the record:

- Bump `pyproject.toml` to `0.3.0`.
- In `docs/README.md`, index these, plus one line per `docs/evidence/` set:
  - `plans/degenerate-page-reocr.md`
  - `plans/gutter-hoist-fix.md`
  - `plans/lorya-comparison.md`
  - the Jev and explorer plans, if merged
  - this file
- `docs/usage-examples.md:226` passes the retired `gemini-2.0-flash`.
- Correct `docs/evidence/2026-09-04-column-selection-criterion/`: the "Where it does not hold"
  Tulsa rows, and the harness's whole-page candidate and count-equality pick (see the
  per-column note above). CLAUDE.md's "validated for travel guides …, unproven for city
  directories" wording rests on it.
- `AGENTS.md` (for Codex) is 182 lines against CLAUDE.md's 302. It lacks the binder,
  degenerate-detection, collection-registry and deferred-work sections. Sync it, or reduce it
  to a pointer.
- `CLAUDE.md:20` says "Deferred until after the re-align and entry-binder work". Both ship in
  0.3, so update it.
- Commit `prompts/harvest/ner_prompt.md`. `harvest_slice.sh` in
  `docs/evidence/2026-09-02-ocr-token-burn/` is tracked on `main` and already calls it. That
  breaks the evidence folder's own rule: run the reproduction from a clean checkout.
- Keep these out of the release:
  - the Colab notebook change (`--batch-size 72 → 10` plus about 220 KB of widget state)
  - `docs/blog-post.md`
  - the slides screenshot

### Found after Phase 2

A second review, and then the smoke run, turned up the following. Each is fixed on
`release-0.3` with tests.

| commit | what |
|---|---|
| `e0bb208` | B2's fix labelled LoC's synthetic manifests ImageService3, so explorer crops asked tile.loc.gov for `max` and got 400. The version is now read from the service URL first, and LoC services are v2. Checked live. |
| `11e3218` | The Colab review notebooks globbed the wrong OCR file pattern and defaulted to a retired model. |
| `1eea5de` | Pages that came back empty were left out of the billed-page count. |
| `c85c644` | A response stopped by a safety or other filter returned no text and was reported as an empty page, with no retry. It now takes the RECITATION ladder and fails if nothing rescues it. |
| `45b1896` | Image downloads did not retry on a timeout, and a page that never arrived still ended in "Done." Timeouts now retry twice, and missing pages are counted from disk and fail the stage. |
| `3314eb5` | `main.py` printed "✗ Failed" and exited 0, so scripts and the dashboard saw success. |
| `2a5d425` | On a 1924 phone book, `fix_entries` flagged 42 of 179 rows as duplicates: office and residence lines, and people sharing a surname. Where neither row has an address, a differing phone number now rules a pair out. No other CSV on disk changes. |
| `cc48163` | `report_cost.py` billed extraction's calls to the OCR retry ladder: it reported 3.00 calls per page on a run with no retries. Extraction's calls are now tagged `ner`, `ner:fallback` and `ner:review`, and retries are counted from the tags. |
| `aeb27b5` | `docs/pipeline-stages.md` and `tools/README.md` brought up to 0.3. |

### End-to-end smoke run (2026-10-17)

The volume was `ia_smoke_0_3_louisburg1924`: four canvases of the 1924 Louisburg–Franklin
County telephone directory on Internet Archive, sliced with `slice_manifest.py`. It ran with
`--no-flex`, on Surya with MPS.

| step | result |
|---|---|
| `pipeline run` | Download, then OCR, extract (178 entries) and explorer. One page timed out at 60 s on every attempt; IA's image server answers that request with a 504 after 60 s. That page exposed the download and exit-code gaps above. |
| `pipeline ocr` | Surya took about 61 s per page. Align auto-detected the OCR model. |
| `pipeline extract` | Re-aligned pages were re-extracted without `--force`: 179 entries, 9 unplaced, 3 reading-order violations. |
| `--export-alto` | 3 ALTO files. |
| `pipeline postprocess` | Fix, then explorer. This step exposed the duplicate false positives. |
| `report_cost.py` | **$0.0299** for 9 calls (3 OCR, 6 NER across two extractions) over 3 pages. No retries. |

## Landing sequence

### Phase 0: commit what exists only on this laptop

Run on `ocr-retry-hardening`:

```bash
git add utils/image_utils.py pipeline/download_images.py pipeline/run_gemini_ocr.py \
        pipeline/run_surya_ocr.py tests/test_image_utils.py tests/test_run_gemini_ocr.py \
        tests/test_download_images.py docs/evidence/2026-10-16-blank-page/
git commit     # the blank-page fix and download repair
git add docs/evidence/2026-09-22-gutter-hoist/ docs/plans/gutter-hoist-fix.md CLAUDE.md
git commit     # gutter hoist recorded as deferred work
git add prompts/harvest/ner_prompt.md
git commit     # the prompt the token-burn evidence calls
git push origin claude/local-ocr-city-directories-ey5741    # 7 commits exist only here
git tag archive/htr-alto-experiments htr-alto-experiments
git push origin archive/htr-alto-experiments
```

### Phase 1: push `main`'s 13 commits

Push them so CI sees them on their own: `git push origin main`. Use a PR instead if you want
CI to run before they land.

### Phase 2: build `release-0.3`

```bash
git switch -c release-0.3 alignment-column-criterion
git cherry-pick 4bb96df..8740c4a     # the 11
#   One conflict, in tools/report_cost.py: keep main's lines (no f-prefix) and add
#   `if exclusion_note: print(exclusion_note)` after the "Tiers:" line.
git cherry-pick <the three Phase 0 commits>
```

Then, in order:

1. One commit each for B1 to B4, F1 to F5, and the opt-out flag if you take it.
2. Merge the Jev and explorer-plan branches. Expect small conflicts in their pointer lines in
   `AGENTS.md` and `CLAUDE.md`.
3. Finish with the docs pass and the version bump.

### Phase 3: PR and validation

Open "Release 0.3" against `main`. Link it from #10, then close #10.

CI must pass in a clean checkout; that is what catches B1.

Optional, CPU-only, with no API spend: re-align one city-directory volume into a copy and run
`tools/audit_entry_bindings.py` on it. The column-criterion evidence names this as the next
measurement, and it would settle whether the opt-out should be the default for city
directories.

### Phase 4: release

Merge with a merge commit, not a squash: the commit messages carry the evidence, and a squash
loses it.

```bash
git switch main && git pull
git tag -a v0.3.0 -m "v0.3.0"
git push origin v0.3.0
gh release create v0.3.0 --title "v0.3.0 — …" --notes-file <notes>
```

Then reply on #5.

### Phase 5: clean up

```bash
git worktree remove .claude/worktrees/epic-cannon
git worktree remove .claude/worktrees/hopeful-clarke
git worktree remove output/green_books_and_related/the_green_book_9ea5d5b0/.claude/worktrees/recursing-chebyshev-fb174c
git branch -d release-0.3 alignment-column-criterion city_directories claude/epic-cannon \
    claude/hopeful-clarke claude/recursing-chebyshev-fb174c rosetta-preflight-and-env-docs \
    t11-stage-registry
# these landed as cherry-picks, which git cannot see as merged, hence -D:
git branch -D ocr-retry-hardening fix/loc-serial-volume-manifest alignment-running-heads \
    htr-alto-experiments
git tag archive/load-external-csv origin/claude/load-external-csv-sMJ70
git push origin archive/load-external-csv
git push origin --delete alignment-running-heads city_directories claude/hopeful-clarke \
    claude/directory-pipeline-aws-plan-ccbt3a claude/directory-pipeline-review-bynegl \
    claude/uv-hugging-face-directory-pipeline-0wm8ei claude/analyze-test-coverage-zOZ5Y \
    claude/load-external-csv-sMJ70 claude/jev-directory-pipeline-ljnhjt \
    claude/data-explorer-improvements-b546iu
```

`git worktree remove` refuses when a worktree holds uncommitted work; look inside before
reaching for `--force`. Delete `ocr-retry-hardening` only after `v0.3.0` exists.

## Decisions (made 2026-10-17)

1. **Per-column alignment ships as the default, with no opt-out flag.** On city directories it
   rarely completes: 4 of 19 panel pages, and 1 of 59 sampled Tulsa pages. Where it ran it
   cost 4 of 217 lines. The "Tulsa bad picks" caveat was a harness artifact.
2. **The structural section column is named `volume_section`** (F1). The explorer keys on it,
   which leaves NER's own `section` field alone.
3. **PR #5:** reply before tagging, close it in favour of a tracking issue, and build PDF input
   as a source in 0.4, crediting the contributor.
4. **Local OCR waits for 0.4.** Push its 7 unpushed commits as a backup in Phase 0.
5. **`main`'s 13 commits land through a PR**, so CI runs first.
6. **The gold panel stays local.** The pilot becomes an evidence script that reads
   `../city-directory-extraction` when present. It contains no gold, and its README reports
   aggregates only, with no gold lines quoted. Why not copy panel pages:
   - The panel has never been published. `city-directory-extraction` is public, but `data/*`
     is gitignored.
   - Our own Surya and Gemini output for those exact pages nearly reproduces the gold: 80% of
     gold lines are character-identical to Surya.
   - `historical-ocr-eval` scores OCR engines against that text.

   The Tulsa box-vs-Surya regression fixture moves to 0.4, alongside the gutter-hoist work that
   needs it.

## Release notes outline

Same shape as v0.2.0's.

**Data safety**

- The blank-page check no longer skips pages with ink.
- Interrupted downloads are detected and fetched again; writes go through a `.part` file.
- Degenerate OCR output is detected, with a retry ladder and a `--max-output-tokens` cap.
- Pages a safety filter blocked are retried, and reported as failed rather than as empty.
- Timed-out downloads are retried, and pages still missing fail the stage.
- Stage recording is gated on success, and `main.py` exits non-zero when a target fails.
- Manual alignments survive `--force`.
- Old `[OCR FAILED]` placeholder files are deleted and retried.

**Placement**

- An entry-to-line binder gated on identity fields, with `unplaced_entries`,
  `audit_entry_bindings.py` and `collate_review_queue.py`.
- Per-column alignment, with running-head hoisting and inversion-aware selection.
- NER reads the raw OCR text on pages with advertisement delimiters.

**Explorer**

- A section switch.
- Rows are embedded once; the 1869 Rowell explorer went from 29 MB to 15 MB.
- Thumbnails are no longer upscaled.
- A `--context-lines` flag.
- CONTENTdm viewer links.

**New stages and tools**

- `--export-alto` (ALTO v3)
- the column-merge detector
- `slice_manifest.py`, `add_page_column.py`, `compare_alignments.py`
- `report_cost.py` (measured usage) and `analyze_retry_ladder.py`
- the collection registry

**Models and cost**

- New defaults.
- Per-generation `thinking_config_for()`.
- Token usage persisted as a run proceeds, tagged by stage and retry rung.

**Sources**

- LoC serial and sub-resource manifests.
- A fix for the LoC title-lookup crash.
- 5xx retries on image downloads.

**Development**

- `AGENTS.md` and Codex skills.
- A warning when an Intel Python runs on Apple Silicon.
- uv docs.
- Tests: 89 → 686.

**Known issues**

- Per-column alignment on city directories.
- The gutter hoist.
- 21 degenerate Green Book pages.

## After 0.3

**Code, for 0.4:**

- the gutter-hoist fix (plan ready). The pilot shows it is also what keeps two-column city
  directories off the per-column path.
- an alignment regression fixture from non-panel Tulsa pages, scored by the box-vs-Surya check.
  Land it before the gutter-hoist fix so the fix has something to move.
- an extraction guard: skip any page the blank check calls blank. OCR written before the check
  existed (2026-06-11) can otherwise still reach NER. The 2026-10-17 quarantine removed the 44
  known cases, but nothing stops a new one.
- column-start location on dense pages. `_locate_column_start` failed on 6 panel pages, five of
  them four-to-six-column, and those pages hold most of the remaining placement errors.
- a placement check that needs no gold: on each page, compare the Gemini text bound to a box with
  Surya's own text at that box. Share of bound lines below 0.5 similarity on the panel:

  | pages | share below 0.5 |
  |---|---|
  | clean pages | under 5% |
  | Polk 1917 | 84.5% |
  | Queens 1933 | 61.5% |
  | Polk 1925 | 53.0% |

  That ranking agrees with the gold-based errors. Today's outlier report keys on unboxed rate,
  so it cannot see a page where everything is boxed onto the wrong lines. Low-quality Surya text
  also scores low, so judge each page against its volume's own distribution, as the
  unmatched-rate report does.
- a Surya segmentation check before alignment. The failing panel pages are under-segmented,
  while clean pages have 0–3 tall boxes and about 0.9 or more boxes per line:

  | failing page | tall boxes | usable boxes per Gemini line |
  |---|---|---|
  | Polk 1917 | 16 | 0.75 |
  | Queens 1933 | 26 | 0.82 |
  | Polk 1925 | 26 | 0.49 |

  `city-directory-extraction` found the opposite failure on Lain 1897 p.0032: 317 micro-boxes
  averaging 16 characters. Mean line length alone does not separate them. Both failures are
  upstream of the matcher, like the `--blank-threshold` region-blob case.
- re-OCR of the degenerate pages (plan ready; needs per-page model selection)
- the `long_line` rule treats any line over 500 characters as a runaway, so it also catches
  reflowed prose. 31 existing pages (0.3%) would trigger needless retries.
- the retry ladder's best-candidate choice puts any clean candidate ahead of any flawed one,
  so a short clean retry beats a long flawed first answer. Measure how often that loses text.
- `collate_review_queue.py` lists refusals only. An entry whose line went to a stronger claim
  is not in the queue.
- `fix_entries`' `flag_unanchored` still describes a row with no box and no address as
  "likely hallucinated". Since the binder, no box usually means a deliberate refusal, so the
  flag's wording and use need revisiting.
- the two open nits above (`find_volumes`, the `redownload_page` fallback)
- the city-directory placement audit
- the local OCR backend (rebase it, and decide the extras conflict)
- section detection (plan)
- a PDF source

**Data, in `green-books` (none of these block the release):**

- **The 47 invented rows were investigated on 2026-10-17 and never published.**
  - They come from pages 3, 106 and 112 of the 1963–64 Green Book, all blank. They exist only in
    that volume's text-only experiment CSV and the three pages' cached `*_entries.json`.
  - They are absent from the main and fixed CSVs, `green_book_entries_all.csv`,
    `travel_guides_all.csv`, green-books' published CSVs, its HF export and its annotations.
    green-books' git history has never held them.
  - **The latent risk:** 44 blank pages still carry invented Gemini text, most with alignment
    files rebuilt on Sep 4. 14 of them would reach NER on the next extraction run:
    - 11 never-extracted pages: NHA 4, Smith's 4, two Green Book volumes 3
    - the three 1963–64 pages, whose caches the Sep 4 re-align made stale
  - The 24 Travelers' Guide pages are outside `included_pages.txt`.
  - **Done 2026-10-17.**
    - All 44 pages' OCR, alignment and entries files (101 files) moved to
      `output/_quarantine/hallucinated-blank-pages-2026-10-17/`. Its README and `MOVED.tsv`
      cover undoing it.
    - The 47 rows were dropped from the text-only CSV (3,216 → 3,169; original kept in the
      quarantine).
    - Extraction can now reach none of these pages.
  - Caveat: `_ocr_backups/green_books_aligned_2026-09-04/aligned.tar.gz` holds 50 of the
    alignment files. Restore the pre-re-align tree only with
    `-X _quarantine/hallucinated-blank-pages-2026-10-17/exclude-from-restore.txt`.
- The five recovered 1786 pages still need align and extract.
- The binder re-bind and thumbnail regeneration in
  `../green-books/docs/alignment-audit-2026-09-04.md` are still to do.
