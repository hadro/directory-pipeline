#!/usr/bin/env python3
"""Audit where a published CSV's canvas_fragment actually points, and A/B the binder.

The question this answers is the one coverage metrics cannot: not "how many lines
got a box" but "is this entry's box on this entry's line". It needs no ground
truth — every aligned line carries both its bbox and the text read inside it, so
an entry whose name and address share no words with the text at its own box is
wrong, and can be found without anyone having labelled anything.

Two modes, usually run together:

  audit  For each CSV row, find the aligned line whose canvas_fragment matches
         and compare the entry's identity text against that line's text.

           no_box      the fragment carries no #xywh — whole-canvas fallback
           orphan_box  the fragment appears in no aligned JSON on disk
           text_zero   box text shares NO words with the entry — reliably wrong
           text_weak   under a third shared — mixed; corpus-wide this is mostly
                       benign (an entry wrapped across two lines, bound correctly
                       to its name line), so do not read it as an error count

  a-b    Re-bind each page with the current utils.entry_binding and report what
         would change: repairs on rows the audit flagged, and regressions on
         rows it did not. This is the before/after checkpoint for any change to
         alignment or binding.

Choosing which alignment to audit against
-----------------------------------------
A page often carries alignments from several models and nothing records which
one produced a given canvas_fragment. Two rules, and the right one depends on
what you are asking:

  --resolve overlap   (default) pick the alignment containing the most of this
                      page's shipped fragments. Identifies the alignment that
                      actually produced the data, which is what an audit of
                      shipped data wants — but it stops working once a re-align
                      changes the fragments.
  --resolve coverage  pick the best alignment, ranked as lines * (lines /
                      (lines + unmatched)), matching _detect_aligned_slug. Use
                      this after a re-align, when the fragments have moved.

An earlier version of this tool took sorted()[0] and got 1,156 phantom
"orphans" — the same alphabetical-first trap that _detect_aligned_slug had.
Both bugs existed because nothing stamps the model into the entries JSON.

Usage:
    python tools/audit_entry_bindings.py output/green_books_and_related \
        --csv ~/github/green-books/green_book_entries_all.csv \
        --csv ~/github/green-books/travel_guides_all.csv \
        --report audit.csv --a-b
"""

import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.entry_binding import (  # noqa: E402
    bind_entries, identity_tokens, score, tokens,
)

csv.field_size_limit(10 ** 9)

ALIGNED_RE = re.compile(r"_(gemini[-\w.]+)_aligned\.json$")
ENTRIES_RE = re.compile(r"_(gemini[-\w.]+)_entries\.json$")
XYWH = re.compile(r"#xywh=\d+,\d+,\d+,\d+")
WEAK_THRESHOLD = 0.34


def churn(aligned_idx: dict, window_s: int = 300) -> int:
    """How many aligned JSONs were written within the last *window_s* seconds.

    Auditing a tree while ``align_ocr --force`` is running over it produces
    numbers that mean nothing: pages already re-aligned carry new bboxes, so the
    CSV's fragments no longer match and every one of them is reported as an
    orphan. Doing exactly that mid-run turned 134 real orphans into 9,588
    phantom ones — which looks like a catastrophic data finding rather than a
    tool pointed at a moving target, so this is worth refusing by default.
    """
    import time
    now = time.time()
    recent = 0
    for paths in aligned_idx.values():
        for p in paths:
            try:
                if now - p.stat().st_mtime < window_s:
                    recent += 1
            except OSError:
                continue
    return recent


def index_tree(root: Path) -> tuple[dict, dict]:
    aligned = collections.defaultdict(list)
    entries = {}
    for p in root.rglob("*_aligned.json"):
        aligned[ALIGNED_RE.sub("", p.name) + ".jpg"].append(p)
    for p in root.rglob("*_entries.json"):
        entries.setdefault(ENTRIES_RE.sub("", p.name) + ".jpg", p)
    return aligned, entries


def _coverage(doc: dict) -> float:
    """lines * (lines / (lines + unmatched)) — see _detect_aligned_slug."""
    n = len(doc.get("lines", []))
    if not n:
        return 0.0
    return n * (n / (n + len(doc.get("unmatched_gemini", []))))


def pick_aligned(paths: list[Path], shipped: set, how: str) -> dict | None:
    best, best_key = None, -1.0
    for p in paths:
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if how == "overlap":
            key = sum(1 for l in doc.get("lines", [])
                      if l.get("canvas_fragment") in shipped)
        else:
            key = _coverage(doc)
        if key > best_key:
            best, best_key = doc, key
    return best


def load_rows(csv_paths: list[Path]) -> dict:
    """image filename → [(row_index, row), …] in CSV (reading) order."""
    pages = collections.OrderedDict()
    for path in csv_paths:
        with path.open(encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                row["_csv"] = path.name
                pages.setdefault(row.get("image", ""), []).append((i, row))
    return pages


def audit(pages: dict, aligned_idx: dict, how: str) -> tuple[list[dict], collections.Counter]:
    flagged, stats = [], collections.Counter()
    for image, rows in pages.items():
        shipped = {r["canvas_fragment"] for _, r in rows}
        doc = pick_aligned(aligned_idx.get(image, []), shipped, how)
        by_frag = {l.get("canvas_fragment"): l for l in doc.get("lines", [])} if doc else {}
        for i, row in rows:
            stats["_total"] += 1
            cf = row.get("canvas_fragment", "")
            flag, line_text, sim = None, "", None
            if not XYWH.search(cf):
                flag = "no_box"
            elif doc is None:
                flag = "no_aligned_json"
            elif cf not in by_frag:
                flag = "orphan_box"
            else:
                line_text = by_frag[cf].get("gemini_text", "")
                ent = set(identity_tokens(row))
                if ent:
                    sim = len(ent & set(tokens(line_text))) / len(ent)
                    if sim == 0:
                        flag = "text_zero"
                    elif sim < WEAK_THRESHOLD:
                        flag = "text_weak"
            if flag:
                stats[flag] += 1
                flagged.append({
                    "csv": row["_csv"], "row": i, "image": image,
                    "volume_year": row.get("volume_year", ""),
                    "name": row.get("name", ""), "address": row.get("address", ""),
                    "category": row.get("category", ""),
                    "canvas_fragment": cf, "line_text": line_text,
                    "sim": "" if sim is None else f"{sim:.2f}", "flags": flag,
                })
    return flagged, stats


def a_b(pages: dict, aligned_idx: dict, entries_idx: dict, flagged: list[dict],
        how: str, sample: int, seed: int) -> tuple[collections.Counter, collections.Counter]:
    """Repairs on flagged pages, regressions on a sample of clean ones."""
    bad = collections.defaultdict(set)
    for f in flagged:
        if f["flags"] in ("text_zero", "no_box"):
            bad[f["image"]].add(f["canvas_fragment"])
    flagged_images = {f["image"] for f in flagged}

    def page(image):
        ep = entries_idx.get(image)
        if not ep:
            return None, None
        try:
            entries = json.loads(ep.read_text(encoding="utf-8")).get("entries", [])
        except (OSError, ValueError):
            return None, None
        shipped = {e.get("canvas_fragment") for e in entries}
        return entries, pick_aligned(aligned_idx.get(image, []), shipped, how)

    repairs = collections.Counter()
    for image in bad:
        entries, doc = page(image)
        if not entries or not doc:
            continue
        lines = doc["lines"]
        for e, b in zip(entries, bind_entries(entries, lines,
                                              canvas_width=doc.get("canvas_width") or 0)):
            if e.get("canvas_fragment") not in bad[image]:
                continue
            if b.line_index is None:
                repairs["still_unbound"] += 1
            elif lines[b.line_index].get("canvas_fragment") == e.get("canvas_fragment"):
                repairs["unchanged"] += 1
            else:
                repairs["repaired"] += 1

    clean = [i for i in entries_idx if i not in flagged_images]
    random.seed(seed)
    regress = collections.Counter()
    for image in random.sample(clean, min(sample, len(clean))):
        entries, doc = page(image)
        if not entries or not doc:
            continue
        lines = doc["lines"]
        by_frag = {l.get("canvas_fragment"): l for l in lines}
        for e, b in zip(entries, bind_entries(entries, lines,
                                              canvas_width=doc.get("canvas_width") or 0)):
            old = e.get("canvas_fragment", "")
            new = lines[b.line_index]["canvas_fragment"] if b.line_index is not None else None
            if new == old:
                regress["same"] += 1
                continue
            if old not in by_frag:
                regress["old_unresolvable"] += 1
                continue
            so = score(identity_tokens(e), tokens(by_frag[old].get("gemini_text", "")))
            sn = b.score if b.line_index is not None else 0.0
            regress["better" if sn > so else "worse" if sn < so else "tie"] += 1
    return repairs, regress


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path, help="Pipeline output tree to audit against")
    ap.add_argument("--csv", type=Path, action="append", required=True,
                    help="Published CSV to audit (repeatable)")
    ap.add_argument("--report", type=Path, help="Write flagged rows here")
    ap.add_argument("--resolve", choices=("overlap", "coverage"), default="overlap",
                    help="How to pick a page's alignment (default: %(default)s)")
    ap.add_argument("--a-b", action="store_true",
                    help="Also report what the current binder would change")
    ap.add_argument("--sample", type=int, default=400,
                    help="Clean pages to sample for regressions (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--allow-unstable", action="store_true",
                    help="Audit even if aligned JSONs are being written right now")
    args = ap.parse_args()

    if not args.output_dir.is_dir():
        print(f"error: {args.output_dir} is not a directory", file=sys.stderr)
        return 2
    missing = [c for c in args.csv if not c.is_file()]
    if missing:
        print(f"error: no such CSV: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2

    print(f"indexing {args.output_dir} …", file=sys.stderr)
    aligned_idx, entries_idx = index_tree(args.output_dir)
    pages = load_rows(args.csv)
    print(f"{len(aligned_idx)} aligned pages, {len(pages)} pages in CSV", file=sys.stderr)

    recent = churn(aligned_idx)
    if recent and not args.allow_unstable:
        print(f"\nrefusing to audit: {recent} aligned JSON(s) were written in the last "
              f"5 minutes.\nAn alignment run appears to be in progress, and auditing a "
              f"half-rewritten tree\nreports every already-re-aligned page as an orphan. "
              f"Wait for it to finish,\nor pass --allow-unstable if you know the churn is "
              f"unrelated.", file=sys.stderr)
        return 3

    flagged, stats = audit(pages, aligned_idx, args.resolve)
    total = stats.pop("_total")
    hard = stats["no_box"] + stats["text_zero"]
    print(f"\naudit ({args.resolve} resolution): {total} rows")
    for k in ("text_zero", "no_box", "orphan_box", "no_aligned_json", "text_weak"):
        if stats[k]:
            print(f"  {stats[k]:>6}  {k}")
    print(f"  → {hard} hard errors ({hard / max(total, 1):.2%}); "
          f"text_weak is mostly benign, do not add it in")

    if args.report and flagged:
        with args.report.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flagged[0].keys()))
            w.writeheader()
            w.writerows(flagged)
        print(f"  wrote {args.report} ({len(flagged)} rows)")

    if args.a_b:
        repairs, regress = a_b(pages, aligned_idx, entries_idx, flagged,
                               args.resolve, args.sample, args.seed)
        comparable = sum(v for k, v in regress.items() if k != "old_unresolvable")
        print("\nA/B against the current binder")
        print(f"  broken bindings: {json.dumps(dict(repairs))}")
        print(f"  clean pages    : {json.dumps(dict(regress))}")
        print(f"  regression rate: {regress['worse'] / max(comparable, 1):.2%}"
              f" ({regress['worse']}/{comparable})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
