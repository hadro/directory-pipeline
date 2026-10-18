#!/usr/bin/env python3
"""Measure which rungs of the OCR retry ladder actually rescue a page.

tools/report_cost.py says what each rung *cost*. This says whether it *worked* —
the other half of deciding whether a rung earns its place. Both are needed: a
rung that is cheap and useless should still go, and a rung that is expensive and
reliable should stay.

It reads run logs rather than needing new instrumentation, so it works on the
archived logs in docs/evidence/ as well as on any new run:

    python tools/analyze_retry_ladder.py docs/evidence/2026-09-02-ocr-token-burn/logs
    python -m pipeline.run_gemini_ocr output/vol 2> run.log && \\
        python tools/analyze_retry_ladder.py run.log

The ladder (pipeline/run_gemini_ocr.py) is: temperature rungs 0.1/0.3/0.7, then
two simple-prompt rungs, then one model escalation. Loop kinds already skip the
temperature rungs, on evidence that none was ever rescued there.

How a rescue is inferred
------------------------
Each rung runs only ``if kind`` — that is, only because the rung before it left
the page still flawed. So reaching a rung is itself proof the previous one
failed, and that is what this counts:

  * temperature rescued  — entered the temperature rungs and never reached the
    simple-prompt rung. More reliable than the absence of a "still problematic"
    line, because a retry returning empty text logs neither.
  * simple rescued       — logged "Simple prompt succeeded".
  * escalation rescued   — logged "Model escalation succeeded".

Only pages whose trace reaches a terminal line ("Done:", "FAILED:", or "All
retries exhausted") are counted. This matters: several of the archived harvest
logs stop mid-ladder because the job was interrupted, and without the check a
page whose log simply ends looks identical to one the temperature rung rescued.
An earlier version of this script did exactly that and reported repetition loops
as 75% rescued by temperature, contradicting the evidence set it was reading.

One quirk of logs written before the ``best()`` rewrite: they print "All retries
exhausted — keeping best available output" even on pages the simple prompt had
just rescued. A page with a recorded rescue is therefore not counted unresolved.

Caveat on kinds: the log carries ``_output_issue``'s human-readable message, not
its ``kind``, so this parser has to match on message text — exactly what
CLAUDE.md tells the *pipeline* never to do. The mapping lives in _KIND_PATTERNS
below and unrecognised messages are reported as "unknown" rather than quietly
bucketed. A page's kind can also change between rungs (one evidence page enters
as "line too long" and leaves as "spaced leader runaway"); the kind that
provoked the *first* retry is the one recorded, since that is what a decision
about the ladder would branch on.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Aliased: this module defines its own classify() for log messages, which used
# to shadow this import and quietly turn the collection filter into a no-op.
from utils.collections import classify as classify_collection  # noqa: E402

# Message prefix -> _output_issue kind. Order matters: the first match wins.
_KIND_PATTERNS = [
    ("RECITATION", "recitation"),
    ("blocked (", "blocked"),
    ("repetition loop", "repetition"),
    ("template loop", "template_loop"),
    ("block loop", "block_loop"),
    ("spaced leader", "spaced_leader"),
    ("line too long", "long_line"),
    ("dot-leader", "leader"),
    ("leader runaway", "leader"),
]

_LOOP_KINDS = {"repetition", "template_loop", "block_loop"}
_LEADER_KINDS = {"leader", "spaced_leader", "long_line"}

_PAGE = r"(\S+\.jpg)"
_RX = {
    "temp_enter": re.compile(rf"^\s*(?P<msg>.+?) — retrying at temperature=\S+: {_PAGE}$"),
    "temp_skip": re.compile(rf"^\s*(?P<msg>.+?) — skipping temperature retries.*: {_PAGE}$"),
    "simple_enter": re.compile(rf"^\s*Trying simple fallback prompt at temperature=\S+: {_PAGE}$"),
    "simple_ok": re.compile(rf"^\s*Simple prompt succeeded at temperature=\S+: {_PAGE}$"),
    "esc_enter": re.compile(rf"^\s*Escalating to \S+: {_PAGE}$"),
    "esc_ok": re.compile(rf"^\s*Model escalation succeeded \(.*\): {_PAGE}$"),
    "unresolved": re.compile(rf"^\s*All retries exhausted — keeping best.*: {_PAGE}$"),
    "lost": re.compile(rf"^\s*All retries exhausted — no output recovered: {_PAGE}$"),
}

# Terminal lines name the output .txt, not the source .jpg; strip the model slug
# to recover the page stem. Reaching one of these is what makes a trace complete.
_DONE = re.compile(r"^\s*\[\d+/\d+\]\s+(?:Done|FAILED):\s+(\S+?)(?:_gemini-\S+)?\.(?:txt|jpg)$")


def classify(message: str) -> str:
    low = message.lower()
    for needle, kind in _KIND_PATTERNS:
        if needle.lower() in low:
            return kind
    return "unknown"


def parse(paths: "list[Path]") -> "dict[str, dict]":
    """Return {page: {kind, temp, simple, simple_ok, esc, esc_ok, unresolved, lost}}."""
    pages: "dict[str, dict]" = defaultdict(
        lambda: {"kind": None, "temp": False, "skipped_temp": False, "simple": False,
                 "simple_ok": False, "esc": False, "esc_ok": False,
                 "unresolved": False, "lost": False, "complete": False}
    )
    done_stems: "set[str]" = set()
    for path in paths:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            d = _DONE.match(raw.rstrip())
            if d:
                done_stems.add(d.group(1))
                continue
            for event, rx in _RX.items():
                m = rx.match(raw)
                if not m:
                    continue
                page = m.group(m.lastindex)
                rec = pages[page]
                if event in ("temp_enter", "temp_skip"):
                    if rec["kind"] is None:
                        rec["kind"] = classify(m.group("msg"))
                    rec["temp" if event == "temp_enter" else "skipped_temp"] = True
                elif event == "simple_enter":
                    rec["simple"] = True
                elif event == "simple_ok":
                    rec["simple_ok"] = True
                elif event == "esc_enter":
                    rec["esc"] = True
                elif event == "esc_ok":
                    rec["esc_ok"] = True
                elif event == "unresolved":
                    rec["unresolved"] = True
                elif event == "lost":
                    rec["lost"] = True
                break

    for page, rec in pages.items():
        stem = page[:-4] if page.endswith(".jpg") else page
        rec["complete"] = bool(
            stem in done_stems or rec["unresolved"] or rec["lost"]
            or rec["simple_ok"] or rec["esc_ok"]
        )
        # Pre-best()-rewrite logs print "exhausted" even after a rescue.
        if rec["simple_ok"] or rec["esc_ok"]:
            rec["unresolved"] = False
    return dict(pages)


def _rate(rescued: int, entered: int) -> str:
    if not entered:
        return "      —"
    return f"{rescued:>3}/{entered:<3} {rescued / entered * 100:>3.0f}%"


def report(pages: dict) -> None:
    partial = [p for p, r in pages.items() if not r["complete"]]
    complete = {p: r for p, r in pages.items() if r["complete"]}

    by_kind: "dict[str, list]" = defaultdict(list)
    for rec in complete.values():
        by_kind[rec["kind"] or "unknown"].append(rec)

    print(f"{len(pages)} page(s) entered the retry ladder; "
          f"{len(complete)} traced to an outcome\n")
    if partial:
        print(f"Excluded {len(partial)} page(s) whose log ends mid-ladder — an interrupted")
        print("run is not a rescue, and counting it as one inverts the result.\n")
    if not complete:
        print("Nothing to report: no page in these logs reached a terminal line.")
        return
    header = f"{'kind':16}{'pages':>6}  {'temperature':>14}  {'simple prompt':>14}  {'escalation':>14}  {'unresolved':>11}"
    print(header)
    print("-" * len(header))

    for kind in sorted(by_kind, key=lambda k: -len(by_kind[k])):
        recs = by_kind[kind]
        t_in = sum(r["temp"] for r in recs)
        # Reaching the simple rung proves the temperature rungs did not rescue.
        t_ok = sum(r["temp"] and not r["simple"] and not r["unresolved"] and not r["lost"]
                   for r in recs)
        s_in = sum(r["simple"] for r in recs)
        s_ok = sum(r["simple_ok"] for r in recs)
        e_in = sum(r["esc"] for r in recs)
        e_ok = sum(r["esc_ok"] for r in recs)
        bad = sum(r["unresolved"] or r["lost"] for r in recs)
        print(f"{kind:16}{len(recs):>6}  {_rate(t_ok, t_in):>14}  {_rate(s_ok, s_in):>14}  "
              f"{_rate(e_ok, e_in):>14}  {bad:>11}")

    skipped = sum(r["skipped_temp"] for r in complete.values())
    if skipped:
        print(f"\n{skipped} page(s) skipped the temperature rungs (loop kinds already bypass them).")

    unknown = len(by_kind.get("unknown", []))
    if unknown:
        print(f"{unknown} page(s) had a message this parser does not recognise — "
              f"extend _KIND_PATTERNS.")

    print("\nRead the rates as rescued/entered. A rung is only entered because the one")
    print("before it left the page flawed, so 'entered' is already a failure count for")
    print("the previous rung.")

    # The two open questions, answered directly where the data allows.
    leader = [r for k, rs in by_kind.items() if k in _LEADER_KINDS for r in rs]
    loop = [r for k, rs in by_kind.items() if k in _LOOP_KINDS for r in rs]
    print()
    for label, recs, rung in (
        ("leader kinds through the temperature rungs", leader, "temp"),
        ("loop kinds through escalation", loop, "esc"),
    ):
        entered = sum(r[rung] for r in recs)
        if rung == "temp":
            ok = sum(r["temp"] and not r["simple"] and not r["unresolved"] and not r["lost"]
                     for r in recs)
        else:
            ok = sum(r["esc_ok"] for r in recs)
        if entered:
            print(f"{label}: {ok}/{entered} rescued ({ok / entered * 100:.0f}%)")
        else:
            print(f"{label}: no data ({len(recs)} page(s) of this class seen)")


def logs_under(root: Path, all_collections: bool = False) -> "list[Path]":
    """Log files under *root*, minus those inside excluded collections.

    A log inside a test collection describes a test, not the corpus.
    """
    found = sorted(root.rglob("*.txt")) + sorted(root.rglob("*.log"))
    if all_collections:
        return found
    return [f for f in found
            if all(classify_collection(part)[0] for part in f.relative_to(root).parts[:-1])]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure which retry-ladder rungs actually rescue a page.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("paths", type=Path, nargs="+",
                    help="Log files, or directories of them")
    ap.add_argument("--all-collections", action="store_true",
                    help="Include logs from collections that "
                         "analysis/collection_registry.txt normally excludes")
    args = ap.parse_args()

    files: "list[Path]" = []
    for p in args.paths:
        if p.is_dir():
            files.extend(logs_under(p, args.all_collections))
        elif p.exists():
            files.append(p)
        else:
            print(f"Error: no such path: {p}", file=sys.stderr)
            return 1
    if not files:
        print("No log files found.", file=sys.stderr)
        return 1

    pages = parse(files)
    if not pages:
        print(f"Read {len(files)} file(s); no retry-ladder activity found.")
        return 0
    print(f"Read {len(files)} log file(s).\n")
    report(pages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
