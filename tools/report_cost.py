#!/usr/bin/env python3
"""Report what Gemini API calls actually cost, from recorded token usage.

This reads measurements, not estimates. Every Gemini response carries
usage_metadata; ``utils.gemini.generate_with_retry`` records it and the OCR and
NER stages persist the totals to ``token_usage`` in each volume's
pipeline_state.json. This script prices those totals.

That distinction is the point. A tokens-x-pages estimate cannot see the
retry ladder: a page blocked for RECITATION and retried at three temperatures
is billed four times, each paying full input tokens, and a blocked response
returns prompt tokens with no output at all. Those calls are in the recorded
totals because they were counted where they happened.

Totals are cumulative across runs, so re-OCR'ing a volume adds to its figure —
which is correct, you paid twice.

Usage
-----
    python tools/report_cost.py output/                     # every volume
    python tools/report_cost.py output/my-volume/           # one volume
    python tools/report_cost.py output/ --by-model          # group by model
    python tools/report_cost.py output/my-volume/ --pages 1900
        project this volume's cost-per-call onto a larger corpus

Rates live in utils/models.py (MODEL_RATES, checked RATES_CHECKED). A model with
no entry there is reported as unpriced rather than guessed at.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.state import read_state  # noqa: E402
from utils.collections import eval_collections, summarize_exclusions  # noqa: E402
from utils.models import (  # noqa: E402
    MODEL_RATES,
    RATES_CHECKED,
    TIER_MULTIPLIERS,
    price_usage,
)

_FIELDS = ("calls", "input", "output", "thoughts", "cached")


def find_volumes(root: Path, all_collections: bool = False) -> "tuple[list[Path], str]":
    """Volumes under *root* holding a pipeline_state.json, minus test collections.

    Returns ``(volumes, note)`` where *note* names what was excluded, so a
    dropped collection is never silent. A path pointing straight at one volume
    is always honoured — the registry governs corpus-wide sweeps, not a
    deliberate request for one directory.
    """
    if (root / "pipeline_state.json").exists():
        return [root], ""
    if all_collections:
        return sorted(p.parent for p in root.rglob("pipeline_state.json")), ""
    kept, dropped = eval_collections(root)
    keep_names = {d.name for d in kept}
    vols = [p.parent for p in sorted(root.rglob("pipeline_state.json"))
            if any(part in keep_names for part in p.parts)]
    return vols, summarize_exclusions(dropped)


def usage_for(volume: Path) -> "dict[str, dict[str, int]]":
    return read_state(volume).get("token_usage", {}) or {}


def pages_for(volume: Path) -> int:
    """Pages this volume actually sent to the API, across all runs."""
    return int(read_state(volume).get("token_usage_pages", 0) or 0)


def _split_key(key: str) -> "tuple[str, str, str]":
    """Split "model|tier|reason". Tolerates the 2-part keys written before
    reason tagging existed, so older state files still price correctly."""
    parts = key.split("|")
    model = parts[0]
    tier = parts[1] if len(parts) > 1 and parts[1] else "standard"
    reason = parts[2] if len(parts) > 2 and parts[2] else "initial"
    return model, tier, reason


def _merge(into: dict, key: str, counts: dict) -> None:
    running = into.setdefault(key, dict.fromkeys(_FIELDS, 0))
    for field in _FIELDS:
        running[field] += int(counts.get(field, 0) or 0)


def _fmt_money(value: "float | None") -> str:
    if value is None:
        return "     unpriced"
    return f"${value:>11.4f}" if value < 1 else f"${value:>11.2f}"


def collapse_reasons(rows: dict) -> dict:
    """Merge "model|tier|reason" keys down to "model|tier" for the main table."""
    out: dict = {}
    for key, counts in rows.items():
        model, tier, _ = _split_key(key)
        _merge(out, f"{model}|{tier}", counts)
    return out


def _print_reason_table(rows: dict, indent: str = "") -> None:
    """What each failure class actually cost — the question estimates cannot answer."""
    by_reason: dict = {}
    for key, counts in rows.items():
        model, tier, reason = _split_key(key)
        bucket = by_reason.setdefault(reason, {"counts": dict.fromkeys(_FIELDS, 0), "cost": 0.0})
        for field in _FIELDS:
            bucket["counts"][field] += int(counts.get(field, 0) or 0)
        cost = price_usage(model, counts, tier)
        if cost:
            bucket["cost"] += cost
    if len(by_reason) <= 1:
        return
    total = sum(b["cost"] for b in by_reason.values()) or 1.0
    print(f"{indent}by reason:")
    header = f"{indent}  {'reason':24}{'calls':>7}{'output':>12}{'cost':>14}{'share':>8}"
    print(header)
    print(f"{indent}  {'-' * (len(header) - len(indent) - 2)}")
    for reason in sorted(by_reason, key=lambda r: -by_reason[r]["cost"]):
        b = by_reason[reason]
        print(f"{indent}  {reason:24}{b['counts']['calls']:>7}"
              f"{b['counts']['output']:>12,}{_fmt_money(b['cost']):>14}"
              f"{b['cost'] / total * 100:>7.1f}%")


def _print_table(rows: "dict[str, dict[str, int]]", indent: str = "") -> "float | None":
    """Print one usage table. Returns the priced total, or None if nothing priced."""
    if not rows:
        print(f"{indent}(no recorded usage)")
        return None
    header = f"{indent}{'model':26}{'tier':9}{'calls':>7}{'input':>12}{'output':>12}{'thinking':>10}{'cost':>14}"
    print(header)
    print(f"{indent}{'-' * (len(header) - len(indent))}")
    total = 0.0
    any_priced = False
    for key in sorted(rows):
        model, tier, _reason = _split_key(key)
        c = rows[key]
        cost = price_usage(model, c, tier)
        if cost is not None:
            total += cost
            any_priced = True
        print(
            f"{indent}{model:26}{tier:9}{c['calls']:>7}{c['input']:>12,}"
            f"{c['output']:>12,}{c['thoughts']:>10,}{_fmt_money(cost):>14}"
        )
    return total if any_priced else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Price recorded Gemini token usage from pipeline_state.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("path", type=Path, help="A volume directory, or a root to search")
    ap.add_argument("--by-model", action="store_true",
                    help="Aggregate across volumes instead of listing each one")
    ap.add_argument("--all-collections", action="store_true",
                    help="Include source probes, evidence slices and test collections "
                         "that analysis/collection_registry.txt normally excludes")
    ap.add_argument("--by-reason", action="store_true",
                    help="Break each volume down by what provoked the call "
                         "(initial, temp:recitation, simple:block_loop, escalate:…)")
    ap.add_argument("--pages", type=int, metavar="N",
                    help="Project the observed cost per call onto N pages")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Error: no such path: {args.path}", file=sys.stderr)
        return 1

    volumes, exclusion_note = find_volumes(args.path, args.all_collections)
    if not volumes:
        print(f"No pipeline_state.json found under {args.path}", file=sys.stderr)
        return 1

    print(f"Rates: utils/models.py MODEL_RATES, checked {RATES_CHECKED}. "
          "Thinking tokens bill at the output rate.")
    print("Tiers: " + ", ".join(f"{t} x{m}" for t, m in sorted(TIER_MULTIPLIERS.items())))
    if exclusion_note:
        print(exclusion_note)
    print()

    combined: dict = {}
    grand = 0.0
    priced_any = False
    measured = 0
    pages = 0

    for volume in volumes:
        rows = usage_for(volume)
        for key, counts in rows.items():
            _merge(combined, key, counts)
        if not rows:
            continue
        measured += 1
        vol_pages = pages_for(volume)
        pages += vol_pages
        if not args.by_model:
            print(volume)
            subtotal = _print_table(collapse_reasons(rows), indent="  ")
            if subtotal is not None:
                priced_any = True
                grand += subtotal
                label = "subtotal"
                if vol_pages:
                    label += f" ({vol_pages:,} pages, ${subtotal / vol_pages:.5f}/page)"
                print(f"  {label:44}{_fmt_money(subtotal):>39}")
            if args.by_reason:
                print()
                _print_reason_table(rows, indent="  ")
            print()

    if args.by_model:
        total = _print_table(collapse_reasons(combined))
        if args.by_reason:
            print()
            _print_reason_table(combined)
        if total is not None:
            priced_any = True
            grand = total
        print()

    if measured == 0:
        print("No token usage recorded yet.")
        print("Usage is written by the OCR and NER stages; volumes processed before")
        print("token accounting existed have none. Re-run a stage to start recording.")
        return 0

    calls = sum(c["calls"] for c in combined.values())
    summary = f"{measured} volume(s) with recorded usage, {calls:,} call(s)"
    if pages:
        summary += f" over {pages:,} page(s) — {calls / pages:.2f} calls/page"
    print(summary)
    if not priced_any:
        return 0

    print(f"Total: {_fmt_money(grand).strip()}")
    unpriced = sorted({_split_key(k)[0] for k in combined} - set(MODEL_RATES))
    if unpriced:
        print(f"Excludes unpriced model(s): {', '.join(unpriced)} "
              f"— add them to MODEL_RATES in utils/models.py.")

    if pages:
        print(f"Per page: ${grand / pages:.5f}")
        if calls > pages:
            naive = grand / calls
            print(f"  (a per-call figure would read ${naive:.5f} and understate by "
                  f"{calls / pages:.2f}x — that gap is the retry ladder)")

    if args.pages:
        print()
        if not pages:
            print("Cannot project: no page count recorded for the measured runs.")
            print("Re-run a stage to record one — projecting from calls alone would")
            print("understate by the retry multiplier.")
        else:
            per_page = grand / pages
            print(f"Observed ${per_page:.5f} per page over {pages:,} page(s).")
            print(f"Projected onto {args.pages:,} pages: ${per_page * args.pages:,.2f}")
            print("The projection carries this run's retry rate with it — a volume that")
            print("blocks or loops more often will cost more per page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
