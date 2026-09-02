# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Compare the CURRENT repetition detector in pipeline/run_gemini_ocr.py against a proposed
digit-normalised variant, over real OCR output.

Why: the current check counts *identical* lines. A degenerating model often repeats a
TEMPLATE with a varying number — "Edgey, 148 W 24th" / "- 122 W 23d st" / "John, 148 W 128th".
Every line differs, so the check passes and the degenerate page is written to disk, where it
poisons anything downstream that counts names. Normalising digit runs to '#' before counting
collapses the template to one line and catches it.

Usage
-----
    # the four pages in this evidence folder
    python3 check_repetition.py pages/*_ocr.txt

    # sweep a whole corpus to check the false-positive rate before changing the threshold
    python3 check_repetition.py ../../../output/harvest_*/*_gemini*.txt
"""
from __future__ import annotations

import argparse
import collections
import re
import sys

# Constants copied verbatim from pipeline/run_gemini_ocr.py (lines 126-128).
REPETITION_THRESHOLD = 0.40
REPETITION_MIN_REPEATS = 20
REPETITION_MIN_LINE_LEN = 4

DIGIT_RUN = re.compile(r"\d+")


def repetition_ratio(text: str, normalise_digits: bool) -> "tuple[float, int, str]":
    """Return (top-line share, line count, the dominant line)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < REPETITION_MIN_REPEATS:
        return 0.0, len(lines), ""
    keys = [DIGIT_RUN.sub("#", ln) if normalise_digits else ln
            for ln in lines if len(ln) >= REPETITION_MIN_LINE_LEN]
    if not keys:
        return 0.0, len(lines), ""
    top, n = collections.Counter(keys).most_common(1)[0]
    return n / len(lines), len(lines), top


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--threshold", type=float, default=REPETITION_THRESHOLD)
    args = ap.parse_args(argv)

    print(f"{'file':44} {'lines':>6} {'current':>8} {'proposed':>9}  verdict")
    caught = missed = 0
    for path in args.files:
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError as e:
            print(f"  {path}: {e}", file=sys.stderr)
            continue
        cur, n, _ = repetition_ratio(text, False)
        new, _, top = repetition_ratio(text, True)
        flag_cur = cur >= args.threshold
        flag_new = new >= args.threshold
        if flag_new and not flag_cur:
            verdict, caught = "CAUGHT ONLY BY PROPOSED", caught + 1
        elif flag_new and flag_cur:
            verdict = "both flag"
        elif flag_cur:
            verdict = "current only (unexpected)"
        else:
            verdict, missed = "clean", missed + 1
        name = path.split("/")[-1]
        print(f"{name[:44]:44} {n:6} {cur:7.0%} {new:8.0%}  {verdict}")
        if flag_new:
            print(f"{'':44} {'':6} {'':7} {'':8}  dominant: {top[:60]!r}")
    print()
    print(f"flagged only by the proposed check: {caught}   not flagged by either: {missed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
