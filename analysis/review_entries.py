#!/usr/bin/env python3
"""Send an extracted entries CSV to Gemini and get back a data-quality report.

Usage:
    python review_entries.py <csv_path> [options]

    python review_entries.py output/woods_directory/entries_gemini-2.0-flash.csv
    python review_entries.py output/woods_directory/entries_gemini-2.0-flash.csv \\
        --ner-prompt output/woods_directory/ner_prompt.md \\
        --model gemini-2.5-pro-preview-03-25 \\
        --out report.md
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3-flash-preview"

SYSTEM_PROMPT = """\
You are a data quality auditor reviewing a CSV of structured records extracted
from a digitized historical document via an automated OCR + named-entity
recognition pipeline. Your job is to identify systematic problems, anomalies,
and patterns that suggest extraction errors — not to flag every minor quirk.

Focus on issues that affect the usefulness or accuracy of the data as a whole:
- Fields whose values look like they belong in a different field (e.g. an
  address in the name column, a phone number in the address column).
- Systematic OCR artifacts that recur across many rows (garbled character
  sequences, encoding problems, repeated misreads of the same abbreviation).
- Fields that are unexpectedly empty across a suspiciously large fraction of
  rows, given what the schema says they should contain.
- Duplicate or near-duplicate entries that may represent double-extraction of
  the same source record.
- Implausible or inconsistent values (dates in the wrong century, negative
  numbers, impossible combinations of field values).
- Evidence of column-merging errors (two fields concatenated into one, or one
  field split across two columns).
- Any other pattern that suggests a systematic pipeline failure rather than
  one-off noise.

Group your findings by issue type. For each issue:
1. Name and describe the issue.
2. Give 2–5 concrete example rows (use the row's name or a key identifier plus
   the offending field value).
3. Estimate roughly how many rows are affected.
4. Suggest what likely caused it and how it might be fixed.

At the end, give a brief overall quality assessment (1–2 sentences).

Be concise. Skip issues that affect only 1–2 rows and are clearly one-off OCR
noise with no systemic cause. Output plain Markdown."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_ner_prompt(csv_path: Path) -> Path | None:
    """Look for ner_prompt.md alongside the CSV or one level up."""
    for d in (csv_path.parent, csv_path.parent.parent):
        p = d / "ner_prompt.md"
        if p.exists():
            return p
    return None


def _call_gemini(client: genai.Client, model: str, user_text: str) -> str:
    max_retries = 4
    delay = 15
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                config=GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
                contents=user_text,
            )
            return response.text or ""
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"  API error (attempt {attempt + 1}): {exc} — retrying in {delay}s…",
                      file=sys.stderr)
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send an extracted entries CSV to Gemini for data-quality review."
    )
    parser.add_argument("csv", help="Path to the entries CSV file")
    parser.add_argument(
        "--ner-prompt", "-n",
        metavar="PATH",
        help="Path to the NER prompt used to produce this CSV (adds schema context). "
             "Auto-discovered from the CSV directory if omitted.",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out", "-o",
        metavar="PATH",
        help="Write report to this file (default: ner_review.md next to the CSV)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Truncate to this many rows before sending (default: 5000). "
             "Use 0 for no limit.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_text = csv_path.read_text(encoding="utf-8")
    lines = csv_text.splitlines()
    total_rows = max(0, len(lines) - 1)  # subtract header

    if args.max_rows and total_rows > args.max_rows:
        print(f"  CSV has {total_rows} rows — truncating to {args.max_rows} for review.",
              file=sys.stderr)
        csv_text = "\n".join(lines[: args.max_rows + 1])
    else:
        print(f"  CSV: {total_rows} rows", file=sys.stderr)

    # ── NER prompt ────────────────────────────────────────────────────────────
    ner_prompt_path = Path(args.ner_prompt) if args.ner_prompt else _find_ner_prompt(csv_path)
    ner_section = ""
    if ner_prompt_path and ner_prompt_path.exists():
        ner_text = ner_prompt_path.read_text(encoding="utf-8").strip()
        ner_section = (
            f"\n\n## NER extraction prompt (schema reference)\n\n"
            f"The following prompt was used to extract records from the source document. "
            f"Use it as the authoritative definition of what each field should contain.\n\n"
            f"---\n{ner_text}\n---"
        )
        print(f"  NER prompt: {ner_prompt_path}", file=sys.stderr)
    else:
        print("  NER prompt: not found (running without schema context)", file=sys.stderr)

    # ── Build user message ────────────────────────────────────────────────────
    user_text = (
        f"## Entries CSV\n\nFile: `{csv_path.name}`  \n"
        f"Total rows: {total_rows}"
        f"{ner_section}"
        f"\n\n## Data\n\n```csv\n{csv_text}\n```"
    )

    # ── Call Gemini ───────────────────────────────────────────────────────────
    client = genai.Client(api_key=api_key)
    print(f"  Sending to {args.model}…", file=sys.stderr)
    report = _call_gemini(client, args.model, user_text)

    # ── Output ────────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else csv_path.parent / "ner_review.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"  Report written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
