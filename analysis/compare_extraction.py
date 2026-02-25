#!/usr/bin/env python3
"""Compare text-only vs multimodal NER extraction on a sample of pages.

Runs both extraction modes on the same pages and produces a side-by-side
comparison to evaluate whether including the page image meaningfully improves
structured entry extraction quality.

Metrics compared per page:
  - Entry count
  - Field null rates (raw_address, phone, notes)
  - Address type distribution (standard / intersection / building / unknown)
  - Advertisement rate

Output:
  extraction_comparison_{slug}.csv   — per-page numeric metrics
  extraction_comparison_{slug}.html  — side-by-side HTML report

Usage
-----
    python compare_extraction.py images/greenbooks/feb978b0 --model gemini-2.0-flash
    python compare_extraction.py images/greenbooks/feb978b0 --model gemini-2.0-flash --pages 5
    python compare_extraction.py images/greenbooks/feb978b0 --dry-run
"""

import argparse
import csv
import json
import os
import sys
import time
import threading
from pathlib import Path

from google import genai
from google.genai.types import GenerateContentConfig, Part

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_PAGES = 10
NER_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "ner_prompt.md"

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def model_slug(model: str) -> str:
    return model.replace("/", "_")


# ---------------------------------------------------------------------------
# Gemini API (mirrors extract_entries.py)
# ---------------------------------------------------------------------------

def _call_gemini(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_text: str,
    image_path: Path | None = None,
) -> str:
    parts: list = []
    if image_path is not None:
        parts.append(Part.from_bytes(data=image_path.read_bytes(), mime_type="image/jpeg"))
    parts.append(Part.from_text(text=user_text))

    max_retries = 5
    delay = 10
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                ),
                contents=parts,
            )
            return response.text or ""
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                _log(f"  Rate limited — retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    return ""


def _parse_json_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if 0 <= start < end:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Extraction helpers (mirrors extract_entries.py)
# ---------------------------------------------------------------------------

def _page_text_from_aligned(aligned: dict) -> str:
    lines = [
        ln.get("gemini_text", "").strip()
        for ln in aligned.get("lines", [])
        if ln.get("gemini_text", "").strip()
    ]
    lines += [t.strip() for t in aligned.get("unmatched_gemini", []) if t.strip()]
    return "\n".join(lines)


def _build_user_message(page_text: str) -> str:
    """Build user message with empty prior context (each page isolated for comparison)."""
    return (
        "## Prior page context\n"
        "State: unknown\n"
        "City: unknown\n"
        "Category: unknown\n\n"
        "## Page text\n"
        f"{page_text}\n\n"
        "Return the JSON extraction."
    )


def _img_path_from_aligned(aligned_path: Path, slug: str) -> Path | None:
    suffix = f"_{slug}_aligned.json"
    name = aligned_path.name
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    candidate = aligned_path.parent / f"{stem}.jpg"
    return candidate if candidate.exists() else None


def _extract(
    client: genai.Client,
    model: str,
    system_prompt: str,
    aligned_path: Path,
    slug: str,
    mode: str,
) -> dict:
    """
    Run extraction in one mode on one page.

    Returns:
      {"entries": [...], "raw": "<raw response>", "status": "ok"|"parse_error"|"api_error:<msg>"}
    """
    try:
        aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"entries": [], "raw": "", "status": f"load_error:{exc}"}

    page_text = _page_text_from_aligned(aligned)
    if not page_text.strip():
        return {"entries": [], "raw": "", "status": "empty"}

    user_msg = _build_user_message(page_text)

    image_path: Path | None = None
    if mode == "multimodal":
        image_path = _img_path_from_aligned(aligned_path, slug)
        if image_path is None:
            _log(f"    Warning: image not found for {aligned_path.name}; mode=text-only fallback")

    try:
        raw = _call_gemini(client, model, system_prompt, user_msg, image_path)
    except Exception as exc:
        return {"entries": [], "raw": "", "status": f"api_error:{exc}"}

    result = _parse_json_response(raw)
    if result is None:
        return {"entries": [], "raw": raw, "status": "parse_error"}

    return {"entries": result.get("entries", []), "raw": raw, "status": "ok"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(entries: list[dict]) -> dict:
    n = len(entries)
    if n == 0:
        return {
            "count": 0,
            "null_address": 0, "null_phone": 0, "null_notes": 0,
            "addr_standard": 0, "addr_intersection": 0,
            "addr_building": 0, "addr_unknown": 0,
            "is_ad": 0,
        }

    null_address = sum(1 for e in entries if not e.get("raw_address"))
    null_phone   = sum(1 for e in entries if not e.get("phone"))
    null_notes   = sum(1 for e in entries if not e.get("notes"))
    addr_types   = [e.get("address_type", "unknown") for e in entries]
    is_ad        = sum(1 for e in entries if e.get("is_advertisement"))

    return {
        "count": n,
        "null_address": null_address,
        "null_phone": null_phone,
        "null_notes": null_notes,
        "addr_standard": addr_types.count("standard"),
        "addr_intersection": addr_types.count("intersection"),
        "addr_building": addr_types.count("building"),
        "addr_unknown": addr_types.count("unknown"),
        "is_ad": is_ad,
    }


def _delta_str(a: int | float, b: int | float) -> str:
    """Format a difference as +N / -N / 0."""
    d = b - a
    if d > 0:
        return f"+{d}"
    if d < 0:
        return str(d)
    return "0"


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "page", "filename",
    "text_count", "mm_count", "delta_count",
    "text_null_addr", "mm_null_addr",
    "text_null_phone", "mm_null_phone",
    "text_addr_standard", "mm_addr_standard",
    "text_addr_intersection", "mm_addr_intersection",
    "text_addr_unknown", "mm_addr_unknown",
    "text_is_ad", "mm_is_ad",
    "text_status", "mm_status",
]


def _metrics_to_csv_row(
    i: int,
    filename: str,
    text_m: dict,
    mm_m: dict,
    text_status: str,
    mm_status: str,
) -> dict:
    return {
        "page": i,
        "filename": filename,
        "text_count": text_m["count"],
        "mm_count": mm_m["count"],
        "delta_count": mm_m["count"] - text_m["count"],
        "text_null_addr": text_m["null_address"],
        "mm_null_addr": mm_m["null_address"],
        "text_null_phone": text_m["null_phone"],
        "mm_null_phone": mm_m["null_phone"],
        "text_addr_standard": text_m["addr_standard"],
        "mm_addr_standard": mm_m["addr_standard"],
        "text_addr_intersection": text_m["addr_intersection"],
        "mm_addr_intersection": mm_m["addr_intersection"],
        "text_addr_unknown": text_m["addr_unknown"],
        "mm_addr_unknown": mm_m["addr_unknown"],
        "text_is_ad": text_m["is_ad"],
        "mm_is_ad": mm_m["is_ad"],
        "text_status": text_status,
        "mm_status": mm_status,
    }


def write_comparison_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_STYLE = """
body { font-family: system-ui, sans-serif; font-size: 14px; margin: 20px; color: #222; }
h1 { font-size: 1.4em; }
h2 { font-size: 1.1em; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
.summary { background: #f5f5f5; border: 1px solid #ddd; padding: 12px 16px; margin-bottom: 24px; }
.summary table { border-collapse: collapse; }
.summary td, .summary th { padding: 4px 12px 4px 0; text-align: left; }
.summary th { font-weight: 600; }
.page-block { margin-bottom: 32px; }
.cols { display: flex; gap: 24px; }
.col { flex: 1; min-width: 0; }
.col h3 { font-size: 1em; margin: 0 0 8px; }
.col h3 span.badge {
  display: inline-block; font-size: .75em; padding: 1px 6px;
  border-radius: 3px; margin-left: 6px; font-weight: normal;
}
.badge.text-only { background: #dbeafe; color: #1e40af; }
.badge.multimodal { background: #dcfce7; color: #166534; }
table.entries { border-collapse: collapse; width: 100%; font-size: 12px; }
table.entries th { background: #f0f0f0; padding: 4px 6px; text-align: left; white-space: nowrap; }
table.entries td { padding: 3px 6px; border-top: 1px solid #eee; vertical-align: top; }
.ad { background: #fef9c3; }
.diff-add { background: #dcfce7; }
.diff-del { background: #fee2e2; }
.parse-error { color: #b91c1c; font-style: italic; padding: 8px; background: #fef2f2; }
.metric-bar { display: flex; gap: 32px; margin-bottom: 8px; font-size: 12px; }
.metric-item label { color: #666; margin-right: 4px; }
.metric-item .val { font-weight: 600; }
.delta-pos { color: #166534; }
.delta-neg { color: #991b1b; }
.delta-zero { color: #666; }
"""

_ENTRY_COLS = [
    "establishment_name", "raw_address", "address_type",
    "city", "state", "category", "phone", "is_advertisement",
]


def _entry_key(e: dict) -> str:
    return e.get("establishment_name", "") + "|" + (e.get("raw_address") or "")


def _html_val(v) -> str:
    if v is None:
        return '<span style="color:#aaa">—</span>'
    if isinstance(v, bool):
        return "yes" if v else ""
    s = str(v)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _entries_table(entries: list[dict], extra_keys: set[str] | None = None) -> str:
    if not entries:
        return "<p><em>No entries extracted.</em></p>"
    rows_html = []
    for e in entries:
        row_class = " class='ad'" if e.get("is_advertisement") else ""
        cells = "".join(f"<td>{_html_val(e.get(c))}</td>" for c in _ENTRY_COLS)
        if extra_keys is not None:
            key = _entry_key(e)
            if key in extra_keys:
                row_class = " class='diff-add'"
        rows_html.append(f"<tr{row_class}>{cells}</tr>")
    header = "".join(f"<th>{c}</th>" for c in _ENTRY_COLS)
    return (
        f"<table class='entries'><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _delta_class(d: int) -> str:
    if d > 0:
        return "delta-pos"
    if d < 0:
        return "delta-neg"
    return "delta-zero"


def _render_metrics_bar(text_m: dict, mm_m: dict) -> str:
    items = [
        ("entries", text_m["count"], mm_m["count"]),
        ("std addr", text_m["addr_standard"], mm_m["addr_standard"]),
        ("no addr", text_m["null_address"], mm_m["null_address"]),
        ("ads", text_m["is_ad"], mm_m["is_ad"]),
    ]
    parts = []
    for label, tv, mv in items:
        d = mv - tv
        dcls = _delta_class(d)
        ds = f"+{d}" if d > 0 else str(d)
        parts.append(
            f"<div class='metric-item'>"
            f"<label>{label}</label>"
            f"<span class='val'>{tv} → {mv}</span> "
            f"<span class='{dcls}'>({ds})</span>"
            f"</div>"
        )
    return f"<div class='metric-bar'>{''.join(parts)}</div>"


def write_comparison_html(
    pages: list[dict],
    item_dir: Path,
    model: str,
    out_path: Path,
) -> None:
    """
    pages: list of {
      filename, text_result, mm_result, text_metrics, mm_metrics
    }
    """
    total_text = sum(p["text_metrics"]["count"] for p in pages)
    total_mm   = sum(p["mm_metrics"]["count"] for p in pages)
    d_total    = total_mm - total_text

    # Summary table
    summary_rows = [
        ("Item directory", str(item_dir)),
        ("Model", model),
        ("Pages compared", str(len(pages))),
        ("Total entries — text-only", str(total_text)),
        ("Total entries — multimodal", str(total_mm)),
        ("Delta", f"{'+' if d_total >= 0 else ''}{d_total}"),
    ]
    summary_html = "<table>" + "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary_rows
    ) + "</table>"

    # Per-page sections
    page_sections = []
    for i, p in enumerate(pages, 1):
        text_r = p["text_result"]
        mm_r   = p["mm_result"]
        text_m = p["text_metrics"]
        mm_m   = p["mm_metrics"]

        # Compute diff: entries only in mm (added), only in text (removed)
        text_keys = {_entry_key(e) for e in text_r.get("entries", [])}
        mm_keys   = {_entry_key(e) for e in mm_r.get("entries", [])}
        mm_added  = mm_keys - text_keys

        bar = _render_metrics_bar(text_m, mm_m)

        text_body: str
        if text_r["status"].startswith("parse_error"):
            text_body = f"<div class='parse-error'>Parse error — raw response:<br><pre>{text_r.get('raw','')[:800]}</pre></div>"
        else:
            text_body = _entries_table(text_r.get("entries", []))

        mm_body: str
        if mm_r["status"].startswith("parse_error"):
            mm_body = f"<div class='parse-error'>Parse error — raw response:<br><pre>{mm_r.get('raw','')[:800]}</pre></div>"
        else:
            mm_body = _entries_table(mm_r.get("entries", []), extra_keys=mm_added)

        text_n = text_m["count"]
        mm_n   = mm_m["count"]
        page_sections.append(f"""
<div class="page-block">
  <h2>Page {i}: {p['filename']}</h2>
  {bar}
  <div class="cols">
    <div class="col">
      <h3>Text-only <span class="badge text-only">{text_n} entries</span></h3>
      {text_body}
    </div>
    <div class="col">
      <h3>Multimodal <span class="badge multimodal">{mm_n} entries</span></h3>
      {mm_body}
    </div>
  </div>
</div>
""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extraction comparison — {item_dir.name}</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<h1>NER extraction comparison: text-only vs multimodal</h1>
<div class="summary">{summary_html}</div>
{''.join(page_sections)}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare text-only vs multimodal NER extraction on a page sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "item_dir",
        help="Item directory containing *_{slug}_aligned.json files",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pages", "-n",
        type=int,
        default=DEFAULT_PAGES,
        metavar="N",
        help=f"Number of pages to compare (default: {DEFAULT_PAGES})",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=str(NER_PROMPT_FILE),
        metavar="FILE",
        help=f"NER system prompt file (default: {NER_PROMPT_FILE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files and show what would be run, without calling the API",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"Error: NER prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    item_dir = Path(args.item_dir)
    if not item_dir.is_dir():
        print(f"Error: not a directory: {item_dir}", file=sys.stderr)
        sys.exit(1)

    slug = model_slug(args.model)
    aligned_files = sorted(item_dir.glob(f"*_{slug}_aligned.json"))[: args.pages]
    if not aligned_files:
        print(f"No *_{slug}_aligned.json files found in {item_dir}", file=sys.stderr)
        sys.exit(1)

    _log(f"\nComparing {len(aligned_files)} page(s) in {item_dir.name}")
    _log(f"Model: {args.model}  |  Modes: text-only vs multimodal")
    if args.dry_run:
        _log("[DRY RUN — no API calls]")
        for f in aligned_files:
            _log(f"  {f.name}")
        return

    client = genai.Client(api_key=api_key)

    pages: list[dict] = []
    for i, aligned_path in enumerate(aligned_files, 1):
        _log(f"\n[{i:02d}/{len(aligned_files)}] {aligned_path.name}")

        _log("    Running text-only ...")
        text_result = _extract(client, args.model, system_prompt, aligned_path, slug, "text-only")
        _log(f"    text-only: {text_result['status']}  {len(text_result['entries'])} entries")

        _log("    Running multimodal ...")
        mm_result = _extract(client, args.model, system_prompt, aligned_path, slug, "multimodal")
        _log(f"    multimodal: {mm_result['status']}  {len(mm_result['entries'])} entries")

        pages.append({
            "filename": aligned_path.name,
            "text_result": text_result,
            "mm_result": mm_result,
            "text_metrics": _metrics(text_result.get("entries", [])),
            "mm_metrics": _metrics(mm_result.get("entries", [])),
        })

    # Write outputs
    csv_path  = item_dir / f"extraction_comparison_{slug}.csv"
    html_path = item_dir / f"extraction_comparison_{slug}.html"

    csv_rows = [
        _metrics_to_csv_row(
            i + 1,
            p["filename"],
            p["text_metrics"],
            p["mm_metrics"],
            p["text_result"]["status"],
            p["mm_result"]["status"],
        )
        for i, p in enumerate(pages)
    ]
    write_comparison_csv(csv_rows, csv_path)
    write_comparison_html(pages, item_dir, args.model, html_path)

    # Print summary
    total_text = sum(r["text_count"] for r in csv_rows)
    total_mm   = sum(r["mm_count"] for r in csv_rows)
    d = total_mm - total_text
    sign = "+" if d >= 0 else ""
    _log(f"\nSummary across {len(pages)} page(s):")
    _log(f"  text-only entries:  {total_text}")
    _log(f"  multimodal entries: {total_mm}  ({sign}{d})")
    _log(f"\n  CSV  → {csv_path}")
    _log(f"  HTML → {html_path}")
    _log("\nDone.")


if __name__ == "__main__":
    main()
