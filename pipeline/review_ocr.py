#!/usr/bin/env python3
"""Review Gemini OCR output for anomalous pages.

Compares each page's line count (and average line length) to a rolling window of
its neighbors and flags pages that deviate significantly.  Writes a self-contained
HTML report with per-page thumbnails for quick visual triage.

Flagged pages are not necessarily errors — title pages, ad pages, and other
non-directory content will also be flagged.  The thumbnail lets you tell at a glance
whether a flag is a legitimate layout difference or an OCR problem worth fixing.

Usage:
    python pipeline/review_ocr.py output/my-collection/item_dir/
    python pipeline/review_ocr.py output/my-collection/item_dir/ --model gemini-2.0-flash
    python pipeline/review_ocr.py output/my-collection/item_dir/ --threshold 0.5 --window 3
"""

import argparse
import base64
import io
import re
import statistics
import sys
import webbrowser
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def _detect_model(item_dir: Path) -> str | None:
    """Auto-detect model slug from *_gemini*.txt files in item_dir."""
    slug_pat = re.compile(r"_(gemini[-\w.]+)\.txt$")
    for p in sorted(item_dir.glob("*_gemini*.txt")):
        m = slug_pat.search(p.name)
        if m:
            return m.group(1)
    return None


def _load_scope(item_dir: Path) -> set[str] | None:
    """Return set of 4-digit page keys from included_pages.txt, or None if absent."""
    scope_path = item_dir / "included_pages.txt"
    if not scope_path.exists():
        return None
    lines = scope_path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def _page_key(path: Path) -> str:
    """Extract the 4-digit page number prefix used for sorting and scope filtering."""
    m = re.match(r"(\d{4})", path.name)
    return m.group(1) if m else path.name


def _find_image(txt_path: Path, slug: str) -> Path | None:
    """Return the source image path for a given OCR txt file, or None."""
    suffix = f"_{slug}.txt"
    if txt_path.name.endswith(suffix):
        stem = txt_path.name[: -len(suffix)]
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = txt_path.parent / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _page_stats(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    line_count = len(lines)
    word_count = len(text.split())
    char_count = len(text)
    avg_line_len = round(sum(len(ln) for ln in lines) / line_count, 1) if line_count else 0.0
    return {
        "line_count": line_count,
        "word_count": word_count,
        "char_count": char_count,
        "avg_line_len": avg_line_len,
    }


def _window_median(values: list[float], idx: int, window: int) -> float | None:
    start = max(0, idx - window)
    end = min(len(values), idx + window + 1)
    neighbors = values[start:idx] + values[idx + 1:end]
    return statistics.median(neighbors) if neighbors else None


def _flag_pages(pages: list[dict], window: int, threshold: float) -> list[dict]:
    line_counts = [p["stats"]["line_count"] for p in pages]
    avg_lens    = [p["stats"]["avg_line_len"] for p in pages]

    for i, page in enumerate(pages):
        reasons: list[str] = []

        ref_lc = _window_median(line_counts, i, window)
        ref_al = _window_median(avg_lens,    i, window)

        page["ref_line_median"]   = ref_lc
        page["ref_avglen_median"] = ref_al

        lc = page["stats"]["line_count"]
        al = page["stats"]["avg_line_len"]

        if ref_lc and ref_lc > 0:
            ratio = lc / ref_lc
            page["line_ratio"] = round(ratio, 3)
            if ratio < threshold or ratio > (1.0 / threshold):
                reasons.append("line_count")
        else:
            page["line_ratio"] = None

        if ref_al and ref_al > 0 and lc > 0:
            ratio_al = al / ref_al
            page["avglen_ratio"] = round(ratio_al, 3)
            if ratio_al < threshold or ratio_al > (1.0 / threshold):
                reasons.append("avg_line_len")
        else:
            page["avglen_ratio"] = None

        page["flagged"]      = bool(reasons)
        page["flag_reasons"] = reasons

    return pages


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------

def _thumbnail_b64(image_path: Path, max_width: int = 180) -> str | None:
    """Return a base64-encoded JPEG thumbnail, or None if unavailable."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not image_path or not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if w > max_width:
                img = img.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=72)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Courier New', monospace; font-size: 12px;
  background: #111; color: #ccc; padding: 24px; max-width: 960px; margin: 0 auto;
}
h1  { font-size: 15px; color: #eee; margin-bottom: 4px; }
.meta { color: #555; margin-bottom: 24px; font-size: 11px; }
h2  { font-size: 11px; color: #666; text-transform: uppercase;
      letter-spacing: 1px; margin-bottom: 10px; }
.bar-wrap { margin-bottom: 20px; }
.bar-chart { display: flex; align-items: flex-end; gap: 1px; height: 80px; }
.bar { flex: 1; min-width: 2px; background: #2a2a2a; }
.bar-chart a { text-decoration: none; }
.bar-chart a .bar { width: 100%; }
.bar.flagged { background: #b8900a; }
.bar.flagged:hover { background: #e0a800; }
.bar-note { color: #444; font-size: 10px; margin-top: 3px; }
.card {
  border: 1px solid #242424; border-radius: 3px;
  padding: 12px; margin-bottom: 12px; background: #161616;
}
.card.flagged { border-color: #6a4e00; }
.card-header { display: flex; gap: 14px; }
.thumb img { width: 120px; border: 1px solid #2a2a2a; border-radius: 2px; display: block; }
.thumb-missing { width: 120px; height: 80px; background: #1a1a1a;
                 border: 1px solid #222; border-radius: 2px; flex-shrink: 0; }
.card-body { flex: 1; }
.flag-title { color: #c8a000; font-weight: bold; font-size: 13px; margin-bottom: 4px; }
.stat-row { color: #555; font-size: 10px; margin-bottom: 6px; }
.neighbors { display: flex; gap: 3px; flex-wrap: wrap; margin: 6px 0 8px; }
.nb {
  padding: 1px 5px; background: #1c1c1c; border: 1px solid #282828;
  border-radius: 2px; font-size: 10px; color: #555;
}
.nb.current { border-color: #b8900a; color: #c8a000; font-weight: bold; }
details { margin-top: 6px; }
summary { cursor: pointer; color: #484848; font-size: 10px; user-select: none;
          padding: 2px 0; }
summary:hover { color: #777; }
pre {
  background: #0e0e0e; border: 1px solid #1e1e1e; border-radius: 2px;
  padding: 8px; font-size: 10px; color: #888; white-space: pre-wrap;
  word-break: break-word; max-height: 280px; overflow-y: auto; margin-top: 4px;
}
table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px; }
th { text-align: left; color: #444; padding: 4px 8px;
     border-bottom: 1px solid #1e1e1e; font-weight: normal; }
td { padding: 3px 8px; border-bottom: 1px solid #181818; color: #666; }
td.flag-cell { color: #b8900a; }
tr:hover td { background: #151515; }
.no-flags { color: #444; font-style: italic; padding: 10px 0; font-size: 11px; }
.section-summary { cursor: pointer; font-size: 11px; color: #444;
                   user-select: none; padding: 4px 0; }
.section-summary:hover { color: #777; }
"""


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _build_html(pages: list[dict], slug: str, item_dir: Path) -> str:
    volume = item_dir.name
    n_pages   = len(pages)
    n_flagged = sum(1 for p in pages if p["flagged"])

    # --- Bar chart ---
    max_lc = max((p["stats"]["line_count"] for p in pages), default=1) or 1
    bars = []
    for p in pages:
        h   = max(1, int(70 * p["stats"]["line_count"] / max_lc))
        cls = "bar flagged" if p["flagged"] else "bar"
        tip = f'{p["page"]}: {p["stats"]["line_count"]} lines'
        bar = f'<div class="{cls}" style="height:{h}px" title="{_esc(tip)}"></div>'
        if p["flagged"]:
            bar = f'<a href="#page-{_esc(p["page"])}" style="flex:1;min-width:2px;display:flex;align-items:flex-end;">{bar}</a>'
        bars.append(bar)
    bar_chart = (
        '<div class="bar-wrap">'
        '<h2>Line counts per page</h2>'
        '<div class="bar-chart">' + "".join(bars) + "</div>"
        '<p class="bar-note">amber = flagged &nbsp;&bull;&nbsp; '
        'hover for page number</p></div>'
    )

    # --- Flagged cards ---
    flagged_html = []
    for i, p in enumerate(pages):
        if not p["flagged"]:
            continue

        # Thumbnail
        img_path = _find_image(p["path"], slug)
        b64 = _thumbnail_b64(img_path) if img_path else None
        if b64:
            thumb = f'<div class="thumb"><img src="data:image/jpeg;base64,{b64}" alt="{_esc(p["page"])}"></div>'
        else:
            thumb = '<div class="thumb-missing"></div>'

        # Neighbor bar
        start = max(0, i - 3)
        end   = min(n_pages, i + 4)
        nbs   = []
        for j in range(start, end):
            q   = pages[j]
            cls = "nb current" if j == i else "nb"
            nbs.append(f'<span class="{cls}">{_esc(q["page"])}={q["stats"]["line_count"]}</span>')
        nb_bar = '<div class="neighbors">' + "".join(nbs) + "</div>"

        # Stats line
        ref   = p.get("ref_line_median")
        ratio = p.get("line_ratio")
        ratio_s = f"{ratio:.2f}" if ratio is not None else "—"
        ref_s   = f"{ref:.0f}"   if ref   is not None else "—"
        reasons = ", ".join(p["flag_reasons"])
        stat_row = (
            f'<div class="stat-row">'
            f'window median: {ref_s} lines &nbsp;&bull;&nbsp; '
            f'ratio: {ratio_s} &nbsp;&bull;&nbsp; '
            f'flagged: {_esc(reasons)} &nbsp;&bull;&nbsp; '
            f'words: {p["stats"]["word_count"]} &nbsp;&bull;&nbsp; '
            f'avg line len: {p["stats"]["avg_line_len"]}'
            f'</div>'
        )

        text_block = (
            f'<details><summary>show text ({p["stats"]["line_count"]} lines)</summary>'
            f'<pre>{_esc(p["text"])}</pre></details>'
        )

        flagged_html.append(f"""<div class="card flagged" id="page-{_esc(p['page'])}">
  <div class="card-header">
    {thumb}
    <div class="card-body">
      <div class="flag-title">&#9888; Page {_esc(p["page"])}</div>
      {stat_row}
      {nb_bar}
      {text_block}
    </div>
  </div>
</div>""")

    flagged_section = (
        "\n".join(flagged_html) if flagged_html
        else '<p class="no-flags">No pages flagged.</p>'
    )

    # --- All-pages table ---
    rows = []
    for p in pages:
        flag_cell = (
            f'<td class="flag-cell">&#9888; {_esc(", ".join(p["flag_reasons"]))}</td>'
            if p["flagged"] else "<td></td>"
        )
        rows.append(
            f'<tr><td>{_esc(p["page"])}</td>'
            f'<td>{p["stats"]["line_count"]}</td>'
            f'<td>{p["stats"]["word_count"]}</td>'
            f'<td>{p["stats"]["avg_line_len"]}</td>'
            f'{flag_cell}</tr>'
        )
    all_pages_table = (
        "<table><thead><tr>"
        "<th>Page</th><th>Lines</th><th>Words</th><th>Avg line len</th><th>Flag</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Review — {_esc(volume)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>OCR Review &mdash; {_esc(volume)}</h1>
<p class="meta">
  model: {_esc(slug)} &nbsp;&bull;&nbsp;
  {n_pages} pages analyzed &nbsp;&bull;&nbsp;
  {n_flagged} flagged
</p>

{bar_chart}

<h2>Flagged pages ({n_flagged})</h2>
{flagged_section}

<details style="margin-top: 24px;">
<summary class="section-summary">All pages ({n_pages})</summary>
{all_pages_table}
</details>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flag anomalous Gemini OCR pages by comparing to their neighbors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("item_dir",
                        help="Directory containing *_{model}.txt OCR files")
    parser.add_argument("--model", default=None,
                        help="Model slug to review (default: auto-detected)")
    parser.add_argument("--window", type=int, default=5,
                        help="Pages on each side for local comparison (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Flag if metric/median < threshold or > 1/threshold "
                             "(default: 0.6, i.e. flag at <60%% or >167%% of median)")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (default: {item_dir}/ocr_review_{model}.html)")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open the report in the browser")
    args = parser.parse_args()

    item_dir = Path(args.item_dir).resolve()
    if not item_dir.is_dir():
        print(f"Error: {item_dir} is not a directory", file=sys.stderr)
        return 1

    slug = _model_slug(args.model) if args.model else _detect_model(item_dir)
    if not slug:
        print("Error: no Gemini OCR txt files found. Run --gemini-ocr first.",
              file=sys.stderr)
        return 1
    print(f"  model:  {slug}", file=sys.stderr)

    scope = _load_scope(item_dir)
    txt_files = sorted(item_dir.glob(f"*_{slug}.txt"), key=_page_key)
    if scope:
        txt_files = [f for f in txt_files if _page_key(f) in scope]
    if not txt_files:
        print(f"Error: no *_{slug}.txt files found in {item_dir}", file=sys.stderr)
        return 1

    print(f"  pages:  {len(txt_files)}", file=sys.stderr)

    pages: list[dict] = []
    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        pages.append({
            "page":  _page_key(txt_path),
            "path":  txt_path,
            "text":  text,
            "stats": _page_stats(text),
        })

    pages     = _flag_pages(pages, args.window, args.threshold)
    n_flagged = sum(1 for p in pages if p["flagged"])

    print(f"  flagged: {n_flagged}", file=sys.stderr)
    for p in pages:
        if p["flagged"]:
            ratio_s = f"{p['line_ratio']:.2f}" if p.get("line_ratio") is not None else "?"
            ref_s   = f"{p['ref_line_median']:.0f}" if p.get("ref_line_median") is not None else "?"
            print(
                f"    ⚠  {p['page']}  "
                f"lines={p['stats']['line_count']}  "
                f"(ratio={ratio_s}, median={ref_s})  "
                f"[{', '.join(p['flag_reasons'])}]",
                file=sys.stderr,
            )

    html = _build_html(pages, slug, item_dir)

    out_path = Path(args.out) if args.out else item_dir / f"ocr_review_{slug}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  report: {out_path}", file=sys.stderr)

    if not args.no_open:
        webbrowser.open(out_path.as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
