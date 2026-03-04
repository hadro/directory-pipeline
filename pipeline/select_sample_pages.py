#!/usr/bin/env python3
"""Generate an HTML page for selecting 4–8 sample pages from a collection volume.

Writes select_pages.html into the item directory (alongside the images) so that
thumbnail <img> tags resolve as relative paths — no server required.

After opening the page in a browser:
  • Click thumbnails to select / deselect pages.
  • The selection panel updates live.  When you are happy, click
    **Download selection.txt** to save a plain-text file listing the selected
    filenames (one per line).  Put that file wherever you like; pass its path
    to generate_prompt.py with --selection.

Usage
-----
    python pipeline/select_sample_pages.py output/the_travelers_guide_e088efa0/
    python pipeline/select_sample_pages.py output/the_travelers_guide_e088efa0/uuid/
    python pipeline/select_sample_pages.py output/the_travelers_guide_e088efa0/ --no-open
"""

import argparse
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Image discovery helpers
# ---------------------------------------------------------------------------

_EXCLUDE = re.compile(
    r"_(?:viz|surya|gemini|tesseract|chandra|left|right|aligned|entries|annotations)"
)


def _source_images(item_dir: Path) -> list[Path]:
    """Return sorted source JPGs, excluding derivative outputs."""
    return sorted(
        p for p in item_dir.glob("*.jpg")
        if not _EXCLUDE.search(p.stem)
    )


def _find_item_dirs(root: Path) -> list[Path]:
    """Given a slug dir (or item dir), return the list of item directories
    that contain source images."""
    # If root itself has images, use it directly.
    if _source_images(root):
        return [root]
    # Otherwise descend one level (slug → item_id sub-dirs).
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    found = [d for d in candidates if _source_images(d)]
    return found


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Select sample pages — {volume}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: #f0f0f0;
      color: #222;
    }}

    /* ── sticky header ─────────────────────────────────────────────── */
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: #1e293b;
      color: #f8fafc;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      gap: 18px;
      box-shadow: 0 2px 6px rgba(0,0,0,.35);
    }}
    header h1 {{ font-size: 15px; font-weight: 600; white-space: nowrap; }}
    header .volume {{ font-size: 12px; color: #94a3b8; flex: 1; min-width: 0;
                      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    #status {{
      font-size: 13px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 999px;
      background: #374151;
      white-space: nowrap;
    }}
    #status.ok  {{ background: #15803d; color: #f0fdf4; }}
    #status.too-few {{ background: #92400e; color: #fef3c7; }}
    #status.too-many {{ background: #9f1239; color: #fff1f2; }}

    .btn {{
      padding: 5px 14px;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      transition: opacity .15s;
    }}
    .btn:hover {{ opacity: .85; }}
    .btn:disabled {{ opacity: .35; cursor: default; }}
    .btn-primary {{ background: #2563eb; color: #fff; }}
    .btn-secondary {{ background: #475569; color: #fff; }}
    .btn-clear {{ background: #64748b; color: #fff; font-size: 12px; }}

    /* ── grid ──────────────────────────────────────────────────────── */
    #grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 8px;
      padding: 14px;
    }}

    .card {{
      background: #fff;
      border: 3px solid transparent;
      border-radius: 8px;
      cursor: pointer;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,.12);
      transition: border-color .1s, transform .1s;
      user-select: none;
    }}
    .card:hover {{ transform: translateY(-2px); box-shadow: 0 3px 8px rgba(0,0,0,.18); }}
    .card.selected {{
      border-color: #2563eb;
      box-shadow: 0 0 0 1px #2563eb, 0 3px 8px rgba(37,99,235,.25);
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 3 / 4;
      object-fit: cover;
      display: block;
    }}
    .card .label {{
      text-align: center;
      padding: 4px 2px 5px;
      font-size: 11px;
      color: #64748b;
      line-height: 1.3;
    }}
    .card.selected .label {{ color: #1d4ed8; font-weight: 600; }}
    .badge {{
      display: none;
      position: absolute;
      top: 4px;
      right: 4px;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: #2563eb;
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      align-items: center;
      justify-content: center;
    }}
    .card-wrap {{ position: relative; }}
    .card.selected .badge {{ display: flex; }}

    /* ── selection panel ────────────────────────────────────────────── */
    footer {{
      position: sticky;
      bottom: 0;
      background: #1e293b;
      color: #f8fafc;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      gap: 12px;
      border-top: 1px solid #334155;
      flex-wrap: wrap;
    }}
    #selection-list {{
      flex: 1;
      font-size: 12px;
      color: #94a3b8;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 120px;
    }}
  </style>
</head>
<body>

<header>
  <h1>Select sample pages</h1>
  <span class="volume">{volume}</span>
  <span id="status" class="too-few">0 selected (need 4–8)</span>
  <button class="btn btn-clear" onclick="clearAll()">Clear</button>
</header>

<div id="grid"></div>

<footer>
  <span id="selection-list">No pages selected.</span>
  <button class="btn btn-secondary" onclick="copyToClipboard()" id="copy-btn" disabled>Copy filenames</button>
  <button class="btn btn-primary"   onclick="downloadSelection()" id="dl-btn" disabled>Download selection.txt</button>
</footer>

<script>
  const IMAGES = {images_json};
  const MIN_SEL = 4;
  const MAX_SEL = 8;

  let selected = new Set();

  // ── render grid ────────────────────────────────────────────────────
  const grid = document.getElementById("grid");
  IMAGES.forEach((fname, i) => {{
    const wrap = document.createElement("div");
    wrap.className = "card-wrap";

    const card = document.createElement("div");
    card.className = "card";
    card.dataset.fname = fname;

    const badge = document.createElement("div");
    badge.className = "badge";
    card.appendChild(badge);

    const img = document.createElement("img");
    img.src = fname;
    img.alt = fname;
    img.loading = "lazy";
    card.appendChild(img);

    const label = document.createElement("div");
    label.className = "label";
    // show page number and bare image ID
    const m = fname.match(/^(\\d+)/);
    label.textContent = m ? "p. " + parseInt(m[1], 10) : fname;
    card.appendChild(label);

    card.addEventListener("click", () => toggleCard(card, badge));
    wrap.appendChild(card);
    grid.appendChild(wrap);
  }});

  function toggleCard(card, badge) {{
    const fname = card.dataset.fname;
    if (selected.has(fname)) {{
      selected.delete(fname);
      card.classList.remove("selected");
    }} else {{
      if (selected.size >= MAX_SEL) {{
        // flash the status indicator
        const st = document.getElementById("status");
        st.style.transition = "none";
        st.style.background = "#7f1d1d";
        setTimeout(() => {{ st.style.transition = ""; updateStatus(); }}, 400);
        return;
      }}
      selected.add(fname);
      card.classList.add("selected");
    }}
    updateBadges();
    updateStatus();
  }}

  function updateBadges() {{
    let n = 0;
    document.querySelectorAll(".card.selected .badge").forEach(b => {{
      b.textContent = ++n;
    }});
  }}

  function updateStatus() {{
    const n = selected.size;
    const st = document.getElementById("status");
    const dlBtn = document.getElementById("dl-btn");
    const cpBtn = document.getElementById("copy-btn");

    if (n === 0) {{
      st.textContent = "0 selected (need 4–8)";
      st.className = "too-few";
    }} else if (n < MIN_SEL) {{
      st.textContent = n + " selected (need " + (MIN_SEL - n) + " more)";
      st.className = "too-few";
    }} else if (n > MAX_SEL) {{
      st.textContent = n + " selected (max " + MAX_SEL + ")";
      st.className = "too-many";
    }} else {{
      st.textContent = n + " selected \u2713";
      st.className = "ok";
    }}

    const ready = n >= MIN_SEL && n <= MAX_SEL;
    dlBtn.disabled = !ready;
    cpBtn.disabled = n === 0;

    const list = document.getElementById("selection-list");
    if (n === 0) {{
      list.textContent = "No pages selected.";
    }} else {{
      list.textContent = [...selected].join(",  ");
    }}
  }}

  function clearAll() {{
    selected.clear();
    document.querySelectorAll(".card.selected").forEach(c => c.classList.remove("selected"));
    updateStatus();
  }}

  function getSelectionText() {{
    return [...selected].join("\\n") + "\\n";
  }}

  function downloadSelection() {{
    const blob = new Blob([getSelectionText()], {{type: "text/plain"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "selection.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  }}

  function copyToClipboard() {{
    navigator.clipboard.writeText([...selected].join("\\n")).then(() => {{
      const btn = document.getElementById("copy-btn");
      const orig = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => btn.textContent = orig, 1500);
    }});
  }}
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_html(item_dir: Path, images: list[Path]) -> Path:
    """Write select_pages.html into item_dir and return its path."""
    # Image filenames relative to item_dir (they're siblings of the HTML file)
    import json
    fnames = [p.name for p in images]
    volume = item_dir.parent.name + "/" + item_dir.name

    html = _HTML_TEMPLATE.format(
        volume=volume,
        images_json=json.dumps(fnames),
    )
    out_path = item_dir / "select_pages.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a browser-based page-selector for choosing sample pages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Slug directory (output/{slug}/) or item directory containing .jpg files",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Generate the HTML but do not open it in a browser",
    )
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    item_dirs = _find_item_dirs(root)
    if not item_dirs:
        print(f"Error: no source .jpg images found under {root}", file=sys.stderr)
        sys.exit(1)

    for item_dir in item_dirs:
        images = _source_images(item_dir)
        out_path = generate_html(item_dir, images)
        print(f"Generated: {out_path}  ({len(images)} pages)", file=sys.stderr)
        if not args.no_open:
            webbrowser.open(out_path.as_uri())

    print(
        "\nSelect 4–8 pages and click 'Download selection.txt'.\n"
        "Then run:\n"
        "  python pipeline/generate_prompt.py <output_dir> "
        "--selection path/to/selection.txt",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
