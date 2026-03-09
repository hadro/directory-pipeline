#!/usr/bin/env python3
"""Interactive browser UI for page selection — two modes in one interface.

Tab 1 — Sample pages: pick 4–10 representative pages for OCR/NER prompt
  calibration.  Saves selection.txt, consumed by generate_prompt.py.

Tab 2 — Scope pages: mark which pages in the volume should be processed by
  subsequent pipeline steps (OCR, extraction).  All pages start selected;
  deselect frontmatter/junk.  Saves included_pages.txt, consumed by
  run_gemini_ocr.py and extract_entries.py.  If included_pages.txt is absent,
  all pages are processed (backward compatible).

Usage
-----
    python pipeline/select_pages.py output/the_travelers_guide_e088efa0/
    python pipeline/select_pages.py output/the_travelers_guide_e088efa0/uuid/
    python pipeline/select_pages.py output/the_travelers_guide_e088efa0/ --no-open
"""

import argparse
import functools
import http.server
import json
import re
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Image discovery helpers
# ---------------------------------------------------------------------------

_EXCLUDE = re.compile(
    r"_(?:viz|surya|gemini|tesseract|chandra|aligned|entries|annotations)"
)
_SPLIT = re.compile(r"_(left|right)$", re.IGNORECASE)


def _source_images(item_dir: Path) -> list[Path]:
    """Return sorted source JPGs.

    When split pages (*_left.jpg / *_right.jpg) exist for a spread, those are
    returned in place of the original double-page spread file.
    """
    all_jpgs = set(item_dir.glob("*.jpg"))

    # Partition into base images and split pages.
    base_jpgs = sorted(
        p for p in all_jpgs
        if not _EXCLUDE.search(p.stem) and not _SPLIT.search(p.stem)
    )
    split_jpgs = {p for p in all_jpgs if _SPLIT.search(p.stem)}

    result: list[Path] = []
    for base in base_jpgs:
        left = item_dir / f"{base.stem}_left.jpg"
        right = item_dir / f"{base.stem}_right.jpg"
        splits = [p for p in (left, right) if p in split_jpgs]
        if splits:
            result.extend(splits)
        else:
            result.append(base)

    return result


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


def _load_txt(path: Path) -> list[str]:
    """Return non-empty, non-comment lines from a text file, or [] if absent."""
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


# ---------------------------------------------------------------------------
# Local HTTP server for direct save
# ---------------------------------------------------------------------------

class _SaveHandler(http.server.SimpleHTTPRequestHandler):
    """Serve images from item_dir and accept POSTs to write selection files."""
    save_path: Path   # selection.txt
    scope_path: Path  # included_pages.txt

    def do_POST(self):
        if self.path in ("/save", "/save-scope"):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length))
                text = body.get("selection", "")
                if self.path == "/save":
                    out = type(self).save_path
                    label = "selection.txt"
                else:
                    out = type(self).scope_path
                    label = "included_pages.txt"
                out.write_text(text, encoding="utf-8")
                payload = json.dumps({"path": str(out)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
                print(f"\n  Saved {label} → {out}", file=sys.stderr)
            except Exception as exc:
                self.send_error(500, str(exc))
        else:
            self.send_error(404)

    def log_message(self, *_):
        pass  # suppress per-request logging


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Select pages — {volume}</title>
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
      gap: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,.35);
      flex-wrap: wrap;
    }}

    /* ── tabs ──────────────────────────────────────────────────────── */
    .tabs {{
      display: flex;
      gap: 2px;
      background: #0f172a;
      border-radius: 6px;
      padding: 3px;
      flex-shrink: 0;
    }}
    .tab-btn {{
      padding: 5px 14px;
      border-radius: 4px;
      border: none;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      background: transparent;
      color: #94a3b8;
      transition: background .15s, color .15s;
      white-space: nowrap;
    }}
    .tab-btn.active {{
      background: #1e40af;
      color: #fff;
    }}
    .tab-btn.active.scope-active {{
      background: #15803d;
    }}

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
    #status.ok       {{ background: #15803d; color: #f0fdf4; }}
    #status.too-few  {{ background: #92400e; color: #fef3c7; }}
    #status.too-many {{ background: #9f1239; color: #fff1f2; }}

    /* ── scope quick-action strip ──────────────────────────────────── */
    #scope-controls {{
      display: none;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #94a3b8;
    }}
    #scope-controls.visible {{ display: flex; }}
    #scope-controls input[type=number] {{
      width: 52px;
      padding: 3px 6px;
      border-radius: 4px;
      border: 1px solid #475569;
      background: #1e293b;
      color: #f8fafc;
      font-size: 12px;
      text-align: center;
    }}
    #scope-controls .btn-scope-quick {{
      padding: 4px 10px;
      border-radius: 4px;
      border: none;
      cursor: pointer;
      font-size: 12px;
      background: #374151;
      color: #f8fafc;
    }}
    #scope-controls .btn-scope-quick:hover {{ background: #475569; }}

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
    .btn-primary  {{ background: #2563eb; color: #fff; }}
    .btn-scope    {{ background: #15803d; color: #fff; }}
    .btn-secondary {{ background: #475569; color: #fff; }}
    .btn-clear    {{ background: #64748b; color: #fff; font-size: 12px; }}

    /* ── size slider ────────────────────────────────────────────────── */
    .size-control {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: #94a3b8;
      margin-left: auto;
    }}
    .size-control input[type=range] {{
      width: 90px;
      accent-color: #2563eb;
      cursor: pointer;
    }}

    /* ── grid ──────────────────────────────────────────────────────── */
    #grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(var(--thumb-size, 140px), 1fr));
      gap: 8px;
      padding: 14px;
    }}

    /* ── cards — sample mode ────────────────────────────────────────── */
    .card {{
      background: #fff;
      border: 3px solid transparent;
      border-radius: 8px;
      cursor: pointer;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,.12);
      transition: border-color .1s, transform .1s, opacity .1s;
      user-select: none;
    }}
    .card:hover {{ transform: translateY(-2px); box-shadow: 0 3px 8px rgba(0,0,0,.18); }}

    /* sample selected */
    .card.sample-selected {{
      border-color: #2563eb;
      box-shadow: 0 0 0 1px #2563eb, 0 3px 8px rgba(37,99,235,.25);
    }}

    /* scope excluded */
    .card.scope-excluded {{
      border-color: #dc2626;
      opacity: 0.45;
    }}
    /* scope included (default in scope mode — soft green ring) */
    body.scope-mode .card:not(.scope-excluded) {{
      border-color: #16a34a;
      box-shadow: 0 0 0 1px #16a34a, 0 1px 3px rgba(22,163,74,.15);
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
    .card.sample-selected .label {{ color: #1d4ed8; font-weight: 600; }}
    body.scope-mode .card.scope-excluded .label {{ color: #dc2626; }}

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
    .card.sample-selected .badge {{ display: flex; }}

    /* ── selection panel / footer ───────────────────────────────────── */
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
  <div class="tabs">
    <button class="tab-btn active" id="tab-sample" onclick="switchTab('sample')">Sample pages</button>
    <button class="tab-btn"        id="tab-scope"  onclick="switchTab('scope')">Scope pages</button>
  </div>
  <span class="volume">{volume}</span>
  <span id="status" class="too-few">0 selected (need 4–10)</span>

  <!-- Scope quick-actions (visible only in scope tab) -->
  <div id="scope-controls">
    <span>Exclude first</span>
    <input type="number" id="exclude-n" value="5" min="1">
    <button class="btn-scope-quick" onclick="excludeFirstN()">pages</button>
    <button class="btn btn-clear" onclick="scopeSelectAll()" style="font-size:11px;padding:3px 9px;">Include all</button>
  </div>

  <button class="btn btn-clear" id="clear-btn" onclick="clearAll()">Clear</button>
  <label class="size-control" title="Thumbnail size">
    🔍
    <input type="range" id="size-slider" min="100" max="400" value="140"
           oninput="document.documentElement.style.setProperty('--thumb-size', this.value + 'px')">
  </label>
</header>

<div id="grid"></div>

<footer>
  <span id="selection-list">No pages selected.</span>
  <button class="btn btn-secondary" onclick="copyToClipboard()" id="copy-btn" disabled>Copy filenames</button>
  <button class="btn btn-primary"   onclick="saveSelection()" id="dl-btn"   disabled id="dl-sample-btn">{save_sample_label}</button>
  <button class="btn btn-scope"     onclick="saveScope()"     id="scope-btn" style="display:none;">{save_scope_label}</button>
</footer>

<script>
  const IMAGES        = {images_json};
  const SAVE_URL      = {save_url_json};
  const SAVE_SCOPE_URL = {save_scope_url_json};
  const PRESELECTED   = new Set({preselected_json});
  const PRESCOPED_EXCL = new Set({prescoped_excl_json}); // filenames to EXCLUDE initially
  const MIN_SEL = 4;
  const MAX_SEL = 10;

  // ── State ──────────────────────────────────────────────────────────
  let currentTab = 'sample';
  let sampleSelected = new Set(PRESELECTED);
  let scopeExcluded  = new Set(PRESCOPED_EXCL);
  let lastClickedIdx = null;  // for shift-click range in scope mode

  // ── render grid ────────────────────────────────────────────────────
  const grid = document.getElementById("grid");
  IMAGES.forEach((fname, i) => {{
    const wrap = document.createElement("div");
    wrap.className = "card-wrap";

    const card = document.createElement("div");
    card.className = "card";
    card.dataset.fname = fname;
    card.dataset.idx = i;

    const badge = document.createElement("div");
    badge.className = "badge";
    card.appendChild(badge);

    const img = document.createElement("img");
    img.src = fname.replace(/%/g, "%25");
    img.alt = fname;
    img.loading = "lazy";
    card.appendChild(img);

    const label = document.createElement("div");
    label.className = "label";
    const m = fname.match(/^(\\d+)/);
    label.textContent = m ? "p. " + parseInt(m[1], 10) : fname;
    card.appendChild(label);

    card.addEventListener("click", (e) => handleCardClick(card, badge, i, e));
    wrap.appendChild(card);
    grid.appendChild(wrap);
  }});

  // Apply pre-existing selections on load
  applyCardStates();

  // ── Tab switching ───────────────────────────────────────────────────
  function switchTab(tab) {{
    currentTab = tab;
    const isSample = tab === 'sample';

    document.getElementById("tab-sample").classList.toggle("active", isSample);
    document.getElementById("tab-scope").classList.toggle("active", !isSample);
    document.getElementById("tab-scope").classList.toggle("scope-active", !isSample);
    document.getElementById("scope-controls").classList.toggle("visible", !isSample);
    document.getElementById("clear-btn").style.display = isSample ? "" : "none";
    document.getElementById("dl-btn").style.display    = isSample ? "" : "none";
    document.getElementById("scope-btn").style.display = isSample ? "none" : "";
    document.body.classList.toggle("scope-mode", !isSample);

    updateStatus();
  }}

  // ── Card click ─────────────────────────────────────────────────────
  function handleCardClick(card, badge, idx, e) {{
    if (currentTab === 'sample') {{
      const fname = card.dataset.fname;
      if (sampleSelected.has(fname)) {{
        sampleSelected.delete(fname);
      }} else {{
        if (sampleSelected.size >= MAX_SEL) {{
          const st = document.getElementById("status");
          st.style.transition = "none";
          st.style.background = "#7f1d1d";
          setTimeout(() => {{ st.style.transition = ""; updateStatus(); }}, 400);
          return;
        }}
        sampleSelected.add(fname);
      }}
    }} else {{
      // Scope mode: shift-click selects a range to EXCLUDE
      const fname = card.dataset.fname;
      if (e.shiftKey && lastClickedIdx !== null) {{
        const lo = Math.min(lastClickedIdx, idx);
        const hi = Math.max(lastClickedIdx, idx);
        const shouldExclude = !scopeExcluded.has(IMAGES[idx]);
        for (let j = lo; j <= hi; j++) {{
          if (shouldExclude) scopeExcluded.add(IMAGES[j]);
          else               scopeExcluded.delete(IMAGES[j]);
        }}
      }} else {{
        if (scopeExcluded.has(fname)) scopeExcluded.delete(fname);
        else                          scopeExcluded.add(fname);
      }}
      lastClickedIdx = idx;
    }}
    applyCardStates();
    updateStatus();
  }}

  function applyCardStates() {{
    document.querySelectorAll(".card").forEach(card => {{
      const fname = card.dataset.fname;
      const badge = card.querySelector(".badge");
      card.classList.toggle("sample-selected", sampleSelected.has(fname));
      card.classList.toggle("scope-excluded",  scopeExcluded.has(fname));
      if (badge) badge.textContent = "";
    }});
    // Re-number sample badges in order
    let n = 0;
    document.querySelectorAll(".card.sample-selected .badge").forEach(b => {{
      b.textContent = ++n;
    }});
  }}

  // ── Status bar ─────────────────────────────────────────────────────
  function updateStatus() {{
    const st    = document.getElementById("status");
    const dlBtn = document.getElementById("dl-btn");
    const cpBtn = document.getElementById("copy-btn");
    const scBtn = document.getElementById("scope-btn");
    const list  = document.getElementById("selection-list");

    if (currentTab === 'sample') {{
      const n = sampleSelected.size;
      if (n === 0) {{
        st.textContent = "0 selected (need 4–10)";
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
      list.textContent = n === 0 ? "No pages selected."
                                 : [...sampleSelected].join(",  ");
    }} else {{
      const total    = IMAGES.length;
      const excluded = scopeExcluded.size;
      const included = total - excluded;
      st.textContent = included + " of " + total + " pages included";
      st.className = included === total ? "ok" : (included === 0 ? "too-few" : "ok");
      scBtn.disabled = false;
      cpBtn.disabled = true;
      list.textContent = excluded === 0
        ? "All " + total + " pages included."
        : excluded + " page" + (excluded === 1 ? "" : "s") + " excluded: " +
          [...scopeExcluded].join(",  ");
    }}
  }}

  function clearAll() {{
    sampleSelected.clear();
    applyCardStates();
    updateStatus();
  }}

  // ── Scope quick-actions ────────────────────────────────────────────
  function excludeFirstN() {{
    const n = parseInt(document.getElementById("exclude-n").value, 10) || 0;
    for (let i = 0; i < Math.min(n, IMAGES.length); i++) {{
      scopeExcluded.add(IMAGES[i]);
    }}
    applyCardStates();
    updateStatus();
  }}

  function scopeSelectAll() {{
    scopeExcluded.clear();
    applyCardStates();
    updateStatus();
  }}

  // ── Text serialisation ─────────────────────────────────────────────
  function getSampleText() {{
    return [...sampleSelected].join("\\n") + "\\n";
  }}

  function getScopeText() {{
    // Write the INCLUDED pages (= all pages minus excluded)
    const included = IMAGES.filter(f => !scopeExcluded.has(f));
    return included.join("\\n") + "\\n";
  }}

  // ── Save handlers ──────────────────────────────────────────────────
  async function saveSelection() {{
    if (SAVE_URL) {{
      try {{
        const r = await fetch(SAVE_URL, {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{selection: getSampleText()}})
        }});
        await r.json();
        const btn = document.getElementById("dl-btn");
        btn.textContent = "Saved \u2713";
        btn.style.background = "#15803d";
        return;
      }} catch(e) {{
        console.warn("Save to folder failed, falling back to download", e);
      }}
    }}
    const blob = new Blob([getSampleText()], {{type: "text/plain"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "selection.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  }}

  async function saveScope() {{
    if (SAVE_SCOPE_URL) {{
      try {{
        const r = await fetch(SAVE_SCOPE_URL, {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{selection: getScopeText()}})
        }});
        await r.json();
        const btn = document.getElementById("scope-btn");
        btn.textContent = "Saved \u2713";
        btn.style.background = "#166534";
        return;
      }} catch(e) {{
        console.warn("Save to folder failed, falling back to download", e);
      }}
    }}
    const blob = new Blob([getScopeText()], {{type: "text/plain"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "included_pages.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  }}

  function copyToClipboard() {{
    const text = currentTab === 'sample'
      ? [...sampleSelected].join("\\n")
      : IMAGES.filter(f => !scopeExcluded.has(f)).join("\\n");
    navigator.clipboard.writeText(text).then(() => {{
      const btn = document.getElementById("copy-btn");
      const orig = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => btn.textContent = orig, 1500);
    }});
  }}

  // Initial status render
  updateStatus();
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTML generator
# ---------------------------------------------------------------------------

def _read_scope_excluded(item_dir: Path, images: list[Path]) -> list[str]:
    """Return filenames to mark as excluded in the scope tab on load.

    If included_pages.txt exists, excluded = all images NOT in that file.
    If it doesn't exist, return [] (all included = no pre-exclusions).
    """
    included_path = item_dir / "included_pages.txt"
    if not included_path.exists():
        return []
    included = set(_load_txt(included_path))
    all_names = {p.name for p in images}
    return [n for n in all_names if n not in included]


def generate_html(
    item_dir: Path,
    images: list[Path],
    save_url: str | None = None,
    save_scope_url: str | None = None,
) -> Path:
    """Write select_pages.html into item_dir and return its path."""
    fnames = [p.name for p in images]
    volume = item_dir.parent.name + "/" + item_dir.name

    # Pre-populate from existing saved files
    preselected = _load_txt(item_dir / "selection.txt")
    prescoped_excl = _read_scope_excluded(item_dir, images)

    save_sample_label = "Save to output folder" if save_url else "Download selection.txt"
    save_scope_label  = "Save scope"            if save_scope_url else "Download included_pages.txt"

    html = _HTML_TEMPLATE.format(
        volume             = volume,
        images_json        = json.dumps(fnames),
        save_url_json      = json.dumps(save_url),
        save_scope_url_json = json.dumps(save_scope_url),
        preselected_json   = json.dumps(preselected),
        prescoped_excl_json = json.dumps(prescoped_excl),
        save_sample_label  = save_sample_label,
        save_scope_label   = save_scope_label,
    )
    out_path = item_dir / "select_pages.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a browser-based page-selector (two-tab: sample + scope).",
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

        if args.no_open:
            out_path = generate_html(item_dir, images)
            print(f"Generated: {out_path}  ({len(images)} pages)", file=sys.stderr)
        else:
            _SaveHandler.save_path  = item_dir / "selection.txt"
            _SaveHandler.scope_path = item_dir / "included_pages.txt"
            handler = functools.partial(_SaveHandler, directory=str(item_dir))
            with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
                server.allow_reuse_address = True
                port = server.server_address[1]
                save_url       = f"http://127.0.0.1:{port}/save"
                save_scope_url = f"http://127.0.0.1:{port}/save-scope"

                out_path = generate_html(item_dir, images, save_url, save_scope_url)
                url = f"http://127.0.0.1:{port}/select_pages.html"
                print(f"Serving: {url}  ({len(images)} pages)", file=sys.stderr)
                print("Press Ctrl+C when done.", file=sys.stderr)

                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                webbrowser.open(url)

                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    server.shutdown()
                    print("", file=sys.stderr)


if __name__ == "__main__":
    main()
