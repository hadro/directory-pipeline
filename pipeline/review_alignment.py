#!/usr/bin/env python3
"""
Annotation tool for reviewing and correcting Surya/Gemini alignment results.

Opens a local web UI where you can:
  - Browse pages sorted by number of unmatched Gemini entries
  - Draw a bounding box around an unmatched region
  - Re-run Surya OCR on just that crop
  - Review proposed matches and save accepted ones back to the _aligned.json

Usage
-----
    uv run review_alignment.py images/
    uv run review_alignment.py images/ --model gemini-2.0-flash --port 5001

Then open http://localhost:5000 in your browser.
"""

import argparse
import difflib
import json
import os
import re
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, send_file
from PIL import Image

load_dotenv()

app = Flask(__name__)

# Project root is one level up from this script (pipeline/)
PROJECT_ROOT: Path = Path(__file__).parent.parent

# Set at startup
IMAGES_ROOT: Path = Path("images")
MODEL: str = "gemini-2.0-flash"

# Surya models – pre-loaded at startup (see main())
_det = None
_rec = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _text_sim(a: str, b: str) -> int:
    """0–100 similarity matching align_ocr._text_sim exactly."""
    an, bn = _normalize(a), _normalize(b)
    if not an and not bn:
        return 50
    if not an or not bn:
        return -80
    return round(difflib.SequenceMatcher(None, an, bn, autojunk=False).ratio() * 100)


def _canvas_fragment(
    canvas_uri: str, bbox: list[int],
    img_w: int, img_h: int, canvas_w: int, canvas_h: int,
) -> str:
    x1, y1, x2, y2 = bbox
    if canvas_w and img_w and canvas_w != img_w:
        sx = canvas_w / img_w
        sy = canvas_h / img_h if img_h else 1.0
        x1, y1, x2, y2 = round(x1*sx), round(y1*sy), round(x2*sx), round(y2*sy)
    return f"{canvas_uri}#xywh={x1},{y1},{max(1, x2-x1)},{max(1, y2-y1)}"


def _load_surya() -> None:
    global _det, _rec
    if _det is not None:
        return
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    _det = DetectionPredictor()
    _rec = RecognitionPredictor(FoundationPredictor())


def _json_suffix() -> str:
    return f"_{MODEL}_aligned.json"


def _stem_from(json_path: Path) -> str:
    return json_path.name[: -len(_json_suffix())]


def _img_path(json_path: Path) -> Path:
    return json_path.parent / f"{_stem_from(json_path)}.jpg"


def _find_pages() -> list[dict]:
    pages = []
    for p in sorted(IMAGES_ROOT.rglob(f"*{_json_suffix()}")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ug = data.get("unmatched_gemini", [])
        img = _img_path(p)
        if not img.exists():
            continue
        # Volume = first path component below IMAGES_ROOT, if there are at
        # least two directory levels (volume/item_dir/file).  Single-level
        # runs (item_dir/file) have an empty volume string.
        rel_parts = p.relative_to(IMAGES_ROOT).parts
        volume = rel_parts[0] if len(rel_parts) >= 3 else ""
        pages.append({
            "json_path": str(p),
            "img_path": str(img),
            "stem": _stem_from(p),
            "item_dir": p.parent.name,
            "volume": volume,
            "unmatched_count": len(ug),
            "aligned_count": len(data.get("lines", [])),
        })
    pages.sort(key=lambda x: -x["unmatched_count"])
    return pages


# ---------------------------------------------------------------------------
# NW alignment (mirrors align_ocr.needleman_wunsch)
# ---------------------------------------------------------------------------

def _nw_align(
    surya_lines: list[dict],
    gemini_texts: list[str],
    gap: int = -40,
) -> list[dict]:
    """
    Align Surya crop lines against the unmatched Gemini text list.
    Returns a list of pair dicts:
      {surya: line_dict | None, gemini: str | None, sim: int}
    """
    seq_a = [ln["text"] for ln in surya_lines]
    seq_b = gemini_texts
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return []

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ch = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap
        ch[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap
        ch[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores = {
                "D": dp[i-1][j-1] + _text_sim(seq_a[i-1], seq_b[j-1]),
                "U": dp[i-1][j] + gap,
                "L": dp[i][j-1] + gap,
            }
            best = max(scores, key=scores.__getitem__)
            dp[i][j] = scores[best]
            ch[i][j] = best

    pairs: list[dict] = []
    i, j = n, m
    while i > 0 or j > 0:
        c = ch[i][j]
        if c == "D":
            sim = _text_sim(seq_a[i-1], seq_b[j-1])
            pairs.append({"surya": surya_lines[i-1], "gemini": seq_b[j-1], "sim": sim})
            i -= 1; j -= 1
        elif c == "U":
            pairs.append({"surya": surya_lines[i-1], "gemini": None, "sim": 0})
            i -= 1
        else:
            pairs.append({"surya": None, "gemini": seq_b[j-1], "sim": 0})
            j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    pages = _find_pages()
    volumes = sorted({p["volume"] for p in pages if p["volume"]})
    return render_template_string(_HTML, pages=pages, volumes=volumes, model=MODEL)


@app.route("/image")
def serve_image():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return "Not found", 404
    return send_file(p)


@app.route("/page_data")
def page_data():
    json_path = Path(request.args.get("json_path", ""))
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    img = _img_path(json_path)
    img_w, img_h = Image.open(img).size  # actual jpg pixel dimensions
    return jsonify({
        "img_path": str(img),
        "img_width": img_w,
        "img_height": img_h,
        "canvas_uri": data.get("canvas_uri", ""),
        "canvas_width": data.get("canvas_width", img_w),
        "canvas_height": data.get("canvas_height", img_h),
        "lines": data.get("lines", []),
        "unmatched_gemini": data.get("unmatched_gemini", []),
    })


@app.route("/annotate", methods=["POST"])
def annotate():
    """Crop the image, run Surya OCR on each box, NW-align combined lines with unmatched_gemini."""
    body = request.json
    json_path = Path(body["json_path"])
    # Accept either 'bboxes' (array) or legacy 'bbox' (single)
    bboxes = body.get("bboxes") or [body["bbox"]]

    _load_surya()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    unmatched = data.get("unmatched_gemini", [])

    img = Image.open(_img_path(json_path)).convert("RGB")
    iw, ih = img.size

    all_surya_lines: list[dict] = []
    for bbox in bboxes:
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(iw, int(bbox[2]))
        y2 = min(ih, int(bbox[3]))
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        crop = img.crop((x1, y1, x2, y2))
        results = _rec([crop], det_predictor=_det, sort_lines=True)
        for ln in results[0].text_lines:
            all_surya_lines.append({
                "bbox": [
                    int(ln.bbox[0]) + x1, int(ln.bbox[1]) + y1,
                    int(ln.bbox[2]) + x1, int(ln.bbox[3]) + y1,
                ],
                "text": ln.text,
                "confidence": round(float(getattr(ln, "confidence", 1.0)), 4),
            })

    if not all_surya_lines:
        return jsonify({"error": "No lines detected (boxes may be too small)"}), 400

    # Sort all lines top-to-bottom across all crops before alignment
    all_surya_lines.sort(key=lambda ln: ln["bbox"][1])

    pairs = _nw_align(all_surya_lines, unmatched)
    return jsonify({"surya_lines": all_surya_lines, "pairs": pairs})


@app.route("/save", methods=["POST"])
def save():
    """Merge accepted pairs into the _aligned.json."""
    body = request.json
    json_path = Path(body["json_path"])
    accepted = body["accepted"]  # [{surya_bbox, gemini_text}, ...]

    data = json.loads(json_path.read_text(encoding="utf-8"))
    lines = data.get("lines", [])
    unmatched = list(data.get("unmatched_gemini", []))

    canvas_uri = data.get("canvas_uri", "")
    cw = data.get("canvas_width", 0)
    ch = data.get("canvas_height", 0)
    img = Image.open(_img_path(json_path))
    iw, ih = img.size

    for pair in accepted:
        bbox = pair["surya_bbox"]
        gem = pair["gemini_text"]
        lines.append({
            "bbox": bbox,
            "canvas_fragment": _canvas_fragment(canvas_uri, bbox, iw, ih, cw, ch),
            "confidence": "manual",
            "gemini_text": gem,
        })
        if gem in unmatched:
            unmatched.remove(gem)

    data["lines"] = lines
    data["unmatched_gemini"] = unmatched
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return jsonify({"ok": True, "remaining_unmatched": len(unmatched)})


@app.route("/done", methods=["POST"])
def done():
    """Shut the server down so the pipeline can continue to the next stage."""
    def _shutdown() -> None:
        import time
        time.sleep(0.4)   # let the JSON response reach the browser first
        os._exit(0)
    threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Alignment Review — {{ model }}</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{display:flex;height:100vh;font:13px/1.4 system-ui,sans-serif;overflow:hidden}

/* ── sidebar ── */
#sidebar{width:270px;min-width:200px;border-right:1px solid #ccc;display:flex;flex-direction:column;background:#fafafa}
#sidebar h2{padding:10px 12px;font-size:13px;font-weight:600;background:#f0f0f0;border-bottom:1px solid #ccc}
#page-list{overflow-y:auto;flex:1}
.pi{padding:7px 12px;cursor:pointer;border-bottom:1px solid #eee;display:flex;flex-direction:column;gap:2px}
.pi:hover{background:#e8f0fe}
.pi.active{background:#d2e3fc}
.pi .cnt{font-weight:700;color:#c62828}
.pi .nm{font-family:monospace;font-size:11px;color:#555;word-break:break-all}

/* ── main ── */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#toolbar{padding:8px 12px;background:#f0f0f0;border-bottom:1px solid #ccc;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
#toolbar span{font-weight:600;font-size:12px}
#status{color:#555;font-size:12px;margin-left:auto}
button{padding:5px 12px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#fff;font-size:12px}
button:disabled{opacity:.45;cursor:default}
button.primary{background:#1a73e8;color:#fff;border-color:#1a73e8}
button.primary:disabled{background:#9ab8f0;border-color:#9ab8f0}
button.danger{background:#d93025;color:#fff;border-color:#d93025}
button.done{background:#2e7d32;color:#fff;border-color:#2e7d32}
button.done:hover:not(:disabled){background:#1b5e20}

/* ── content area ── */
#content{flex:1;display:flex;overflow:hidden}
#canvas-wrap{flex:1;overflow:auto;background:#888;padding:10px;display:flex;align-items:flex-start;justify-content:center}
canvas{cursor:crosshair;display:block;background:#fff}

/* ── instructions panel ── */
#instructions{background:#fff;border-radius:6px;padding:28px 32px;max-width:580px;margin:auto;line-height:1.6;color:#333}
#instructions h2{font-size:16px;margin-bottom:14px;color:#1a1a1a}
#instructions ol{padding-left:20px}
#instructions li{margin-bottom:8px;font-size:13px}
#instructions code{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:12px}

/* ── right panel ── */
#right{width:310px;min-width:220px;border-left:1px solid #ccc;display:flex;flex-direction:column;overflow:hidden}
#right h3{padding:8px 12px;font-size:12px;font-weight:600;background:#f0f0f0;border-bottom:1px solid #ccc}
#unmatched-list{flex:1;overflow-y:auto;padding:6px}
.ug{padding:4px 6px;border-radius:3px;font-size:12px;cursor:default;user-select:text}
.ug+.ug{margin-top:2px}
.ug.highlight{background:#fff3cd}

/* ── results panel ── */
#results{border-top:2px solid #1a73e8;display:flex;flex-direction:column;overflow:hidden;max-height:55%}
#results h3{padding:8px 12px;font-size:12px;font-weight:600;background:#e8f0fe;border-bottom:1px solid #c5d9f7}
#pairs-list{overflow-y:auto;padding:6px;flex:1}
.pair{display:flex;gap:6px;align-items:flex-start;padding:3px 4px;border-radius:3px;margin-bottom:3px}
.pair:hover{background:#f5f5f5}
.pair input[type=checkbox]{margin-top:2px;flex-shrink:0}
.pair .surya-txt{color:#1a5276;font-size:11px;flex:1;min-width:0;word-break:break-word}
.pair .arrow{color:#888;font-size:11px;flex-shrink:0}
.pair .gemini-txt{font-size:11px;flex:1;min-width:0;word-break:break-word}
.pair .sim{font-size:10px;color:#888;flex-shrink:0}
.pair.gap-a .surya-txt{text-decoration:line-through;opacity:.5}
.pair.gap-b .gemini-txt{text-decoration:line-through;opacity:.5}
.pair.good{background:#e8f5e9}
#results-footer{padding:6px 8px;border-top:1px solid #ddd;background:#f9f9f9;display:flex;gap:8px}
</style>
</head>
<body>

<div id="sidebar">
  <h2>Pages with unmatched entries</h2>

  <!-- Volume filter -->
  {% if volumes %}
  <div style="padding:6px 8px;border-bottom:1px solid #ccc">
    <select id="volume-select" onchange="filterPages()"
      style="width:100%;padding:4px 6px;border:1px solid #ccc;border-radius:3px;font-size:12px">
      <option value="">All volumes</option>
      {% for v in volumes %}
      <option value="{{ v }}">{{ v }}</option>
      {% endfor %}
    </select>
  </div>
  {% endif %}

  <!-- Search + min-count -->
  <div style="padding:6px 8px;border-bottom:1px solid #ccc;display:flex;gap:6px;align-items:center">
    <input id="filter-input" type="text" placeholder="Search page…"
      style="flex:1;padding:4px 6px;border:1px solid #ccc;border-radius:3px;font-size:12px"
      oninput="filterPages()">
    <label style="font-size:11px;white-space:nowrap" title="Minimum unmatched count">
      ≥<input id="min-count" type="number" value="1" min="1"
        style="width:36px;padding:2px 4px;border:1px solid #ccc;border-radius:3px;font-size:11px;margin-left:3px"
        oninput="filterPages()">
    </label>
  </div>

  <div id="page-list">
  {% for p in pages %}
  <div class="pi"
       onclick="loadPage(this)"
       data-count="{{ p.unmatched_count }}"
       data-text="{{ p.stem }} {{ p.item_dir }}"
       data-volume="{{ p.volume }}"
       data-json-path="{{ p.json_path }}"
       data-img-path="{{ p.img_path }}"
       data-stem="{{ p.stem }}"
       data-item-dir="{{ p.item_dir }}"
       data-aligned-count="{{ p.aligned_count }}"
       title="{{ p.item_dir }}">
    <span class="cnt">{{ p.unmatched_count }} unmatched</span>
    <span class="nm">{{ p.stem }}</span>
    <span style="font-size:11px;color:#888">{{ p.volume or p.item_dir }}</span>
  </div>
  {% endfor %}
  {% if not pages %}
  <p style="padding:12px;color:#888">No pages with unmatched entries found.</p>
  {% endif %}
  </div>
</div>

<div id="main">
  <div id="toolbar">
    <span id="page-name">← select a page</span>
    <button id="clear-btn" onclick="clearBoxes()" disabled>Clear box</button>
    <button id="undo-btn" onclick="undoBox()" disabled>Undo last</button>
    <button id="run-btn" class="primary" onclick="runOcr()" disabled>Run Surya on box</button>
    <span id="status"></span>
    <button id="done-btn" class="done" onclick="finishReview()">Done reviewing</button>
  </div>
  <div id="content">
    <div id="canvas-wrap">
      <!-- Instructions shown before a page is selected -->
      <div id="instructions">
        <h2>How to use this tool</h2>
        <ol>
          <li><strong>Pick a volume</strong> from the dropdown (top of sidebar), then click any page in the list.</li>
          <li>The page image loads with <span style="color:#1e9e1e;font-weight:600">green boxes</span> overlaid on every already-matched line.</li>
          <li>Look at the <em>Unmatched Gemini text</em> panel on the right to see what entries are missing bounding boxes.</li>
          <li><strong>Draw a box</strong> around the region that contains those missing entries — click and drag on the image.</li>
          <li>Click <strong>Run Surya on box</strong>. Surya re-runs OCR on just that crop.</li>
          <li>The <em>Proposed matches</em> panel appears. Each row shows a Surya-detected line → Gemini text pair with a similarity score.<br>
              High-confidence pairs (≥ 40 %) are pre-checked in <span style="color:#2e7d32;font-weight:600">green</span>. Uncheck any you disagree with, or check extra rows manually.</li>
          <li>Click <strong>Save accepted</strong>. Matched pairs are written back to the <code>_aligned.json</code> and the unmatched count updates immediately.</li>
          <li>Repeat steps 4–7 for any remaining unmatched entries, then move on to the next page.</li>
        </ol>
        <p style="margin-top:12px;color:#888;font-size:12px">
          Tip: use the <strong>≥</strong> filter to show only pages with many unmatched entries,
          and the <strong>search box</strong> to find a specific page by name.
        </p>
      </div>
      <canvas id="cv" style="display:none"></canvas>
    </div>
    <div id="right">
      <h3>Unmatched Gemini text (<span id="ug-count">0</span>)</h3>
      <div id="unmatched-list"></div>
      <div id="results" style="display:none">
        <h3>Proposed matches — check to accept</h3>
        <div id="pairs-list"></div>
        <div id="results-footer">
          <button class="primary" onclick="saveAccepted()" id="save-btn">Save accepted</button>
          <button onclick="cancelResults()">Discard</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let pageInfo = null, pageData = null;
let img = new Image();
let scale = 1;
let drag = null, boxes = [];   // boxes: array of {x1,y1,x2,y2} in original image coords

// ── canvas events ──────────────────────────────────────────────────────────
cv.addEventListener('mousedown', e => {
  if (!pageData) return;
  const r = cv.getBoundingClientRect();
  drag = { x: e.clientX - r.left, y: e.clientY - r.top };
  render();
});
cv.addEventListener('mousemove', e => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  drag.x2 = e.clientX - r.left;
  drag.y2 = e.clientY - r.top;
  render();
});
cv.addEventListener('mouseup', e => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  const ex = e.clientX - r.left, ey = e.clientY - r.top;
  const b = {
    x1: Math.round(Math.min(drag.x, ex) / scale),
    y1: Math.round(Math.min(drag.y, ey) / scale),
    x2: Math.round(Math.max(drag.x, ex) / scale),
    y2: Math.round(Math.max(drag.y, ey) / scale),
  };
  drag = null;
  if ((b.x2 - b.x1) > 20 && (b.y2 - b.y1) > 10) {
    boxes.push(b);
  }
  updateBoxButtons();
  render();
});

// ── rendering ──────────────────────────────────────────────────────────────
function render() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (img.complete && img.naturalWidth) {
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
  }
  if (!pageData) return;

  // Matched lines — green
  ctx.strokeStyle = 'rgba(30,160,30,0.65)';
  ctx.lineWidth = 1;
  for (const ln of pageData.lines) {
    const [x1,y1,x2,y2] = ln.bbox.map(v => v * scale);
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
  }

  // Live drag box — dashed orange
  if (drag && drag.x2 != null) {
    ctx.save();
    ctx.setLineDash([5,3]);
    ctx.strokeStyle = 'rgba(240,120,0,0.7)';
    ctx.lineWidth = 1.5;
    const x = Math.min(drag.x, drag.x2), y = Math.min(drag.y, drag.y2);
    ctx.strokeRect(x, y, Math.abs(drag.x2-drag.x), Math.abs(drag.y2-drag.y));
    ctx.restore();
  }

  // Committed boxes — solid orange with index labels when multiple
  boxes.forEach((b, i) => {
    ctx.strokeStyle = 'rgba(240,100,0,0.9)';
    ctx.lineWidth = 2;
    ctx.strokeRect(b.x1*scale, b.y1*scale, (b.x2-b.x1)*scale, (b.y2-b.y1)*scale);
    if (boxes.length > 1) {
      ctx.fillStyle = 'rgba(240,100,0,0.9)';
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText(String(i + 1), b.x1*scale + 3, b.y1*scale + 14);
    }
  });
}

// ── filter sidebar ─────────────────────────────────────────────────────────
function filterPages() {
  const q   = document.getElementById('filter-input').value.toLowerCase();
  const min = parseInt(document.getElementById('min-count').value) || 1;
  const volSel = document.getElementById('volume-select');
  const vol = volSel ? volSel.value : '';
  document.querySelectorAll('.pi').forEach(el => {
    const txt = el.dataset.text.toLowerCase();
    const cnt = parseInt(el.dataset.count);
    const elVol = el.dataset.volume || '';
    const volOk = !vol || elVol === vol;
    el.style.display = (cnt >= min && volOk && (!q || txt.includes(q))) ? '' : 'none';
  });
}

// ── load page ──────────────────────────────────────────────────────────────
function loadPage(el) {
  const info = {
    json_path:     el.dataset.jsonPath,
    img_path:      el.dataset.imgPath,
    stem:          el.dataset.stem,
    item_dir:      el.dataset.itemDir,
    unmatched_count: parseInt(el.dataset.count),
    aligned_count: parseInt(el.dataset.alignedCount),
  };
  pageInfo = info;
  boxes = []; drag = null;
  updateBoxButtons();
  document.getElementById('results').style.display = 'none';
  document.getElementById('page-name').textContent = info.stem;
  setStatus('Loading…');

  // Highlight active item
  document.querySelectorAll('.pi').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');

  fetch('/page_data?json_path=' + encodeURIComponent(info.json_path))
    .then(r => r.json())
    .then(data => {
      pageData = data;
      updateUnmatched(data.unmatched_gemini);
      // Show canvas, hide instructions
      document.getElementById('instructions').style.display = 'none';
      document.getElementById('cv').style.display = 'block';
      img = new Image();
      img.onload = () => {
        const wrap = document.getElementById('canvas-wrap');
        // Use actual jpg dimensions (img_width/img_height) — not canvas_width/
        // canvas_height — because bboxes are in jpg pixel space and the IIIF
        // canvas is often a different size (e.g. 2560×2560 square vs a
        // portrait jpg of 2048×2796).
        scale = Math.min((wrap.clientWidth - 20) / data.img_width, 1.0);
        cv.width  = Math.round(data.img_width  * scale);
        cv.height = Math.round(data.img_height * scale);
        render();
        setStatus('');
      };
      img.src = '/image?path=' + encodeURIComponent(data.img_path);
    })
    .catch(err => setStatus('Error: ' + err));
}

function updateUnmatched(list) {
  document.getElementById('ug-count').textContent = list.length;
  const el = document.getElementById('unmatched-list');
  el.innerHTML = list.map((t, i) =>
    `<div class="ug" id="ug-${i}">${escHtml(t)}</div>`
  ).join('');
}

// ── box management ─────────────────────────────────────────────────────────
function updateBoxButtons() {
  const n = boxes.length;
  document.getElementById('run-btn').disabled = n === 0;
  document.getElementById('clear-btn').disabled = n === 0;
  document.getElementById('undo-btn').disabled = n === 0;
  document.getElementById('run-btn').textContent =
    n <= 1 ? 'Run Surya on box' : `Run Surya on ${n} boxes`;
  document.getElementById('clear-btn').textContent =
    n <= 1 ? 'Clear box' : `Clear all (${n})`;
}

function clearBoxes() {
  boxes = [];
  document.getElementById('results').style.display = 'none';
  updateBoxButtons();
  render();
}

function undoBox() {
  boxes.pop();
  updateBoxButtons();
  render();
}

function runOcr() {
  if (!boxes.length || !pageInfo) return;
  setStatus('Running Surya on ' + boxes.length + ' crop(s)…');
  document.getElementById('run-btn').disabled = true;

  fetch('/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      json_path: pageInfo.json_path,
      bboxes: boxes.map(b => [b.x1, b.y1, b.x2, b.y2]),
    }),
  })
  .then(r => r.json())
  .then(result => {
    if (result.error) { setStatus('Error: ' + result.error); return; }
    setStatus('Done — ' + result.surya_lines.length + ' Surya lines detected.');
    showPairs(result.pairs);
  })
  .catch(err => { setStatus('Error: ' + err); document.getElementById('run-btn').disabled = false; });
}

// ── pairs UI ───────────────────────────────────────────────────────────────
let _pairs = [];

function showPairs(pairs) {
  _pairs = pairs;
  const el = document.getElementById('pairs-list');
  el.innerHTML = pairs.map((p, i) => {
    const s = p.surya, g = p.gemini;
    const cls = s && g ? (p.sim >= 40 ? 'pair good' : 'pair') : (s ? 'pair gap-b' : 'pair gap-a');
    const checked = s && g && p.sim >= 40 ? 'checked' : '';
    const surTxt = s ? escHtml(s.text) : '<em style="color:#aaa">—</em>';
    const gemTxt = g ? escHtml(g) : '<em style="color:#aaa">—</em>';
    const simTxt = s && g ? `${p.sim}%` : '';
    return `<div class="${cls}" id="pair-${i}">
      <input type="checkbox" ${checked} ${!s || !g ? 'disabled' : ''} id="chk-${i}">
      <span class="surya-txt">${surTxt}</span>
      <span class="arrow">→</span>
      <span class="gemini-txt">${gemTxt}</span>
      <span class="sim">${simTxt}</span>
    </div>`;
  }).join('');
  document.getElementById('results').style.display = 'flex';

  // Highlight unmatched entries that appear in the pairs
  document.querySelectorAll('.ug').forEach(el => el.classList.remove('highlight'));
  for (const p of pairs) {
    if (p.gemini) {
      document.querySelectorAll('.ug').forEach(el => {
        if (el.textContent === p.gemini) el.classList.add('highlight');
      });
    }
  }
}

function cancelResults() {
  document.getElementById('results').style.display = 'none';
  document.getElementById('run-btn').disabled = boxes.length === 0;
  document.querySelectorAll('.ug').forEach(el => el.classList.remove('highlight'));
}

function saveAccepted() {
  const accepted = [];
  _pairs.forEach((p, i) => {
    const chk = document.getElementById(`chk-${i}`);
    if (chk && chk.checked && p.surya && p.gemini) {
      accepted.push({ surya_bbox: p.surya.bbox, gemini_text: p.gemini });
    }
  });
  if (!accepted.length) { setStatus('Nothing checked.'); return; }

  setStatus('Saving…');
  fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ json_path: pageInfo.json_path, accepted }),
  })
  .then(r => r.json())
  .then(result => {
    if (!result.ok) { setStatus('Save failed.'); return; }
    setStatus(`Saved ${accepted.length} match(es). ${result.remaining_unmatched} unmatched remain.`);
    // Refresh page data to update overlays
    fetch('/page_data?json_path=' + encodeURIComponent(pageInfo.json_path))
      .then(r => r.json())
      .then(data => {
        pageData = data;
        updateUnmatched(data.unmatched_gemini);
        // Update sidebar count
        document.querySelectorAll('.pi.active .cnt').forEach(el => {
          el.textContent = data.unmatched_gemini.length + ' unmatched';
        });
        render();
      });
    cancelResults();
    boxes = [];
    updateBoxButtons();
  })
  .catch(err => setStatus('Error: ' + err));
}

// ── utils ──────────────────────────────────────────────────────────────────
function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function finishReview() {
  if (!confirm('Finished reviewing? This will shut down the server and let the pipeline continue.')) return;
  const btn = document.getElementById('done-btn');
  btn.disabled = true;
  setStatus('Shutting down…');
  fetch('/done', { method: 'POST' })
    .then(() => setStatus('Done. Server shut down — you may close this tab.'))
    .catch(() => setStatus('Done. Server shut down — you may close this tab.'));
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global IMAGES_ROOT, MODEL

    parser = argparse.ArgumentParser(
        description="Annotation tool for reviewing Surya/Gemini alignment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images_dir",
        help="Root images directory (e.g. images/ or images/greenbooks)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.0-flash",
        help="Gemini model name used in aligned JSON filenames (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to listen on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    IMAGES_ROOT = Path(args.images_dir)
    if not IMAGES_ROOT.exists():
        print(f"Error: directory not found: {IMAGES_ROOT}", file=sys.stderr)
        sys.exit(1)

    MODEL = args.model

    print(f"  images: {IMAGES_ROOT.resolve()}", flush=True)
    print(f"  model:  {MODEL}", flush=True)
    print("Loading Surya models… (this takes ~30 s)", flush=True)
    _load_surya()
    print("Models ready.", flush=True)
    print(f"Alignment review server — http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
