#!/usr/bin/env python3
"""Interactive field-value explorer for pipeline entry CSVs.

Reads entries_{model}.csv (or an output directory containing one) and writes a
self-contained HTML file with:
  - Auto-detected facet filters for low-cardinality fields (category, state, etc.)
  - Full-text search across all fields
  - Interactive bar charts for each facet field (clicking a bar filters the table)
  - Page density strip (entries per IIIF canvas — shows document structure)
  - Field fill-rate overview (which columns have reliable data)
  - Filterable results table with a detail panel
  - IIIF page thumbnails in the detail panel (when manifest.json files are present)
  - Export filtered subset as CSV

Works for any document type — field schema is introspected from the CSV header.
No geocoding required; runs directly after --extract-entries.

Usage:
    python pipeline/explore_entries.py output/functional_directory_.../
    python pipeline/explore_entries.py output/some_item/entries_gemini-2.0-flash.csv
    python pipeline/explore_entries.py output/some_item/ --out my_explorer.html
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import iiif_utils

# ---------------------------------------------------------------------------
# Manifest metadata extraction
# ---------------------------------------------------------------------------

def _iiif_label(obj: object) -> str:
    """Return the first plain-text string from a IIIF v3 language map."""
    if not obj:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return _strip_html(obj[0]) if obj else ""
    if isinstance(obj, dict):
        for lang in ("en", "none"):
            vals = obj.get(lang)
            if vals:
                v = vals[0] if isinstance(vals, list) else vals
                return _strip_html(str(v))
        for vals in obj.values():
            v = vals[0] if isinstance(vals, list) else vals
            return _strip_html(str(v))
    return ""


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", str(s)).strip()


def _looks_like_id(s: str) -> bool:
    """True when s is a bare numeric/UUID identifier rather than a human title."""
    s = s.strip()
    return bool(re.match(r"^[\d\-a-f]{6,}$", s, re.I))


def _extract_item_meta(manifest: dict) -> dict:
    """Return a flat dict of display metadata from a IIIF v3 manifest.

    Keys: title, date, institution, homepage_url, collection, genre
    All values are plain strings; missing values are empty strings.
    """
    manifest_id: str = manifest.get("id") or manifest.get("@id") or ""

    # ── Title ────────────────────────────────────────────────────────────────
    raw_label = _iiif_label(manifest.get("label", {}))
    title = "" if _looks_like_id(raw_label) else raw_label

    # ── Flat metadata lookup (case-insensitive key) ──────────────────────────
    meta: dict[str, list[str]] = {}
    for item in manifest.get("metadata", []):
        key = _iiif_label(item.get("label", {})).lower()
        vals_obj = item.get("value", {})
        if isinstance(vals_obj, dict):
            raw = next(iter(vals_obj.values()), [])
        elif isinstance(vals_obj, list):
            raw = vals_obj
        else:
            raw = [str(vals_obj)]
        vals = [_strip_html(v) for v in (raw if isinstance(raw, list) else [raw])]
        meta[key] = vals

    # Override title from explicit metadata field when manifest label is an id
    if not title:
        for k in ("title", "Title"):
            if k.lower() in meta:
                title = meta[k.lower()][0]
                break

    # ── Date ─────────────────────────────────────────────────────────────────
    date = ""
    for k in ("date issued", "dates / origin", "date"):
        if k in meta:
            v = meta[k][0]
            # NYPL wraps as "Date Issued: 1903" — strip the label prefix
            if ":" in v:
                v = v.split(":", 1)[1].strip()
            date = v
            break

    # ── Institution & item homepage ───────────────────────────────────────────
    institution = ""
    homepage_url = ""

    provider = manifest.get("provider", [])
    if provider:
        p = provider[0]
        institution = _iiif_label(p.get("label", {}))
        ph = p.get("homepage", [])
        if ph:
            homepage_url = ph[0].get("id", "") if isinstance(ph[0], dict) else ph[0]

    # Item-level homepage overrides provider homepage
    item_hp = manifest.get("homepage", [])
    if item_hp:
        h = item_hp[0]
        homepage_url = h.get("id", homepage_url) if isinstance(h, dict) else h

    # Detect institution from manifest URI when provider is absent
    if not institution:
        if "nypl.org" in manifest_id:
            institution = "The New York Public Library"
        elif "loc.gov" in manifest_id:
            institution = "Library of Congress"
        elif "archive.org" in manifest_id:
            institution = "Internet Archive"

    # For LC: construct item page URL from manifest ID
    if "loc.gov" in manifest_id and not homepage_url:
        m_loc = re.match(r"(https://www\.loc\.gov/item/[^/]+)/", manifest_id)
        if m_loc:
            homepage_url = m_loc.group(1) + "/"

    # For NYPL: construct Digital Collections link from UUID if missing
    if "nypl.org" in manifest_id and not homepage_url:
        m_nypl = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                           manifest_id, re.I)
        if m_nypl:
            homepage_url = f"https://digitalcollections.nypl.org/items/{m_nypl.group(1)}"

    # ── Collection ───────────────────────────────────────────────────────────
    collection = ""
    for k in ("collection",):
        if k in meta:
            for c in meta[k]:
                # Skip slugs: must contain a space or non-alphanumeric char
                # (filters out IA slugs like "ColumbiaUniversityLibraries")
                if len(c) > 10 and re.search(r"[^a-zA-Z0-9]", c):
                    collection = c
                    break
            if collection:
                break

    # ── Genre / resource type ────────────────────────────────────────────────
    genre = ""
    for k in ("genres", "resource type", "mediatype"):
        if k in meta:
            genre = meta[k][0]
            break

    return {
        "title": title,
        "date": date,
        "institution": institution,
        "homepage_url": homepage_url,
        "collection": collection,
        "genre": genre,
    }


def _extract_loc_item_json_meta(item_json_path: Path) -> dict:
    """Extract display metadata from a saved LC item JSON (/?fo=json response).

    download_images.py saves this file alongside manifest.json for LC items.
    """
    try:
        data = json.loads(item_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    item = data.get("item", {})
    homepage_url = data.get("url") or f"https://www.loc.gov/item/{item_json_path.parent.name}/"
    date = str(item.get("date", ""))
    fmt = item.get("format") or item.get("type", [])
    genre = fmt[0] if isinstance(fmt, list) and fmt else str(fmt) if fmt else ""
    return {
        "title": item.get("title", ""),
        "date": date,
        "institution": "Library of Congress",
        "homepage_url": homepage_url,
        "collection": "",
        "genre": genre,
    }


def _find_item_meta(search_root: Path) -> dict:
    """Find the first manifest.json under search_root and extract item metadata.

    For LC items, also checks for item.json saved by download_images.py.
    """
    for manifest_path in sorted(search_root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta = _extract_item_meta(manifest)

            # For LC the synthetic manifest label is just the item ID;
            # check for item.json saved by the download step.
            if not meta.get("title") or _looks_like_id(meta.get("title", "")):
                item_json_path = manifest_path.parent / "item.json"
                if item_json_path.exists():
                    lc_meta = _extract_loc_item_json_meta(item_json_path)
                    if lc_meta.get("title"):
                        meta.update({k: v for k, v in lc_meta.items() if v})

            if meta.get("title") or meta.get("homepage_url"):
                return meta
        except Exception:
            continue
    return {"title": "", "date": "", "institution": "", "homepage_url": "",
            "collection": "", "genre": ""}


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

# Columns that are internal/provenance — excluded from facets and search display
_ID_FIELDS = {"canvas_fragment", "image", "line_text"}

# Max cardinality for a column to be treated as a facet (checkbox filter + chart)
_FACET_MAX_CARDINALITY = 50
# Min fill rate (fraction non-empty) for a facet to be shown
_FACET_MIN_FILL = 0.02
# Max top values to embed for a facet
_FACET_TOP_N = 25


def _classify_fields(rows: list[dict]) -> list[dict]:
    """Return field metadata list, one dict per CSV column.

    Each dict: {name, fill_rate, cardinality, type, top_values}
    type is "id" | "facet" | "search"
    top_values is a list of [value, count] pairs (for facet fields only)
    """
    if not rows:
        return []
    all_cols = list(rows[0].keys())
    n = len(rows)
    meta = []
    for col in all_cols:
        values = [r.get(col, "") for r in rows]
        nonempty = [v for v in values if v.strip()]
        fill_rate = len(nonempty) / n if n else 0
        counts = Counter(nonempty)
        cardinality = len(counts)

        if col in _ID_FIELDS:
            field_type = "id"
            top_values = []
        elif cardinality <= _FACET_MAX_CARDINALITY and fill_rate >= _FACET_MIN_FILL:
            field_type = "facet"
            top_values = [[v, c] for v, c in counts.most_common(_FACET_TOP_N)]
        else:
            field_type = "search"
            top_values = []

        meta.append({
            "name": col,
            "fill_rate": round(fill_rate, 4),
            "cardinality": cardinality,
            "type": field_type,
            "top_values": top_values,
        })
    return meta


# ---------------------------------------------------------------------------
# IIIF canvas map (reused from map_entries.py)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline Data Explorer — {title_json_safe}</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/dist/plot.umd.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; height: 100%; font-family: system-ui, sans-serif; font-size: 14px; background: #f8f8f6; color: #222; }}
a {{ color: #1a6ebd; }}
#app {{ display: flex; height: 100dvh; overflow: hidden; }}

/* ── Sidebar ── */
#sidebar {{ width: 240px; min-width: 200px; max-width: 300px; background: #fff; border-right: 1px solid #ddd; display: flex; flex-direction: column; overflow: hidden; }}
#sidebar-header {{ padding: 14px 16px 10px; border-bottom: 1px solid #eee; }}
#sidebar-header h1 {{ margin: 0 0 5px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: #1a6ebd; }}
#sidebar-header h2 {{ margin: 0 0 2px; font-size: 14px; font-weight: 700; line-height: 1.3; word-break: break-word; }}
#sidebar-header .sub {{ font-size: 12px; color: #595959; }}
#volume-wrap {{ margin-top: 10px; display: none; background: #eef3fb; border: 1px solid #5a86b8; border-radius: 5px; padding: 8px 10px; }}
#volume-select {{ width: 100%; font-size: 12px; padding: 4px 6px; border: 1px solid #767676; border-radius: 4px; background: #fff; color: #333; cursor: pointer; }}
#sidebar-scroll {{ overflow-y: auto; flex: 1; padding: 10px 14px 20px; }}

.facet-group {{ margin-bottom: 14px; }}
.facet-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: #555; margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center; }}
.facet-clear {{ font-size: 11px; font-weight: 400; text-transform: none; color: #1a6ebd; cursor: pointer; letter-spacing: 0; background: none; border: none; padding: 0; }}
.facet-item {{ display: flex; align-items: center; gap: 6px; margin-bottom: 3px; cursor: pointer; user-select: none; }}
.facet-item input[type=checkbox] {{ accent-color: #1a6ebd; cursor: pointer; flex-shrink: 0; }}
.facet-item .val {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 13px; }}
.facet-item .cnt {{ font-size: 11px; color: #595959; flex-shrink: 0; }}
.facet-more {{ font-size: 11px; color: #595959; margin-top: 3px; cursor: pointer; background: none; border: none; padding: 0; text-align: left; }}
.facet-more:hover {{ color: #1a6ebd; }}

/* ── Main panel ── */
#main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
#toolbar {{ padding: 10px 16px; border-bottom: 1px solid #ddd; background: #fff; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
#search {{ flex: 1; min-width: 180px; padding: 6px 10px; border: 1px solid #767676; border-radius: 4px; font-size: 13px; }}
#count-label {{ font-size: 12px; color: #666; white-space: nowrap; }}
#export-btn {{ padding: 5px 12px; background: #1a6ebd; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; white-space: nowrap; }}
#export-btn:hover {{ background: #155a9e; }}
#meta-strip {{ margin-left: auto; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; flex-shrink: 0; }}
.meta-chip {{ font-size: 11px; background: #f0f0ee; border: 1px solid #ddd; border-radius: 10px; padding: 2px 8px; color: #555; white-space: nowrap; }}
.meta-link {{ color: #1a6ebd; text-decoration: none; }}
.meta-link:hover {{ background: #e0e8ff; border-color: #1a6ebd; }}

#charts-row {{ padding: 10px 16px 0; display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; border-bottom: 1px solid #eee; background: #fff; }}
#charts-row.collapsed {{ display: none; }}
#charts-toggle {{ padding: 4px 10px; background: none; border: 1px solid #767676; border-radius: 4px; cursor: pointer; font-size: 12px; color: #595959; white-space: nowrap; }}
#charts-toggle:hover {{ border-color: #222; color: #222; }}
.chart-block {{ display: flex; flex-direction: column; }}
.chart-block h3 {{ margin: 0 0 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: #555; }}

thead tr.filter-row th {{ position: sticky; top: 33px; background: #f8f8f6; padding: 3px 6px; border-bottom: 1px solid #ddd; }}
.col-filter-input {{ width: 100%; font-size: 11px; padding: 2px 5px; border: 1px solid #767676; border-radius: 3px; box-sizing: border-box; }}
.col-filter-input:focus {{ outline: 2px solid #1a6ebd; outline-offset: 0; border-color: #1a6ebd; }}

#content {{ flex: 1; display: flex; overflow: hidden; }}

/* ── Table ── */
#table-wrap {{ flex: 1; overflow-y: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
thead th {{ position: sticky; top: 0; background: #f0f0ee; border-bottom: 2px solid #ddd; padding: 7px 10px; text-align: left; font-weight: 600; white-space: nowrap; cursor: pointer; user-select: none; }}
thead th:hover {{ background: #e4e4e2; }}
thead th.sorted-asc::after {{ content: " ↑"; color: #1a6ebd; }}
thead th.sorted-desc::after {{ content: " ↓"; color: #1a6ebd; }}
tbody tr {{ border-bottom: 1px solid #eee; cursor: pointer; }}
tbody tr:hover {{ background: #f5f8ff; }}
tbody tr.selected {{ background: #ddeeff; }}
td {{ padding: 6px 10px; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
td.source-cell {{ text-align: center; }}
.src-btn {{ background: #f0f0f0; border: 1px solid #767676; border-radius: 3px; padding: 2px 7px; cursor: pointer; font-size: 11px; color: #333; white-space: nowrap; }}
.src-btn:hover {{ background: #fff; color: #1a6ebd; border-color: #1a6ebd; }}
#no-results {{ padding: 40px; text-align: center; color: #595959; font-size: 14px; display: none; }}

/* ── Detail panel ── */
#detail {{ width: 300px; min-width: 260px; max-width: 360px; background: #fff; border-left: 1px solid #ddd; overflow-y: auto; display: none; }}
#detail.open {{ display: block; }}
#detail-inner {{ padding: 14px; }}
#detail-inner h2 {{ margin: 0 0 10px; font-size: 14px; font-weight: 700; line-height: 1.3; }}
#detail-thumb {{ width: 100%; border-radius: 4px; margin-bottom: 10px; display: block; border: 1px solid #ddd; }}
.detail-fields {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; font-size: 12px; }}
.df-key {{ color: #666; font-weight: 600; white-space: nowrap; }}
.df-val {{ word-break: break-word; }}
.detail-source {{ margin-top: 12px; }}
.detail-source a {{ font-size: 12px; }}
#detail-close {{ float: right; background: none; border: none; font-size: 18px; cursor: pointer; color: #595959; line-height: 1; margin: -4px -4px 0 0; }}
#detail-close:hover {{ color: #333; }}
#mobile-header {{ display: none; }}

/* ── Mobile (≤640 px) ───────────────────────────────────────────────────── */
@media (max-width: 640px) {{
  #sidebar {{
    position: fixed; top: 0; left: 0;
    width: 85vw; max-width: 300px; height: 100dvh;
    z-index: 200; min-width: 0;
    transform: translateX(-100%);
    transition: transform 0.22s ease;
    box-shadow: 2px 0 12px rgba(0,0,0,0.18);
  }}
  #app.sidebar-open #sidebar {{ transform: translateX(0); }}
  #sidebar-backdrop {{
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.35); z-index: 199;
  }}
  #app.sidebar-open #sidebar-backdrop {{ display: block; }}
  #filter-btn {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 10px; background: none;
    border: 1px solid #767676; border-radius: 4px;
    cursor: pointer; font-size: 13px; color: #595959;
    white-space: nowrap; flex-shrink: 0;
  }}
  #table-wrap {{ overflow-x: auto; }}
  #detail {{
    position: fixed; inset: 0;
    width: 100%; max-width: 100%; min-width: 0;
    height: 100dvh; z-index: 300;
    overflow-y: auto; border-left: none;
  }}
  #detail-close {{ font-size: 24px; padding: 4px 8px; }}
  #meta-strip {{ margin-left: 0; order: 3; width: 100%; }}
  #mobile-header {{
    display: block; padding: 8px 12px 6px;
    background: #fff; border-bottom: 1px solid #eee;
  }}
  #mobile-title {{
    font-size: 13px; font-weight: 700; line-height: 1.3;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  #mobile-sub {{ font-size: 11px; color: #595959; margin-bottom: 4px; }}
  #mobile-volume-select {{
    display: none; width: 100%; font-size: 12px;
    padding: 4px 6px; border: 1px solid #767676; border-radius: 4px;
    background: #fff; color: #333; margin-top: 4px;
  }}
  #mobile-volume-select.visible {{ display: block; }}
}}
</style>
</head>
<body>
<div id="app">

<!-- Sidebar -->
<div id="sidebar">
  <div id="sidebar-header">
    <h1><a href="https://github.com/hadro/directory-pipeline" target="_blank" style="color:inherit;text-decoration:none;">Pipeline Data Explorer</a></h1>
    <h2 id="doc-title"></h2>
    <div class="sub" id="doc-sub"></div>
    <div id="volume-wrap">
      <label for="volume-select" style="display:block;font-size:11px;color:#595959;margin-bottom:3px;text-transform:uppercase;letter-spacing:.04em;">Select volume</label>
      <select id="volume-select"></select>
    </div>
  </div>
  <div id="sidebar-scroll">
    <div id="fill-rate-section">
      <div class="facet-label" style="margin-bottom:6px;">Field coverage</div>
      <div id="fill-rate-chart"></div>
    </div>
    <hr style="margin:12px 0; border:none; border-top:1px solid #eee;">
    <div id="facet-controls"></div>
  </div>
</div>
<div id="sidebar-backdrop" aria-hidden="true"></div>

<!-- Main -->
<div id="main">
  <div id="mobile-header">
    <div id="mobile-title"></div>
    <div id="mobile-sub"></div>
    <select id="mobile-volume-select" aria-label="Select volume"></select>
  </div>
  <div id="toolbar">
    <button id="filter-btn" style="display:none" aria-label="Open filters" aria-expanded="false" aria-controls="sidebar">&#9776; Filters</button>
    <input id="search" type="search" placeholder="Search all fields…" autocomplete="off" aria-label="Search all fields">
    <span id="count-label" aria-live="polite" aria-atomic="true"></span>
    <div id="meta-strip"></div>
    <button id="charts-toggle" aria-expanded="true" aria-controls="charts-row">Hide charts ▲</button>
    <button id="export-btn">Export CSV</button>
  </div>
  <div id="charts-row"></div>
  <div id="content">
    <div id="table-wrap">
      <table id="results-table">
        <thead id="table-head"></thead>
        <tbody id="table-body"></tbody>
      </table>
      <div id="no-results">No entries match the current filters.</div>
    </div>
    <div id="detail">
      <div id="detail-inner">
        <button id="detail-close" aria-label="Close detail panel">×</button>
        <h2 id="detail-title"></h2>
        <img id="detail-thumb" alt="" style="display:none">
        <div class="detail-fields" id="detail-fields"></div>
        <div class="detail-source" id="detail-source"></div>
      </div>
    </div>
  </div>
</div>
</div>

<script>
const ALL_INITIAL_ENTRIES = {entries_json};
const VOLUMES     = {volumes_json};
const FIELD_META  = {field_meta_json};
const CANVAS_MAP  = {canvas_map_json};
const DOC_TITLE   = {title_json};
const DOC_META    = {doc_meta_json};
let ALL_ENTRIES = ALL_INITIAL_ENTRIES;

// ── State ──────────────────────────────────────────────────────────────────
const facetState = {{}};   // {{fieldName: Set<value>}}
let searchQuery = "";
let sortCol = null;
let sortDir = 1;  // 1 = asc, -1 = desc
let selectedIdx = null;
let colFilters = {{}};  // fieldName → filter string
const showAllFacets = new Set();  // field names with expanded value list

// ── Boot ───────────────────────────────────────────────────────────────────
document.getElementById("doc-title").textContent = DOC_META.title || DOC_TITLE;
document.getElementById("doc-sub").textContent =
    ALL_ENTRIES.length.toLocaleString() + " entries";
document.getElementById("mobile-title").textContent = DOC_META.title || DOC_TITLE;
document.getElementById("mobile-sub").textContent =
    ALL_ENTRIES.length.toLocaleString() + " entries";

// Metadata strip in top-right toolbar
function renderMetaStrip(meta) {{
  const parts = [];
  if (meta.date)         parts.push(`<span class="meta-chip">${{meta.date}}</span>`);
  if (meta.institution)  parts.push(`<span class="meta-chip">${{meta.institution}}</span>`);
  if (meta.genre)        parts.push(`<span class="meta-chip">${{meta.genre}}</span>`);
  if (meta.homepage_url)
    parts.push(`<a class="meta-chip meta-link" href="${{meta.homepage_url}}" target="_blank">Item page ↗</a>`);
  const strip = document.getElementById("meta-strip");
  if (strip) strip.innerHTML = parts.join("");
}}
renderMetaStrip(DOC_META);

let facetFields = FIELD_META.filter(f => f.type === "facet");
let displayFields = FIELD_META.filter(f => f.type !== "id");

// Init facet state
facetFields.forEach(f => {{ facetState[f.name] = new Set(); }});

// ── Volume selector ─────────────────────────────────────────────────────────
(function () {{
  const volLabels = Object.keys(VOLUMES);
  if (volLabels.length === 0) return;
  const wrap = document.getElementById("volume-wrap");
  const sel  = document.getElementById("volume-select");
  wrap.style.display = "block";
  volLabels.forEach(label => {{
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label + "  (" + VOLUMES[label].rows.length.toLocaleString() + ")";
    sel.appendChild(opt);
  }});
  // Set h1, mobile title, and meta strip to first volume on load
  document.getElementById("doc-title").textContent = volLabels[0];
  document.getElementById("mobile-title").textContent = volLabels[0];
  renderMetaStrip(VOLUMES[volLabels[0]].doc_meta || {{}});

  // Build mobile volume select
  const msel = document.getElementById("mobile-volume-select");
  volLabels.forEach(label => {{
    const opt = document.createElement("option");
    opt.value = label; opt.textContent = label;
    msel.appendChild(opt);
  }});
  msel.classList.add("visible");
  msel.addEventListener("change", () => {{
    sel.value = msel.value;
    sel.dispatchEvent(new Event("change"));
  }});

  sel.addEventListener("change", () => {{
    const vol = VOLUMES[sel.value];
    ALL_ENTRIES = vol.rows;
    facetFields = vol.field_meta.filter(f => f.type === "facet");
    displayFields = vol.field_meta.filter(f => f.type !== "id");
    Object.keys(facetState).forEach(k => delete facetState[k]);
    facetFields.forEach(f => {{ facetState[f.name] = new Set(); }});
    searchQuery = "";
    sortCol = null;
    sortDir = 1;
    selectedIdx = null;
    colFilters = {{}};
    showAllFacets.clear();
    document.getElementById("search").value = "";
    document.getElementById("detail").classList.remove("open");
    document.getElementById("doc-title").textContent = sel.value;
    document.getElementById("doc-sub").textContent =
      ALL_ENTRIES.length.toLocaleString() + " entries";
    document.getElementById("mobile-title").textContent = sel.value;
    document.getElementById("mobile-sub").textContent =
      ALL_ENTRIES.length.toLocaleString() + " entries";
    msel.value = sel.value;
    renderMetaStrip(vol.doc_meta || {{}});
    renderFillRate();
    renderAll();
  }});
}})();

// ── Filtering ──────────────────────────────────────────────────────────────
function getFiltered() {{
  const q = searchQuery.trim().toLowerCase();
  return ALL_ENTRIES.filter(row => {{
    // Search filter
    if (q) {{
      const haystack = displayFields.map(f => (row[f.name] || "")).join(" ").toLowerCase();
      if (!haystack.includes(q)) return false;
    }}
    // Facet filters
    for (const f of facetFields) {{
      const sel = facetState[f.name];
      if (sel.size === 0) continue;
      if (!sel.has(row[f.name] || "")) return false;
    }}
    // Column header filters
    for (const [col, val] of Object.entries(colFilters)) {{
      if (!val) continue;
      if (!(row[col] || "").toLowerCase().includes(val.toLowerCase())) return false;
    }}
    return true;
  }});
}}

// ── Sort ───────────────────────────────────────────────────────────────────
function getSorted(rows) {{
  if (!sortCol) return rows;
  return [...rows].sort((a, b) => {{
    const av = (a[sortCol] || "").toLowerCase();
    const bv = (b[sortCol] || "").toLowerCase();
    return av < bv ? -sortDir : av > bv ? sortDir : 0;
  }});
}}

// ── Canvas page extraction ─────────────────────────────────────────────────
function canvasId(row) {{
  const cf = row.canvas_fragment || "";
  return cf.includes("#") ? cf.split("#")[0] : cf;
}}

// ── Render: table body only (preserves thead / focus) ──────────────────────
function renderTableBody() {{
  const filtered = getFiltered();
  const sorted   = getSorted(filtered);

  document.getElementById("count-label").textContent =
    `${{sorted.length.toLocaleString()}} of ${{ALL_ENTRIES.length.toLocaleString()}} entries`;

  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";
  sorted.slice(0, 2000).forEach(row => {{
    const tr = document.createElement("tr");
    const isSelected = ALL_ENTRIES.indexOf(row) === selectedIdx;
    if (isSelected) tr.classList.add("selected");
    tr.setAttribute("tabindex", "0");
    tr.setAttribute("aria-selected", isSelected ? "true" : "false");
    displayFields.forEach(f => {{
      const td = document.createElement("td");
      td.textContent = row[f.name] || "";
      td.title = row[f.name] || "";
      tr.appendChild(td);
    }});
    const tdSrc = document.createElement("td");
    tdSrc.className = "source-cell";
    if (row.canvas_fragment) {{
      const btn = document.createElement("button");
      btn.className = "src-btn";
      btn.textContent = "▶ source";
      btn.title = row.canvas_fragment;
      btn.addEventListener("click", e => {{
        e.stopPropagation();
        openSource(row.canvas_fragment);
      }});
      tdSrc.appendChild(btn);
    }}
    tr.appendChild(tdSrc);
    const rowHandler = () => {{
      selectedIdx = ALL_ENTRIES.indexOf(row);
      showDetail(row);
      renderTableBody();
    }};
    tr.addEventListener("click", rowHandler);
    tr.addEventListener("keydown", e => {{
      if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); rowHandler(); }}
    }});
    tbody.appendChild(tr);
  }});

  const noResults = document.getElementById("no-results");
  noResults.style.display = sorted.length === 0 ? "block" : "none";
}}

// ── Render: table head (sort row + filter row) ──────────────────────────────
function renderTableHead() {{
  const head = document.getElementById("table-head");
  head.innerHTML = "";

  // Sort row
  const tr = document.createElement("tr");
  displayFields.forEach(f => {{
    const th = document.createElement("th");
    th.textContent = f.name.replace(/_/g, " ");
    th.dataset.col = f.name;
    th.setAttribute("tabindex", "0");
    if (sortCol === f.name) {{
      th.className = sortDir > 0 ? "sorted-asc" : "sorted-desc";
      th.setAttribute("aria-sort", sortDir > 0 ? "ascending" : "descending");
    }} else {{
      th.setAttribute("aria-sort", "none");
    }}
    const sortHandler = () => {{
      if (sortCol === f.name) sortDir = -sortDir;
      else {{ sortCol = f.name; sortDir = 1; }}
      renderTableHead();
      renderTableBody();
    }};
    th.addEventListener("click", sortHandler);
    th.addEventListener("keydown", e => {{
      if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); sortHandler(); }}
    }});
    tr.appendChild(th);
  }});
  const thSrc = document.createElement("th");
  thSrc.textContent = "Source";
  tr.appendChild(thSrc);
  head.appendChild(tr);

  // Filter row — inputs call renderTableBody only, preserving focus
  const filterRow = document.createElement("tr");
  filterRow.className = "filter-row";
  displayFields.forEach(f => {{
    const fth = document.createElement("th");
    if (f.type === "facet" && f.top_values.length > 0) {{
      const sel = document.createElement("select");
      sel.className = "col-filter-input";
      sel.setAttribute("aria-label", `Filter by ${{f.name.replace(/_/g, " ")}}`);
      const opt0 = document.createElement("option");
      opt0.value = ""; opt0.textContent = "(all)";
      sel.appendChild(opt0);
      f.top_values.forEach(([val]) => {{
        const opt = document.createElement("option");
        opt.value = val; opt.textContent = val || "(empty)";
        if (colFilters[f.name] === val) opt.selected = true;
        sel.appendChild(opt);
      }});
      sel.addEventListener("change", () => {{ colFilters[f.name] = sel.value; renderTableBody(); }});
      fth.appendChild(sel);
    }} else {{
      const inp = document.createElement("input");
      inp.type = "text"; inp.className = "col-filter-input"; inp.placeholder = "filter\u2026";
      inp.setAttribute("aria-label", `Filter by ${{f.name.replace(/_/g, " ")}}`);
      inp.value = colFilters[f.name] || "";
      inp.addEventListener("input", () => {{ colFilters[f.name] = inp.value; renderTableBody(); }});
      fth.appendChild(inp);
    }}
    filterRow.appendChild(fth);
  }});
  filterRow.appendChild(document.createElement("th"));  // Source column
  head.appendChild(filterRow);
}}

// ── Render: table (head + body) ─────────────────────────────────────────────
function renderTable() {{
  renderTableHead();
  renderTableBody();
}}

// ── Resolve canvas_fragment → human-browseable viewer URL ──────────────────
function viewerUrl(canvas_fragment) {{
  if (!canvas_fragment) return "";
  const cid = canvas_fragment.includes("#") ? canvas_fragment.split("#")[0] : canvas_fragment;

  // CANVAS_MAP entry[3] is a pre-built viewer URL (e.g. NYPL Digital Collections)
  const entry = CANVAS_MAP[cid];
  if (entry && entry[3]) return entry[3];

  // Internet Archive: https://iiif.archive.org/iiif/{{identifier}}${{page}}/canvas
  const iaMatch = cid.match(/https:\\/\\/iiif\\.archive\\.org\\/iiif\\/([^$]+)\\$(\\d+)\\/canvas/);
  if (iaMatch) {{
    return `https://archive.org/details/${{iaMatch[1]}}/page/n${{iaMatch[2]}}/mode/2up`;
  }}

  // Library of Congress: https://www.loc.gov/item/{{id}}/canvas/{{num}}
  const locMatch = cid.match(/https:\\/\\/www\\.loc\\.gov\\/item\\/([^/]+)\\/canvas\\/(\\d+)/);
  if (locMatch) {{
    return `https://www.loc.gov/item/${{locMatch[1]}}/?sp=${{parseInt(locMatch[2]) + 1}}`;
  }}

  // Fall back to the raw canvas URI
  return canvas_fragment;
}}

function openSource(canvas_fragment) {{
  const url = viewerUrl(canvas_fragment);
  if (url) window.open(url, "_blank");
}}

// ── Detail panel ───────────────────────────────────────────────────────────
function thumbUrl(row) {{
  const cf = row.canvas_fragment || "";
  if (!cf) return "";
  const cid = cf.includes("#") ? cf.split("#")[0] : cf;
  const svc = CANVAS_MAP[cid];
  if (!svc) return "";
  if (cf.includes("#xywh=")) {{
    const xywh = cf.split("#xywh=")[1].split(",").map(Number);
    if (xywh.length !== 4 || xywh[2] <= 0 || xywh[3] <= 0) return "";
    const [cw, ch] = [svc[1], svc[2]];
    if (!cw || !ch) return "";
    const pad = 40, tw = 400;
    const rx = Math.max(0, xywh[0] - pad);
    const ry = Math.max(0, xywh[1] - pad);
    const rw = (xywh[0] + xywh[2] + pad) - rx;
    const rh = (xywh[1] + xywh[3] + pad) - ry;
    const xp = (rx / cw * 100).toFixed(4);
    const yp = (ry / ch * 100).toFixed(4);
    const wp = (rw / cw * 100).toFixed(4);
    const hp = (rh / ch * 100).toFixed(4);
    return `${{svc[0]}}/pct:${{xp}},${{yp}},${{wp}},${{hp}}/${{tw}},/0/default.jpg`;
  }}
  return `${{svc[0]}}/full/400,/0/default.jpg`;
}}

function showDetail(row) {{
  const panel = document.getElementById("detail");
  panel.classList.add("open");

  const nameField = displayFields.find(f => f.name.includes("name") || f.name.includes("title"));
  document.getElementById("detail-title").textContent =
    (nameField && row[nameField.name]) || "(entry)";

  const img = document.getElementById("detail-thumb");
  const url = thumbUrl(row);
  if (url) {{
    img.src = url;
    img.style.display = "block";
    img.onerror = () => {{ img.style.display = "none"; }};
  }} else {{
    img.style.display = "none";
  }}

  const fields = document.getElementById("detail-fields");
  fields.innerHTML = "";
  displayFields.forEach(f => {{
    const val = row[f.name] || "";
    if (!val) return;
    const k = document.createElement("div");
    k.className = "df-key";
    k.textContent = f.name.replace(/_/g, " ");
    const v = document.createElement("div");
    v.className = "df-val";
    v.textContent = val;
    fields.appendChild(k);
    fields.appendChild(v);
  }});

  const src = document.getElementById("detail-source");
  if (row.canvas_fragment) {{
    src.innerHTML = `<a href="${{viewerUrl(row.canvas_fragment)}}" target="_blank">Open source page ↗</a>`;
  }} else {{
    src.innerHTML = "";
  }}
}}

document.getElementById("detail-close").addEventListener("click", () => {{
  document.getElementById("detail").classList.remove("open");
  selectedIdx = null;
  renderTable();
}});

// ── Render: facet sidebar ──────────────────────────────────────────────────
function renderFacetSidebar() {{
  const container = document.getElementById("facet-controls");
  container.innerHTML = "";
  facetFields.forEach(f => {{
    const grp = document.createElement("div");
    grp.className = "facet-group";

    const lbl = document.createElement("div");
    lbl.className = "facet-label";
    const labelText = document.createElement("span");
    labelText.textContent = f.name.replace(/_/g, " ");
    lbl.appendChild(labelText);
    if (facetState[f.name].size > 0) {{
      const clr = document.createElement("button");
      clr.className = "facet-clear";
      clr.textContent = "clear";
      clr.setAttribute("aria-label", `Clear ${{f.name.replace(/_/g, " ")}} filter`);
      clr.addEventListener("click", () => {{
        facetState[f.name].clear();
        renderAll();
      }});
      lbl.appendChild(clr);
    }}
    grp.appendChild(lbl);

    const showAll = showAllFacets.has(f.name);
    const topN = showAll ? f.top_values.length : Math.min(10, f.top_values.length);
    f.top_values.slice(0, topN).forEach(([val, cnt]) => {{
      const item = document.createElement("label");
      item.className = "facet-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = facetState[f.name].has(val);
      cb.addEventListener("change", () => {{
        if (cb.checked) facetState[f.name].add(val);
        else facetState[f.name].delete(val);
        renderAll();
      }});
      const valSpan = document.createElement("span");
      valSpan.className = "val";
      valSpan.textContent = val || "(empty)";
      valSpan.title = val;
      const cntSpan = document.createElement("span");
      cntSpan.className = "cnt";
      cntSpan.textContent = cnt.toLocaleString();
      item.appendChild(cb);
      item.appendChild(valSpan);
      item.appendChild(cntSpan);
      grp.appendChild(item);
    }});

    if (!showAll && f.top_values.length > 10) {{
      const more = document.createElement("button");
      more.className = "facet-more";
      more.textContent = `+ ${{f.top_values.length - 10}} more…`;
      more.setAttribute("aria-label", `Show ${{f.top_values.length - 10}} more ${{f.name.replace(/_/g, " ")}} values`);
      more.addEventListener("click", () => {{
        showAllFacets.add(f.name);
        renderFacetSidebar();
      }});
      grp.appendChild(more);
    }}
    container.appendChild(grp);
  }});
}}

// ── Render: fill-rate chart ────────────────────────────────────────────────
function renderFillRate() {{
  const el = document.getElementById("fill-rate-chart");
  const data = FIELD_META
    .filter(f => f.type !== "id")
    .map(f => ({{ name: f.name.replace(/_/g, " "), fill: f.fill_rate }}))
    .sort((a, b) => b.fill - a.fill);

  const plot = Plot.plot({{
    width: 200, height: data.length * 16 + 20,
    marginLeft: 100, marginRight: 10, marginTop: 4, marginBottom: 16,
    x: {{ domain: [0, 1], label: null, tickFormat: "%", ticks: 2 }},
    y: {{ label: null }},
    marks: [
      Plot.barX(data, {{
        x: "fill", y: "name",
        fill: d => d.fill >= 0.8 ? "#59a14f" : d.fill >= 0.4 ? "#f28e2b" : "#e15759",
        sort: {{ y: "x", reverse: true }}
      }}),
      Plot.text(data, {{
        x: "fill", y: "name",
        text: d => (d.fill * 100).toFixed(0) + "%",
        dx: 4, frameAnchor: "left",
        fontSize: 9, fill: "#444",
      }}),
      Plot.ruleX([0]),
    ],
    style: {{ fontSize: "11px" }},
  }});
  el.innerHTML = "";
  el.appendChild(plot);
}}

// ── Render: facet bar charts ───────────────────────────────────────────────
function renderCharts() {{
  const row = document.getElementById("charts-row");
  row.innerHTML = "";

  const filtered = getFiltered();

  facetFields.slice(0, 4).forEach(f => {{
    // Count filtered values
    const counts = {{}};
    filtered.forEach(r => {{
      const v = r[f.name] || "(empty)";
      counts[v] = (counts[v] || 0) + 1;
    }});
    const data = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .map(([v, c]) => ({{ v, c }}));

    if (data.length === 0) return;

    const block = document.createElement("div");
    block.className = "chart-block";
    const h3 = document.createElement("h3");
    h3.textContent = f.name.replace(/_/g, " ");
    block.appendChild(h3);

    const barHeight = 18;
    const plot = Plot.plot({{
      width: 220,
      height: data.length * barHeight + 24,
      marginLeft: 110, marginRight: 30, marginTop: 2, marginBottom: 20,
      x: {{ label: null, grid: false }},
      y: {{ label: null, tickSize: 0 }},
      marks: [
        Plot.barX(data, {{
          x: "c", y: "v",
          fill: d => facetState[f.name].has(d.v) ? "#1a6ebd" : "#aac4e6",
          sort: {{ y: "x", reverse: true }},
          title: d => `${{d.v}}: ${{d.c}}`,
          rx: 2,
        }}),
        Plot.text(data, {{
          x: "c", y: "v", text: d => d.c.toString(),
          dx: 4, frameAnchor: "left",
          fontSize: 10, fill: "#555",
        }}),
      ],
      style: {{ fontSize: "11px", cursor: "pointer" }},
    }});

    // Make bars clickable as filters
    plot.addEventListener("click", e => {{
      const rect = e.target.closest("rect");
      if (!rect) return;
      const titleEl = rect.querySelector("title");
      const title = titleEl ? titleEl.textContent : "";
      const val = title.split(":")[0].trim();
      if (!val || val === "(empty)") return;
      if (facetState[f.name].has(val)) facetState[f.name].delete(val);
      else facetState[f.name].add(val);
      renderAll();
    }});

    // Make bars keyboard-accessible
    plot.querySelectorAll("rect").forEach(rect => {{
      const titleEl = rect.querySelector("title");
      if (!titleEl) return;
      rect.setAttribute("tabindex", "0");
      rect.setAttribute("role", "button");
      rect.setAttribute("aria-label", titleEl.textContent);
      rect.addEventListener("keydown", e => {{
        if (e.key === "Enter" || e.key === " ") {{
          e.preventDefault();
          rect.dispatchEvent(new MouseEvent("click", {{ bubbles: true }}));
        }}
      }});
    }});
    block.appendChild(plot);
    row.appendChild(block);
  }});
}}

// ── Export CSV ─────────────────────────────────────────────────────────────
document.getElementById("export-btn").addEventListener("click", () => {{
  const filtered = getSorted(getFiltered());
  const cols = FIELD_META.map(f => f.name);
  const lines = [cols.map(c => JSON.stringify(c || "")).join(",")];
  filtered.forEach(row => {{
    lines.push(cols.map(c => JSON.stringify(row[c] || "")).join(","));
  }});
  const blob = new Blob([lines.join("\\n")], {{ type: "text/csv" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (DOC_TITLE || "entries").replace(/\\s+/g, "_") + "_filtered.csv";
  a.click();
}});

// ── Charts toggle ──────────────────────────────────────────────────────────
document.getElementById("charts-toggle").addEventListener("click", () => {{
  const row = document.getElementById("charts-row");
  const btn = document.getElementById("charts-toggle");
  const collapsed = row.classList.toggle("collapsed");
  btn.textContent = collapsed ? "Show charts ▼" : "Hide charts ▲";
  btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
}});

// ── Mobile sidebar toggle ──────────────────────────────────────────────────
(function () {{
  const filterBtn = document.getElementById("filter-btn");
  const backdrop  = document.getElementById("sidebar-backdrop");
  const app       = document.getElementById("app");
  function openSidebar()  {{ app.classList.add("sidebar-open");    filterBtn.setAttribute("aria-expanded", "true");  }}
  function closeSidebar() {{ app.classList.remove("sidebar-open"); filterBtn.setAttribute("aria-expanded", "false"); }}
  filterBtn.addEventListener("click", () => app.classList.contains("sidebar-open") ? closeSidebar() : openSidebar());
  backdrop.addEventListener("click", closeSidebar);
  document.getElementById("facet-controls").addEventListener("change", () => {{
    if (window.innerWidth <= 640) closeSidebar();
  }});
}})();

// ── Search input ───────────────────────────────────────────────────────────
let searchTimer;
document.getElementById("search").addEventListener("input", e => {{
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {{
    searchQuery = e.target.value;
    renderAll();
  }}, 180);
}});

// ── Render all ─────────────────────────────────────────────────────────────
function renderAll() {{
  renderFacetSidebar();
  renderCharts();
  renderTable();
}}

renderFillRate();
if (window.innerWidth <= 640) {{
  document.getElementById("charts-row").classList.add("collapsed");
  const _ctBtn = document.getElementById("charts-toggle");
  _ctBtn.textContent = "Show charts \u25BC";
  _ctBtn.setAttribute("aria-expanded", "false");
}}
renderAll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Build canvas thumbnail map (for detail panel)
# ---------------------------------------------------------------------------

def _build_canvas_thumb_map(
    search_root: Path,
) -> dict[str, list]:
    """Return {canvas_id: [service_id, canvas_w, canvas_h, viewer_url]} for JS.

    viewer_url is a human-browseable page for sources that have one
    (currently NYPL Digital Collections). IA and LoC viewer URLs are derived
    from the canvas URI itself in the JS viewerUrl() function.
    """
    mapping: dict[str, list] = {}
    for manifest_path in search_root.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        item_uuid = manifest_path.parent.name
        for i, canvas in enumerate(iiif_utils.iter_canvases(manifest)):
            svc_id = canvas["service_id"]
            viewer_url = ""
            if svc_id.startswith("https://iiif.nypl.org/"):
                viewer_url = (
                    f"https://digitalcollections.nypl.org/items/{item_uuid}"
                    f"?canvasIndex={i}"
                )
            mapping[canvas["canvas_id"]] = [
                svc_id,
                canvas["canvas_width"],
                canvas["canvas_height"],
                viewer_url,
            ]

    # Override canvas dimensions from aligned JSON coordinate space
    for aligned_path in search_root.rglob("*_aligned.json"):
        try:
            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        canvas_id = aligned.get("canvas_uri") or aligned.get("canvas_id") or ""
        aw = int(aligned.get("canvas_width") or 0)
        ah = int(aligned.get("canvas_height") or 0)
        if canvas_id and aw and ah and canvas_id in mapping:
            mapping[canvas_id][1] = aw
            mapping[canvas_id][2] = ah

    return mapping


# ---------------------------------------------------------------------------
# Find CSV
# ---------------------------------------------------------------------------

def _find_csv(path: Path, slug: str | None) -> Path:
    if path.is_file():
        return path
    if slug:
        hits = sorted(path.rglob(f"entries_{slug}.csv"))
        if hits:
            return hits[0]
    # Auto-discover any entries_*.csv (excluding geocoded)
    hits = [h for h in sorted(path.rglob("entries_*.csv"))
            if "_geocoded" not in h.name and "_explorer" not in h.name]
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"No entries_*.csv found under {path}. Run --extract-entries first."
    )


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _safe_json(obj: object) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("</script>", r"<\/script>").replace("<!--", r"<\!--")


def build_html(
    rows: list[dict],
    field_meta: list[dict],
    canvas_map: dict,
    title: str,
    doc_meta: dict | None = None,
    volumes: dict[str, list[dict]] | None = None,
) -> str:
    return _HTML_TEMPLATE.format(
        entries_json     = _safe_json(list(rows)),
        field_meta_json  = _safe_json(field_meta),
        canvas_map_json  = _safe_json(canvas_map),
        title_json       = _safe_json(title),
        title_json_safe  = title.replace('"', '&quot;'),
        doc_meta_json    = _safe_json(doc_meta or {}),
        volumes_json     = _safe_json(
            {k: {"rows": list(v["rows"]), "field_meta": v["field_meta"], "doc_meta": v.get("doc_meta", {})}
             for k, v in (volumes or {}).items()}
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an interactive field-value explorer from an entries CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help="Path to an entries CSV, or a directory containing one.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        metavar="MODEL",
        help="Model slug in the CSV filename. Auto-detected from entries_*.csv if omitted.",
    )
    parser.add_argument(
        "--out", "-o",
        metavar="FILE",
        help="Output HTML path (default: <csv_stem>_explorer.html next to the CSV)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Root of the images directory — used to find IIIF manifests for thumbnails. "
             "Defaults to the CSV's parent directory.",
    )
    parser.add_argument(
        "--title",
        default=None,
        metavar="TEXT",
        help="Document title shown in the explorer header (auto-derived from path if omitted)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # Auto-detect model slug
    if args.model is None:
        src = Path(args.source)
        candidates = [h for h in sorted((src.rglob("entries_*.csv") if src.is_dir() else [src]))
                      if "_geocoded" not in h.name and "_explorer" not in h.name]
        for c in candidates:
            m = re.match(r"entries_(.+)\.csv", c.name)
            if m:
                args.model = m.group(1)
                break

    slug = args.model.replace("/", "_") if args.model else None

    # Find all matching CSVs — a collection run produces one per item subdirectory.
    src = Path(args.source)
    if src.is_file():
        all_csvs = [src]
    elif slug:
        all_csvs = sorted(src.rglob(f"entries_{slug}.csv"))
    else:
        all_csvs = [h for h in sorted(src.rglob("entries_*.csv"))
                    if "_geocoded" not in h.name and "_explorer" not in h.name]
    if not all_csvs:
        print(
            f"No entries_*.csv found under {src}. Run --extract-entries first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    csv_path = all_csvs[0]  # used for title/meta fallback
    collection_mode = len(all_csvs) > 1

    # For a collection, write the explorer one level up (the slug directory),
    # not inside an individual item subdirectory.
    if args.out:
        out_path = Path(args.out)
    elif collection_mode:
        common_dir = Path(os.path.commonpath([str(c.parent) for c in all_csvs]))
        out_path = common_dir / f"entries_{slug or 'all'}_explorer.html"
    else:
        out_path = csv_path.with_name(csv_path.stem + "_explorer.html")

    # Build per-volume data; each entry carries its own rows + field_meta so the
    # explorer can show correct facets and column sets when switching volumes.
    volumes: dict[str, dict] = {}
    for p in all_csvs:
        with p.open(encoding="utf-8", newline="") as fh:
            vol_rows = list(csv.DictReader(fh))
        if collection_mode:
            item_dir = p.parent
            meta = _find_item_meta(item_dir)
            label: str = meta.get("title") or item_dir.name
            if label in volumes:
                label = f"{label} ({item_dir.name})"
            volumes[label] = {"rows": vol_rows, "field_meta": _classify_fields(vol_rows), "doc_meta": meta}
        else:
            volumes[p.stem] = {"rows": vol_rows, "field_meta": _classify_fields(vol_rows)}

    if collection_mode:
        volumes = dict(sorted(volumes.items(), key=lambda x: x[0].lower()))

    first_vol = next(iter(volumes.values()))
    rows = first_vol["rows"]
    field_meta = first_vol["field_meta"]
    all_rows = [r for vol in volumes.values() for r in vol["rows"]]

    if not args.quiet:
        if collection_mode:
            print(
                f"Loaded {len(all_rows)} entries across {len(all_csvs)} volumes",
                file=sys.stderr,
            )
        else:
            print(f"Loaded {len(rows)} entries from {csv_path.name}", file=sys.stderr)

    # Derive title from path — for a collection use the slug directory name
    title = args.title
    if not title:
        title_root = out_path.parent if collection_mode else csv_path
        for part in reversed(title_root.parts):
            if part not in ("output", ".") and not part.endswith(".csv"):
                title = part.replace("_", " ").strip()
                break
        title = title or csv_path.stem

    # Build IIIF canvas map for thumbnails — search from the slug dir for collections
    output_root = Path(args.output_dir) if args.output_dir else (
        out_path.parent if collection_mode else csv_path.parent
    )
    canvas_map: dict = {}
    doc_meta: dict = {}
    if output_root.is_dir():
        canvas_map = _build_canvas_thumb_map(output_root)
        doc_meta = _find_item_meta(output_root)
        if canvas_map and not args.quiet:
            print(
                f"IIIF: loaded {len(canvas_map)} canvas→service mappings",
                file=sys.stderr,
            )
        elif not canvas_map and not args.quiet:
            print(
                "IIIF: no manifests found — thumbnails disabled. "
                "Pass --output-dir to specify the images directory.",
                file=sys.stderr,
            )

    # Use manifest title when the path-derived title looks like a bare identifier
    if doc_meta.get("title") and _looks_like_id(title):
        title = doc_meta["title"]

    html = build_html(rows, field_meta, canvas_map, title, doc_meta,
                      volumes=volumes if collection_mode else None)
    out_path.write_text(html, encoding="utf-8")
    if not args.quiet:
        print(f"Explorer written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
