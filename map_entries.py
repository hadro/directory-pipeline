#!/usr/bin/env python3
"""Visualize Green Book entries on an interactive map with search & filter.

Reads a pre-geocoded entries CSV produced by geocode_entries.py and builds
a self-contained HTML file with a Leaflet map and a sidebar for filtering
by name/address, state, and category.

Run geocode_entries.py first to produce entries_{slug}_geocoded.csv:
    python geocode_entries.py images/green_book_1962_9ab2e8f0/ --model gemini-2.0-flash

Then build the map:
    python map_entries.py images/green_book_1962_9ab2e8f0/ --model gemini-2.0-flash
    python map_entries.py path/to/entries_gemini-2.0-flash_geocoded.csv
    python map_entries.py path/to/entries.csv --out my_map.html
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "formal_accommodations":   "#4e79a7",
    "informal_accommodations": "#76b7b2",
    "eating_drinking":         "#f28e2b",
    "barber_beauty":           "#e15759",
    "service_station":         "#59a14f",
    "other":                   "#b07aa1",
}
CATEGORY_LABELS = {
    "formal_accommodations":   "Hotels / Motels",
    "informal_accommodations": "Tourist Homes",
    "eating_drinking":         "Restaurants / Bars",
    "barber_beauty":           "Barber / Beauty",
    "service_station":         "Service Stations",
    "other":                   "Other",
}

_JITTER = 0.018   # ~2 km — applied only to city-level fallback coords


# ---------------------------------------------------------------------------
# Read geocoded CSV
# ---------------------------------------------------------------------------

def _prepare_entries(rows: list[dict]) -> tuple[list[dict], int]:
    """Convert geocoded CSV rows to minimal dicts for the map.

    Expects rows to have lat, lon, geocode_level columns (from geocode_entries.py).
    City-level entries get a small random jitter so they don't stack.
    """
    rng = random.Random(42)
    entries = []
    skipped = 0

    for row in rows:
        lat_str = row.get("lat", "")
        lon_str = row.get("lon", "")
        if not lat_str or not lon_str:
            skipped += 1
            continue

        lat = float(lat_str)
        lon = float(lon_str)

        if row.get("geocode_level") == "city":
            lat += rng.uniform(-_JITTER, _JITTER)
            lon += rng.uniform(-_JITTER, _JITTER)

        entries.append({
            "name":     row.get("establishment_name", ""),
            "address":  (row.get("raw_address") or "").strip(),
            "city":     row.get("city", ""),
            "state":    row.get("state", ""),
            "category": row.get("category", "other"),
            "page":     row.get("image", "").replace(".jpg", ""),
            "lat":      round(lat, 6),
            "lon":      round(lon, 6),
        })

    return entries, skipped


# ---------------------------------------------------------------------------
# HTML generation (unchanged)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Green Book Map</title>
<link rel="stylesheet"
  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet"
  href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet"
  href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ display: flex; height: 100vh; font-family: system-ui, -apple-system, sans-serif;
       font-size: 13px; color: #333; }}

/* ── Sidebar ── */
#sidebar {{
  width: 272px; flex-shrink: 0; background: #fff;
  border-right: 1px solid #ddd;
  display: flex; flex-direction: column;
  overflow: hidden;
}}
#sidebar-header {{
  padding: 14px 16px 10px;
  border-bottom: 1px solid #eee;
}}
#sidebar-header h1 {{ font-size: 15px; font-weight: 700; }}
#sidebar-header p  {{ font-size: 11px; color: #888; margin-top: 2px; }}
#sidebar-body {{
  padding: 12px 14px;
  overflow-y: auto;
  flex: 1;
  display: flex; flex-direction: column; gap: 14px;
}}

/* ── Form elements ── */
label.field-label {{
  display: block; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .04em;
  color: #666; margin-bottom: 5px;
}}
input[type=text], select {{
  width: 100%; padding: 6px 8px; font-size: 13px;
  border: 1px solid #ccc; border-radius: 4px;
  outline: none;
}}
input[type=text]:focus, select:focus {{ border-color: #4e79a7; }}

/* ── Category checkboxes ── */
.cat-row {{
  display: flex; align-items: center; gap: 7px;
  padding: 3px 0; cursor: pointer; user-select: none;
}}
.cat-row input {{ flex-shrink: 0; cursor: pointer; }}
.cat-dot {{
  width: 11px; height: 11px; border-radius: 50%;
  flex-shrink: 0;
}}
.cat-label {{ font-size: 13px; }}
#cat-toggle {{
  font-size: 11px; color: #4e79a7; cursor: pointer;
  text-decoration: underline; margin-top: 3px;
  display: inline-block;
}}

/* ── Footer count + actions ── */
#sidebar-footer {{
  border-top: 1px solid #eee; padding: 10px 14px;
  display: flex; flex-direction: column; gap: 6px;
}}
#result-count {{ font-size: 12px; color: #555; }}
#fit-btn {{
  padding: 6px 0; background: #4e79a7; color: #fff;
  border: none; border-radius: 4px; cursor: pointer;
  font-size: 13px; font-weight: 600;
}}
#fit-btn:hover {{ background: #3a6590; }}
#reset-btn {{
  padding: 5px 0; background: #f5f5f5; color: #555;
  border: 1px solid #ddd; border-radius: 4px; cursor: pointer;
  font-size: 12px;
}}
#reset-btn:hover {{ background: #e8e8e8; }}

/* ── Map ── */
#map {{ flex: 1; }}

/* ── Popup ── */
.popup-name  {{ font-weight: 700; font-size: 13px; }}
.popup-addr  {{ color: #444; margin: 2px 0; }}
.popup-cat   {{ color: #777; font-style: italic; font-size: 11px; }}
.popup-page  {{ color: #aaa; font-size: 10px; margin-top: 3px; }}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>Negro Motorist Green Book</h1>
    <p id="subtitle">Loading…</p>
  </div>

  <div id="sidebar-body">

    <div>
      <label class="field-label" for="search">Search</label>
      <input id="search" type="text" placeholder="Name, address, or city…"/>
    </div>

    <div>
      <label class="field-label" for="state-select">State / Territory</label>
      <select id="state-select">
        <option value="">All</option>
      </select>
    </div>

    <div>
      <label class="field-label">Category</label>
      <div id="cat-checks"></div>
      <span id="cat-toggle">Select none</span>
    </div>

  </div>

  <div id="sidebar-footer">
    <div id="result-count">…</div>
    <button id="fit-btn">Fit map to results</button>
    <button id="reset-btn">Reset view (continental US)</button>
  </div>
</div>

<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script>
const ALL_ENTRIES    = {entries_json};
const CAT_COLORS     = {cat_colors_json};
const CAT_LABELS     = {cat_labels_json};
const TOTAL_ENTRIES  = ALL_ENTRIES.length;
const YEAR_LABEL     = {year_label_json};

const map = L.map('map', {{ preferCanvas: true }}).setView([38.5, -96.0], 5);
L.tileLayer(
  'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{ attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 19 }}
).addTo(map);

let clusterGroup = L.markerClusterGroup({{ chunkedLoading: true, maxClusterRadius: 40 }});
map.addLayer(clusterGroup);
let visibleLatLngs = [];

document.getElementById('subtitle').textContent =
  YEAR_LABEL + ' · ' + TOTAL_ENTRIES.toLocaleString() + ' entries';

const stateSet = new Set(ALL_ENTRIES.map(e => e.state));
const stateSelect = document.getElementById('state-select');
[...stateSet].sort().forEach(s => {{
  const opt = document.createElement('option');
  opt.value = s; opt.textContent = s;
  stateSelect.appendChild(opt);
}});

const catChecks = document.getElementById('cat-checks');
Object.keys(CAT_LABELS).forEach(cat => {{
  const row = document.createElement('label');
  row.className = 'cat-row';
  row.innerHTML =
    `<input type="checkbox" class="cat-cb" value="${{cat}}" checked>` +
    `<span class="cat-dot" style="background:${{CAT_COLORS[cat]}}"></span>` +
    `<span class="cat-label">${{CAT_LABELS[cat]}}</span>`;
  catChecks.appendChild(row);
}});

let allChecked = true;
document.getElementById('cat-toggle').addEventListener('click', function() {{
  allChecked = !allChecked;
  document.querySelectorAll('.cat-cb').forEach(cb => cb.checked = allChecked);
  this.textContent = allChecked ? 'Select none' : 'Select all';
  updateMap();
}});

function esc(s) {{
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;')
                  .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function updateMap() {{
  const q        = document.getElementById('search').value.toLowerCase().trim();
  const state    = document.getElementById('state-select').value;
  const activeCats = new Set(
    [...document.querySelectorAll('.cat-cb:checked')].map(cb => cb.value)
  );

  clusterGroup.clearLayers();
  visibleLatLngs = [];

  for (const e of ALL_ENTRIES) {{
    if (state && e.state !== state) continue;
    if (!activeCats.has(e.category)) continue;
    if (q) {{
      const hay = (e.name + ' ' + e.address + ' ' + e.city).toLowerCase();
      if (!hay.includes(q)) continue;
    }}

    const color = CAT_COLORS[e.category] || '#888';
    const marker = L.circleMarker([e.lat, e.lon], {{
      radius: 7, color: color, fillColor: color,
      fillOpacity: 0.85, weight: 1.5
    }});

    marker.bindTooltip(esc(e.name), {{ direction: 'top', offset: [0, -4] }});
    marker.bindPopup(
      `<div class="popup-name">${{esc(e.name)}}</div>` +
      (e.address ? `<div class="popup-addr">${{esc(e.address)}}</div>` : '') +
      `<div class="popup-addr">${{esc(e.city)}}, ${{esc(e.state)}}</div>` +
      `<div class="popup-cat">${{esc(CAT_LABELS[e.category] || e.category)}}</div>` +
      `<div class="popup-page">${{esc(e.page)}}</div>`,
      {{ maxWidth: 280 }}
    );

    clusterGroup.addLayer(marker);
    visibleLatLngs.push([e.lat, e.lon]);
  }}

  const n = visibleLatLngs.length;
  document.getElementById('result-count').textContent =
    n.toLocaleString() + ' of ' + TOTAL_ENTRIES.toLocaleString() + ' entries shown';
}}

document.getElementById('fit-btn').addEventListener('click', () => {{
  if (visibleLatLngs.length === 0) return;
  if (visibleLatLngs.length === 1) {{
    map.setView(visibleLatLngs[0], 16);
  }} else {{
    map.fitBounds(L.latLngBounds(visibleLatLngs), {{ padding: [30, 30] }});
  }}
}});

document.getElementById('reset-btn').addEventListener('click', () => {{
  map.setView([38.5, -96.0], 5);
}});

let _debounce;
document.getElementById('search').addEventListener('input', () => {{
  clearTimeout(_debounce);
  _debounce = setTimeout(updateMap, 180);
}});
document.getElementById('state-select').addEventListener('change', updateMap);
document.querySelectorAll('.cat-cb').forEach(cb =>
  cb.addEventListener('change', () => {{
    const all  = [...document.querySelectorAll('.cat-cb')].every(c => c.checked);
    const none = [...document.querySelectorAll('.cat-cb')].every(c => !c.checked);
    if (all)  {{ allChecked = true;  document.getElementById('cat-toggle').textContent = 'Select none'; }}
    if (none) {{ allChecked = false; document.getElementById('cat-toggle').textContent = 'Select all'; }}
    updateMap();
  }})
);

updateMap();
</script>
</body>
</html>
"""


def build_html(entries: list[dict], title_year: str) -> str:
    def safe_json(obj: object) -> str:
        s = json.dumps(obj, ensure_ascii=False)
        return s.replace("</script>", r"<\/script>").replace("<!--", r"<\!--")

    return _HTML_TEMPLATE.format(
        entries_json    = safe_json(entries),
        cat_colors_json = safe_json(CATEGORY_COLORS),
        cat_labels_json = safe_json(CATEGORY_LABELS),
        year_label_json = safe_json(title_year),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_csv(path: Path, slug: str) -> Path:
    """Prefer the geocoded CSV; fall back to the plain entries CSV."""
    if path.is_file():
        return path
    # Geocoded CSV first
    hits = sorted(path.rglob(f"entries_{slug}_geocoded.csv"))
    if hits:
        return hits[0]
    # Plain entries CSV (geocode_level column will be absent → all skipped)
    hits = sorted(path.rglob(f"entries_{slug}.csv"))
    if hits:
        print(
            f"Warning: no geocoded CSV found — run geocode_entries.py first.\n"
            f"  Found {hits[0].name} but it has no lat/lon columns.",
            file=sys.stderr,
        )
        return hits[0]
    raise FileNotFoundError(
        f"No entries_{slug}_geocoded.csv found under {path}. "
        f"Run geocode_entries.py first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize geocoded Green Book entries on an interactive map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source",
        help="Path to a geocoded entries CSV, or a directory containing one.")
    parser.add_argument("--model", "-m", default="gemini-2.0-flash", metavar="MODEL",
        help="Model slug used in the CSV filename (default: gemini-2.0-flash)")
    parser.add_argument("--out", "-o", metavar="FILE",
        help="Output HTML path (default: <csv_stem>.html next to the CSV)")
    parser.add_argument("--no-ads", action="store_true",
        help="Exclude advertisement entries")
    parser.add_argument("--year", default="", metavar="YEAR",
        help="Year label shown in the sidebar (e.g. 1962)")
    args = parser.parse_args()

    slug     = args.model.replace("/", "_")
    csv_path = _find_csv(Path(args.source), slug)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".html")

    # Infer year from directory name
    import re
    year = args.year
    if not year:
        for part in reversed(csv_path.parts):
            m = re.search(r"\b(19\d\d)\b", part)
            if m:
                year = m.group(1)
                break
    year_label = f"Green Book {year}" if year else "Green Book"

    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if args.no_ads:
        rows = [r for r in rows if r.get("is_advertisement", "").lower() != "true"]

    print(f"Loaded {len(rows)} entries from {csv_path.name}", file=sys.stderr)

    entries, skipped = _prepare_entries(rows)
    print(
        f"Map: {len(entries)} entries placed, {skipped} skipped (no geocode).",
        file=sys.stderr,
    )

    html = build_html(entries, year_label)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
