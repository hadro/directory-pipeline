#!/usr/bin/env python3
"""Visualize Green Book entries on an interactive map with search & filter.

Reads a pre-geocoded entries CSV produced by geocode_entries.py and builds
a self-contained HTML file with a Leaflet map and a sidebar for filtering
by name/address, state, and category.

Run geocode_entries.py first to produce entries_{slug}_geocoded.csv:
    python pipeline/geo/geocode_entries.py output/green_book_1962_9ab2e8f0/ --model gemini-2.0-flash

Then build the map:
    python pipeline/geo/map_entries.py output/green_book_1962_9ab2e8f0/ --model gemini-2.0-flash
    python pipeline/geo/map_entries.py path/to/entries_gemini-2.0-flash_geocoded.csv
    python pipeline/geo/map_entries.py path/to/entries.csv --out my_map.html
"""

import argparse
import base64
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import iiif_utils

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
# IIIF thumbnail support
# ---------------------------------------------------------------------------

def _build_canvas_service_map(
    search_root: Path,
) -> dict[str, tuple[str, int, int, str]]:
    """Walk search_root for manifest.json files.

    Returns {canvas_id: (service_id, canvas_w, canvas_h, viewer_url)}.

    canvas_w / canvas_h reflect the coordinate space that canvas_fragment
    values in aligned.json were generated in — NOT necessarily the manifest
    canvas dimensions (which may have been updated to natural image dims by
    export_entry_boxes.py --update-manifest).  We read the canonical values
    from *_aligned.json files so that _iiif_region_url's pct: calculation
    remains correct regardless of whether the manifest has been updated.

    viewer_url is a deep-link into the institution's viewer (e.g. NYPL Digital
    Collections with ?canvasIndex=N); empty string for unknown institutions.
    """
    mapping: dict[str, tuple[str, int, int, str]] = {}
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
            mapping[canvas["canvas_id"]] = (
                svc_id,
                canvas["canvas_width"],
                canvas["canvas_height"],
                viewer_url,
            )

    # Override canvas_w / canvas_h with the coordinate space recorded in each
    # *_aligned.json file.  align_ocr.py writes canvas_width / canvas_height as
    # the coordinate space it used when generating canvas_fragment values.
    # This may differ from the manifest dimensions if the manifest was later
    # updated to natural image dims by export_entry_boxes.py --update-manifest.
    for aligned_path in search_root.rglob("*_aligned.json"):
        try:
            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        canvas_id = aligned.get("canvas_uri") or aligned.get("canvas_id") or ""
        aw = int(aligned.get("canvas_width") or 0)
        ah = int(aligned.get("canvas_height") or 0)
        if canvas_id and aw and ah and canvas_id in mapping:
            svc_id, _, _, viewer_url = mapping[canvas_id]
            mapping[canvas_id] = (svc_id, aw, ah, viewer_url)

    return mapping


def _iiif_region_url(
    canvas_fragment: str,
    service_map: dict[str, tuple[str, int, int, str]],
    thumb_width: int = 220,
    pad: int = 40,
) -> str:
    """Convert a canvas_fragment (#xywh=) to a IIIF Image API region URL.

    Uses pct: addressing: x_pct = x / canvas_w * 100.  Because canvas_w
    comes from *_aligned.json (the same coordinate space that align_ocr.py
    used when writing canvas_fragment), the pct: values are always relative
    to the correct coordinate space, regardless of whether the manifest canvas
    dimensions have been updated to natural image dims.

    Adds padding so there is visual context around the target text line.
    Returns '' if the fragment can't be resolved to a known image service.
    """
    if "#xywh=" not in canvas_fragment:
        return ""
    canvas_id, xywh = canvas_fragment.split("#xywh=", 1)
    entry = service_map.get(canvas_id)
    if not entry:
        return ""
    service_id, canvas_w, canvas_h = entry[0], entry[1], entry[2]
    if not service_id or not canvas_w or not canvas_h:
        return ""
    try:
        x, y, w, h = [int(v) for v in xywh.split(",")]
    except ValueError:
        return ""
    if w <= 0 or h <= 0:
        return ""
    # Apply padding in canvas-pixel space, clamping at 0
    rx = max(0, x - pad)
    ry = max(0, y - pad)
    rw = (x + w + pad) - rx
    rh = (y + h + pad) - ry
    # Convert to pct: so the request targets the correct region of the
    # server's native image regardless of resolution differences.
    x_pct = rx / canvas_w * 100
    y_pct = ry / canvas_h * 100
    w_pct = rw / canvas_w * 100
    h_pct = rh / canvas_h * 100
    return (
        f"{service_id}/pct:{x_pct:.4f},{y_pct:.4f},"
        f"{w_pct:.4f},{h_pct:.4f}/{thumb_width},/0/default.jpg"
    )


def _content_state_url(
    canvas_fragment: str,
    manifest_url: str,
    viewer_base_url: str,
) -> str:
    """Build a IIIF Content State deep-link URL for the given canvas fragment.

    Encodes a minimal W3C Annotation as Base64url and appends it as
    ?iiif-content= to viewer_base_url.  Mirador 3.3+ opens to the specified
    canvas and region when this parameter is present.

    Returns '' if any required input is missing.
    """
    if not canvas_fragment or not manifest_url or not viewer_base_url:
        return ""
    if "#xywh=" not in canvas_fragment:
        return ""
    canvas_id, _xywh = canvas_fragment.split("#xywh=", 1)
    # Use bare canvas_id (no #xywh= fragment) for target.id.
    # Mirador 3.3.0 does an exact string match for canvas lookup; including
    # the fragment causes the lookup to fail and the main view never renders.
    # motivation must be a string per the IIIF Content State 1.0 spec.
    state = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "type": "Annotation",
        "motivation": "contentState",
        "target": {
            "id": canvas_id,
            "type": "Canvas",
            "partOf": [{"id": manifest_url, "type": "Manifest"}],
        },
    }
    encoded = (
        base64.urlsafe_b64encode(
            json.dumps(state, separators=(",", ":")).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{viewer_base_url.rstrip('/')}?iiif-content={encoded}"


# ---------------------------------------------------------------------------
# Read geocoded CSV
# ---------------------------------------------------------------------------

def _prepare_entries(
    rows: list[dict],
    service_map: dict[str, tuple[str, int, int, str]] | None = None,
    viewer_base_url: str = "",
    manifest_url: str = "",
) -> tuple[list[dict], int]:
    """Convert geocoded CSV rows to minimal dicts for the map.

    Expects rows to have lat, lon, geocode_level columns (from geocode_entries.py).
    City-level entries get a small random jitter so they don't stack.
    If service_map is provided, a IIIF thumbnail URL is added to each entry.
    If viewer_base_url and manifest_url are also provided, each thumbnail links
    to a IIIF Content State deep-link that opens the entry in the viewer.
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

        entry: dict = {
            "name":     row.get("establishment_name", ""),
            "address":  (row.get("raw_address") or "").strip(),
            "city":     row.get("city", ""),
            "state":    row.get("state", ""),
            "category": row.get("category", "other"),
            "page":     row.get("image", "").replace(".jpg", ""),
            "lat":      round(lat, 6),
            "lon":      round(lon, 6),
        }

        if service_map is not None:
            cf = row.get("canvas_fragment", "")
            thumb = _iiif_region_url(cf, service_map) if cf else ""
            if thumb:
                entry["thumb"] = thumb
                # Prefer a Content State deep-link into the self-hosted viewer
                if viewer_base_url and manifest_url:
                    cs_link = _content_state_url(cf, manifest_url, viewer_base_url)
                    if cs_link:
                        entry["thumb_link"] = cs_link
                # Fallback: institutional viewer deep-link (page-level only)
                if "thumb_link" not in entry:
                    canvas_id = cf.split("#xywh=", 1)[0] if "#xywh=" in cf else cf
                    map_entry = service_map.get(canvas_id)
                    if map_entry and map_entry[3]:
                        entry["thumb_link"] = map_entry[3]

        entries.append(entry)

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
.popup-thumb {{
  display: block; width: 100%; border-radius: 3px;
  margin-bottom: 6px; border: 1px solid #e0e0e0;
}}
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
      (e.thumb ? (e.thumb_link
        ? `<a href="${{e.thumb_link}}" target="_blank" rel="noopener"><img class="popup-thumb" src="${{e.thumb}}" alt="source scan"/></a>`
        : `<img class="popup-thumb" src="${{e.thumb}}" alt="source scan"/>`) : '') +
      `<div class="popup-name">${{esc(e.name)}}</div>` +
      (e.address ? `<div class="popup-addr">${{esc(e.address)}}</div>` : '') +
      `<div class="popup-addr">${{esc(e.city)}}, ${{esc(e.state)}}</div>` +
      `<div class="popup-cat">${{esc(CAT_LABELS[e.category] || e.category)}}</div>` +
      `<div class="popup-page">${{esc(e.page)}}</div>`,
      {{ maxWidth: 300 }}
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
    parser.add_argument("--output-dir", default=None, metavar="DIR",
        help="Root of the images directory tree — used to find IIIF manifests "
             "and add source-scan thumbnails to map popups. "
             "Defaults to the CSV's parent directory.")
    parser.add_argument("--viewer-url", default="", metavar="URL",
        help="Base URL of a self-hosted IIIF viewer (e.g. "
             "https://hadro.github.io/green-book-iiif-test/). "
             "When set, thumbnail images in map popups link to a IIIF Content "
             "State deep-link that opens the viewer at the exact entry location. "
             "A manifest.json is assumed to live at <viewer-url>/manifest.json "
             "unless --manifest-url overrides it.")
    parser.add_argument("--manifest-url", default="", metavar="URL",
        help="Explicit URL of the IIIF manifest served alongside the viewer. "
             "Defaults to <viewer-url>/manifest.json.")
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

    # Build IIIF canvas → image-service map for popup thumbnails
    output_root = Path(args.output_dir) if args.output_dir else csv_path.parent
    service_map: dict[str, str] | None = None
    if output_root.is_dir():
        service_map = _build_canvas_service_map(output_root)
        if service_map:
            print(
                f"IIIF: loaded {len(service_map)} canvas→service mappings "
                f"from {output_root}",
                file=sys.stderr,
            )
        else:
            print(
                "IIIF: no manifests found — map popups will not include thumbnails. "
                "Pass --output-dir to specify the images directory.",
                file=sys.stderr,
            )

    viewer_base_url = args.viewer_url.rstrip("/")
    manifest_url = (
        args.manifest_url
        or (f"{viewer_base_url}/manifest.json" if viewer_base_url else "")
    )

    entries, skipped = _prepare_entries(
        rows,
        service_map=service_map,
        viewer_base_url=viewer_base_url,
        manifest_url=manifest_url,
    )
    print(
        f"Map: {len(entries)} entries placed, {skipped} skipped (no geocode).",
        file=sys.stderr,
    )

    html = build_html(entries, year_label)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
