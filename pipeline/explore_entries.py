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
    manifest_id: str = (manifest.get("id") or manifest.get("@id") or "").rstrip("?")

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
        "manifest_url": manifest_id,
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
    # Extract viewer resource URL (e.g. https://www.loc.gov/resource/rbc0001.2026batch96169559/)
    resource_url = ""
    resources = item.get("resources") or data.get("resources") or []
    if resources and isinstance(resources[0], dict):
        resource_url = resources[0].get("url", "")
    return {
        "title": item.get("title", ""),
        "date": date,
        "institution": "Library of Congress",
        "homepage_url": homepage_url,
        "collection": "",
        "genre": genre,
        "resource_url": resource_url,
    }


def _find_item_meta(search_root: Path) -> dict:
    """Find the first manifest.json under search_root and extract item metadata.

    For LC items, also checks for item.json saved by download_images.py.
    """
    for manifest_path in sorted(search_root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta = _extract_item_meta(manifest)

            # For LC items, check for item.json saved by the download step.
            # Always read it (even when manifest already has a title) so we
            # can pick up fields like resource_url that the manifest lacks.
            item_json_path = manifest_path.parent / "item.json"
            if "loc.gov" in manifest.get("id", "") and item_json_path.exists():
                lc_meta = _extract_loc_item_json_meta(item_json_path)
                # Only override title/date/genre when manifest has none / looks like an ID
                if not meta.get("title") or _looks_like_id(meta.get("title", "")):
                    if lc_meta.get("title"):
                        meta.update({k: v for k, v in lc_meta.items() if v})
                else:
                    # Always pull resource_url (and other enrichment fields not in manifest)
                    for key in ("resource_url", "date", "collection", "genre"):
                        if lc_meta.get(key) and not meta.get(key):
                            meta[key] = lc_meta[key]
                    if lc_meta.get("resource_url"):
                        meta["resource_url"] = lc_meta["resource_url"]

            if meta.get("title") or meta.get("homepage_url"):
                return meta
        except Exception:
            continue
    return {"title": "", "date": "", "institution": "", "homepage_url": "",
            "collection": "", "genre": "", "manifest_url": ""}


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
            if "year" in col.lower() or "date" in col.lower():
                top_values.sort(key=lambda x: x[0])
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

# The page template lives in explore_entries.html next to this file — extracted from an
# inline string so it gets HTML/JS syntax highlighting and reviewable diffs.
# Generated output is unaffected; the template still ships inside the package.
_HTML_TEMPLATE = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")


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
# Build annotation index (for canvas_fragment enrichment)
# ---------------------------------------------------------------------------

def _load_annotation_index(item_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Return {canvas_base_uri: [(body_text, target_with_xywh), ...]} from all
    *_annotations.json files in item_dir (reading order preserved).
    *_entry_annotations.json files are excluded (those are entry-level, not line-level).
    """
    index: dict[str, list[tuple[str, str]]] = {}
    for p in sorted(item_dir.rglob("*_annotations.json")):
        if p.name.endswith("_entry_annotations.json"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for ann in data.get("items", []):
            target = ann.get("target", "")
            if "#xywh=" not in target:
                continue
            body = ann.get("body", {})
            text = body.get("value", "") if isinstance(body, dict) else ""
            base = target.split("#")[0]
            index.setdefault(base, []).append((text.lower(), target))
    return index


def _enrich_rows_with_annotations(
    rows: list[dict], ann_index: dict[str, list[tuple[str, str]]]
) -> list[dict]:
    """For each row whose canvas_fragment lacks #xywh=, find the best-matching
    line annotation on the same canvas via token overlap and replace canvas_fragment
    with the annotation target (which includes the bounding box).

    Each annotation may be claimed by at most one row (greedy best-first within
    each canvas).  Single-character tokens are excluded from matching to avoid
    false positives from initials appearing in unrelated lines.
    """
    def _tokens(s: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) >= 2}

    # Track which annotation indices have already been claimed, per canvas
    canvas_used: dict[str, set[int]] = {}

    enriched = []
    for row in rows:
        cf = row.get("canvas_fragment", "")
        if "#xywh=" in cf or not cf:
            enriched.append(row)
            continue

        base = cf.split("#")[0]
        candidates = ann_index.get(base)
        if not candidates:
            enriched.append(row)
            continue

        query = _tokens(
            " ".join(v for k, v in row.items() if k not in _ID_FIELDS and v)
        )
        if not query:
            enriched.append(row)
            continue

        used = canvas_used.setdefault(base, set())
        best_score, best_idx, best_target = 0, -1, ""
        for i, (ann_text, ann_target) in enumerate(candidates):
            if i in used:
                continue
            score = len(query & _tokens(ann_text))
            if score > best_score:
                best_score, best_idx, best_target = score, i, ann_target

        if best_score >= 2:
            row = dict(row)
            row["canvas_fragment"] = best_target
            used.add(best_idx)

        enriched.append(row)
    return enriched


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
    viewer_url: str = "",
    manifest_url: str = "",
    full_page_thumbs: bool = False,
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
        viewer_url_json  = _safe_json(viewer_url),
        manifest_url_json = _safe_json(manifest_url),
        full_page_thumbs_json = _safe_json(full_page_thumbs),
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
        nargs="+",
        help="One or more paths to entries CSVs or directories containing them.",
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
        "--viewer-url",
        default="",
        metavar="URL",
        help="Base URL of a self-hosted IIIF viewer (e.g. https://hadro.github.io/my-repo/). "
             "When set, the detail panel shows a 'View in Mirador ↗' link that opens the "
             "viewer at the exact canvas via IIIF Content State. "
             "A manifest.json is assumed at <viewer-url>/manifest.json unless --manifest-url overrides it.",
    )
    parser.add_argument(
        "--manifest-url",
        default="",
        metavar="URL",
        help="Explicit URL of the IIIF manifest served with the viewer. "
             "Defaults to <viewer-url>/manifest.json.",
    )
    parser.add_argument(
        "--full-page-thumbs",
        action="store_true",
        help="Show full-page thumbnails instead of cropped bounding-box regions "
             "(useful when canvas_fragment bboxes are unreliable or absent)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    sources = [Path(s) for s in args.source]

    # Auto-detect model slug from the first source that has a matching CSV.
    if args.model is None:
        for src in sources:
            candidates = [h for h in sorted((src.rglob("entries_*.csv") if src.is_dir() else [src]))
                          if "_geocoded" not in h.name and "_explorer" not in h.name]
            for c in candidates:
                m = re.match(r"entries_(.+)\.csv", c.name)
                if m:
                    args.model = m.group(1)
                    break
            if args.model:
                break

    slug = args.model.replace("/", "_") if args.model else None

    # Find all matching CSVs across all sources.
    def _find_csvs(src: Path) -> list[Path]:
        if src.is_file():
            return [src]
        elif slug:
            return sorted(src.rglob(f"entries_{slug}.csv"))
        else:
            return [h for h in sorted(src.rglob("entries_*.csv"))
                    if "_geocoded" not in h.name and "_explorer" not in h.name]

    all_csvs = []
    seen = set()
    for src in sources:
        for p in _find_csvs(src):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                all_csvs.append(p)
    all_csvs.sort(key=lambda p: str(p))

    if not all_csvs:
        print(
            f"No entries_*.csv found in: {', '.join(str(s) for s in sources)}. "
            "Run --extract-entries first.",
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
        ann_index = _load_annotation_index(p.parent)
        if ann_index:
            vol_rows = _enrich_rows_with_annotations(vol_rows, ann_index)
            if not args.quiet:
                enriched_n = sum(1 for r in vol_rows if "#xywh=" in (r.get("canvas_fragment") or ""))
                print(f"  Annotation enrichment: {enriched_n}/{len(vol_rows)} entries have bounding boxes", file=sys.stderr)
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

    # Build IIIF canvas map for thumbnails — for multi-source collections, scan
    # each CSV's parent directory so manifests from all sources are found.
    canvas_map: dict = {}
    doc_meta: dict = {}
    if args.output_dir:
        output_root = Path(args.output_dir)
        if output_root.is_dir():
            canvas_map = _build_canvas_thumb_map(output_root)
            doc_meta = _find_item_meta(output_root)
    elif len(sources) > 1:
        # Multi-source: scan each CSV's parent dir individually and merge.
        for p in all_csvs:
            canvas_map.update(_build_canvas_thumb_map(p.parent))
        if not doc_meta:
            doc_meta = _find_item_meta(all_csvs[0].parent)
    else:
        output_root = out_path.parent if collection_mode else csv_path.parent
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

    viewer_url = args.viewer_url.rstrip("/")
    manifest_url = (
        args.manifest_url
        or (f"{viewer_url}/manifest.json" if viewer_url else "")
    )

    html = build_html(rows, field_meta, canvas_map, title, doc_meta,
                      volumes=volumes if collection_mode else None,
                      viewer_url=viewer_url,
                      manifest_url=manifest_url,
                      full_page_thumbs=args.full_page_thumbs)
    out_path.write_text(html, encoding="utf-8")
    if not args.quiet:
        print(f"Explorer written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
