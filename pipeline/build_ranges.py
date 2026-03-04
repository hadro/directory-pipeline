#!/usr/bin/env python3
"""Build IIIF Presentation API v3 Range structures (table of contents) from geocoded entries.

Reads entries_{model}_geocoded.csv and manifest.json from an item directory,
groups entries by State > City > Category in display order (alphabetical), and
produces a IIIF v3 Range hierarchy that any compliant viewer (Universal Viewer,
Mirador) will render as a navigable table of contents.

Each node in the hierarchy targets the canvas where the first entry in that
group appears in document order (canvas index, then y-coordinate within page).

Outputs
-------
  ranges_{model}.json        — standalone structures array, loadable or mergeable
  With --update-manifest:
    manifest.json is updated in-place and a backup is written to manifest_bak.json

Depth levels
------------
  1  State only              (e.g. "Alabama")
  2  State > City            (e.g. "Alabama > Birmingham")
  3  State > City > Category (default)

Usage
-----
    python pipeline/build_ranges.py output/green_book_1947_4bea2040/uuid/
    python pipeline/build_ranges.py output/green_book_1947_4bea2040/uuid/ \\
        --model gemini-2.0-flash --depth 2
    python pipeline/build_ranges.py output/green_book_1947_4bea2040/uuid/ \\
        --update-manifest --base-url https://example.org/iiif
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import iiif_utils


# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------

CATEGORY_LABELS: dict[str, str] = {
    "formal_accommodations":   "Hotels / Motels",
    "informal_accommodations": "Tourist Homes",
    "eating_drinking":         "Restaurants / Bars",
    "barber_beauty":           "Barber / Beauty",
    "service_station":         "Service Stations",
    "other":                   "Other",
}


def _title(s: str) -> str:
    """'NEW YORK' → 'New York'; already-mixed case preserved."""
    if not s:
        return s
    if s == s.upper():
        return s.title()
    return s


def _cat_label(cat: str) -> str:
    return CATEGORY_LABELS.get(cat) or _title(cat.replace("_", " ")) or "Other"


def _slugify(s: str) -> str:
    """Produce a URL-safe slug from a string."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "x"


# ---------------------------------------------------------------------------
# Canvas ordering helpers
# ---------------------------------------------------------------------------

def _build_canvas_order(manifest: dict) -> dict[str, int]:
    """Return {canvas_id: position_index} from a manifest."""
    return {c["canvas_id"]: i for i, c in enumerate(iiif_utils.iter_canvases(manifest))}


def _doc_sort_key(canvas_fragment: str, canvas_order: dict[str, int]) -> tuple[int, int]:
    """
    Sort key that places entries in document order: (canvas_index, y_coord).
    Entries with unresolvable fragments sort last.
    """
    if "#xywh=" not in canvas_fragment:
        canvas_id = canvas_fragment
        y = 0
    else:
        canvas_id, xywh = canvas_fragment.split("#xywh=", 1)
        parts = xywh.split(",")
        try:
            y = int(parts[1]) if len(parts) >= 2 else 0
        except ValueError:
            y = 0
    return (canvas_order.get(canvas_id, 999_999), y)


def _first_canvas_id(fragments: list[str], canvas_order: dict[str, int]) -> str | None:
    """
    Return the canvas_id of the earliest-appearing fragment in document order.
    """
    best: tuple[int, int] | None = None
    best_cid: str | None = None
    for frag in fragments:
        key = _doc_sort_key(frag, canvas_order)
        if best is None or key < best:
            best = key
            best_cid = (
                frag.split("#xywh=", 1)[0] if "#xywh=" in frag else frag
            )
    return best_cid


# ---------------------------------------------------------------------------
# Range building
# ---------------------------------------------------------------------------

def _canvas_ref(canvas_id: str) -> dict:
    """IIIF v3 inline canvas reference for a Range's items list."""
    return {"id": canvas_id, "type": "Canvas"}


def _range(range_id: str, label: str, items: list[dict]) -> dict:
    r: dict = {"type": "Range", "label": {"en": [label]}, "items": items}
    if range_id:
        r["id"] = range_id
    return r


def build_structures(
    rows: list[dict],
    canvas_order: dict[str, int],
    depth: int,
    base_url: str,
) -> list[dict]:
    """
    Build the IIIF v3 structures array from geocoded CSV rows.

    Hierarchy (controlled by depth):
      1 → State
      2 → State > City
      3 → State > City > Category

    Display order is alphabetical at every level.
    Each node targets the first document-order canvas for that group.
    """

    def rid(*parts: str) -> str:
        if not base_url:
            return ""
        slug = "/".join(_slugify(p) for p in parts if p)
        return f"{base_url}/range/{slug}"

    # Collect: state → city → category → [canvas_fragments]
    tree: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in rows:
        state = _title(row.get("state", "").strip())
        city = _title(row.get("city", "").strip())
        cat = row.get("category", "other").strip()
        cf = row.get("canvas_fragment", "").strip()
        if not state or not cf:
            continue
        tree[state][city][cat].append(cf)

    state_ranges: list[dict] = []

    for state in sorted(tree):
        city_tree = tree[state]

        if depth == 1:
            # Leaf: point to first canvas for this state
            all_frags = [f for c in city_tree.values() for cats in c.values() for f in cats]
            cid = _first_canvas_id(all_frags, canvas_order)
            if not cid:
                continue
            state_ranges.append(
                _range(rid(state), state, [_canvas_ref(cid)])
            )
            continue

        city_ranges: list[dict] = []

        for city in sorted(city_tree):
            cat_tree = city_tree[city]

            if depth == 2:
                all_frags = [f for cats in cat_tree.values() for f in cats]
                cid = _first_canvas_id(all_frags, canvas_order)
                if not cid:
                    continue
                city_ranges.append(
                    _range(rid(state, city), city, [_canvas_ref(cid)])
                )
                continue

            # depth == 3: add category level
            cat_ranges: list[dict] = []
            for cat in sorted(cat_tree, key=_cat_label):
                frags = cat_tree[cat]
                cid = _first_canvas_id(frags, canvas_order)
                if not cid:
                    continue
                cat_ranges.append(
                    _range(rid(state, city, cat), _cat_label(cat), [_canvas_ref(cid)])
                )
            if not cat_ranges:
                continue
            # City node points to first canvas in any of its categories
            all_frags = [f for frags in cat_tree.values() for f in frags]
            cid = _first_canvas_id(all_frags, canvas_order)
            city_ranges.append(
                _range(rid(state, city), city, [_canvas_ref(cid)] + cat_ranges)
            )

        if not city_ranges:
            continue

        # State node points to first canvas in any of its cities
        all_frags = [f for c in city_tree.values() for cats in c.values() for f in cats]
        cid = _first_canvas_id(all_frags, canvas_order)
        state_ranges.append(
            _range(rid(state), state, [_canvas_ref(cid)] + city_ranges)
        )

    if not state_ranges:
        return []

    root = _range(
        rid("toc"),
        "Table of Contents",
        state_ranges,
    )
    return [root]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IIIF v3 Range structures (table of contents) from geocoded entries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "item_dir",
        help="Item directory containing manifest.json and entries_{model}_geocoded.csv",
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.0-flash",
        metavar="MODEL",
        help="Model slug used in the geocoded CSV filename (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=[1, 2, 3],
        default=3,
        metavar="N",
        help="Hierarchy depth: 1=State, 2=State>City, 3=State>City>Category (default: 3)",
    )
    parser.add_argument(
        "--base-url",
        default="",
        metavar="URL",
        help=(
            "Base URL for Range @id fields "
            "(e.g. https://example.org/iiif). "
            "Omit to produce Ranges without id fields."
        ),
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help=(
            "Write the structures array into manifest.json "
            "(backs up original to manifest_bak.json)"
        ),
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing ranges file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    item_dir = Path(args.item_dir)
    if not item_dir.is_dir():
        print(f"Error: not a directory: {item_dir}", file=sys.stderr)
        sys.exit(1)

    slug = args.model.replace("/", "_")

    # Locate geocoded CSV
    csv_path = item_dir / f"entries_{slug}_geocoded.csv"
    if not csv_path.exists():
        # Try recursive search
        found = list(item_dir.rglob(f"entries_{slug}_geocoded.csv"))
        if not found:
            print(
                f"Error: no entries_{slug}_geocoded.csv found under {item_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        csv_path = found[0]
        item_dir = csv_path.parent

    # Locate manifest
    manifest_path = item_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {item_dir}", file=sys.stderr)
        sys.exit(1)

    # Check output
    out_path = item_dir / f"ranges_{slug}.json"
    if out_path.exists() and not args.force:
        print(
            f"  [skip] already exists: {out_path.name}  (use --force to overwrite)",
            file=sys.stderr,
        )
        sys.exit(0)

    # Load manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading manifest: {exc}", file=sys.stderr)
        sys.exit(1)

    canvas_order = _build_canvas_order(manifest)
    if not canvas_order:
        print("Error: no canvases found in manifest.", file=sys.stderr)
        sys.exit(1)

    # Load CSV
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        print(f"Error reading CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(
            f"Building depth-{args.depth} Ranges from {len(rows)} entries "
            f"across {len(canvas_order)} canvases…",
            file=sys.stderr,
        )

    structures = build_structures(rows, canvas_order, args.depth, args.base_url)

    if not structures:
        print("Warning: no Ranges produced (no entries with canvas_fragments?).", file=sys.stderr)
        sys.exit(0)

    # Count nodes
    def _count(node: dict) -> int:
        return 1 + sum(
            _count(child)
            for child in node.get("items", [])
            if isinstance(child, dict) and child.get("type") == "Range"
        )

    n_ranges = sum(_count(r) for r in structures)

    # Write standalone ranges file
    out_path.write_text(
        json.dumps(structures, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"  Wrote {n_ranges} Range nodes → {out_path.name}", file=sys.stderr)

    # Optionally update manifest in-place
    if args.update_manifest:
        bak_path = item_dir / "manifest_bak.json"
        bak_path.write_text(
            manifest_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        manifest["structures"] = structures
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not args.quiet:
            print(
                f"  Updated manifest.json with structures  "
                f"(backup → {bak_path.name})",
                file=sys.stderr,
            )

    if not args.quiet:
        print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
