"""
Combines all *_fixed.csv files within a green book collection directory into a
single normalized CSV with a unified schema.

Usage:
    python analysis/combine_volumes.py <collection_dir> [--output combined.csv]

Example:
    python analysis/combine_volumes.py \
        output/green_books_and_related/the_green_book_9ea5d5b0 \
        --output output/green_books_and_related/the_green_book_9ea5d5b0/combined.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Unified schema column order
# ---------------------------------------------------------------------------
UNIFIED_COLUMNS = [
    # Provenance
    "volume_id",
    "volume_title",
    "volume_year",
    "publication",
    # Core entry fields
    "name",
    "address",
    "state",
    "city",
    "sub_region",    # borough/county/country subdivision (1963-64, 1966-67)
    "category",
    "notes",
    # Sparse optional fields
    "phone",
    "proprietor",
    "amenities_services",
    "rates",
    "personnel",
    "reference_number",
    "is_advertisement",
    "is_recommended",
    # Source
    "canvas_fragment",
    "image",
    # Quality flags
    "flag_state_invalid",
    "flag_state_eq_city",
    "flag_name_address",
    "flag_header_row",
    "flag_duplicate",
    "flag_unanchored",
]

# ---------------------------------------------------------------------------
# Per-volume column mappings: source_col -> unified_col
# Columns not listed here are dropped (e.g. redundant city_state after parsing).
# ---------------------------------------------------------------------------
DEFAULT_MAPPING = {
    "name":                 "name",
    "establishment_name":   "name",
    "business_name":        "name",
    "subject":              "name",        # Smith's Tourist Guide (laws + listings share one field)
    "address":              "address",
    "state":                "state",
    "state_country":        "state",
    "state_or_region":      "state",
    "state_region":         "state",
    "state_province":       "state",       # Travelguide 1949, 1952, 1955
    "state_or_province":    "state",       # Hackley & Harrison
    "location_state_country": "state",
    "city":                 "city",
    "city_borough":         "city",        # Travelguide 1962-63
    "location_city":        "city",
    "category":             "category",
    "section":              "category",
    "section_category":     "category",
    "business_type":        "category",
    "notes":                "notes",
    "description":          "notes",
    "details":              "notes",
    "additional_info":      "notes",
    "note":                 "notes",       # Travelguide 1955
    "phone":                "phone",
    "proprietor":           "proprietor",
    "manager_proprietor":   "proprietor",  # Travelguide 1954, 1956
    "manager_owner":        "proprietor",  # Travelguide 1953, 1955, 1957
    "amenities_services":   "amenities_services",
    "rates":                "rates",
    "price":                "rates",       # Travelguide 1953, 1957, 1962-63
    "price_info":           "rates",
    "rates_and_capacity":   "rates",       # Travelguide 1949
    "personnel":            "personnel",
    "reference_number":     "reference_number",
    "is_display_ad":        "is_advertisement",
    "is_advertisement":     "is_advertisement",
    "is_advertiser":        "is_advertisement",
    "is_recommended":       "is_recommended",
    "is_starred":           "is_recommended",
    "is_approved":          "is_recommended",  # Travelguide (various years)
    "is_go_approved":       "is_recommended",  # Go Guide
    "is_subscriber":        "is_recommended",  # Travelers Guide, Hackley & Harrison
    "is_nha_member":        "is_recommended",  # NHA Directory
    "canvas_fragment":      "canvas_fragment",
    "image":                "image",
    "flag_state_invalid":   "flag_state_invalid",
    "flag_state_eq_city":   "flag_state_eq_city",
    "flag_name_address":    "flag_name_address",
    "flag_header_row":      "flag_header_row",
    "flag_duplicate":       "flag_duplicate",
    "flag_unanchored":      "flag_unanchored",
}

# Volumes with bespoke handling keyed by UUID
VOLUME_OVERRIDES = {
    # 1937 — uses location (section heading) + city_state ("City, St." string)
    "fc1bcc10-1a76-0132-8925-58d385a7bbd0": {
        "mapping_overrides": {
            "location":   "category",   # section heading, not a state
            # city_state is parsed separately in transform_row()
        },
        "drop": {"city_state"},
    },
    # 1963-64 international — location_primary=state/continent,
    #                          location_secondary=country (non-US only)
    "c4391ba0-1a76-0132-7cae-58d385a7bbd0": {
        "mapping_overrides": {
            "location_primary":   "state",
            "location_secondary": "sub_region",
        },
    },
    # 1966-67 international — same structure as 1963-64
    "fdb5a8d0-1a75-0132-d947-58d385a7bbd0": {
        "mapping_overrides": {
            "location_state_country": "state",
            "location_city":          "city",
        },
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATE_ABBREV = re.compile(
    r"^(?P<city>.+?),\s*(?P<state>[A-Z][a-z]?\.\s*[A-Z]\.?|[A-Z]{2})$"
)


def parse_city_state(value: str) -> tuple[str, str]:
    """Parse '`City, N. Y.`' or 'City, NY' into (city, state)."""
    value = value.strip()
    if not value:
        return "", ""
    m = _STATE_ABBREV.match(value)
    if m:
        return m.group("city").strip(), m.group("state").strip()
    # Fallback: split on last comma
    parts = value.rsplit(",", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return value, ""


# Human-readable publication series name, keyed by collection directory slug.
# Used to populate the `publication` column in combined CSVs.
_COLLECTION_LABELS: dict[str, str] = {
    "the_green_book_9ea5d5b0":                           "The Negro Motorist Green Book",
    "travelguide_634f3af0":                              "Travelguide",
    "go_guide_to_pleasant_motoring_e6743f00":            "Go, Guide to Pleasant Motoring",
    "the_travelers_guide_e088efa0":                      "The Travelers Guide",
    "n_h_a_directory_and_guide_to_travelers_b33397d0":   "NHA Directory and Guide to Travelers",
    "smith_s_tourist_guide_of_necessary_infor_e20bf5b0": "Smith's Tourist Guide",
    "hackley_harrison_s_hotel_and_apartment_g_4f7822b0": "Hackley & Harrison's Hotel and Apartment Guide",
}


def load_manifest_label(volume_dir: Path) -> str:
    manifest = volume_dir / "manifest.json"
    if not manifest.exists():
        return ""
    try:
        data = json.loads(manifest.read_text())
        label = data.get("label", {})
        if isinstance(label, dict):
            val = next(iter(label.values()))
            return val[0] if isinstance(val, list) else str(val)
        return str(label)
    except Exception:
        return ""


def load_manifest_date(volume_dir: Path) -> str:
    """Extract 'Date Issued' from manifest metadata; returns 4-digit year string or ''."""
    manifest = volume_dir / "manifest.json"
    if not manifest.exists():
        return ""
    try:
        data = json.loads(manifest.read_text())
        for item in data.get("metadata", []):
            lbl = item.get("label", {})
            lbl_str = (next(iter(lbl.values()))[0] if isinstance(lbl, dict) else str(lbl))
            if lbl_str == "Date Issued":
                val = item.get("value", {})
                val_str = (next(iter(val.values()))[0] if isinstance(val, dict) else str(val))
                m = re.search(r"\b(1[89]\d{2})\b", val_str)
                return m.group(1) if m else ""
    except Exception:
        pass
    return ""


def extract_year(title: str) -> str:
    m = re.search(r"\b(19\d{2})", title)
    return m.group(1) if m else ""


def build_mapping(volume_id: str, source_cols: list[str]) -> dict[str, str]:
    mapping = dict(DEFAULT_MAPPING)
    overrides = VOLUME_OVERRIDES.get(volume_id, {})
    mapping.update(overrides.get("mapping_overrides", {}))
    return mapping


def transform_row(
    row: dict,
    volume_id: str,
    mapping: dict[str, str],
) -> dict:
    out = {col: "" for col in UNIFIED_COLUMNS}
    drop = VOLUME_OVERRIDES.get(volume_id, {}).get("drop", set())

    for src_col, value in row.items():
        if src_col in drop:
            continue
        unified = mapping.get(src_col)
        if unified and unified in out:
            if not out[unified]:  # first writer wins; avoids clobbering
                out[unified] = value

    # Special parse: 1937 city_state → city + state
    if volume_id == "fc1bcc10-1a76-0132-8925-58d385a7bbd0":
        raw = row.get("city_state", "")
        city, state = parse_city_state(raw)
        if city:
            out["city"] = city
        if state:
            out["state"] = state

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def combine(collection_dir: Path, output_path: Path) -> None:
    csv_files = sorted(collection_dir.glob("*/entries_*_fixed.csv"))
    if not csv_files:
        sys.exit(f"No *_fixed.csv files found under {collection_dir}")

    publication = _COLLECTION_LABELS.get(collection_dir.name, "")

    rows_written = 0
    with output_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()

        for csv_path in csv_files:
            volume_id = csv_path.parent.name
            title = load_manifest_label(csv_path.parent)
            year = extract_year(title) or load_manifest_date(csv_path.parent)

            with csv_path.open(encoding="utf-8") as in_fh:
                reader = csv.DictReader(in_fh)
                mapping = build_mapping(volume_id, reader.fieldnames or [])
                count = 0
                for row in reader:
                    out_row = transform_row(row, volume_id, mapping)
                    out_row["volume_id"] = volume_id
                    out_row["volume_title"] = title
                    out_row["volume_year"] = year
                    out_row["publication"] = publication
                    writer.writerow(out_row)
                    count += 1
                rows_written += count
                print(f"  {year}  {volume_id}  {count:>5} rows  {title}")

    print(f"\nWrote {rows_written} total rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine green book volume CSVs.")
    parser.add_argument("collection_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <collection_dir>/combined.csv)",
    )
    args = parser.parse_args()

    collection_dir = args.collection_dir.resolve()
    if not collection_dir.is_dir():
        sys.exit(f"Not a directory: {collection_dir}")

    output_path = args.output or collection_dir / "combined.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Combining volumes in {collection_dir}\n")
    combine(collection_dir, output_path)


if __name__ == "__main__":
    main()
