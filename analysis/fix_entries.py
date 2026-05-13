#!/usr/bin/env python3
"""Post-processing fixes for extracted entries CSVs.

Applies tier-A quality improvements without re-running the NER model:
  1. Unicode normalization — Greek/Cyrillic lookalikes → ASCII
  2. Category normalization — free-form variants → canonical taxonomy
  3. Flag: state not in known list
  4. Flag: state == city (city promoted to state field)
  5. Flag: name == address (address extracted as business name)
  6. Deduplication — fuzzy-match on (name, address, city_state)

Usage:
    # Fix one volume's CSV (writes <stem>_fixed.csv by default):
    python analysis/fix_entries.py output/.../entries_gemini-3.1-flash-lite.csv

    # Fix every Green Book volume in-place:
    python analysis/fix_entries.py --all --inplace

    # Dry-run: print a summary without writing anything:
    python analysis/fix_entries.py output/.../entries_gemini-3.1-flash-lite.csv --dry-run
"""

import argparse
import csv
import difflib
import glob
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Unicode normalization
# ---------------------------------------------------------------------------

# Greek / Cyrillic characters that OCR confuses with Latin equivalents.
# Ordered longest-match-first to avoid partial replacements.
_UNICODE_MAP: dict[str, str] = {
    # Unicode replacement characters (U+FFFD) from encoding round-trips.
    # Three in a row = em dash (U+2014, 3 UTF-8 bytes each corrupted individually).
    "���": "-",
    # Greek uppercase
    "Α": "A",  # Α
    "Β": "B",  # Β
    "Γ": "G",  # Γ (rare but observed)
    "Ε": "E",  # Ε
    "Ζ": "Z",  # Ζ
    "Η": "H",  # Η
    "Ι": "I",  # Ι
    "Κ": "K",  # Κ
    "Μ": "M",  # Μ
    "Ν": "N",  # Ν
    "Ο": "O",  # Ο
    "Ρ": "P",  # Ρ
    "Τ": "T",  # Τ
    "Υ": "Y",  # Υ
    "Χ": "X",  # Χ
    # Greek lowercase lookalikes
    "ο": "o",  # ο
    "ν": "v",  # ν (looks like v)
    # Cyrillic uppercase
    "А": "A",  # А
    "В": "B",  # В
    "Е": "E",  # Е
    "К": "K",  # К
    "М": "M",  # М
    "Н": "H",  # Н
    "О": "O",  # О
    "Р": "P",  # Р
    "С": "C",  # С
    "Т": "T",  # Т
    "Х": "X",  # Х
    # Cyrillic lowercase
    "а": "a",  # а
    "е": "e",  # е
    "о": "o",  # о
    "р": "p",  # р (looks like p)
    "с": "c",  # с
    "х": "x",  # х
    # Smart quotes / dashes that survive into CSV
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "½": "1/2",  # ½ → 1/2
}

_UNICODE_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_MAP))


def normalize_unicode(text: str) -> tuple[str, int]:
    """Replace known lookalike characters; return (fixed_text, count_replaced)."""
    count = [0]

    def _replace(m: re.Match) -> str:
        count[0] += 1
        return _UNICODE_MAP[m.group(0)]

    result = _UNICODE_RE.sub(_replace, text)
    return result, count[0]


def normalize_unicode_row(row: dict, extra_fields: tuple[str, ...] = ()) -> tuple[dict, int]:
    """Apply unicode normalization to all text fields in a CSV row."""
    TEXT_FIELDS = ("state", "city", "category", "name", "address", "notes")
    all_fields = set(TEXT_FIELDS) | set(extra_fields)
    total = 0
    for field in all_fields:
        if row.get(field):
            row[field], n = normalize_unicode(row[field])
            total += n
    return row, total


# ---------------------------------------------------------------------------
# 2. Category normalization
# ---------------------------------------------------------------------------

# Canonical category names for the Green Book schema.
CANONICAL_CATEGORIES: list[str] = [
    "HOTELS",
    "RESTAURANTS",
    "BEAUTY PARLORS",
    "SERVICE STATIONS",
    "TOURIST HOMES",
    "BARBER SHOPS",
    "TAVERNS",
    "DRUG STORES",
    "THEATERS",
    "TAXI CABS",
    "YMCA",
    "YWCA",
    "NIGHT CLUBS",
    "BEAUTY CULTURE SCHOOLS",
    "TRAILER PARKS & CAMPS",
    "ROOMING HOUSES",
    "SUMMER RESORTS",
    "ROAD HOUSES",
    "RESTAURANTS & GRILLS",
    "TAILOR SHOPS",
    "UNDERTAKERS",
    "DRIVE-IN RESTAURANTS",
    "RECREATION",
    "CHURCHES",
    "SCHOOLS",
    "AIRPORTS",
    "TRAVEL AGENCIES",
    "REAL ESTATE",
    "WINE & LIQUOR STORES",
    "CLEANERS",
    "SANITARIUMS",
    "SIGHTSEEING",
    "DANCE HALLS",
    "VACATION RESORTS",
    "PUBLISHERS",
    "REPRESENTATIVE",
    "WANTED",
    "ADVERTISEMENT",
]

# Explicit variant → canonical mapping for known OCR errors / aliases.
_CATEGORY_EXPLICIT: dict[str, str] = {
    # OCR garbles
    "BEATY PARLORS": "BEAUTY PARLORS",
    "BEAUTY PARLOHS": "BEAUTY PARLORS",
    "BEAUTY PARLOS": "BEAUTY PARLORS",
    "BJTY PARLORS": "BEAUTY PARLORS",
    "BARBEH SHOPS": "BARBER SHOPS",
    "BARBER SHOPS.": "BARBER SHOPS",
    "BARBER STORES": "BARBER SHOPS",
    "TAVEVRNS": "TAVERNS",
    "TAVERNS NIGHT CLUBS": "TAVERNS",
    "TAILORS.": "TAILOR SHOPS",
    "TAILORS": "TAILOR SHOPS",
    "TAILOR SHOP": "TAILOR SHOPS",
    "DHUG STORES": "DRUG STORES",
    "DRUG STORE": "DRUG STORES",
    "STOOHOS": "DRUG STORES",  # OCR garble observed in reviews
    "SERVICE STATONS": "SERVICE STATIONS",
    "SERVICE STATION": "SERVICE STATIONS",
    "TOUHIST HOMES": "TOURIST HOMES",
    "TOUSIST HOMES": "TOURIST HOMES",
    "TOURISTS HOMES": "TOURIST HOMES",
    "TOURIST-HOMES": "TOURIST HOMES",
    "TOURIST HOME": "TOURIST HOMES",
    "TOURIST": "TOURIST HOMES",
    "THEATRE": "THEATERS",
    "THEATRES": "THEATERS",
    "THEATER": "THEATERS",
    # Normalise plural/singular
    "HOTEL": "HOTELS",
    "RESTAURANT": "RESTAURANTS",
    "BEAUTY PARLOR": "BEAUTY PARLORS",
    "BEAUTY SHOP": "BEAUTY PARLORS",
    "BEAUTY SHOPS": "BEAUTY PARLORS",
    "BEAUTY SALON": "BEAUTY PARLORS",
    "BARBER SHOP": "BARBER SHOPS",
    "TAVERN": "TAVERNS",
    "TAXI": "TAXI CABS",
    "TAXI CAB": "TAXI CABS",
    "TAXI'S": "TAXI CABS",
    "TAXIS": "TAXI CABS",
    "TAXICABS": "TAXI CABS",
    "ROAD HOUSE": "ROAD HOUSES",
    "ROOMING HOUSE": "ROOMING HOUSES",
    "SANITARIUM": "SANITARIUMS",
    "SUMMER RESORT": "SUMMER RESORTS",
    "RESORT": "SUMMER RESORTS",
    "RESORTS": "SUMMER RESORTS",
    "VACATION RESORT": "VACATION RESORTS",
    "VACATION GUIDE": "VACATION RESORTS",
    "SCHOOL OF BEAUTY CULTURE": "BEAUTY CULTURE SCHOOLS",
    "SCHOOLS OF BEAUTY CULTURE": "BEAUTY CULTURE SCHOOLS",
    "BEAUTY SCHOOL": "BEAUTY CULTURE SCHOOLS",
    "BEAUTY SCHOOLS": "BEAUTY CULTURE SCHOOLS",
    "Y.M.C.A": "YMCA",
    "Y.M.C.A.": "YMCA",
    "Y. M. C. A.": "YMCA",
    "Y.W.C.A": "YWCA",
    "Y.W.C.A.": "YWCA",
    "TRAILER PARK": "TRAILER PARKS & CAMPS",
    "TRAILER PARKS & CAMPS": "TRAILER PARKS & CAMPS",
    "TRAILER PARKS AND CAMPS": "TRAILER PARKS & CAMPS",
    "TRAILER FARKS & CAMPS": "TRAILER PARKS & CAMPS",
    "TRAILERS PARK": "TRAILER PARKS & CAMPS",
    "CABIN CAMPS": "TOURIST CABINS",
    "TOURIST CABINS": "TOURIST CABINS",
    "TOURIST CABIN": "TOURIST CABINS",
    "DRIVE IN": "DRIVE-IN RESTAURANTS",
    "DRIVE INN": "DRIVE-IN RESTAURANTS",
    "DRIVE INNS": "DRIVE-IN RESTAURANTS",
    "DRIVE IN RESTAURANTS": "DRIVE-IN RESTAURANTS",
    "UNDERTAKER": "UNDERTAKERS",
    "NIGHT CLUB": "NIGHT CLUBS",
    "TRAVEL AGENCY": "TRAVEL AGENCIES",
    "TRAVEL SERVICE": "TRAVEL AGENCIES",
    "TRAVEL SERVICES": "TRAVEL AGENCIES",
    "WINE AND LIQUOR STORE": "WINE & LIQUOR STORES",
    "WINE AND LIQUOR STORES": "WINE & LIQUOR STORES",
    "WINES & LIQUOR STORES": "WINE & LIQUOR STORES",
    "WINE & LIQUOR STORE": "WINE & LIQUOR STORES",
    "UNIVERSITIES AND COLLEGES": "SCHOOLS",
    "RECREATION CENTER": "RECREATION",
    "RECREATION CLUBS": "RECREATION",
    "RECREATION PARK": "RECREATION",
    "RECREATION PARKS": "RECREATION",
    "AMUSEMENT CENTERS": "RECREATION",
    "AMUSEMENTS": "RECREATION",
}

_CANONICAL_UPPER = [c.upper() for c in CANONICAL_CATEGORIES]


def normalize_category(raw: str) -> tuple[str, bool]:
    """Return (normalized_category, was_changed).

    1. Check explicit mapping table.
    2. If already canonical, return as-is.
    3. Fuzzy-match against canonical list (cutoff 0.75).
    4. Fall back to uppercased original.
    """
    if not raw or not raw.strip():
        return raw, False
    stripped = raw.strip().upper()

    explicit = _CATEGORY_EXPLICIT.get(stripped)
    if explicit:
        changed = explicit != raw.strip()
        return explicit, changed

    if stripped in _CANONICAL_UPPER:
        idx = _CANONICAL_UPPER.index(stripped)
        canonical = CANONICAL_CATEGORIES[idx]
        changed = canonical != raw.strip()
        return canonical, changed

    matches = difflib.get_close_matches(stripped, _CANONICAL_UPPER, n=1, cutoff=0.75)
    if matches:
        idx = _CANONICAL_UPPER.index(matches[0])
        return CANONICAL_CATEGORIES[idx], True

    return raw.strip(), False


# ---------------------------------------------------------------------------
# 2b. Category inference from business name
# ---------------------------------------------------------------------------

# Ordered keyword rules for name-based category inference.
# Longer/more-specific phrases come before shorter ones (first match wins).
_INFER_RULES: list[tuple[re.Pattern, str]] = [
    # Tourist homes — multi-word phrases first
    (re.compile(r'\btourist home\b',    re.I), "TOURIST HOMES"),
    (re.compile(r'\btourist court\b',   re.I), "TOURIST HOMES"),
    (re.compile(r'\btourist cabin\b',   re.I), "TOURIST HOMES"),
    (re.compile(r'\btourist camp\b',    re.I), "TOURIST HOMES"),
    (re.compile(r'\bauto court\b',      re.I), "TOURIST HOMES"),
    (re.compile(r'\brooming house\b',   re.I), "ROOMING HOUSES"),
    (re.compile(r'\bboarding house\b',  re.I), "ROOMING HOUSES"),
    # Service stations — multi-word first
    (re.compile(r'\bservice station\b', re.I), "SERVICE STATIONS"),
    (re.compile(r'\bfilling station\b', re.I), "SERVICE STATIONS"),
    # Night clubs
    (re.compile(r'\bnight club\b',      re.I), "NIGHT CLUBS"),
    (re.compile(r'\bnightclub\b',       re.I), "NIGHT CLUBS"),
    (re.compile(r'\bsupper club\b',     re.I), "NIGHT CLUBS"),
    # Road houses
    (re.compile(r'\broad house\b',      re.I), "ROAD HOUSES"),
    # Drive-in restaurants
    (re.compile(r'\bdrive.in\b',        re.I), "DRIVE-IN RESTAURANTS"),
    # Restaurants
    (re.compile(r'\brestaurant\b',      re.I), "RESTAURANTS"),
    (re.compile(r'\bcaf[eé]\b',         re.I), "RESTAURANTS"),
    (re.compile(r'\bdiner\b',           re.I), "RESTAURANTS"),
    (re.compile(r'\bgrill\b',           re.I), "RESTAURANTS"),
    (re.compile(r'\blunchroom\b',       re.I), "RESTAURANTS"),
    (re.compile(r'\bluncheonette\b',    re.I), "RESTAURANTS"),
    (re.compile(r'\bcafeteria\b',       re.I), "RESTAURANTS"),
    (re.compile(r'\bkitchen\b',         re.I), "RESTAURANTS"),
    (re.compile(r'\beating\b',          re.I), "RESTAURANTS"),
    # Hotels / lodging
    (re.compile(r'\bhotel\b',           re.I), "HOTELS"),
    (re.compile(r'\bmotel\b',           re.I), "HOTELS"),
    (re.compile(r'\binn\b',             re.I), "HOTELS"),
    (re.compile(r'\blodge\b',           re.I), "HOTELS"),
    (re.compile(r'\blodging\b',         re.I), "HOTELS"),
    # Beauty
    (re.compile(r'\bbeauty\b',          re.I), "BEAUTY PARLORS"),
    (re.compile(r'\bhair salon\b',      re.I), "BEAUTY PARLORS"),
    # Barbers
    (re.compile(r'\bbarber\b',          re.I), "BARBER SHOPS"),
    # Taverns
    (re.compile(r'\btavern\b',          re.I), "TAVERNS"),
    # Drug stores
    (re.compile(r'\bdrug\b',            re.I), "DRUG STORES"),
    # Taxi
    (re.compile(r'\btaxi\b',            re.I), "TAXI CABS"),
    # Garages
    (re.compile(r'\bgarage\b',          re.I), "GARAGES"),
    # Tailor shops
    (re.compile(r'\btailor\b',          re.I), "TAILOR SHOPS"),
]

# Category values that are candidates for keyword inference (lowercase comparison).
_INFER_TARGET_CATEGORIES: set[str] = {
    "",
    "general",
    "hotels - motels - tourist homes - restaurants",
}


def infer_category_from_name(name: str, notes: str = "") -> str | None:
    """Return inferred canonical category from name (and notes), or None if no rule matches."""
    text = name + " " + notes
    for pattern, category in _INFER_RULES:
        if pattern.search(text):
            return category
    return None


# ---------------------------------------------------------------------------
# 3–5. Flag checks
# ---------------------------------------------------------------------------

US_STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO",
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO",
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE",
    "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI",
    "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY",
    "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
    "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA",
    "SOUTH DAKOTA", "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA",
    "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING",
    # Territories & DC
    "DISTRICT OF COLUMBIA", "D.C.", "DC", "PUERTO RICO", "VIRGIN ISLANDS",
    # International regions seen in Green Books
    "BERMUDA", "BAHAMAS", "CANADA", "MEXICO", "CUBA", "JAMAICA",
    "TRINIDAD", "BARBADOS", "HAITI", "MARTINIQUE", "PANAMA",
    "WEST INDIES", "ONTARIO", "QUEBEC", "NOVA SCOTIA",
    # Additional international regions
    "COSTA RICA", "CARIBBEAN", "CENTRAL AMERICA",
    # Abbreviations
    "ALA", "ALAS", "ARIZ", "ARK", "CAL", "CALIF", "COLO", "CONN",
    "DEL", "FLA", "GA", "IDA", "ILL", "IND", "KAN", "KANS", "KY",
    "LA", "ME", "MD", "MASS", "MICH", "MINN", "MISS", "MO", "MONT",
    "NEB", "NEBR", "NEV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OKL", "OKLA", "ORE", "OREG", "PA", "RI", "SC", "SD", "TENN",
    "TEX", "UTAH", "VT", "VA", "WASH", "WIS", "WISC", "WYO",
}

# Verbose variants, OCR typos, city-as-state, and national park names → canonical state
STATE_ALIASES: dict[str, str] = {
    # New York variants
    "NEW YORK STATE":    "NEW YORK",
    "NEW YORK CITY":     "NEW YORK",
    "STATEN ISLAND":     "NEW YORK",
    "LONG ISLAND":       "NEW YORK",
    "WESTCHESTER":       "NEW YORK",
    # Washington variants
    "WASHINGTON STATE":  "WASHINGTON",
    "YAKIMA":            "WASHINGTON",
    "MOUNT RAINIER":     "WASHINGTON",
    "OLYMPIC":           "WASHINGTON",
    # New Jersey cities
    "ATLANTIC CITY":     "NEW JERSEY",
    "ASBURY PARK":       "NEW JERSEY",
    "ABSECON":           "NEW JERSEY",
    # International full-address variants
    "SAN JOSE, COSTA RICA, CENTRAL AMERICA": "COSTA RICA",
    # OCR typos / punctuation variants
    "MISISSIPPI":        "MISSISSIPPI",
    "ARKINGSAS":         "ARKANSAS",
    "CONN.":             "CONNECTICUT",
    "TENNESSEE RIVER TENNESSEE": "TENNESSEE",
    # National parks (vacation guide section sets state = park name)
    "HOT SPRINGS":       "ARKANSAS",
    "ISLE ROYALE":       "MICHIGAN",
    "KINGS CANYON":      "CALIFORNIA",
    "LASSEN VOLCANIC":   "CALIFORNIA",
    "MAMMOTH CAVE":      "KENTUCKY",
    "MESA VERDE":        "COLORADO",
    "MOUNT MCKINLEY":    "ALASKA",
    "ROCKY MOUNTAIN":    "COLORADO",
    "SEQUOIA":           "CALIFORNIA",
    "SHENANDOAH":        "VIRGINIA",
    "WIND CAVE":         "SOUTH DAKOTA",
    "YELLOWSTONE":       "WYOMING",
    "YOSEMITE":          "CALIFORNIA",
    "ZION":              "UTAH",
}

# Known city names that appear as state values due to Issue 6.
# Not exhaustive — the state-validity flag catches most cases.
_COMMON_CITIES = {
    "NEW YORK CITY", "LOS ANGELES", "CHICAGO", "HOUSTON", "PHILADELPHIA",
    "ATLANTA", "MIAMI", "DETROIT", "BOSTON", "SAN FRANCISCO",
    "BALTIMORE", "CLEVELAND", "PITTSBURGH", "MEMPHIS", "NEW ORLEANS",
    "BIRMINGHAM", "RICHMOND", "WASHINGTON",
}


def detect_schema(fieldnames: list[str]) -> dict[str, str]:
    """Map logical field roles to actual CSV column names.

    Handles both the old schema (state, city, category) and newer volumes
    that use location_state_country, location_city, section_category.
    Returns keys: state, city, category, name, address (only those found).
    """
    candidates = {
        "state":    ("state", "location_primary", "location_state", "location_state_country"),
        "city":     ("city", "location_city"),
        "category": ("category", "section_category"),
        "name":     ("name", "business_name", "establishment_name"),
        "address":  ("address",),
    }
    return {
        role: next((c for c in cols if c in fieldnames), cols[0])
        for role, cols in candidates.items()
    }


def normalize_state(val: str) -> str:
    """Return canonical state name for val, or val unchanged if not in STATE_ALIASES."""
    return STATE_ALIASES.get(val.strip().upper(), val.strip())


def flag_state_invalid(state: str) -> bool:
    """True if state is not in the known state/country list."""
    if not state:
        return False
    s = state.strip().upper()
    return s not in US_STATES


def flag_state_eq_city(state: str, city: str) -> bool:
    """True when state and city are identical (city promoted to state)."""
    if not state or not city:
        return False
    return state.strip().upper() == city.strip().upper()


def flag_header_row(name: str, address: str, city: str, category: str) -> bool:
    """True if the row is a section header extracted as an entry.

    Two cases:
    1. Name is empty — blank rows that slipped through NER.
    2. Name is a bare state/country name (in US_STATES) with no address,
       city, or category — a geographic section divider, not a real entry.
    """
    name = (name or "").strip()
    if not name:
        return True
    if not (address or "").strip() and not (city or "").strip() and not (category or "").strip():
        return name.upper() in US_STATES
    return False


def flag_name_eq_address(name: str, address: str) -> bool:
    """True when the name field is a substring of the address or vice-versa."""
    if not name or not address:
        return False
    n = name.strip().lower()
    a = address.strip().lower()
    if n == a:
        return True
    # Address-as-name: name looks like a street (starts with a number or ends
    # in a street suffix).
    street_suffixes = (
        " st", " st.", " ave", " ave.", " blvd", " rd", " rd.", " dr",
        " dr.", " ln", " ln.", " hwy", " hwy.", " pl", " pl.",
    )
    if any(n.endswith(s) for s in street_suffixes):
        return True
    if re.match(r"^\d+\s", n):  # starts with a house number
        return True
    return False


# ---------------------------------------------------------------------------
# 6. Deduplication
# ---------------------------------------------------------------------------

def _completeness_score(row: dict) -> tuple[int, int]:
    """Score a row by data completeness. Higher is better.

    Returns (non-empty field count, total char length of name+address) so that
    a display-ad row with a full address beats a sparse directory-listing row.
    """
    key_fields = ["name", "address", "city", "state", "category", "phone"]
    non_empty = sum(1 for f in key_fields if (row.get(f) or "").strip())
    char_len = sum(len((row.get(f) or "").strip()) for f in ["name", "address"])
    return (non_empty, char_len)


def find_duplicates(
    rows: list[dict],
    threshold: float = 0.85,
    city_col: str = "city",
    state_col: str = "state",
    name_col: str = "name",
) -> set[int]:
    """Return indices of rows that are near-duplicates of a more-complete row.

    Blocks by (city, state), fuzzy-matches on (name|address) within each block.
    Near-duplicate pairs are merged into clusters via union-find; within each
    cluster the most complete row is kept (scored by non-empty field count then
    total character length of name+address). All others are flagged.
    """
    from collections import defaultdict

    blocks: dict[tuple, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        block_key = (
            (row.get(city_col) or "").strip().lower(),
            (row.get(state_col) or "").strip().lower(),
        )
        blocks[block_key].append(idx)

    # Precompute match keys once per row.
    keys: list[str] = [
        (row.get(name_col) or "").strip().lower()
        + "|"
        + (row.get("address") or "").strip().lower()
        for row in rows
    ]

    # Collect all near-duplicate pairs across blocks.
    # Skip blocks larger than MAX_BLOCK to avoid O(n²) blowup on large cities.
    _MAX_BLOCK = 150
    pairs: list[tuple[int, int]] = []
    for block_indices in blocks.values():
        if len(block_indices) > _MAX_BLOCK:
            continue
        for pos_i, i in enumerate(block_indices):
            for j in block_indices[pos_i + 1:]:
                if difflib.SequenceMatcher(None, keys[i], keys[j]).ratio() >= threshold:
                    pairs.append((i, j))

    if not pairs:
        return set()

    # Union-find to group transitively linked duplicates into clusters
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)

    # Group all seen indices by their cluster root
    seen = {idx for pair in pairs for idx in pair}
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in seen:
        clusters[find(idx)].append(idx)

    # Within each cluster keep the most complete row; flag the rest
    duplicate_indices: set[int] = set()
    for cluster in clusters.values():
        best = max(cluster, key=lambda i: _completeness_score(rows[i]))
        duplicate_indices.update(idx for idx in cluster if idx != best)

    return duplicate_indices


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_csv(
    src: Path,
    dst: Path,
    dry_run: bool = False,
    dedup_threshold: float = 0.85,
    infer_categories: bool = False,
) -> dict:
    """Apply all fixes to src, write result to dst. Return summary stats."""

    rows: list[dict] = []
    with src.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    schema = detect_schema(fieldnames)
    state_col    = schema.get("state", "state")
    city_col     = schema.get("city", "city")
    category_col = schema.get("category", "category")
    name_col     = schema.get("name", "name")

    # Collect the actual column names so unicode normalization covers them all
    schema_fields = tuple(schema.values())

    stats = {
        "total": len(rows),
        "unicode_fixes": 0,
        "category_fixes": 0,
        "inferred_categories": 0,
        "state_corrections": 0,
        "flag_state_invalid": 0,
        "flag_state_eq_city": 0,
        "flag_name_address": 0,
        "flag_header_row": 0,
        "flag_duplicate": 0,
        "flag_unanchored": 0,
    }

    # ── Apply transformations ────────────────────────────────────────────────
    for row in rows:
        # 1. Unicode normalization
        row, n = normalize_unicode_row(row, extra_fields=schema_fields)
        stats["unicode_fixes"] += n

        # 2. Category normalization
        if row.get(category_col):
            normalized, changed = normalize_category(row[category_col])
            if changed:
                row[category_col] = normalized
                stats["category_fixes"] += 1

        # 2b. Category inference from business name
        if infer_categories:
            current_cat = row.get(category_col, "").strip().lower()
            if current_cat in _INFER_TARGET_CATEGORIES:
                inferred = infer_category_from_name(
                    row.get(name_col, ""),
                    row.get("notes", ""),
                )
                if inferred:
                    row[category_col] = inferred
                    stats["inferred_categories"] += 1

    # ── State normalization ──────────────────────────────────────────────────
    for row in rows:
        raw = (row.get(state_col) or "").strip()
        corrected = normalize_state(raw)
        if corrected != raw:
            row[state_col] = corrected
            stats["state_corrections"] += 1

    # ── Flags ────────────────────────────────────────────────────────────────
    for row in rows:
        row["flag_state_invalid"] = ""
        row["flag_state_eq_city"] = ""
        row["flag_name_address"] = ""
        row["flag_header_row"] = ""
        row["flag_duplicate"] = ""

        if flag_state_invalid(row.get(state_col, "")):
            row["flag_state_invalid"] = "1"
            stats["flag_state_invalid"] += 1

        if flag_state_eq_city(row.get(state_col, ""), row.get(city_col, "")):
            row["flag_state_eq_city"] = "1"
            stats["flag_state_eq_city"] += 1

        if flag_name_eq_address(row.get(name_col, ""), row.get("address", "")):
            row["flag_name_address"] = "1"
            stats["flag_name_address"] += 1

        if flag_header_row(
            row.get(name_col, ""), row.get("address", ""),
            row.get(city_col, ""), row.get(category_col, ""),
        ):
            row["flag_header_row"] = "1"
            stats["flag_header_row"] += 1

    # ── Deduplication ────────────────────────────────────────────────────────
    dup_indices = find_duplicates(
        rows, threshold=dedup_threshold,
        city_col=city_col, state_col=state_col, name_col=name_col,
    )
    for idx in dup_indices:
        rows[idx]["flag_duplicate"] = "1"
    stats["flag_duplicate"] = len(dup_indices)

    # ── Unanchored entries (likely hallucinated) ─────────────────────────────
    # Entries with no #xywh= in canvas_fragment AND no address couldn't be
    # matched to any line on the page — strong signal of NER hallucination.
    # Entries with an address but no xywh are usually legitimate (e.g. NYC
    # attraction guide pages where full-page canvas URI is appropriate).
    flag_unanchored = 0
    for row in rows:
        row["flag_unanchored"] = ""
        cf = row.get("canvas_fragment", "")
        has_address = bool((row.get("address") or "").strip())
        if cf and "#xywh=" not in cf and not has_address:
            row["flag_unanchored"] = "1"
            flag_unanchored += 1
    stats["flag_unanchored"] = flag_unanchored

    # ── Write output ─────────────────────────────────────────────────────────
    out_fieldnames = fieldnames + [
        f for f in ("flag_state_invalid", "flag_state_eq_city",
                    "flag_name_address", "flag_header_row",
                    "flag_duplicate", "flag_unanchored")
        if f not in fieldnames
    ]

    if not dry_run:
        with dst.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=out_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return stats


def _print_stats(path: Path, stats: dict) -> None:
    total = stats["total"]
    print(f"\n{path.parent.name}/{path.name}")
    print(f"  rows:              {total}")
    print(f"  unicode fixes:     {stats['unicode_fixes']}")
    print(f"  category fixes:    {stats['category_fixes']}")
    print(f"  inferred cats:     {stats['inferred_categories']}")
    print(f"  state corrections: {stats['state_corrections']}")
    print(f"  flag_state_invalid:{stats['flag_state_invalid']:>5}  "
          f"({100*stats['flag_state_invalid']/max(total,1):.1f}%)")
    print(f"  flag_state=city:   {stats['flag_state_eq_city']:>5}  "
          f"({100*stats['flag_state_eq_city']/max(total,1):.1f}%)")
    print(f"  flag_name=address: {stats['flag_name_address']:>5}  "
          f"({100*stats['flag_name_address']/max(total,1):.1f}%)")
    print(f"  flag_header_row:   {stats['flag_header_row']:>5}  "
          f"({100*stats['flag_header_row']/max(total,1):.1f}%)")
    print(f"  flag_duplicate:    {stats['flag_duplicate']:>5}  "
          f"({100*stats['flag_duplicate']/max(total,1):.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process an entries CSV: normalize unicode/categories, add QA flags, deduplicate."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        help="Path to entries CSV. Omit when using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every entries_*.csv under the Green Books collection.",
    )
    parser.add_argument(
        "--collection",
        default="output/green_books_and_related/the_green_book_9ea5d5b0",
        help="Collection root when using --all.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the source CSV (default: write <stem>_fixed.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing any files.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for duplicate detection (default: 0.85).",
    )
    parser.add_argument(
        "--infer-categories",
        action="store_true",
        help=(
            "Infer canonical category from entry name for rows where category "
            "is 'General', the combined Hotels/Motels heading, or empty."
        ),
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite",
        help="Model slug used in entries CSV filename (default: gemini-3.1-flash-lite).",
    )
    args = parser.parse_args()

    if args.all:
        pattern = f"{args.collection}/*/entries_{args.model}.csv"
        paths = sorted(Path(p) for p in glob.glob(pattern))
        if not paths:
            print(f"No CSVs found matching: {pattern}", file=sys.stderr)
            sys.exit(1)
    elif args.csv:
        paths = [Path(args.csv)]
    else:
        parser.print_help()
        sys.exit(1)

    aggregate: dict = {
        "total": 0, "unicode_fixes": 0, "category_fixes": 0,
        "inferred_categories": 0, "state_corrections": 0,
        "flag_state_invalid": 0, "flag_state_eq_city": 0,
        "flag_name_address": 0, "flag_header_row": 0, "flag_duplicate": 0,
    }

    for src in paths:
        if not src.exists():
            print(f"Not found: {src}", file=sys.stderr)
            continue

        dst = src if args.inplace else src.with_stem(src.stem + "_fixed")

        stats = process_csv(src, dst, dry_run=args.dry_run,
                            dedup_threshold=args.dedup_threshold,
                            infer_categories=args.infer_categories)
        _print_stats(src, stats)

        if not args.dry_run:
            print(f"  → wrote {dst}")

        for k in aggregate:
            aggregate[k] += stats[k]

    if len(paths) > 1:
        print(f"\n{'='*50}")
        print(f"AGGREGATE ({len(paths)} volumes)")
        _print_stats(Path("total"), aggregate)


if __name__ == "__main__":
    main()
