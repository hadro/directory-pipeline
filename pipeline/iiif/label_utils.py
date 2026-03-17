"""Shared label-building utilities for IIIF annotation export scripts.

Used by export_entry_boxes.py and export_annotations.py to build human-readable
annotation body text and to parse entry schema field names from NER prompt files.
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# NER prompt schema parsing
# ---------------------------------------------------------------------------

# Fields that carry geographic/structural context rather than entry content.
CONTEXT_FIELDS: frozenset[str] = frozenset({
    "state", "city", "town", "alpha_group", "section", "category",
    "subcategory", "volume_year", "page_context",
})

# Keywords that identify name-like fields.
NAME_KEYWORDS: tuple[str, ...] = ("name", "firm", "establishment", "business", "company")

# Keywords that identify address-like fields.
ADDR_KEYWORDS: tuple[str, ...] = ("address", "addr", "location", "street")


def parse_ner_label_fields(ner_prompt_path: Path) -> tuple[list[str], list[str]]:
    """Return (name_fields, addr_fields) parsed from a ner_prompt.md entry schema.

    Scans bullet-list lines of the form ``- `field_name`: description`` inside the
    prompt to discover which fields represent the primary label and the address for
    this collection type.  Returns empty lists if the file cannot be read or no
    schema bullets are found — callers should fall back to built-in defaults.
    """
    try:
        text = ner_prompt_path.read_text(encoding="utf-8")
    except OSError:
        return [], []

    # Match field names in bullet list items (- or *).
    # Handles two common formats: `field_name`: ... and **field_name**: ...
    raw = re.findall(
        r"^\s*[-*]\s+(?:`(\w+)`|\*\*(\w+)\*\*)\s*:",
        text, re.MULTILINE,
    )
    # findall with groups returns tuples; collapse to the non-empty capture.
    fields = [a or b for a, b in raw]

    # Strip context/structural fields; keep only entry payload fields.
    payload = [f for f in fields if f not in CONTEXT_FIELDS]

    name_fields = [f for f in payload if any(kw in f.lower() for kw in NAME_KEYWORDS)]
    addr_fields = [f for f in payload if any(kw in f.lower() for kw in ADDR_KEYWORDS)]
    return name_fields, addr_fields


# ---------------------------------------------------------------------------
# Entry label building
# ---------------------------------------------------------------------------

# Default field name priority lists — cover Green Book, brewery guides,
# phone directories, and other common collection types.
DEFAULT_NAME_FIELDS: list[str] = ["establishment_name", "business_name", "name", "firm_name"]
DEFAULT_ADDR_FIELDS: list[str] = ["raw_address", "address"]


def build_entry_label(
    entry: dict,
    name_fields: list[str] | None = None,
    addr_fields: list[str] | None = None,
) -> str:
    """Return a one-line label for an entry: name — address — city, state.

    Tries each field name in *name_fields* and *addr_fields* in order, using
    the first non-empty value found.  Falls back to :data:`DEFAULT_NAME_FIELDS`
    and :data:`DEFAULT_ADDR_FIELDS` when the lists are omitted.
    """
    if name_fields is None:
        name_fields = DEFAULT_NAME_FIELDS
    if addr_fields is None:
        addr_fields = DEFAULT_ADDR_FIELDS

    name = next((entry[f] for f in name_fields if entry.get(f)), None)
    parts = [name or ""]

    addr = next((entry[f] for f in addr_fields if entry.get(f)), None)
    if addr:
        parts.append(addr)

    city_state = ", ".join(filter(None, [
        entry.get("city"),
        entry.get("state") or entry.get("town"),
    ]))
    if city_state:
        parts.append(city_state)

    return " — ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Missing NER prompt warning / confirmation
# ---------------------------------------------------------------------------

def handle_missing_ner_prompt() -> None:
    """Warn that no NER prompt was found and, in interactive sessions, ask the
    user to confirm before proceeding with built-in field-name defaults.

    Calls ``sys.exit(0)`` if the user declines.  In non-interactive (piped)
    contexts, prints a warning and returns so scripts and CI don't hang.
    """
    if sys.stdin.isatty():
        print(
            "Warning: no ner_prompt.md found in this directory.\n"
            "  Annotation labels will use built-in field-name defaults\n"
            "  (establishment_name / business_name / name …).\n"
            "  Use --ner-prompt PATH to supply one.\n"
            "Proceed anyway? [y/N] ",
            end="", file=sys.stderr,
        )
        answer = input().strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.", file=sys.stderr)
            sys.exit(0)
    else:
        print(
            "Warning: no ner_prompt.md found — proceeding with built-in "
            "field-name defaults.  Pass --ner-prompt PATH to override.",
            file=sys.stderr,
        )
