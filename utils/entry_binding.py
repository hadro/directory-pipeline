#!/usr/bin/env python3
"""Bind extracted entries to the aligned OCR line they came from.

Two places need to agree about which aligned line an entry belongs to:
``extract_entries`` sets ``canvas_fragment`` as it writes each entry, and
``tools/patch_canvas_fragments`` retro-fits fragments into a CSV extracted
before alignment ran. They used to carry separate copies of the matcher and
both copies had the same bug, so the matcher lives here, defined once.

What went wrong before
----------------------
The old matcher tried an entry's known text fields, then fell through to *every*
string field longer than three characters, guarded by a denylist of four field
names (``section``, ``subsection``, ``state``, ``city``). Two failures followed:

1. ``category`` was not on the denylist, and in a Green Book a category value is
   the literal section heading printed on the page. An entry whose name and
   address were already taken fell through to its category, exact-matched the
   heading line, and stole the heading's box. Measured across the published
   corpus this was the single largest cause of misplaced boxes — 489 of ~900
   attributable cases, ahead of every other field combined.
2. The denylist is a list of *names*, but each collection's NER prompt invents
   its own. ``location_city``, ``state_country``, ``state_or_province`` and
   ``state_region`` all walked straight past a guard aimed at ``city``/``state``.

A denylist cannot govern an open vocabulary. This module uses an allowlist
instead, and it admits fields by *shape* rather than by exact name so that a new
collection calling its field ``ship_name`` still works while one calling it
``mileage`` does not.

How a line is chosen
--------------------
Every scoring decision here was measured against the published corpus rather
than reasoned about, and two plausible alternatives lost:

* **Token-set F1, floor 0.34.** Recall-weighted scoring — counting how much of
  the entry appears in the line, without penalising a line that carries more —
  looks better suited to directory lines that legitimately run long. It scored
  worse on both axes (more regressions, fewer repairs). F1 stayed.
* **Global one-to-one assignment, greedy by descending score.** Constraining the
  assignment to be monotonic in line order — entries arrive in reading order, so
  their lines ought to as well — is intuitive and was much worse: it refused
  bindings scoring a perfect 1.0, because NER entry order does not reliably
  track aligned-line order. Reading order is reported by :func:`order_violations`
  as a diagnostic and never constrains the match.

A minimum-overlap guard rejects agreement resting on one short token, which is
what let an entry named "Ave" or "Hall" bind to the first line containing that
substring.

Refusing is a result
--------------------
An entry with no line above the floor gets ``None``, not a best guess. The
failure being fixed is a confidently wrong box; a missing box is visible and
routes to review, and :func:`review_queue` reports each refusal with its best
sub-threshold candidate so a human sees the near miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Fields whose value is the entry's own identity text. Matched by shape, not by
# an exhaustive list of names: any field ending in `_name` qualifies, which
# covers firm_name / business_name / establishment_name / ship_name across
# collections without admitting the context and detail fields that caused the
# original bug.
IDENTITY_EXACT = ("line_text", "name", "subject", "title", "caption")
IDENTITY_SUFFIX = ("_name",)

# Address-like fields, appended to the identity text because a directory line
# usually prints name and address together and the address disambiguates
# entries sharing a name.
ADDRESS_EXACT = ("address", "street", "location_address")
ADDRESS_SUFFIX = ("_address",)

# Never identity, whatever their shape. `section_name`/`category_name` would
# otherwise slip through the `_name` suffix rule and reintroduce the heading
# steal this module exists to prevent.
NEVER_IDENTITY = frozenset({
    "section_name", "category_name", "subsection_name", "state_name",
    "city_name", "region_name", "country_name", "column_name",
})

# Score below which a line is not considered a match at all.
DEFAULT_FLOOR = 0.34

# Agreement resting on a single short token is coincidence, not a match.
MIN_SHARED_TOKENS = 2
MIN_LONE_TOKEN_LEN = 5

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_XYWH = re.compile(r"#xywh=(\d+),(\d+),(\d+),(\d+)")


def tokens(text: str) -> list[str]:
    """Lowercase and split on anything that is not alphanumeric."""
    return [t for t in _NON_ALNUM.sub(" ", (text or "").lower()).split() if t]


def _is_identity(field: str) -> bool:
    if field in NEVER_IDENTITY:
        return False
    return field in IDENTITY_EXACT or field.endswith(IDENTITY_SUFFIX)


def _is_address(field: str) -> bool:
    if field in NEVER_IDENTITY:
        return False
    return field in ADDRESS_EXACT or field.endswith(ADDRESS_SUFFIX)


def _first_field(entry: dict, pick) -> str:
    """The first populated field satisfying *pick*, in the entry's own order."""
    for field, value in entry.items():
        if isinstance(value, str) and value.strip() and pick(field):
            return value
    return ""


def identity_text(entry: dict) -> str:
    """The entry's own text: its name, plus its address when it has one.

    Takes the first populated field of each kind rather than concatenating every
    match, so an entry carrying both `name` and `firm_name` does not get its
    name counted twice.
    """
    parts = [t for t in (_first_field(entry, _is_identity),
                         _first_field(entry, _is_address)) if t]
    return " ".join(parts)


def identity_tokens(entry: dict) -> list[str]:
    return tokens(identity_text(entry))


def identity_variants(entry: dict) -> list[list[str]]:
    """Token sets worth trying for one entry, strongest first.

    A directory entry frequently wraps: the name prints on one line and the
    address indents onto the next, and each line gets its own bbox. Scoring only
    "name + address" then charges the entry for address tokens that were never
    on the line it belongs to — measured on the corpus, ``Hotel Donnelly`` +
    ``1st & Broadway`` scores 0.33 against ``Hotel Donnelly-200 rooms-E. $2.00
    up`` and is refused, though the name is right there.

    So name-and-address is tried first, because it discriminates between
    entries sharing a name, and the name alone is tried as a fallback.
    """
    full = identity_tokens(entry)
    name_only = tokens(_first_field(entry, _is_identity))
    variants = [full]
    if name_only and name_only != full:
        variants.append(name_only)
    return variants


def _shared(entry_tokens: list[str], line_tokens: list[str]) -> "set[str] | None":
    """Tokens common to both, or None when the overlap is too thin to mean anything."""
    if not entry_tokens or not line_tokens:
        return None
    shared = set(entry_tokens) & set(line_tokens)
    if not shared:
        return None
    if len(shared) < MIN_SHARED_TOKENS and max(len(t) for t in shared) < MIN_LONE_TOKEN_LEN:
        return None
    return shared


def score(entry_tokens: list[str], line_tokens: list[str]) -> float:
    """Token-set F1 between an entry's identity text and a line, 0.0 if too thin.

    Used for *ranking* candidates. Whether a candidate is acceptable at all is a
    separate question — see :func:`is_match`, because F1 alone rejects some
    correct bindings.
    """
    shared = _shared(entry_tokens, line_tokens)
    if shared is None:
        return 0.0
    precision = len(shared) / len(set(line_tokens))
    recall = len(shared) / len(set(entry_tokens))
    return 2 * precision * recall / (precision + recall)


def contains_entry(entry_tokens: list[str], line_tokens: list[str]) -> bool:
    """True when every token of the entry's identity text appears in the line.

    Directory lines carry more than the entry does — rates, room codes, phone
    numbers — and F1 charges the entry for all of it. A Green Book listing like
    ``Broadmoor Hotel-(450B) E. $5.25 up A. $1`` shares every token of the name
    "Broadmoor Hotel" yet scores 0.36, under the floor, purely because of the
    tariff notation. Measured on the corpus, refusing those cost more correct
    bindings than it prevented wrong ones.

    Complete containment of a multi-token name is not something that happens by
    accident, and MIN_SHARED_TOKENS keeps a lone short token from qualifying.
    """
    shared = _shared(entry_tokens, line_tokens)
    return shared is not None and len(shared) == len(set(entry_tokens))


def is_match(entry_tokens: list[str], line_tokens: list[str],
             floor: float = DEFAULT_FLOOR) -> bool:
    """Whether a line is an acceptable candidate: strong agreement, or containment."""
    return (score(entry_tokens, line_tokens) >= floor
            or contains_entry(entry_tokens, line_tokens))


def is_edge_bleed(canvas_fragment: str, canvas_width: int, edge_pct: float = 0.05) -> bool:
    """True when the bbox is a narrow strip hard against the left or right edge.

    Two-page spread scans include a sliver of the facing page at the extreme
    edge; Surya detects those slivers as text lines. A percentage keeps the
    threshold proportional across collections scanned at different resolutions.
    """
    if not canvas_fragment or not canvas_width:
        return False
    m = _XYWH.search(canvas_fragment)
    if not m:
        return False
    x, w = int(m.group(1)), int(m.group(3))
    edge_px = canvas_width * edge_pct
    if w >= edge_px:
        return False
    return x < edge_px or (x + w) > (canvas_width - edge_px)


@dataclass(frozen=True)
class Binding:
    """One entry's outcome. ``line_index is None`` means deliberately unbound."""
    entry_index: int
    line_index: int | None
    score: float
    best_rejected: int | None = None       # highest-scoring line below the floor
    best_rejected_score: float = 0.0


def bind_entries(
    entries: list[dict],
    lines: list[dict],
    *,
    floor: float = DEFAULT_FLOOR,
    canvas_width: int = 0,
) -> list[Binding]:
    """Assign each entry at most one aligned line, and each line at most one entry.

    Greedy over all (entry, line) pairs scoring at or above *floor*, highest
    first, which yields a stable one-to-one assignment: the strongest agreement
    on the page is settled before any weaker claim can take its line. Entries
    with no candidate above the floor come back unbound, carrying their best
    rejected candidate for review.
    """
    entry_variants = [identity_variants(e) for e in entries]
    line_tokens = [tokens(ln.get("gemini_text", "")) for ln in lines]

    eligible = [
        j for j, ln in enumerate(lines)
        if not is_edge_bleed(ln.get("canvas_fragment", ""), canvas_width)
    ]

    scored: list[tuple[float, int, int]] = []
    best_below: dict[int, tuple[float, int]] = {}
    for i in range(len(entries)):
        for j in eligible:
            # Best over the entry's variants: name-and-address discriminates, and
            # name alone rescues an entry whose address wrapped to another line.
            s = max(score(v, line_tokens[j]) for v in entry_variants[i])
            # Acceptable via strong agreement OR complete containment; ranked by
            # F1 either way, so a contained-but-diluted match settles after any
            # stronger claim on the same line rather than pre-empting it.
            if any(is_match(v, line_tokens[j], floor) for v in entry_variants[i]):
                scored.append((s, i, j))
            elif s > best_below.get(i, (0.0, -1))[0]:
                best_below[i] = (s, j)

    # Sort by descending score; ties broken by position so the result does not
    # depend on dict or glob ordering.
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))

    taken_entries: dict[int, tuple[int, float]] = {}
    taken_lines: set[int] = set()
    for s, i, j in scored:
        if i in taken_entries or j in taken_lines:
            continue
        taken_entries[i] = (j, s)
        taken_lines.add(j)

    out: list[Binding] = []
    for i in range(len(entries)):
        if i in taken_entries:
            j, s = taken_entries[i]
            out.append(Binding(i, j, s))
        else:
            s, j = best_below.get(i, (0.0, None))
            out.append(Binding(i, None, 0.0, best_rejected=j, best_rejected_score=s))
    return out


def order_violations(bindings: list[Binding]) -> int:
    """Count backwards jumps in line order across the bound entries.

    A diagnostic only. Entries are emitted in reading order and their lines
    usually follow, but not reliably enough to constrain the assignment — doing
    so was measured and rejected. A high count is worth surfacing for review.
    """
    seq = [b.line_index for b in bindings if b.line_index is not None]
    return sum(1 for a, b in zip(seq, seq[1:]) if b < a)


def review_queue(entries: list[dict], lines: list[dict], bindings: list[Binding]) -> list[dict]:
    """The entries this page could not place, with their nearest miss."""
    queue = []
    for b in bindings:
        if b.line_index is not None:
            continue
        near = lines[b.best_rejected] if b.best_rejected is not None else None
        queue.append({
            "entry_index": b.entry_index,
            "entry_text": identity_text(entries[b.entry_index]),
            "nearest_line": (near or {}).get("gemini_text", ""),
            "nearest_fragment": (near or {}).get("canvas_fragment", ""),
            "nearest_score": round(b.best_rejected_score, 3),
        })
    return queue
