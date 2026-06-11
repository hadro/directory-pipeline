#!/usr/bin/env python3
"""Extract structured directory entries from aligned OCR output using Gemini NER.

Reads *_{model}_aligned.json files in sorted (page) order and calls the Gemini
API to identify and extract structured entries: establishment name, address,
city, state, category, advertisement flag, and phone number.

Cross-page context (last active state / city / category heading) is carried
forward between pages so entries at the top of a page that inherit context
from the prior page are correctly attributed.

Input (per page):
  {stem}_{model_slug}_aligned.json   — Gemini-corrected lines with bounding boxes
                                       (preferred; includes precise canvas fragments)
  {stem}_{model_slug}.txt            — raw Gemini OCR text (fallback; canvas_fragment
                                       will be the bare canvas URI, not a spatial rect)

Output (per page):
  {stem}_{model_slug}_entries.json   — extracted entries with canvas fragments

Output (per item directory):
  entries_{model_slug}.csv           — flat CSV of all entries across all pages

Usage
-----
Core path (no alignment required):
    python pipeline/extract_entries.py output/greenbooks/feb978b0

With aligned JSON (adds precise bounding-box fragments):
    python pipeline/extract_entries.py output/greenbooks/feb978b0 --mode multimodal
    python pipeline/extract_entries.py output/greenbooks/ --force
    python pipeline/extract_entries.py output/greenbooks/feb978b0 --dry-run
    python pipeline/extract_entries.py output/woods_directory_73644404 \
  --model gemini-2.5-flash-preview-04-17 \
  --aligned-model gemini-2.0-flash \
  --force

"""

import argparse
import csv
import difflib
import io
import json
import re
import sys
import threading
from pathlib import Path

from PIL import Image

from dotenv import load_dotenv
from google import genai


load_dotenv()
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

from utils.gemini import flex_http_options, generate_with_retry, get_client
from utils.models import DEFAULT_NER_MODEL, FALLBACK_MODEL, model_slug

# Sparse-page thresholds: pages below BOTH limits are skipped before the NER call.
# A blank or nearly-blank page passes enough ghost/bleed text to trigger hallucinations.
# "Both" (AND) logic keeps short-but-real single-entry pages from being silently dropped.
MIN_OCR_CHARS = 100       # characters in the assembled page text
MIN_ALIGNED_LINES = 3     # lines in the aligned JSON (process_page path)
MIN_SURYA_BBOXES = 5      # Surya bbox detections (process_page_txt path)
NER_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "ner_prompt.md"


def _load_scope(output_root: Path) -> "set[str] | None":
    """Return the set of image stems to process, or None (= process all).

    Reads included_pages.txt from output_root or output_root.parent.
    Returns None when no file is found (backward-compatible: process everything).
    Stems are stored without extension so they can be matched against both
    aligned JSON and raw .txt filenames.
    """
    for d in (output_root.resolve(), output_root.resolve().parent):
        p = d / "included_pages.txt"
        if p.exists():
            lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.startswith("#")]
            if lines:
                return {Path(l).stem for l in lines}
    return None


def _find_sibling_ner_prompts(output_root: Path) -> list[Path]:
    """Scan directories adjacent to output_root for ner_prompt.md files.

    Checks immediate siblings at output_root.parent and output_root.parent.parent
    (to handle both slug-level and item-level layouts). Excludes output_root itself.
    Returns up to 5 paths sorted by modification time (newest first).
    """
    found: list[Path] = []
    root = output_root.resolve()
    for search_parent in (root.parent, root.parent.parent):
        if not search_parent.is_dir():
            continue
        for d in search_parent.iterdir():
            if not d.is_dir() or d == root or d == root.parent:
                continue
            p = d / "ner_prompt.md"
            if p.exists():
                found.append(p)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:5]


def _find_ner_prompt(output_root: Path) -> Path:
    """Return a volume-specific ner_prompt.md if one exists alongside the images,
    otherwise return the global NER_PROMPT_FILE fallback.

    Search order:
      1. output_root itself (slug-level prompt)
      2. output_root.parent (legacy location)
      3. Any immediate subdir of output_root (per-item prompt from --generate-prompts)
    """
    root = output_root.resolve()
    for candidate_dir in (root, root.parent):
        p = candidate_dir / "ner_prompt.md"
        if p.exists():
            return p
    # Per-item prompts written by --generate-prompts into each item subdir
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                p = sub / "ner_prompt.md"
                if p.exists():
                    return p
    return NER_PROMPT_FILE

def _detect_aligned_slug(item_dir: Path) -> str | None:
    """Scan item_dir for aligned JSON or OCR txt files and return the model slug found.

    Searches for *_aligned.json first, then *.txt, extracting the model slug
    (e.g. 'gemini-2.0-flash') from the filename suffix. Returns None if nothing
    that looks like pipeline output is found.
    """
    import re
    slug_pat = re.compile(r"_(gemini[-\w.]+)_aligned\.json$")
    for p in sorted(item_dir.glob("*_aligned.json")):
        m = slug_pat.search(p.name)
        if m:
            return m.group(1)
    txt_pat = re.compile(r"_(gemini[-\w.]+)\.txt$")
    for p in sorted(item_dir.glob("*.txt")):
        m = txt_pat.search(p.name)
        if m:
            return m.group(1)
    return None


def _load_canvas_uris(item_dir: Path) -> list[str]:
    """Return canvas URIs in page order from manifest.json in item_dir.

    Supports IIIF v2 (sequences[0].canvases) and v3 (items).
    Returns [] if the manifest is absent or unreadable.
    """
    p = item_dir / "manifest.json"
    if not p.exists():
        return []
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    # IIIF v2
    seqs = m.get("sequences", [])
    if seqs:
        return [c.get("@id", "") for c in seqs[0].get("canvases", [])]
    # IIIF v3
    return [c.get("id", "") for c in m.get("items", [])]


def _infer_fields(entries: list[dict]) -> list[str]:
    """Return column order inferred from the union of all entry keys (insertion order preserved)."""
    return list(dict.fromkeys(k for e in entries for k in e))

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

# Gemini tiles images at 768 px; staying at or below this boundary keeps each
# page image to exactly one tile (258 tokens) regardless of original resolution.
_MULTIMODAL_MAX_PX = 768


def _resize_image_bytes(image_path: Path, max_px: int = _MULTIMODAL_MAX_PX) -> bytes:
    """Return JPEG bytes for image_path resized to fit within max_px × max_px."""
    img = Image.open(image_path)
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _call_gemini(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_text: str,
    image_path: Path | None = None,
    service_tier: str | None = None,
) -> str:
    """Call Gemini with optional image. Returns raw response text."""
    parts: list = []
    if image_path is not None:
        parts.append(Part.from_bytes(data=_resize_image_bytes(image_path), mime_type="image/jpeg"))
    parts.append(Part.from_text(text=user_text))

    response = generate_with_retry(
        client,
        model=model,
        config=GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=65536,
            thinking_config=ThinkingConfig(thinking_budget=0),
            http_options=flex_http_options(service_tier),
        ),
        contents=parts,
        log=_log,
    )
    return response.text or ""


def _strip_fence(text: str) -> str:
    """Strip markdown code fences from a model response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


def _parse_json_response(text: str) -> dict | None:
    """Parse a JSON object from a model response, stripping any markdown fences."""
    text = _strip_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract a JSON object if there's surrounding prose
        start = text.find("{")
        end = text.rfind("}") + 1
        if 0 <= start < end:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _recover_partial_json(text: str) -> dict | None:
    """Salvage complete entries from a response truncated by the model's token limit.

    When a page has many entries the model's output can be cut off mid-entry.
    This function uses raw_decode to walk the entries array and collect every
    complete entry that was emitted before the truncation point.

    Returns a dict with "entries" and "page_context" keys (like a normal
    response), or None if no complete entries could be recovered.
    """
    text = _strip_fence(text)
    decoder = json.JSONDecoder()

    # Recover page_context (appears before entries; it's a shallow object)
    page_context: dict = {}
    ctx_pos = text.find('"page_context"')
    if ctx_pos != -1:
        colon = text.find(":", ctx_pos + len('"page_context"'))
        brace = text.find("{", colon) if colon != -1 else -1
        if brace != -1:
            try:
                page_context, _ = decoder.raw_decode(text, brace)
            except json.JSONDecodeError:
                pass

    # Find the entries array
    entries_pos = text.find('"entries"')
    if entries_pos == -1:
        return None
    array_open = text.find("[", entries_pos)
    if array_open == -1:
        return None

    # Walk the array, parsing one complete object at a time
    entries: list[dict] = []
    i = array_open + 1
    while i < len(text):
        ch = text[i]
        if ch in " \t\n\r,":
            i += 1
            continue
        if ch == "]":
            break  # Normal end of array (response was not truncated)
        if ch == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                entries.append(obj)
                i = end
            except json.JSONDecodeError:
                break  # Truncation point — stop here
        else:
            break

    if not entries:
        return None
    return {"entries": entries, "page_context": page_context}


# ---------------------------------------------------------------------------
# Page text and context
# ---------------------------------------------------------------------------

def _normalize_line(text: str) -> str:
    """Collapse dot-leader runs and long whitespace to a single separator.

    Directory pages use long runs of dots or spaces between a name and an
    address (e.g. "Blue Moon Restaurant ........... 1811 Conover Street").
    These inflate output token counts dramatically when the model echoes them
    back verbatim in line_text fields.  Replace any run of 3+ dots or 3+
    spaces (after stripping leading/trailing whitespace) with a single tab,
    which is compact and still separates the two fields visually.
    """
    text = text.strip()
    # Collapse runs of 3+ dots (possibly mixed with spaces) to a tab
    text = re.sub(r'[.\s]{3,}', '\t', text)
    # Collapse any remaining runs of 2+ spaces to a single space
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _page_text_from_aligned(aligned: dict) -> str:
    """Build a plain-text page transcript from an aligned JSON, in line order."""
    lines = [
        _normalize_line(ln.get("gemini_text", ""))
        for ln in aligned.get("lines", [])
        if ln.get("gemini_text", "").strip()
    ]
    # Append unmatched lines, skipping short fragments (≤8 chars are typically
    # edge-bleed noise from adjacent pages that cause NER hallucinations).
    lines += [_normalize_line(t) for t in aligned.get("unmatched_gemini", []) if len(t.strip()) > 8]
    return "\n".join(lines)


def _build_user_message(page_text: str, prior_context: dict) -> str:
    ctx_lines = "\n".join(f"{k}: {v}" for k, v in prior_context.items()) or "(none)"
    return (
        f"## Prior page context\n"
        f"{ctx_lines}\n\n"
        f"## Page text\n"
        f"{page_text}\n\n"
        f"Return the JSON extraction."
    )


def _img_path_from_aligned(aligned_path: Path, slug: str) -> Path | None:
    """Reconstruct the source image path from an aligned JSON filename."""
    # Pattern: {stem}_{slug}_aligned.json  →  {stem}.jpg
    suffix = f"_{slug}_aligned.json"
    name = aligned_path.name
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    candidate = aligned_path.parent / f"{stem}.jpg"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Canvas fragment linking
# ---------------------------------------------------------------------------

def _find_fragment(line_text: str, aligned_lines: list[dict]) -> tuple[str | None, str | None]:
    """Find the canvas_fragment for an entry by matching its line_text to aligned lines.

    Tries three strategies in order:
    1. Exact match against gemini_text
    2. Substring match — handles the common Green Book format where the aligned
       line includes the city name and dot leaders before the establishment name,
       e.g. "LAKE GEORGE....Woodbine Cottage, 75 Dieskau Street"
    3. Fuzzy match as a last resort

    Returns (canvas_fragment, matched_gemini_text) or (None, None).
    """
    if not line_text or not aligned_lines:
        return None, None
    # Exact match
    for ln in aligned_lines:
        if ln.get("gemini_text", "") == line_text:
            return ln.get("canvas_fragment"), ln.get("gemini_text", "")
    # Substring match
    lt_lower = line_text.lower()
    for ln in aligned_lines:
        if lt_lower in ln.get("gemini_text", "").lower():
            return ln.get("canvas_fragment"), ln.get("gemini_text", "")
    # Fuzzy match (handles minor whitespace / OCR differences)
    candidates = [ln.get("gemini_text", "") for ln in aligned_lines]
    matches = difflib.get_close_matches(line_text, candidates, n=1, cutoff=0.6)
    if matches:
        for ln in aligned_lines:
            if ln.get("gemini_text", "") == matches[0]:
                return ln.get("canvas_fragment"), ln.get("gemini_text", "")
    return None, None


def _bbox_height(canvas_fragment: str) -> int | None:
    """Parse the height from a IIIF canvas fragment URI (#xywh=x,y,w,h)."""
    import re
    m = re.search(r"#xywh=\d+,\d+,\d+,(\d+)", canvas_fragment or "")
    return int(m.group(1)) if m else None


def _is_edge_bleed(canvas_fragment: str, canvas_width: int,
                   edge_pct: float = 0.05) -> bool:
    """Return True if the bbox is a narrow strip abutting the left or right edge.

    Two-page spread scans include a sliver of the facing page at the extreme
    left or right of the image. Surya detects these as text lines and the NW
    alignment can assign an entry to one. Reject any bbox that is both
    narrower than *edge_pct* of the canvas width AND starts/ends within
    *edge_pct* of either edge. Using a percentage keeps the threshold
    proportional across collections with different scan resolutions.
    """
    if not canvas_fragment or not canvas_width:
        return False
    import re
    m = re.search(r"#xywh=(\d+),\d+,(\d+),\d+", canvas_fragment)
    if not m:
        return False
    x, w = int(m.group(1)), int(m.group(2))
    edge_px = canvas_width * edge_pct
    if w >= edge_px:
        return False
    return x < edge_px or (x + w) > (canvas_width - edge_px)



# ---------------------------------------------------------------------------
# Per-page processing helpers
# ---------------------------------------------------------------------------

def _load_cached_result(
    out_path: Path,
    prior_context: dict,
    force: bool,
    dry_run: bool,
    source_path: "Path | None" = None,
) -> "dict | None":
    """Return cached per-page result from *out_path*, or None if the cache
    should be bypassed (force/dry_run), is absent, or is corrupt.

    If *source_path* is given (the aligned JSON the cache was built from),
    the cache is also invalidated when the source is newer — so re-aligning
    a page automatically triggers a fresh NER run.
    """
    if force or dry_run or not out_path.exists() or out_path.stat().st_size == 0:
        return None
    if source_path and source_path.exists():
        if source_path.stat().st_mtime > out_path.stat().st_mtime:
            return None
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return {
            "entries": data.get("entries", []),
            "page_context": data.get("page_context", prior_context),
            "status": "skipped",
        }
    except Exception:
        return None  # corrupt file — re-run


def _run_ner_call(
    client: genai.Client,
    page_name: str,
    model: str,
    system_prompt: str,
    user_msg: str,
    image_path: "Path | None",
    error_path: Path,
    fallback_model: "str | None",
    service_tier: "str | None" = None,
) -> "tuple[dict | None, bool, str | None]":
    """Call Gemini NER with primary + optional fallback model, then partial recovery.

    Returns (result, partial, api_error) where:
      - result      parsed response dict, or None on total failure
      - partial     True if result came from partial JSON recovery
      - api_error   non-empty string if the primary API call itself raised
    On unrecoverable parse failure, writes the raw response to *error_path*.
    On success, deletes *error_path* if it exists from a prior run.
    """
    # Primary API call
    try:
        last_raw = _call_gemini(client, model, system_prompt, user_msg, image_path, service_tier)
    except Exception as exc:
        return None, False, str(exc)
    result = _parse_json_response(last_raw)

    # Fallback when primary model output is unparseable
    if result is None and fallback_model and fallback_model != model:
        _log(
            f"    [{page_name}] primary model exhausted — retrying once"
            f" with fallback model {fallback_model}"
        )
        try:
            last_raw = _call_gemini(client, fallback_model, system_prompt, user_msg, image_path, service_tier)
            result = _parse_json_response(last_raw)
        except Exception as exc:
            _log(f"    [{page_name}] fallback model error: {exc}")

    # Partial JSON recovery for truncated responses
    partial = False
    if result is None:
        result = _recover_partial_json(last_raw)
        if result is None:
            error_path.write_text(last_raw, encoding="utf-8")
            return None, False, None
        partial = True
        _log(
            f"    [{page_name}] WARNING: response was truncated; "
            f"recovered {len(result.get('entries', []))} entries (may be incomplete)"
        )

    if error_path.exists():
        error_path.unlink()
    return result, partial, None


# ---------------------------------------------------------------------------
# Suspicious-entry review
# ---------------------------------------------------------------------------

def _flag_suspicious(entries: list[dict]) -> list[tuple[int, str]]:
    """Return (index, reason) pairs for entries that warrant a second look.

    Three signals:
    1. Isolated state run — a run of 1–2 consecutive entries whose geographic
       context field differs from both neighbouring runs, which are longer.
       Catches NER drift without firing on legitimate state transitions (which
       appear as large blocks at page boundaries).
    2. Empty name — every common name field is blank or None.
    3. Probable heading — short all-caps string with no address field, likely
       a section header that the NER mistook for an establishment.
    """
    def _state(e: dict) -> str | None:
        return e.get("location_primary") or e.get("state")

    def _name(e: dict) -> str:
        for k in ("name", "establishment_name", "firm_name", "business_name"):
            v = e.get(k)
            if v and str(v).strip():
                return str(v).strip()
        return ""

    flags: dict[int, list[str]] = {}

    # Signal 1: isolated state runs
    if entries:
        runs: list[list] = []  # [state, start_idx, end_idx]
        for i, e in enumerate(entries):
            s = _state(e)
            if runs and runs[-1][0] == s:
                runs[-1][2] = i
            else:
                runs.append([s, i, i])

        for j, (state, start, end) in enumerate(runs):
            if state is None:
                continue
            run_len = end - start + 1
            if run_len > 2:
                continue
            prev = runs[j - 1] if j > 0 else None
            nxt = runs[j + 1] if j < len(runs) - 1 else None
            # Require both neighbours — don't flag legitimate transitions at page edges
            if prev is None or nxt is None:
                continue
            if prev[0] == state or nxt[0] == state:
                continue
            # At least one neighbour must be a substantial block
            if (prev[2] - prev[1] + 1) <= 2 and (nxt[2] - nxt[1] + 1) <= 2:
                continue
            reason = f"location '{state}' isolated amid '{prev[0]}'/'{nxt[0]}' entries"
            for idx in range(start, end + 1):
                flags.setdefault(idx, []).append(reason)

    # Signal 2: empty name
    for i, e in enumerate(entries):
        if not _name(e):
            flags.setdefault(i, []).append("name field is empty")

    # Signal 3: probable heading
    for i, e in enumerate(entries):
        n = _name(e)
        address = str(e.get("address") or "").strip()
        if (n and n == n.upper() and len(n.split()) <= 4
                and not any(c.isdigit() for c in n) and not address):
            flags.setdefault(i, []).append(f"possible heading: '{n}' (all-caps, no address)")

    return [(idx, "; ".join(reasons)) for idx, reasons in sorted(flags.items())]


def _build_review_message(
    page_text: str,
    entries: list[dict],
    suspicious: list[tuple[int, str]],
) -> str:
    """Build the targeted re-check user message."""
    lines = [
        "## Review request\n",
        f"{len(suspicious)} entr{'y' if len(suspicious) == 1 else 'ies'} from this "
        "page were flagged for verification. Review the page text and image, then "
        "confirm or correct each flagged entry.\n",
    ]
    for idx, reason in suspicious:
        entry = entries[idx]
        core = {k: v for k, v in entry.items()
                if k not in ("canvas_fragment", "image", "review_note")}
        lines.append(f"[Entry {idx}] {reason}")
        lines.append(json.dumps(core, ensure_ascii=False))
        lines.append("")

    lines += [
        "Return JSON with a 'reviews' list. For each flagged entry include its index plus:",
        '  "correct": true               — entry is correct as extracted',
        "  corrected field key/value pairs — only the fields that need changing",
        '  "not_an_entry": true           — this is a heading or non-listing item\n',
        'Example: {"reviews": [{"index": 0, "correct": true}, '
        '{"index": 3, "location_primary": "NEW YORK"}, '
        '{"index": 7, "not_an_entry": true}]}\n',
        "## Page text",
        page_text,
    ]
    return "\n".join(lines)


def _run_review_call(
    client: genai.Client,
    page_name: str,
    model: str,
    system_prompt: str,
    review_msg: str,
    image_path: "Path | None",
    service_tier: "str | None" = None,
) -> "list[dict] | None":
    """Single-shot re-check call for suspicious entries.

    Always sends the image when available — even if the primary call was
    text-only — so the model has visual context for layout-dependent
    judgements (heading vs. entry, column breaks, etc.).
    Returns the 'reviews' list, or None on any failure.
    """
    try:
        raw = _call_gemini(client, model, system_prompt, review_msg, image_path, service_tier)
    except Exception as exc:
        _log(f"    [{page_name}] review call failed: {exc}")
        return None
    result = _parse_json_response(raw)
    if result is None:
        _log(f"    [{page_name}] review response unparseable — skipping")
        return None
    reviews = result.get("reviews")
    return reviews if isinstance(reviews, list) else None


def _apply_review_corrections(
    entries: list[dict],
    reviews: list[dict],
    suspicious: list[tuple[int, str]],
) -> int:
    """Merge model corrections into entries in-place; return count of changes.

    Adds 'review_note' to every flagged entry:
      "verified"              — model confirmed the entry is correct
      "corrected: f1, f2"    — model updated these fields
      "flagged: not_an_entry" — model says this is a heading / non-listing
    """
    flagged = {idx for idx, _ in suspicious}
    corrected = 0
    for review in reviews:
        idx = review.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(entries):
            continue
        if idx not in flagged:
            continue
        entry = entries[idx]
        if review.get("not_an_entry"):
            entry["review_note"] = "flagged: not_an_entry"
            corrected += 1
        elif review.get("correct"):
            entry["review_note"] = "verified"
        else:
            changed = [k for k in review if k != "index"]
            for k in changed:
                entry[k] = review[k]
            if changed:
                entry["review_note"] = "corrected: " + ", ".join(changed)
                corrected += 1
            else:
                entry["review_note"] = "verified"
    return corrected


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def process_page(
    client: genai.Client,
    aligned_path: Path,
    model: str,
    system_prompt: str,
    prior_context: dict,
    mode: str,
    force: bool,
    dry_run: bool,
    aligned_model: str | None = None,
    fallback_model: str | None = FALLBACK_MODEL,
    service_tier: str | None = None,
    review_suspicious: bool = False,
) -> dict:
    """
    Extract entries from one aligned JSON page.

    aligned_model: if set, the model whose slug appears in aligned_path's name.
    The output file is always tagged with the NER model slug.
    review_suspicious: if True, flag potentially mis-extracted entries after the
    primary NER call and make a targeted re-check call using the page image.

    Returns:
      {
        "entries": [...],
        "page_context": {...},
        "status": "ok" | "partial_recovery" | "skipped" | "empty" | "error:<msg>" | "parse_error",
      }
    """
    slug = model_slug(model)
    aligned_slug = model_slug(aligned_model) if aligned_model else slug
    suffix = f"_{aligned_slug}_aligned.json"
    stem = aligned_path.name[: -len(suffix)]
    out_path = aligned_path.parent / f"{stem}_{slug}_entries.json"

    if cached := _load_cached_result(out_path, prior_context, force, dry_run,
                                      source_path=aligned_path):
        return cached

    # Load aligned JSON
    try:
        aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"entries": [], "page_context": prior_context, "status": f"error:{exc}"}

    page_text = _page_text_from_aligned(aligned)
    if not page_text.strip():
        return {"entries": [], "page_context": prior_context, "status": "empty"}

    # Sparse-page guard: skip NER when both the aligned line count and the OCR
    # character count are below their thresholds. Blank/missing pages often leak
    # enough ghost text from the backing page to fool the model into hallucinating
    # entries; requiring BOTH signals to fail keeps legitimate short pages safe.
    n_lines = len(aligned.get("lines", []))
    if n_lines < MIN_ALIGNED_LINES and len(page_text) < MIN_OCR_CHARS:
        _log(
            f"    [{aligned_path.name}] sparse page ({n_lines} aligned lines, "
            f"{len(page_text)} chars) — skipping NER"
        )
        return {"entries": [], "page_context": prior_context, "status": "sparse_page"}

    if dry_run:
        return {
            "entries": [],
            "page_context": prior_context,
            "status": f"dry_run ({len(page_text.splitlines())} lines)",
        }

    user_msg = _build_user_message(page_text, prior_context)

    # Resolve image path for multimodal mode
    image_path: Path | None = None
    if mode == "multimodal":
        image_path = _img_path_from_aligned(aligned_path, aligned_slug)
        if image_path is None:
            _log(f"    Warning: image not found for {aligned_path.name}; falling back to text-only")

    error_path = aligned_path.parent / f"{stem}_{slug}_entries_error.txt"
    result, partial, api_error = _run_ner_call(
        client, aligned_path.name, model, system_prompt, user_msg,
        image_path, error_path, fallback_model, service_tier,
    )
    if api_error:
        return {"entries": [], "page_context": prior_context, "status": f"api_error:{api_error}"}
    if result is None:
        return {"entries": [], "page_context": prior_context, "status": "parse_error"}

    entries = result.get("entries", [])
    new_context = result.get("page_context") or prior_context

    # Propagate context fields into entries that are missing them.
    for entry in entries:
        for key, val in prior_context.items():
            if key not in entry and val:
                entry[key] = val

    # Optional second-pass review of suspicious entries.
    # Always tries multimodal (fetches image even for text-only primary calls)
    # so the model has visual context for layout-dependent decisions.
    if review_suspicious and entries:
        suspicious = _flag_suspicious(entries)
        if suspicious:
            review_img = image_path or _img_path_from_aligned(aligned_path, aligned_slug)
            review_msg = _build_review_message(page_text, entries, suspicious)
            reviews = _run_review_call(
                client, aligned_path.name, model, system_prompt,
                review_msg, review_img, service_tier,
            )
            if reviews:
                n_corrected = _apply_review_corrections(entries, reviews, suspicious)
                if n_corrected:
                    _log(
                        f"    [{aligned_path.name}] reviewed {len(suspicious)} suspicious"
                        f" → {n_corrected} corrected"
                    )

    # Link canvas fragments from the aligned JSON.
    # Try known text fields first, then every non-trivial string field in order until
    # one matches — this handles schemas where line_text is absent and the most
    # distinctive text (e.g. ship_name) isn't the first field in the dict.
    aligned_lines = aligned.get("lines", [])
    canvas_uri = aligned.get("canvas_uri", "")
    canvas_width = aligned.get("canvas_width") or 0
    used_fragments: set[str] = set()  # prevent multiple entries sharing one bounding box
    # Fields that represent inherited context headings, not the entry's own line text.
    # Matching against these causes the first entry after a heading to steal the
    # heading's bounding box (e.g. "Alabama" substring-matches "ALABAMA." heading,
    # "Brooklyn" substring-matches "BROOKLYN." heading).
    _HEADING_FIELDS = {"section", "subsection", "state", "city"}
    for entry in entries:
        cf = None
        # Prioritise specific entry-text fields; skip heading/context fields entirely.
        candidates = [
            entry.get("line_text", ""),
            entry.get("establishment_name", ""),
            entry.get("firm_name", ""),
            entry.get("business_name", ""),
            entry.get("name", ""),
            entry.get("address", "") or "",
        ]
        candidates += [
            v for k, v in entry.items()
            if k not in _HEADING_FIELDS and isinstance(v, str) and len(v) > 3
        ]
        for lt in candidates:
            if lt:
                candidate_uri, _ = _find_fragment(lt, aligned_lines)
                if (candidate_uri and candidate_uri not in used_fragments
                        and not _is_edge_bleed(candidate_uri, canvas_width)):
                    cf = candidate_uri
                    break
        if cf:
            used_fragments.add(cf)
        entry["canvas_fragment"] = cf or canvas_uri  # fall back to full canvas
        entry["image"] = aligned.get("image", "")

    output = {
        "image": aligned.get("image", ""),
        "model": model,
        "mode": mode,
        "prior_context": prior_context,
        "page_context": new_context,
        "entries": entries,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"entries": entries, "page_context": new_context, "status": "partial_recovery" if partial else "ok"}


def process_page_txt(
    client: genai.Client,
    txt_path: Path,
    canvas_uri: str,
    model: str,
    system_prompt: str,
    prior_context: dict,
    mode: str,
    force: bool,
    dry_run: bool,
    aligned_model: str | None = None,
    fallback_model: str | None = FALLBACK_MODEL,
    service_tier: str | None = None,
    review_suspicious: bool = False,
) -> dict:
    """Extract entries from one raw Gemini .txt page (no alignment data).

    Used when *_aligned.json files are not available. canvas_fragment for all
    entries is set to the bare canvas_uri — a link to the IIIF canvas page
    without spatial coordinates.

    Returns the same structure as process_page().
    """
    slug = model_slug(model)
    aligned_slug = model_slug(aligned_model) if aligned_model else slug
    suffix = f"_{aligned_slug}.txt"
    stem = txt_path.name[: -len(suffix)]
    out_path = txt_path.parent / f"{stem}_{slug}_entries.json"

    if cached := _load_cached_result(out_path, prior_context, force, dry_run):
        return cached

    # Read and normalise the raw text
    try:
        raw = txt_path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"entries": [], "page_context": prior_context, "status": f"error:{exc}"}

    page_text = "\n".join(_normalize_line(ln) for ln in raw.splitlines() if ln.strip())
    if not page_text.strip():
        return {"entries": [], "page_context": prior_context, "status": "empty"}

    # Sparse-page guard: cross-check OCR text length against Surya bbox count.
    # Both must be below threshold to skip — preserving legitimate short pages.
    if len(page_text) < MIN_OCR_CHARS:
        surya_path = txt_path.parent / f"{stem}_surya.json"
        surya_count = 0
        if surya_path.exists():
            try:
                surya_data = json.loads(surya_path.read_text(encoding="utf-8"))
                surya_count = len(surya_data.get("bboxes", surya_data.get("lines", [])))
            except Exception:
                pass
        if surya_count < MIN_SURYA_BBOXES:
            _log(
                f"    [{txt_path.name}] sparse page ({surya_count} surya bboxes, "
                f"{len(page_text)} chars) — skipping NER"
            )
            return {"entries": [], "page_context": prior_context, "status": "sparse_page"}

    if dry_run:
        return {
            "entries": [],
            "page_context": prior_context,
            "status": f"dry_run ({len(page_text.splitlines())} lines)",
        }

    user_msg = _build_user_message(page_text, prior_context)

    # Resolve image path for multimodal mode
    image_path: Path | None = None
    if mode == "multimodal":
        candidate = txt_path.parent / f"{stem}.jpg"
        if candidate.exists():
            image_path = candidate
        else:
            _log(f"    Warning: image not found for {txt_path.name}; falling back to text-only")

    error_path = txt_path.parent / f"{stem}_{slug}_entries_error.txt"
    result, partial, api_error = _run_ner_call(
        client, txt_path.name, model, system_prompt, user_msg,
        image_path, error_path, fallback_model, service_tier,
    )
    if api_error:
        return {"entries": [], "page_context": prior_context, "status": f"api_error:{api_error}"}
    if result is None:
        return {"entries": [], "page_context": prior_context, "status": "parse_error"}

    entries = result.get("entries", [])
    new_context = result.get("page_context") or prior_context

    if review_suspicious and entries:
        suspicious = _flag_suspicious(entries)
        if suspicious:
            # For txt-path pages, look for a sibling .jpg even in text-only mode
            review_img = image_path or (
                txt_path.parent / f"{stem}.jpg"
                if (txt_path.parent / f"{stem}.jpg").exists() else None
            )
            review_msg = _build_review_message(page_text, entries, suspicious)
            reviews = _run_review_call(
                client, txt_path.name, model, system_prompt,
                review_msg, review_img, service_tier,
            )
            if reviews:
                n_corrected = _apply_review_corrections(entries, reviews, suspicious)
                if n_corrected:
                    _log(
                        f"    [{txt_path.name}] reviewed {len(suspicious)} suspicious"
                        f" → {n_corrected} corrected"
                    )

    # No alignment data — canvas_fragment is always the bare canvas URI
    for entry in entries:
        entry["canvas_fragment"] = canvas_uri
        entry["image"] = f"{stem}.jpg"

    output = {
        "image": f"{stem}.jpg",
        "model": model,
        "mode": mode,
        "prior_context": prior_context,
        "page_context": new_context,
        "entries": entries,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"entries": entries, "page_context": new_context, "status": "partial_recovery" if partial else "ok"}


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

def process_item(
    client: genai.Client,
    item_dir: Path,
    model: str,
    system_prompt: str,
    mode: str,
    force: bool,
    dry_run: bool,
    quiet: bool,
    aligned_model: str | None = None,
    fallback_model: str | None = FALLBACK_MODEL,
    service_tier: str | None = None,
    review_suspicious: bool = False,
    sections: list | None = None,
    all_files: list[str] | None = None,
    slug_dir: Path | None = None,
) -> list[dict]:
    """
    Process all aligned JSON pages in an item directory in page order.
    Returns all extracted entries across all pages.

    aligned_model: if set, read *_{aligned_slug}_aligned.json files but write
    output tagged with the NER model slug.  Useful when OCR was run with one
    model and NER is run with another.
    fallback_model: if set, used when the primary model appears to have hit its
    output token limit (all retries produce unparseable responses).
    """
    slug = model_slug(model)
    aligned_slug = model_slug(aligned_model) if aligned_model else slug

    scope = _load_scope(item_dir)

    aligned_files = sorted(item_dir.glob(f"*_{aligned_slug}_aligned.json"))
    if scope is not None and aligned_files:
        aligned_suffix = f"_{aligned_slug}_aligned.json"
        in_scope = [f for f in aligned_files
                    if Path(f.name[: -len(aligned_suffix)]).stem in scope
                    or f.name[: -len(aligned_suffix)] in scope]
        if not in_scope:
            _log(
                f"  Warning: scope filter (included_pages.txt) excluded all "
                f"{len(aligned_files)} aligned file(s) in {item_dir} — check that "
                f"its filenames match the *_aligned.json stems (extension or "
                f"prefix mismatch?). Falling back to text-only extraction "
                f"(no bounding boxes)."
            )
        aligned_files = in_scope
    elif not aligned_files and not quiet:
        _log(
            f"  Note: no *_{aligned_slug}_aligned.json files in {item_dir} — "
            f"alignment has not been run; extracting text-only (no bounding boxes)."
        )
    if not aligned_files:
        # Fallback: read raw Gemini text files (no bounding boxes)
        txt_suffix = f"_{aligned_slug}"
        txt_files = sorted(item_dir.glob(f"*_{aligned_slug}.txt"))
        canvas_uris = _load_canvas_uris(item_dir)
        # Map page stem → manifest canvas URI by position in the full unfiltered list
        # so scope filtering doesn't shift indices and assign wrong canvases.
        # Group files by their 4-digit sequence prefix so that split-spread halves
        # (e.g. _0011_left and _0011_right, both prefix "0012") map to the same
        # original canvas rather than advancing the index twice.
        _stem_to_canvas: dict[str, str] = {}
        _prefix_to_canvas_idx: dict[str, int] = {}
        for f in txt_files:
            stem = f.stem[: -len(txt_suffix)]
            prefix = stem[:4]
            if prefix not in _prefix_to_canvas_idx:
                _prefix_to_canvas_idx[prefix] = len(_prefix_to_canvas_idx)
            idx = _prefix_to_canvas_idx[prefix]
            _stem_to_canvas[stem] = canvas_uris[idx] if idx < len(canvas_uris) else ""
        if scope is not None:
            txt_files = [f for f in txt_files
                         if f.stem[: -len(txt_suffix)] in scope]
        if not txt_files:
            return []
        all_entries: list[dict] = []
        context: dict = {}
        context_file = item_dir / f"extraction_context_{slug}.json"
        if not force and context_file.exists():
            try:
                context = json.loads(context_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not quiet:
            _log(f"  {item_dir.name}: {len(txt_files)} page(s) [text-only fallback], model={model}")
        for i, txt_path in enumerate(txt_files):
            canvas_uri = _stem_to_canvas.get(txt_path.stem[: -len(txt_suffix)], "")
            page_prompt = system_prompt
            if sections and all_files:
                from utils.section_utils import is_section_boundary, prompt_for_page
                stem = txt_path.stem[: -len(f"_{aligned_slug}")]
                img_name = f"{stem}.jpg"
                if is_section_boundary(img_name, sections, all_files):
                    context = {}
                    if not quiet:
                        _log(f"    [section boundary] {img_name} — context reset")
                if slug_dir:
                    ppath = prompt_for_page(img_name, sections, all_files, slug_dir, "ner_prompt")
                    if ppath.exists():
                        page_prompt = ppath.read_text(encoding="utf-8")
            result = process_page_txt(
                client, txt_path, canvas_uri, model, page_prompt,
                context, mode, force, dry_run, aligned_model,
                fallback_model=fallback_model, service_tier=service_tier,
                review_suspicious=review_suspicious,
            )
            status = result["status"]
            n = len(result["entries"])
            ctx = result.get("page_context", context)
            if not quiet:
                ctx_str = " > ".join(str(v) for v in ctx.values()) if ctx else "?"
                _log(f"  [{i + 1:03d}/{len(txt_files)}] {txt_path.name}  {status}  {n} entries  [{ctx_str}]")
            context = ctx
            all_entries.extend(result.get("entries", []))
            if not dry_run:
                context_file.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
        return all_entries

    all_entries: list[dict] = []
    context: dict = {}

    # Resume: load persisted context from a previous run
    context_file = item_dir / f"extraction_context_{slug}.json"
    if not force and context_file.exists():
        try:
            context = json.loads(context_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not quiet:
        _log(f"  {item_dir.name}: {len(aligned_files)} page(s), mode={mode}")

    aligned_suffix = f"_{aligned_slug}_aligned.json"
    for i, aligned_path in enumerate(aligned_files, 1):
        # Auto-retry pages whose last run produced a parse error, even without --force
        stem = aligned_path.name[: -len(aligned_suffix)]
        error_path = aligned_path.parent / f"{stem}_{slug}_entries_error.txt"
        page_force = force
        if not force and error_path.exists():
            _log(f"    [{aligned_path.name}] prior parse-error file detected — forcing retry")
            page_force = True

        page_prompt = system_prompt
        if sections and all_files:
            from utils.section_utils import is_section_boundary, prompt_for_page
            img_name = f"{stem}.jpg"
            if is_section_boundary(img_name, sections, all_files):
                context = {}
                if not quiet:
                    _log(f"    [section boundary] {img_name} — context reset")
            if slug_dir:
                ppath = prompt_for_page(img_name, sections, all_files, slug_dir, "ner_prompt")
                if ppath.exists():
                    page_prompt = ppath.read_text(encoding="utf-8")

        result = process_page(
            client, aligned_path, model, page_prompt,
            context, mode, page_force, dry_run, aligned_model,
            fallback_model=fallback_model, service_tier=service_tier,
            review_suspicious=review_suspicious,
        )
        status = result["status"]
        n = len(result["entries"])
        ctx = result.get("page_context", context)

        if not quiet:
            ctx_str = " > ".join(str(v) for v in ctx.values()) if ctx else "?"
            _log(f"  [{i:03d}/{len(aligned_files)}] {aligned_path.name}  {status}  {n} entries  [{ctx_str}]")

        context = ctx
        all_entries.extend(result.get("entries", []))

        # Persist context so runs can be resumed
        if not dry_run:
            context_file.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")

    return all_entries


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(entries: list[dict], out_path: Path) -> None:
    """Write a flat CSV, inferring column order from the entries themselves."""
    fields = _infer_fields(entries)
    if not fields:
        out_path.write_text("", encoding="utf-8")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for e in entries:
            writer.writerow({k: e.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured entries from aligned OCR using Gemini NER.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help=(
            "Item images directory containing *_aligned.json files "
            "(e.g. output/greenbooks/feb978b0), or a parent directory to "
            "process all item subdirectories."
        ),
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_NER_MODEL,
        metavar="MODEL",
        help=f"Gemini model for NER (default: {DEFAULT_NER_MODEL})",
    )
    parser.add_argument(
        "--aligned-model",
        default=None,
        metavar="MODEL",
        help=(
            "Model whose slug appears in the *_aligned.json filenames "
            "(default: same as --model). Use this when OCR was run with a "
            "different model than the one doing NER, e.g. "
            "--model gemini-2.5-flash --aligned-model gemini-2.0-flash"
        ),
    )
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL,
        metavar="MODEL",
        help=(
            "Model to try when the primary model fails all parse attempts "
            "(typically because it hit its output token limit on dense pages). "
            f"Default: {FALLBACK_MODEL}. Pass empty string to disable."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["text-only", "multimodal"],
        default="text-only",
        help=(
            "text-only: send corrected text only; "
            "multimodal: also send the page image (default: text-only)"
        ),
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        metavar="FILE",
        help=(
            "NER system prompt file. If omitted, looks for ner_prompt.md in the "
            "images directory (volume-specific), then falls back to "
            f"{NER_PROMPT_FILE} (global default)."
        ),
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-run even if output files already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files and show what would be run, without calling the API",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-page progress output",
    )
    parser.add_argument(
        "--flex",
        action="store_true",
        help=(
            "Use Gemini Flex inference (service_tier='flex'): ~50%% cheaper, "
            "1–15 min latency per request, best-effort availability."
        ),
    )
    parser.add_argument(
        "--review-suspicious",
        action="store_true",
        help=(
            "After each page's primary NER call, flag entries with isolated "
            "geographic context, empty names, or probable headings, then make "
            "a targeted re-check call using the page image. Adds one extra API "
            "call per flagged page (typically 5–15%% of pages)."
        ),
    )
    parser.add_argument(
        "--sections",
        metavar="PATH",
        help=(
            "Path to sections.txt marking structural section boundaries. "
            "When provided, extraction context is reset at each section boundary "
            "and the NER prompt switches to the section-specific ner_prompt_{label}.md "
            "if available, falling back to ner_prompt.md."
        ),
    )
    args = parser.parse_args()

    aligned_model = args.aligned_model
    slug = model_slug(args.model)
    aligned_slug = model_slug(aligned_model) if aligned_model else slug
    output_root = Path(args.output_dir)

    # NER prompt — explicit flag > volume-specific > global default
    if args.prompt:
        prompt_path = Path(args.prompt)
    else:
        prompt_path = _find_ner_prompt(output_root)
    if not prompt_path.exists():
        print(f"Error: NER prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8")
    if not args.quiet:
        if prompt_path == NER_PROMPT_FILE:
            siblings = _find_sibling_ner_prompts(output_root)
            lines = [
                f"WARNING: no volume-specific NER prompt found — using generic fallback: {prompt_path}",
            ]
            if siblings:
                lines.append("         Found NER prompt(s) in nearby directories that may be reusable:")
                for s in siblings:
                    lines.append(f"           {s}")
                lines.append(f"         Re-run with: --prompt {siblings[0]}")
            else:
                lines.append("         Run generate_prompt.py first for better extraction quality.")
            print("\n".join(lines), file=sys.stderr)
        else:
            print(f"NER prompt: {prompt_path}", file=sys.stderr)
    if not output_root.exists():
        print(f"Error: directory not found: {output_root}", file=sys.stderr)
        sys.exit(1)

    client = get_client() if not args.dry_run else None  # type: ignore[assignment]

    # Discover item directories: either the given dir itself (if it has aligned
    # JSONs or raw txt files directly) or its immediate subdirectories.
    def _find_item_dirs(slug: str) -> tuple[list[Path], str]:
        pat_aligned = f"*_{slug}_aligned.json"
        pat_txt = f"*_{slug}.txt"
        direct = list(output_root.glob(pat_aligned)) or list(output_root.glob(pat_txt))
        if direct:
            return [output_root], slug
        dirs = [
            d for d in sorted(output_root.iterdir())
            if d.is_dir() and (list(d.glob(pat_aligned)) or list(d.glob(pat_txt)))
        ]
        return dirs, slug

    item_dirs, aligned_slug = _find_item_dirs(aligned_slug)

    if not item_dirs:
        # Auto-detect: no files found for the expected slug — scan for any pipeline
        # output files and infer the OCR model slug from their filenames.
        detected = _detect_aligned_slug(output_root)
        if not detected:
            for d in sorted(output_root.iterdir()):
                if d.is_dir():
                    detected = _detect_aligned_slug(d)
                    if detected:
                        break
        if detected and detected != aligned_slug:
            print(
                f"NOTE: no OCR files found for model '{aligned_slug}'.\n"
                f"      Auto-detected OCR model '{detected}' in {output_root.name} — using that.\n"
                f"      Pass --aligned-model {detected} to suppress this message.",
                file=sys.stderr,
            )
            aligned_model = detected
            aligned_slug = detected
            item_dirs, aligned_slug = _find_item_dirs(aligned_slug)

    if not item_dirs:
        print(
            f"No aligned JSON or Gemini text files found under {output_root}\n"
            f"  (looked for *_{aligned_slug}_aligned.json and *_{aligned_slug}.txt)",
            file=sys.stderr,
        )
        sys.exit(1)

    fallback_model = args.fallback_model or None  # empty string → disabled
    service_tier = "flex" if args.flex else None
    if service_tier and not args.quiet:
        print("Flex inference enabled (service_tier='flex')", file=sys.stderr)

    review_suspicious = args.review_suspicious
    if review_suspicious and not args.quiet:
        print("Suspicious-entry review enabled (--review-suspicious)", file=sys.stderr)

    # --- Sections support ----------------------------------------------------
    sections = None
    all_files: list[str] = []
    sections_slug_dir: Path | None = None

    if args.sections:
        from utils.section_utils import load_sections

        sections_path = Path(args.sections)
        if not sections_path.exists():
            print(f"Error: sections file not found: {sections_path}", file=sys.stderr)
            sys.exit(1)

        # Build all_files using same logic as run_gemini_ocr.py:
        # splits (_left/_right) replace the original spread file.
        _raw_jpgs = sorted(
            p for item_dir in item_dirs
            for p in item_dir.glob("*.jpg")
            if not p.stem.endswith("_viz")
        )
        _all_imgs: list[Path] = []
        for _p in _raw_jpgs:
            if _p.stem.endswith("_left") or _p.stem.endswith("_right"):
                _all_imgs.append(_p)
                continue
            if (_p.with_name(f"{_p.stem}_left.jpg").exists()
                    and _p.with_name(f"{_p.stem}_right.jpg").exists()):
                continue
            _all_imgs.append(_p)
        all_files = [p.name for p in _all_imgs]

        try:
            sections = load_sections(sections_path, all_files)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Slug dir: look for ner_prompt_*.md files starting from output_root
        root = output_root.resolve()
        for candidate in (root, root.parent):
            if any(candidate.glob("ner_prompt_*.md")) or (candidate / "ner_prompt.md").exists():
                sections_slug_dir = candidate
                break
        if sections_slug_dir is None:
            sections_slug_dir = root

        if not args.quiet:
            print(
                f"Sections mode: {len(sections)} section(s): "
                f"{', '.join(s['label'] for s in sections)} "
                f"(prompts in {sections_slug_dir})",
                file=sys.stderr,
            )

    print(
        f"\nExtracting entries: {len(item_dirs)} item dir(s), model={args.model}"
        + (f" (aligned by {aligned_model})" if aligned_model else "")
        + (f", fallback={fallback_model}" if fallback_model and fallback_model != args.model else "")
        + f", mode={args.mode}"
        + (" +review" if review_suspicious else "")
        + (f" +sections({len(sections)})" if sections else "")
        + (" [DRY RUN]" if args.dry_run else ""),
        file=sys.stderr,
    )

    for item_dir in item_dirs:
        # Per-item prompt: if the item dir has its own ner_prompt.md (written by
        # --generate-prompts per-volume), use it; otherwise use the resolved prompt.
        item_prompt_path = item_dir / "ner_prompt.md"
        if item_prompt_path.exists() and item_prompt_path != Path(args.prompt or ""):
            item_prompt = item_prompt_path.read_text(encoding="utf-8")
            if not args.quiet:
                print(f"  NER prompt (per-item): {item_prompt_path}", file=sys.stderr)
        else:
            item_prompt = system_prompt
        entries = process_item(
            client, item_dir, args.model, item_prompt,
            args.mode, args.force, args.dry_run, args.quiet, aligned_model,
            fallback_model=fallback_model, service_tier=service_tier,
            review_suspicious=review_suspicious,
            sections=sections, all_files=all_files or None, slug_dir=sections_slug_dir,
        )
        if not args.dry_run:
            csv_path = item_dir / f"entries_{slug}.csv"
            write_csv(entries, csv_path)
            print(
                f"  → {len(entries)} entries total → {csv_path}",
                file=sys.stderr,
            )

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
