#!/usr/bin/env python3
"""Align Gemini OCR text with Surya bounding boxes using Needleman-Wunsch.

For each image that has both {stem}_{model}.txt (Gemini) and
{stem}_surya.json (Surya), produces {stem}_{model}_aligned.json with
line-level bounding boxes from Surya and corrected text from Gemini.

Reading-order correction
------------------------
OCR lines are re-sorted by detected column (left→right) then top-to-bottom
within each column before alignment.

IIIF canvas URIs and dimensions are read from manifest.json cached by
download_images.py in each item directory.

Confidence tiers
----------------
  line   — matched with line-level bboxes from Surya
  manual — user-confirmed in --review-alignment
  crop   — per-line crop OCR (no NW alignment); same aligned-JSON shape with
           "method": "linecrop" (produced by an experimental line-crop tool)

Output JSON
-----------
  {
    "image": "0001_58030238.jpg",
    "model": "gemini-3.1-flash-lite",
    "canvas_uri": "https://...",
    "canvas_width": 2048,
    "canvas_height": 3000,
    "lines": [
      {
        "bbox": [x1, y1, x2, y2],
        "canvas_fragment": "canvas_uri#xywh=x,y,w,h",
        "confidence": "line",
        "gemini_text": "corrected line text"
      }
    ],
    "unmatched_gemini": ["lines with no OCR match"]
  }

Usage
-----
    python align_ocr.py output/greenbooks --model gemini-2.0-flash
    python align_ocr.py output/greenbooks --model gemini-2.0-flash --workers 4
    python align_ocr.py output/greenbooks --model gemini-2.0-flash --force
"""

import argparse
import difflib
import functools
import json
import os
import re
import statistics
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from utils import iiif_utils
from utils.models import DEFAULT_OCR_MODEL, model_slug, discover_ocr_slug
from pipeline.state import get_ocr_model

_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# IIIF image-service helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _fetch_info_json(info_url: str) -> tuple[int, int]:
    """Fetch a IIIF info.json and return (width, height). Cached by URL.

    Cached by info_url so manifests that use a shared identifier as their
    canvas namespace (e.g. Tulsa Library 1922, where every canvas_id starts
    with the same manifest ID) only hit the network once regardless of how
    many canvases reference it.
    """
    import time as _time
    for attempt in range(2):
        try:
            with urllib.request.urlopen(info_url, timeout=15) as resp:
                info = json.loads(resp.read())
            return int(info.get("width") or 0), int(info.get("height") or 0)
        except Exception as exc:
            if attempt == 0:
                _log(f"  Warning: could not fetch {info_url} (attempt 1): {exc} — retrying")
                _time.sleep(2)
            else:
                _log(f"  Warning: could not fetch {info_url} (attempt 2): {exc} — giving up")
    return 0, 0


@functools.lru_cache(maxsize=256)
def _get_natural_dims(canvas_id: str) -> tuple[int, int]:
    """Return (width, height) of the natural image by fetching info.json.

    Parses the IIIF Image API service base from *canvas_id* (a URL of the form
    ``https://host/prefix/identifier/region/size/rotation/quality.fmt``), then
    delegates to _fetch_info_json which is cached by URL.  Returns (0, 0) on
    any error.
    """
    parts = urlparse(canvas_id)
    path_parts = parts.path.split("/")
    # IIIF Image API path: /{prefix}/{identifier}/{region}/…
    # Service base = scheme + host + first 4 path segments (index 0–3, where 0 is "")
    if len(path_parts) < 5:
        return 0, 0
    service_base = f"{parts.scheme}://{parts.netloc}" + "/".join(path_parts[:4])
    return _fetch_info_json(f"{service_base}/info.json")

# ---------------------------------------------------------------------------
# Tunable alignment parameters
# ---------------------------------------------------------------------------

# Needleman-Wunsch gap penalty (must be negative).
# -40 was calibrated on Green Book scans; lower magnitude = more gaps allowed.
_NW_GAP = -40
# Second-pass alignment uses a more permissive gap penalty to catch lines that
# the first pass left unmatched.  Only applied when ≥ _PASS2_MIN_LINES remain.
_NW_GAP_PASS2    = -20
_PASS2_MIN_LINES =  5

# Pages whose median Surya detection confidence falls below this threshold are
# flagged needs_review=True in the aligned JSON so --review-alignment can
# surface them for manual inspection.
_LOW_CONFIDENCE_PAGE_THRESHOLD = 0.5

# Pages where Surya detects significantly more lines than Gemini produced are
# flagged possible_column_merge=True.  A ratio near 2.0 is the signature of
# Gemini reading across two columns and merging each visual row into one line.
# Pages with fewer than _MERGE_MIN_LINES (e.g. ads, covers) are excluded
# because short pages have naturally noisy ratios.
# _MERGE_RATIO_MAX caps the upper end: extremely high ratios (> 5×) indicate
# a map or diagram where Surya over-detects tiny labels — not a column merge.
_MERGE_RATIO_THRESHOLD = 1.8
_MERGE_RATIO_MAX       = 5.0
_MERGE_MIN_LINES       = 10

# Anchor matching: commit Gemini↔Surya line pairs that are near-verbatim
# matches BEFORE running the global NW.  This prevents the aligner from
# consuming heading words (city names, state names, category lines) on the
# wrong Gemini lines — the failure mode where e.g. "CHERAW" entries mapped to
# CHARLESTON bounding boxes because the global NW preferred nearby wrong words
# over paying the gap cost to reach the real CHERAW Surya line.
_ANCHOR_MIN_LEN    = 3     # minimum normalised chars to consider a line as an anchor
_ANCHOR_MAX_SUFFIX = 6     # extra trailing chars allowed in prefix match
_ANCHOR_SIM_HIGH   = 0.92  # similarity threshold when prefix match doesn't apply


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------

def parse_surya(
    path: Path, min_confidence: float = 0.35
) -> tuple[tuple[int, int, int, int], list[dict], "float | None"]:
    """
    Parse a Surya JSON file produced by run_surya_ocr.py.

    Returns (page_bbox, lines, median_confidence) where each line is:
        {'bbox': [x1, y1, x2, y2], 'text': str, 'surya_confidence': float}

    median_confidence is the median detection confidence across ALL lines on the
    page (before threshold filtering), or None if no confidence values are
    present.  Lines whose confidence is below min_confidence are excluded from
    the returned list but still contribute to the median so the page-level
    quality estimate reflects the raw scan quality.

    page_bbox is (0, 0, image_width, image_height).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    w = data.get("image_width", 0)
    h = data.get("image_height", 0)
    page_bbox = (0, 0, w, h)

    raw_lines = [ln for ln in data.get("lines", []) if ln.get("bbox")]
    # Compute page-level median confidence from ALL detected lines (pre-filter)
    # so the quality estimate is not inflated by the threshold.
    all_confs = [float(ln["confidence"]) for ln in raw_lines if "confidence" in ln]
    if all_confs:
        sorted_confs = sorted(all_confs)
        mid = len(sorted_confs) // 2
        median_conf: "float | None" = (
            sorted_confs[mid]
            if len(sorted_confs) % 2 == 1
            else (sorted_confs[mid - 1] + sorted_confs[mid]) / 2
        )
    else:
        median_conf = None

    lines = [
        {
            "bbox": ln["bbox"],
            "text": ln.get("text", ""),
            "surya_confidence": round(float(ln.get("confidence", 1.0)), 4),
        }
        for ln in raw_lines
        if ln.get("confidence", 1.0) >= min_confidence
    ]
    return page_bbox, lines, median_conf


def _expand_multiline_surya_lines(lines: list[dict]) -> list[dict]:
    """Split Surya lines whose text contains embedded newlines into individual entries.

    Surya sometimes groups a tall region of adjacent directory entries into a
    single detection with newline-separated text.  Since NW alignment is 1-to-1
    at line granularity, only one Gemini line gets a bbox while the rest become
    unmatched.  Splitting the region bbox proportionally by sub-line count gives
    each Gemini line a valid candidate to align against.
    """
    result: list[dict] = []
    for ln in lines:
        text = ln.get("text", "")
        sub_texts = [t.strip() for t in text.split("\n") if t.strip()]
        if len(sub_texts) <= 1:
            result.append(ln)
            continue
        x1, y1, x2, y2 = ln["bbox"]
        h_per = (y2 - y1) / len(sub_texts)
        conf = ln.get("surya_confidence", 1.0)
        for i, sub_text in enumerate(sub_texts):
            result.append({
                "bbox": [x1, round(y1 + i * h_per), x2, round(y1 + (i + 1) * h_per)],
                "text": sub_text,
                "surya_confidence": conf,
            })
    return result


# ---------------------------------------------------------------------------
# Reading-order correction
# ---------------------------------------------------------------------------

# Minimum line bbox height in pixels — absolute floor used when there are too
# few lines to compute a reliable median.  Lines below the effective threshold
# are dot-leader artifacts or horizontal rules detected as text;
# coordinates.
_MIN_LINE_HEIGHT = 15


def filter_short_lines(lines: list[dict], min_height: int | None = None) -> list[dict]:
    """Remove lines whose bounding box height is below an adaptive threshold.

    When *min_height* is not provided, the threshold is computed as 35% of the
    median line height across all input lines, with *_MIN_LINE_HEIGHT* as a
    floor.  This keeps the filter proportional to the scan resolution instead
    of relying on a fixed pixel constant.
    """
    if not lines:
        return lines
    if min_height is None:
        heights = [ln["bbox"][3] - ln["bbox"][1] for ln in lines]
        median_h = statistics.median(heights)
        min_height = max(_MIN_LINE_HEIGHT, round(median_h * 0.35))
    return [ln for ln in lines if (ln["bbox"][3] - ln["bbox"][1]) >= min_height]


def filter_margin_lines(
    lines: list[dict], img_w: int, margin: float = 0.05
) -> list[dict]:
    """Drop Surya lines whose entire bbox falls within a left or right scan margin.

    Book scans often capture a sliver of the facing page at one or both edges.
    Surya correctly detects this text with high confidence, but it should not be
    aligned to the main page's Gemini output.  Lines are excluded when they start
    past the right margin boundary (x1 > img_w * (1 - margin)) or end before the
    left margin boundary (x2 < img_w * margin).  The default 5 % margin is wide
    enough to catch all facing-page bleed observed across tested volumes while
    leaving no legitimate content at risk.
    """
    if not lines or not img_w or margin <= 0:
        return lines
    left_limit  = img_w * margin
    right_limit = img_w * (1 - margin)
    return [
        ln for ln in lines
        if ln["bbox"][2] >= left_limit and ln["bbox"][0] <= right_limit
    ]


_READING_ORDER_BAND = 50  # pixels — horizontal band height for y-band sort.
# Lines within the same 50 px band are sorted left-column-first; bands are
# sorted top-to-bottom.  This keeps row pairs (y-difference ≤ ~35 px) in the
# same band while placing centred section headings (which have ≥ 80 px of
# whitespace above/below them) in their own band ahead of body content —
# preventing them from being pushed to the end of the line sequence by a pure
# column-major sort.


def sort_by_reading_order(lines: list[dict], page_width: int) -> list[dict]:
    """
    Re-sort Surya lines into correct reading order for multi-column pages.

    Default (row-major y-band sort): lines are grouped into 50 px horizontal
    bands (sorted top-to-bottom), and within each band sorted left-column-first.
    This correctly places centred page headings (e.g. state names) before
    the body-text columns they head.

    True two-column pages (column-major): when lines cluster into exactly two
    distinct columns and each column contains at least 20% of all lines, Gemini
    reads the entire left column top-to-bottom before the entire right column.
    In that case we emit left-column lines (sorted by y) followed by right-column
    lines (sorted by y) to match Gemini's reading order.

    Column breaks are detected in two stages:

    Stage 1 – consecutive x1 gap: looks for a gap > 8% of page width between
    adjacent sorted x1 values.  This handles the common case where the
    inter-column gutter is empty.  Also handles pages where a few bridge
    lines (ads, page numbers) reduce the max consecutive gap to ~8–10% of
    page width — lower than the old 10% threshold which missed these.

    Stage 2 – bimodal peak detection (fallback): when stage 1 finds no break
    or a degenerate break (< 10% of lines in the smaller "column", e.g. a
    lone page-number at extreme x1), this stage builds a histogram of x1
    values and detects two dense clusters separated by ≥ 20% of page width,
    each containing ≥ 15% of lines.  The column break is placed at the
    midpoint of the gap between the two peaks.  A bridge-sparsity guard
    (< 10% of lines in the inter-cluster zone) prevents this from running
    on pages with dense centered advertisement content that straddles both
    column margins — splitting that content across columns would confuse NW.

    Ad strip handling: Gemini marks advertisement blocks with
    ``=== ADVERTISEMENT ===`` delimiters, and those lines are excluded from
    the NW anchor set (see ``_find_anchors``).  Reordering Surya bboxes for
    ad strips is not attempted here — the thresholds needed to detect genuine
    vertical-margin strips reliably are too close to normal wrapped-line tails
    that appear in the same x zone.
    """
    if not lines or page_width <= 0:
        return lines

    x1_vals = sorted(line["bbox"][0] for line in lines)

    # Stage 1: Find column breaks via consecutive x1 gaps > 8% of page width.
    threshold = max(page_width * 0.08, 20)
    col_breaks: list[float] = []
    prev = x1_vals[0]
    for x in x1_vals[1:]:
        if x - prev > threshold:
            col_breaks.append((prev + x) / 2.0)
        prev = x

    # Stage 2: Bimodal peak detection fallback.  Runs when stage 1 found no
    # breaks, OR when its break is spurious — e.g. a page-number outlier at
    # high x1 creates a break that puts 99% of lines in one "column" and only
    # the page number in the other.  We detect this by checking whether the
    # smallest column produced by stage-1 would hold < 10% of lines.
    def _smallest_col_frac(breaks: list[float]) -> float:
        if not breaks:
            return 0.0
        hist: dict[int, int] = {}
        for ln in lines:
            ci = next(
                (j for j, b in enumerate(breaks) if ln["bbox"][0] < b), len(breaks)
            )
            hist[ci] = hist.get(ci, 0) + 1
        return min(hist.values()) / len(lines)

    if not col_breaks or _smallest_col_frac(col_breaks) < 0.10:
        _bin = max(page_width // 20, 30)  # ~5% of page width
        _hist: dict[int, int] = {}
        for _x in x1_vals:
            _b = (_x // _bin) * _bin
            _hist[_b] = _hist.get(_b, 0) + 1
        if len(_hist) >= 2:
            _sorted_peaks = sorted(_hist.items(), key=lambda t: t[1], reverse=True)
            (_p1x, _p1n), (_p2x, _p2n) = _sorted_peaks[0], _sorted_peaks[1]
            _lx, _rx = min(_p1x, _p2x), max(_p1x, _p2x)
            _gap = _rx - (_lx + _bin)
            _n = len(x1_vals)
            # Count lines in the "bridge" zone between the two peak clusters.
            # If the bridge is dense (≥ 10% of lines), applying column-major
            # sort would split the bridge content across both columns, confusing
            # NW alignment.  In that case, fall back to y-band sort instead.
            _bridge_n = sum(
                1 for _x in x1_vals if _lx + _bin <= _x < _rx
            )
            if (
                _p1n >= _n * 0.15
                and _p2n >= _n * 0.15
                and _gap >= page_width * 0.20
                and _bridge_n < _n * 0.10
            ):
                col_breaks = [(_lx + _bin + _rx) / 2.0]

    def col_index(x1: int) -> int:
        for i, boundary in enumerate(col_breaks):
            if x1 < boundary:
                return i
        return len(col_breaks)

    # N-column layout: Gemini reads column-major (entire column 0 top-to-bottom,
    # then column 1, etc.).  Detect when two or more columns each hold ≥ 10% of
    # lines and together account for ≥ 80% of lines.  The 10% per-column floor
    # (down from 20%) allows 4+ column layouts where 20% each would be impossible.
    # This tolerates spurious extra col_breaks caused by a few noise lines with
    # extreme x1 values by requiring ≥ 80% total coverage.
    if col_breaks:
        from collections import Counter as _Counter
        col_counts = _Counter(col_index(ln["bbox"][0]) for ln in lines)
        substantial = sorted(
            c for c, n in col_counts.items() if n / len(lines) >= 0.10
        )
        if len(substantial) >= 2:
            cols = [
                sorted(
                    [ln for ln in lines if col_index(ln["bbox"][0]) == c],
                    key=lambda ln: ln["bbox"][1],
                )
                for c in substantial
            ]
            if sum(len(col) for col in cols) / len(lines) >= 0.80:
                return [ln for col in cols for ln in col]

    return sorted(
        lines,
        key=lambda ln: (
            ln["bbox"][1] // _READING_ORDER_BAND,  # y-band (top → bottom)
            col_index(ln["bbox"][0]),               # column (left → right)
            ln["bbox"][1],                          # y within band
        ),
    )


# ---------------------------------------------------------------------------
# Needleman-Wunsch alignment
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _text_sim(a: str, b: str) -> int:
    """
    Integer similarity score for NW: 0..100.
    Positive = prefer alignment; negative = prefer gap (_NW_GAP = -40).
    Break-even with gap at ratio ≈ 0.4.
    """
    an = _normalize(a)
    bn = _normalize(b)
    if not an and not bn:
        return 50
    if not an or not bn:
        return -80  # strongly prefer gapping empty vs. non-empty
    ratio = difflib.SequenceMatcher(None, an, bn, autojunk=False).ratio()
    return round(ratio * 100)


def needleman_wunsch(
    seq_a: list,
    seq_b: list,
    sim_fn,
    gap: int = _NW_GAP,
) -> list[tuple[int | None, int | None]]:
    """
    Global sequence alignment via Needleman-Wunsch.

    Returns a list of (idx_a, idx_b) pairs; None in either position = gap.
    Uses a choice matrix for clean traceback without floating-point re-computation.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return []

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ch = [[""] * (m + 1) for _ in range(n + 1)]  # 'D'=diagonal, 'U'=up, 'L'=left

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
        ch[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
        ch[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores = {
                "D": dp[i - 1][j - 1] + sim_fn(seq_a[i - 1], seq_b[j - 1]),
                "U": dp[i - 1][j] + gap,
                "L": dp[i][j - 1] + gap,
            }
            best = max(scores, key=scores.__getitem__)
            dp[i][j] = scores[best]
            ch[i][j] = best

    # Traceback
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        c = ch[i][j]
        if c == "D":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif c == "U":
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# IIIF manifest / canvas lookup
# ---------------------------------------------------------------------------

def _extract_image_id(image_name: str) -> str:
    """
    Extract the IIIF image ID from a filename.
        '0001_58030238.jpg'       → '58030238'
        '0001_58030238_left.jpg'  → '58030238'
    """
    stem = Path(image_name).stem
    for suffix in ("_left", "_right"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else stem


def load_canvas_info(
    manifest_path: Path, image_id: str
) -> tuple[str, int, int]:
    """
    Return (canvas_uri, canvas_width, canvas_height) for the canvas whose
    IIIF Image Service ID ends with image_id.
    Handles both IIIF Presentation v2 and v3.
    Returns ('', 0, 0) if not found or manifest is missing.
    """
    if not manifest_path.exists():
        return "", 0, 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for canvas in iiif_utils.iter_canvases(manifest):
            if canvas["image_id"] == image_id:
                return (
                    canvas["canvas_id"],
                    canvas["canvas_width"],
                    canvas["canvas_height"],
                )
    except Exception:  # noqa: BLE001
        pass
    return "", 0, 0


def _canvas_fragment(
    canvas_uri: str,
    bbox: list[int],
    img_w: int,
    img_h: int,
    canvas_w: int,
    canvas_h: int,
) -> str:
    """
    Build a IIIF xywh fragment string, scaling pixel coords to canvas space
    if the canvas and image dimensions differ.

    When called from align_image, canvas_w/canvas_h are the *natural* image
    dimensions (fetched from info.json), not the manifest canvas size.  Mirador
    maps annotation xywh directly to image pixel space, so outputting natural
    pixel coords is the correct target.

    Uses non-uniform (independent-axis) scaling — matching how Mirador/OSD
    renders a painting annotation that fills the full canvas.  For the common
    case where the pipeline download resolution equals the natural dimensions,
    sx = sy = 1 and no scaling occurs.
    """
    x1, y1, x2, y2 = bbox
    if canvas_w and img_w and (canvas_w != img_w or canvas_h != img_h):
        sx = canvas_w / img_w
        sy = canvas_h / img_h if img_h else sx
        x1 = round(x1 * sx)
        y1 = round(y1 * sy)
        x2 = round(x2 * sx)
        y2 = round(y2 * sy)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return f"{canvas_uri}#xywh={x1},{y1},{w},{h}"


# ---------------------------------------------------------------------------
# Word-level alignment helpers
# ---------------------------------------------------------------------------

# Legacy fallback — _build_word_aligned_lines now computes an adaptive
# tolerance (2.5× median line height, floor 50 px) instead of this constant.
_INTRA_LINE_Y_TOLERANCE = 100  # pixels


def _union_bbox(bboxes: list[list[int]]) -> list[int]:
    """Return the union bounding box of a collection of bboxes."""
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _y_center(bbox: list[int]) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _median(values: list[float]) -> float:
    vs = sorted(values)
    n = len(vs)
    if n % 2 == 1:
        return vs[n // 2]
    return (vs[n // 2 - 1] + vs[n // 2]) / 2.0


def _find_anchors(
    ocr_line_texts: list[str],
    gemini_lines: list[str],
    non_anchor_gem_indices: "set[int] | None" = None,
) -> list[tuple[int, int]]:
    """
    Find (ocr_line_idx, gem_line_idx) anchor pairs where a Gemini line is a
    near-verbatim match to an entire OCR line.

    City/state headings and category lines typically appear verbatim in both
    Gemini and the OCR source.  Committing those as anchors before the NW pass
    prevents the aligner from consuming them on wrong Gemini lines — the
    failure mode where, e.g., "CHERAW" entries mapped to CHARLESTON bounding
    boxes because the global NW preferred nearby wrong words over paying the gap
    cost to reach the real CHERAW OCR line at a later sequence position.

    ocr_line_texts: raw (un-normalised) text strings, one per OCR line.
    non_anchor_gem_indices: optional set of Gemini line indices that must not
        become anchors (e.g. lines inside ``=== ADVERTISEMENT ===`` blocks).
        Those lines are still included in the NW pass — they just cannot pin
        the alignment.

    Returns pairs in monotonically increasing order on both axes.  Each
    OCR line and each Gemini line appears in at most one anchor.
    """
    tess_line_texts = [_normalize(t) for t in ocr_line_texts]
    gem_norms = [_normalize(g) for g in gemini_lines]
    _non_anchor = non_anchor_gem_indices or set()

    # Lines whose normalised text appears more than once in the Gemini sequence
    # are ambiguous anchors: we cannot tell which Gemini occurrence corresponds
    # to which Surya line, so we skip them entirely.
    from collections import Counter
    gem_counts = Counter(n for n in gem_norms if len(n) >= _ANCHOR_MIN_LEN)

    anchors: list[tuple[int, int]] = []
    used_tess: set[int] = set()
    last_tess = -1

    for gi, gnorm in enumerate(gem_norms):
        if len(gnorm) < _ANCHOR_MIN_LEN:
            continue
        if gi in _non_anchor:  # inside an advertisement block — skip
            continue
        if gem_counts[gnorm] > 1:  # repeated line — skip to avoid mis-anchoring
            continue
        best_score = -1.0
        best_ti = -1
        for ti, tnorm in enumerate(tess_line_texts):
            if ti in used_tess or ti <= last_tess:
                continue
            if len(tnorm) < _ANCHOR_MIN_LEN:
                continue
            # Strategy 1: prefix match — one string is a clean prefix of the other
            # (e.g. "SELLERSVILLE oun" → normalized "sellersvilleoun" vs "sellersville").
            # Add a tiny bonus so prefix matches always beat sim-only matches.
            short, long_ = (tnorm, gnorm) if len(tnorm) <= len(gnorm) else (gnorm, tnorm)
            if long_.startswith(short) and (len(long_) - len(short)) <= _ANCHOR_MAX_SUFFIX:
                score = 1.0 - (len(long_) - len(short)) / max(len(long_), 1) + 0.01
                if score > best_score:
                    best_score = score
                    best_ti = ti
                continue  # no need to also compute sim for prefix matches
            # Strategy 2: high similarity for minor OCR noise within the string.
            # Threshold 0.92 rejects Tourist-Home-style false positives
            # ("mrsmayestouristhome" ↔ "mrsigraysontouristhome" scores only 0.83).
            sim = difflib.SequenceMatcher(None, tnorm, gnorm, autojunk=False).ratio()
            if sim >= _ANCHOR_SIM_HIGH and sim > best_score:
                best_score = sim
                best_ti = ti
        if best_ti >= 0:
            anchors.append((best_ti, gi))
            used_tess.add(best_ti)
            last_tess = best_ti

    return anchors


def _build_line_aligned_lines(
    sorted_surya_lines: list[dict],
    gemini_lines: list[str],
    fragment_fn,
    non_anchor_gem_indices: "set[int] | None" = None,
) -> tuple[list[dict], list[str]]:
    """
    Align Gemini text to Surya line bboxes using anchored Needleman-Wunsch
    at line granularity.

    Each Surya line is treated as a single unit and matched against one Gemini
    line.  The output confidence is "line".

    Anchoring works the same way as in _build_word_aligned_lines: city/state
    headings and category lines that appear near-verbatim in both sources are
    committed as fixed anchors before the NW pass, preventing drift across
    long pages.

    Returns (result_lines, unmatched_gemini).
    """
    if not sorted_surya_lines or not gemini_lines:
        return [], list(gemini_lines)

    surya_texts = [ln["text"] for ln in sorted_surya_lines]
    anchors = _find_anchors(surya_texts, gemini_lines, non_anchor_gem_indices)

    def _nw_lines(s_start: int, s_end: int, g_start: int, g_end: int):
        """NW over line-index slices; return pairs with absolute indices."""
        if s_start >= s_end or g_start >= g_end:
            return []
        pairs = needleman_wunsch(
            surya_texts[s_start:s_end],
            gemini_lines[g_start:g_end],
            _text_sim,
        )
        return [
            (si + s_start if si is not None else None,
             gi + g_start if gi is not None else None)
            for si, gi in pairs
        ]

    all_pairs: list[tuple[int | None, int | None]] = []
    prev_sl = 0
    prev_gl = 0
    for surya_li, gem_li in anchors:
        all_pairs.extend(_nw_lines(prev_sl, surya_li, prev_gl, gem_li))
        all_pairs.extend(_nw_lines(surya_li, surya_li + 1, gem_li, gem_li + 1))
        prev_sl = surya_li + 1
        prev_gl = gem_li + 1
    all_pairs.extend(_nw_lines(prev_sl, len(surya_texts), prev_gl, len(gemini_lines)))

    result_lines: list[dict] = []
    matched_gi: set[int] = set()
    matched_si: set[int] = set()

    for si, gi in all_pairs:
        if gi is None:
            continue  # Surya line with no Gemini counterpart
        if si is None:
            continue  # will be collected as unmatched below
        matched_gi.add(gi)
        matched_si.add(si)
        bbox = list(sorted_surya_lines[si]["bbox"])
        entry: dict = {
            "bbox": bbox,
            "canvas_fragment": fragment_fn(bbox),
            "confidence": "line",
            "gemini_text": gemini_lines[gi],
        }
        sc = sorted_surya_lines[si].get("surya_confidence")
        if sc is not None:
            entry["surya_confidence"] = sc
        result_lines.append(entry)

    # Any Gemini line not matched by a Surya line is unmatched.
    # Drop short fragments (≤8 chars) — these are typically edge-bleed noise
    # from adjacent pages that cause NER to hallucinate entries.
    unmatched_gemini = [
        gemini_lines[gi]
        for gi in range(len(gemini_lines))
        if gi not in matched_gi and len(gemini_lines[gi].strip()) > 8
    ]

    # ── Remaining-column passes ───────────────────────────────────────────
    # When sort_by_reading_order falls back to Y-band interleaving, the first
    # NW pass may consume only one column, leaving the others entirely
    # unmatched.  Loop: each iteration re-sorts unmatched Surya lines by
    # Y-centre (recovering one column's worth of order) and runs a more
    # permissive NW pass.  Terminates when either side is below the minimum
    # threshold, or when a pass produces no new matches (prevents infinite
    # loops on pages where remaining lines are simply unmatchable).
    while True:
        unmatched_si_list = [
            si for si in range(len(sorted_surya_lines)) if si not in matched_si
        ]
        if len(unmatched_si_list) < _PASS2_MIN_LINES or len(unmatched_gemini) < _PASS2_MIN_LINES:
            break

        rem_surya_indexed = sorted(
            ((si, sorted_surya_lines[si]) for si in unmatched_si_list),
            key=lambda pair: (pair[1]["bbox"][1] + pair[1]["bbox"][3]) / 2,
        )
        rem_orig_si = [si for si, _ in rem_surya_indexed]
        rem_surya   = [ln for _, ln in rem_surya_indexed]
        rem_texts   = [ln["text"] for ln in rem_surya]
        rem_gemini  = list(unmatched_gemini)

        rem_pairs = needleman_wunsch(rem_texts, rem_gemini, _text_sim, gap=_NW_GAP_PASS2)

        rem_matched_gi: set[int] = set()
        any_matched = False
        for si2, gi2 in rem_pairs:
            if si2 is None or gi2 is None:
                continue
            rem_matched_gi.add(gi2)
            matched_si.add(rem_orig_si[si2])
            any_matched = True
            bbox = list(rem_surya[si2]["bbox"])
            rem_entry: dict = {
                "bbox": bbox,
                "canvas_fragment": fragment_fn(bbox),
                "confidence": "line",
                "gemini_text": rem_gemini[gi2],
            }
            sc = rem_surya[si2].get("surya_confidence")
            if sc is not None:
                rem_entry["surya_confidence"] = sc
            result_lines.append(rem_entry)

        unmatched_gemini = [
            rem_gemini[gi2]
            for gi2 in range(len(rem_gemini))
            if gi2 not in rem_matched_gi and len(rem_gemini[gi2].strip()) > 8
        ]

        if not any_matched:
            break

    return result_lines, unmatched_gemini


# ---------------------------------------------------------------------------
# Canvas-fragment resolution (shared by NW align and line-crop fallback)
# ---------------------------------------------------------------------------

def resolve_fragment_fn(
    image_path: Path, img_w: int, img_h: int
) -> tuple["callable", dict]:
    """Build the canvas_fragment() closure and resolve canvas metadata for an image.

    Returns ``(fragment_fn, meta)`` where ``fragment_fn(bbox)`` maps a
    pixel-space bbox to a ``canvas_uri#xywh=…`` fragment string, and ``meta`` is
    a dict with keys ``canvas_uri``, ``canvas_width``, ``canvas_height``, and
    ``coords_from_fallback`` (all already coerced to ``value or None``).

    Used by :func:`align_image`; factored out so any alternative line source
    (e.g. a per-line crop OCR path) can reuse the identical scaling and emit
    canvas-space coordinates from the same Surya pixel bboxes.

    Resolution rules (unchanged from the original inline logic):

    * Natural image dimensions are fetched from the IIIF ``info.json`` so output
      coords land in image pixel space (Mirador maps ``xywh`` directly there).
    * A near-square ``info.json`` response for a clearly non-square downloaded
      image is treated as a server placeholder and replaced with the downloaded
      dimensions.
    * On ``info.json`` failure, fall back to the manifest canvas size (warning
      loudly if that too is square).
    * Split ``_left`` / ``_right`` images read their ``_split.json`` sidecar so
      crop coords are translated back to full-spread space before scaling.
    """
    stem = image_path.stem
    image_id = _extract_image_id(image_path.name)
    manifest_path = image_path.parent / "manifest.json"
    canvas_uri, canvas_w, canvas_h = load_canvas_info(manifest_path, image_id)

    # Fetch natural image dimensions so canvas_fragment outputs native pixel
    # coords (Mirador maps xywh directly to image pixel space).
    nat_w, nat_h = (0, 0)
    coords_from_fallback = False
    if canvas_uri:
        nat_w, nat_h = _get_natural_dims(canvas_uri)
    # Sanity-check: some NYPL images return a square placeholder (e.g.
    # 2560×2560) from info.json even though the actual image is portrait or
    # landscape.  Detect this by comparing the info.json aspect ratio to the
    # downloaded image's aspect ratio.  If info.json returned a nearly-square
    # result but the downloaded image is clearly non-square (ratio differs by
    # more than 15 %), treat it as a bad response and fall back to downloaded
    # image dimensions so coordinates land in the right place.
    if nat_w and nat_h and img_w and img_h:
        nat_ratio = nat_w / nat_h
        img_ratio = img_w / img_h
        _is_square = abs(nat_ratio - 1.0) < 0.02          # info.json ≈ square
        _img_non_square = abs(img_ratio - 1.0) > 0.15     # downloaded image is not
        if _is_square and _img_non_square:
            _log(
                f"  *** WARNING: {image_path.name}: info.json returned a square"
                f" ({nat_w}x{nat_h}) but the downloaded image is non-square"
                f" ({img_w}x{img_h}, ratio={img_ratio:.3f}). This is likely a"
                f" server-side placeholder. Falling back to downloaded image"
                f" dimensions for canvas coordinates."
            )
            nat_w, nat_h = img_w, img_h
            coords_from_fallback = True
    if not nat_w:
        nat_w, nat_h = canvas_w, canvas_h
        coords_from_fallback = True
        if canvas_w and canvas_w == canvas_h:
            _log(
                f"  *** WARNING: {image_path.name}: info.json fetch failed after retries."
                f" Falling back to manifest canvas size ({canvas_w}x{canvas_h}), which is"
                f" SQUARE — almost certainly a placeholder. Bounding-box coordinates will"
                f" be in the wrong space. Re-run --align-ocr --force once the network is"
                f" stable, or run tools/rescale_canvas_fragments.py to fix in place."
            )

    # For split images (_left / _right), read the sidecar to get the
    # x_offset that maps split-image coordinates back to full-spread
    # canvas coordinates before any canvas-vs-download scaling.
    split_x_offset = 0
    split_y_offset = 0
    full_img_w = img_w   # width of the full downloaded image pre-split
    full_img_h = img_h
    for _suffix in ("_left", "_right"):
        if stem.endswith(_suffix):
            _split_json = image_path.with_name(
                f"{stem[: -len(_suffix)]}_split.json"
            )
            if _split_json.exists():
                try:
                    _sidecar = json.loads(
                        _split_json.read_text(encoding="utf-8")
                    )
                    full_img_w = _sidecar.get("original_width", img_w)
                    full_img_h = _sidecar.get("original_height", img_h)
                    _side = _suffix.lstrip("_")
                    for _page in _sidecar.get("pages", []):
                        if _page.get("side") == _side:
                            split_x_offset = _page.get("x_offset", 0)
                            split_y_offset = _page.get("y_offset", 0)
                            break
                except Exception:
                    pass
            break

    def fragment(bbox: list[int]) -> str:
        if not canvas_uri:
            return ""
        # Translate split-image pixel coords to full-spread coords,
        # then scale to canvas space (handles download-resolution caps).
        offset_bbox = [
            bbox[0] + split_x_offset,
            bbox[1] + split_y_offset,
            bbox[2] + split_x_offset,
            bbox[3] + split_y_offset,
        ]
        return _canvas_fragment(
            canvas_uri, offset_bbox, full_img_w, full_img_h, nat_w, nat_h
        )

    meta = {
        "canvas_uri": canvas_uri or None,
        "canvas_width": nat_w or None,
        "canvas_height": nat_h or None,
        "coords_from_fallback": coords_from_fallback or None,
    }
    return fragment, meta


# ---------------------------------------------------------------------------
# Per-image alignment
# ---------------------------------------------------------------------------

def align_image(
    image_path: Path,
    model: str,
    force: bool = False,
    quiet: bool = False,
    min_surya_confidence: float = 0.35,
) -> tuple[str, bool]:
    """
    Align Gemini OCR text and Surya bounding boxes for one image.

    Returns (status, possible_column_merge) where status is one of
    'ok' / 'skipped' / 'missing' / 'failed' and possible_column_merge is True
    when the Surya:Gemini line-count ratio exceeds _MERGE_RATIO_THRESHOLD.
    """
    slug = model_slug(model)
    stem = image_path.stem
    gemini_txt  = image_path.parent / f"{stem}_{slug}.txt"
    surya_path  = image_path.parent / f"{stem}_surya.json"
    out_path    = image_path.parent / f"{stem}_{slug}_aligned.json"

    if out_path.exists() and not force:
        return "skipped", False

    use_surya = surya_path.exists()

    if not gemini_txt.exists() or not use_surya:
        return "missing", False

    try:
        # Parse OCR source ----------------------------------------------------
        surya_median_conf: "float | None" = None
        page_bbox, lines, surya_median_conf = parse_surya(
            surya_path, min_confidence=min_surya_confidence
        )
        lines = _expand_multiline_surya_lines(lines)

        img_w, img_h = page_bbox[2], page_bbox[3]
        page_w = img_w - page_bbox[0]

        lines = filter_short_lines(lines)
        lines = filter_margin_lines(lines, img_w)
        lines = sort_by_reading_order(lines, page_w)

        # Gemini text ---------------------------------------------------------
        # Parse line-by-line, filtering delimiter markers but tracking which
        # content lines fall inside advertisement blocks.  Ad-block lines are
        # excluded from anchor selection (they often contain repeated text such
        # as city/state footers that would create false anchors) but are still
        # included in the NW pass so Surya ad bboxes can be consumed as gaps.
        gemini_text = gemini_txt.read_text(encoding="utf-8")
        gemini_lines: list[str] = []
        gemini_ad_indices: set[int] = set()
        _in_ad = False
        for _raw_ln in gemini_text.splitlines():
            _ln = _raw_ln.strip()
            if not _ln:
                continue
            if re.match(r"^=== .+ ===$", _ln):
                _in_ad = _ln.startswith("=== ADVERTISEMENT")
                continue
            if _in_ad:
                gemini_ad_indices.add(len(gemini_lines))
            gemini_lines.append(_ln)

        # IIIF canvas info ----------------------------------------------------
        fragment, canvas_meta = resolve_fragment_fn(image_path, img_w, img_h)

        # NW alignment --------------------------------------------------------
        result_lines, unmatched_gemini = _build_line_aligned_lines(
            lines, gemini_lines, fragment, gemini_ad_indices
        )

        needs_review = (
            surya_median_conf is not None
            and surya_median_conf < _LOW_CONFIDENCE_PAGE_THRESHOLD
        )

        # Flag pages where Gemini may have merged two columns per line.
        n_surya  = len(lines)
        n_gemini = len(gemini_lines)
        ratio = n_surya / n_gemini if n_gemini >= _MERGE_MIN_LINES else 0.0
        possible_column_merge = (
            n_surya  >= _MERGE_MIN_LINES
            and _MERGE_RATIO_THRESHOLD < ratio <= _MERGE_RATIO_MAX
        )

        result = {
            "image": image_path.name,
            "model": model,
            "canvas_uri": canvas_meta["canvas_uri"],
            "canvas_width": canvas_meta["canvas_width"],
            "canvas_height": canvas_meta["canvas_height"],
            "coords_from_fallback": canvas_meta["coords_from_fallback"],
            "surya_median_confidence": round(surya_median_conf, 4) if surya_median_conf is not None else None,
            "needs_review": needs_review,
            "possible_column_merge": possible_column_merge,
            "lines": result_lines,
            "unmatched_gemini": unmatched_gemini,
        }
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return "ok", possible_column_merge

    except Exception as exc:  # noqa: BLE001
        if not quiet:
            _log(f"  Error aligning {image_path.name}: {exc}")
        return "failed", False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align Gemini OCR text with Surya line bboxes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Root images directory (e.g. output/greenbooks)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        metavar="MODEL",
        help=(
            "Gemini model name used to generate the .txt files "
            f"(default: {DEFAULT_OCR_MODEL}, auto-detected from output files if omitted)."
        ),
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=min(4, os.cpu_count() or 1),
        metavar="N",
        help="Parallel workers (default: 4)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process images that already have an aligned JSON file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-image progress output",
    )
    parser.add_argument(
        "--min-surya-confidence",
        type=float,
        default=0.35,
        metavar="THRESHOLD",
        help=(
            "Skip Surya lines whose detection confidence is below this threshold "
            "(0.0–1.0). Filters ghost detections and noisy lines before NW alignment. "
            "Default: 0.35. Set to 0.0 to keep all lines."
        ),
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    if not output_root.exists():
        print(f"Error: directory not found: {output_root}", file=sys.stderr)
        sys.exit(1)

    if args.model is None:
        args.model = (
            get_ocr_model(output_root)
            or discover_ocr_slug(output_root)
            or DEFAULT_OCR_MODEL
        )
        if not getattr(args, "quiet", False):
            print(f"  Auto-detected OCR model: {args.model}", file=sys.stderr)

    # Same split-aware image selection used across the pipeline
    all_jpgs = sorted(output_root.rglob("*.jpg"))
    images = []
    for p in all_jpgs:
        # Skip visualization output files
        if p.stem.endswith("_viz"):
            continue
        # Always include split output files
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        left = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        images.append(p)

    if not images:
        print(f"No .jpg files found under {output_root}", file=sys.stderr)
        sys.exit(0)

    total = len(images)
    if not args.quiet:
        print(
            f"Aligning {total} image(s) — model={args.model}, {args.workers} worker(s)…",
            file=sys.stderr,
        )

    counts = {"ok": 0, "skipped": 0, "missing": 0, "failed": 0}
    completed = 0
    flagged_merge: list[Path] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(align_image, img, args.model, args.force, args.quiet): img
            for img in images
        }
        for future in as_completed(futures):
            image_path = futures[future]
            completed += 1
            try:
                status, merge_flag = future.result()
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                merge_flag = False
                _log(f"Warning: exception for {image_path.name}: {exc}")

            counts[status] += 1
            if merge_flag:
                slug = model_slug(args.model)
                flagged_merge.append(
                    image_path.parent / f"{image_path.stem}_{slug}.txt"
                )

            if not args.quiet:
                slug = model_slug(args.model)
                out_name = f"{image_path.stem}_{slug}_aligned.json"
                if status == "skipped":
                    _log(f"[{completed:04d}/{total}] Skipped (exists): {out_name}")
                elif status == "ok":
                    suffix = " [merged-columns?]" if merge_flag else ""
                    _log(f"[{completed:04d}/{total}] Done:    {out_name}{suffix}")
                elif status == "missing":
                    _log(
                        f"[{completed:04d}/{total}] Missing: Surya JSON "
                        f"and/or Gemini txt for {image_path.name}"
                    )
                else:
                    _log(f"[{completed:04d}/{total}] FAILED:  {image_path.name}")

    if not args.quiet:
        parts = [f"{counts['ok']} aligned", f"{counts['skipped']} skipped"]
        if counts["missing"]:
            parts.append(
                f"{counts['missing']} missing (need both Surya JSON and Gemini txt)"
            )
        if counts["failed"]:
            parts.append(f"{counts['failed']} failed")
        print(f"\nDone. {total} image(s): {', '.join(parts)}.", file=sys.stderr)

    if flagged_merge and not args.quiet:
        pct = len(flagged_merge) / max(counts["ok"], 1) * 100
        print(
            f"\n⚠  {len(flagged_merge)} page(s) ({pct:.0f}% of aligned) may have "
            f"merged columns (Surya:Gemini line ratio {_MERGE_RATIO_THRESHOLD}–{_MERGE_RATIO_MAX}).\n"
            "Re-run Gemini OCR on these before extracting entries:",
            file=sys.stderr,
        )
        for p in sorted(flagged_merge):
            print(f"  {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
