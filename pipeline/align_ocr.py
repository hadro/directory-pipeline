#!/usr/bin/env python3
"""Align Gemini OCR text with Surya (or Tesseract) bounding boxes using Needleman-Wunsch.

Primary backend: Surya (line-level)
------------------------------------
For each image that has both {stem}_{model}.txt (Gemini) and
{stem}_surya.json (Surya), produces {stem}_{model}_aligned.json with
line-level bounding boxes from Surya and corrected text from Gemini.

Legacy backend: Tesseract (word-level)
----------------------------------------
If no _surya.json is found but a _tesseract.hocr file exists, the aligner
falls back to word-level Tesseract alignment for backward compatibility.

Reading-order correction
------------------------
OCR lines are re-sorted by detected column (left→right) then top-to-bottom
within each column before alignment.

IIIF canvas URIs and dimensions are read from manifest.json cached by
download_images.py in each item directory.

Confidence tiers
----------------
  line — matched with line-level bboxes from Surya (primary)
  word — matched with word-level bboxes from Tesseract (legacy fallback)

Output JSON
-----------
  {
    "image": "0001_58030238.jpg",
    "model": "gemini-2.0-flash",
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
import json
import os
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import iiif_utils

DEFAULT_MODEL = "gemini-2.0-flash"


def _discover_ocr_slug(output_root: Path) -> str | None:
    """Scan *output_root* for Gemini OCR files and return the most-common model slug."""
    from collections import Counter
    counts: Counter[str] = Counter()
    for pattern, regex in [
        ("*_aligned.json", re.compile(r"_(gemini-[^_]+)_aligned\.json$")),
        ("*.txt",          re.compile(r"_(gemini-[^_]+)\.txt$")),
    ]:
        for candidate in (output_root, *[d for d in sorted(output_root.iterdir()) if d.is_dir()]):
            for f in candidate.glob(pattern):
                m = regex.search(f.name)
                if m:
                    counts[m.group(1)] += 1
            if counts:
                return counts.most_common(1)[0][0]
    return None

_print_lock = threading.Lock()

BBOX_RE = re.compile(r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")

# ---------------------------------------------------------------------------
# IIIF image-service helpers
# ---------------------------------------------------------------------------

# In-process cache: service_base → (nat_w, nat_h).
# Shared across threads; worst case two threads fetch the same URL once.
_info_cache: dict[str, tuple[int, int]] = {}


def _get_natural_dims(canvas_id: str) -> tuple[int, int]:
    """Return (width, height) of the natural image by fetching info.json.

    Parses the IIIF Image API service base from *canvas_id* (a URL of the form
    ``https://host/prefix/identifier/region/size/rotation/quality.fmt``), then
    fetches ``{service_base}/info.json``.  Results are cached in-process.
    Returns (0, 0) on any error.
    """
    parts = urlparse(canvas_id)
    path_parts = parts.path.split("/")
    # IIIF Image API path: /{prefix}/{identifier}/{region}/…
    # Service base = scheme + host + first 4 path segments (index 0–3, where 0 is "")
    if len(path_parts) < 5:
        return 0, 0
    service_base = f"{parts.scheme}://{parts.netloc}" + "/".join(path_parts[:4])
    if service_base in _info_cache:
        return _info_cache[service_base]
    info_url = f"{service_base}/info.json"
    try:
        with urllib.request.urlopen(info_url, timeout=15) as resp:
            info = json.loads(resp.read())
        w = int(info.get("width") or 0)
        h = int(info.get("height") or 0)
    except Exception as exc:
        _log(f"  Warning: could not fetch {info_url}: {exc}")
        w, h = 0, 0
    _info_cache[service_base] = (w, h)
    return w, h

# Needleman-Wunsch gap penalty (must be negative)
_NW_GAP = -40
# Second-pass alignment: more permissive penalty + minimum size to bother
_NW_GAP_PASS2    = -20
_PASS2_MIN_LINES =  5
# Pages whose median Surya detection confidence falls below this threshold are
# flagged needs_review=True in the aligned JSON so --review-alignment can
# surface them for manual inspection.
_LOW_CONFIDENCE_PAGE_THRESHOLD = 0.5

# Anchor matching: commit Gemini↔Tesseract line pairs that are near-verbatim
# matches BEFORE running the global NW.  This prevents the aligner from
# consuming heading words (city names, state names, category lines) on the
# wrong Gemini lines — the failure mode where e.g. "CHERAW" entries mapped to
# CHARLESTON bounding boxes because the global NW preferred nearby wrong words
# over paying the gap cost to reach the real CHERAW Tesseract line.
_ANCHOR_MIN_LEN = 3     # minimum normalised chars to consider a line as an anchor
_ANCHOR_MAX_SUFFIX = 6  # extra trailing chars allowed in prefix match (Tesseract noise)
_ANCHOR_SIM_HIGH = 0.92 # similarity threshold when prefix match doesn't apply


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def model_slug(model: str) -> str:
    return model.replace("/", "_")


# ---------------------------------------------------------------------------
# hOCR parsing
# ---------------------------------------------------------------------------

class _HocrParser(HTMLParser):
    """Parse Tesseract hOCR output into page bbox and a flat list of line dicts."""

    def __init__(self) -> None:
        super().__init__()
        self.page_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.lines: list[dict] = []
        self._span_depth = 0
        self._line: dict | None = None
        self._line_depth = -1
        self._word: dict | None = None
        self._word_depth = -1
        self._word_text: list[str] = []

    @staticmethod
    def _parse_bbox(title: str) -> tuple[int, int, int, int] | None:
        m = BBOX_RE.search(title)
        return (
            int(m.group(1)), int(m.group(2)),
            int(m.group(3)), int(m.group(4)),
        ) if m else None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        d = dict(attrs)
        cls = d.get("class", "")
        title = d.get("title", "")
        if tag == "span":
            self._span_depth += 1
        bbox = self._parse_bbox(title)
        if "ocr_page" in cls and bbox:
            self.page_bbox = bbox
        elif any(c in cls for c in ("ocr_line", "ocr_caption", "ocr_textfloat", "ocr_header")) and bbox and tag == "span":
            self._line = {"bbox": list(bbox), "words": []}
            self._line_depth = self._span_depth
        elif "ocrx_word" in cls and bbox and tag == "span":
            self._word = {"bbox": list(bbox)}
            self._word_depth = self._span_depth
            self._word_text = []

    def handle_data(self, data: str) -> None:
        if self._word is not None:
            self._word_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "span":
            return
        if self._word is not None and self._span_depth == self._word_depth:
            text = "".join(self._word_text).strip()
            if text and self._line is not None:
                self._word["text"] = text
                self._line["words"].append(self._word)
            self._word = None
            self._word_text = []
        elif self._line is not None and self._span_depth == self._line_depth:
            if self._line["words"]:
                self.lines.append(self._line)
            self._line = None
        self._span_depth -= 1


def parse_hocr(path: Path) -> tuple[tuple[int, int, int, int], list[dict]]:
    """
    Parse a Tesseract hOCR file.

    Returns (page_bbox, lines) where each line is:
        {'bbox': [x1, y1, x2, y2], 'words': [{'bbox': [...], 'text': str}, ...]}
    """
    parser = _HocrParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.page_bbox, parser.lines


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


# ---------------------------------------------------------------------------
# Reading-order correction
# ---------------------------------------------------------------------------

# Minimum line bbox height in pixels.  Lines below this are dot-leader
# artifacts or horizontal rules that Tesseract mis-OCRs as text; letting them
# into the word pool creates false-positive matches with garbage coordinates.
_MIN_LINE_HEIGHT = 15


def filter_short_lines(lines: list[dict], min_height: int = _MIN_LINE_HEIGHT) -> list[dict]:
    """Remove hOCR lines whose bounding box height is below min_height pixels."""
    return [ln for ln in lines if (ln["bbox"][3] - ln["bbox"][1]) >= min_height]


_READING_ORDER_BAND = 50  # pixels — horizontal band height for y-band sort.
# Lines within the same 50 px band are sorted left-column-first; bands are
# sorted top-to-bottom.  This keeps row pairs (y-difference ≤ ~35 px) in the
# same band while placing centred section headings (which have ≥ 80 px of
# whitespace above/below them) in their own band ahead of body content —
# preventing them from being pushed to the end of the Tesseract word sequence
# by a pure column-major sort.


def sort_by_reading_order(lines: list[dict], page_width: int) -> list[dict]:
    """
    Re-sort hOCR lines into correct reading order for multi-column pages.

    Default (row-major y-band sort): lines are grouped into 50 px horizontal
    bands (sorted top-to-bottom), and within each band sorted left-column-first.
    This correctly places centred page headings (e.g. state names) before
    the body-text columns they head.

    True two-column pages (column-major): when Tesseract lines cluster into
    exactly two distinct columns and each column contains at least 20% of all
    lines, Gemini reads the entire left column top-to-bottom before the entire
    right column.  In that case we emit left-column lines (sorted by y) followed
    by right-column lines (sorted by y) to match Gemini's reading order.

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

# Words whose y-center deviates more than this from their Gemini line's
# median word y-center are discarded as spatial outliers before bbox union.
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

    Returns pairs in monotonically increasing order on both axes.  Each
    OCR line and each Gemini line appears in at most one anchor.
    """
    tess_line_texts = [_normalize(t) for t in ocr_line_texts]
    gem_norms = [_normalize(g) for g in gemini_lines]

    # Lines whose normalised text appears more than once in the Gemini sequence
    # are ambiguous anchors: we cannot tell which Gemini occurrence corresponds
    # to which Tesseract line, so we skip them entirely.
    from collections import Counter
    gem_counts = Counter(n for n in gem_norms if len(n) >= _ANCHOR_MIN_LEN)

    anchors: list[tuple[int, int]] = []
    used_tess: set[int] = set()
    last_tess = -1

    for gi, gnorm in enumerate(gem_norms):
        if len(gnorm) < _ANCHOR_MIN_LEN:
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
            # Strategy 1: prefix match — one string is a clean prefix of the other.
            # Handles Tesseract appending garbage characters to a clean heading
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


def _build_word_aligned_lines(
    sorted_tess_lines: list[dict],
    gemini_lines: list[str],
    fragment_fn,
) -> tuple[list[dict], list[str]]:
    """
    Align Gemini text to Tesseract bounding boxes using anchored
    Needleman-Wunsch.

    Approach
    --------
    0.  Find "anchor" pairs: Gemini lines that are near-verbatim matches to an
        entire Tesseract line (city/state headings, category lines, etc.).
        Committing those as fixed anchors prevents the NW from misassigning
        heading words to wrong Gemini lines when the reading orders drift apart.
    1.  Split both flat word/token sequences into segments at anchor points.
    2.  Run an independent NW pass within each segment (and for each anchor
        line itself, to handle minor OCR noise in the heading text).
    3.  Group the matched Tesseract words back by their Gemini line index.
    4.  Apply an intra-line y-consistency filter: discard words whose
        y-center is more than _INTRA_LINE_Y_TOLERANCE px from the median
        y-center of all matched words in that line.
    5.  Lines with no remaining matches are added to unmatched_gemini.

    Why this is better than a single global NW pass
    ------------------------------------------------
    A single global NW can map "CHERAW" (Gemini line 33) to whatever Tesseract
    words happen to be cheapest at that position in the flat sequence, even
    though the real "CHERAW" Tesseract line is 20 positions later.  Anchoring
    "CHERAW" first splits the problem: the NW segment before the anchor handles
    CHARLESTON entries, and the segment after handles the rest of CHERAW —
    each is short enough that gap costs dominate correctly.

    Returns (result_lines, unmatched_gemini).
    """
    if not sorted_tess_lines or not gemini_lines:
        return [], list(gemini_lines)

    # Flat Tesseract word list + per-line boundary indices.
    # tess_line_starts[i]   = first flat index of words for Tess line i.
    # tess_line_starts[i+1] = first flat index of words for Tess line i+1
    #                         (= exclusive end of line i).  Sentinel appended.
    tess_words: list[dict] = []
    tess_line_starts: list[int] = []
    for tl in sorted_tess_lines:
        tess_line_starts.append(len(tess_words))
        tess_words.extend(tl["words"])
    tess_line_starts.append(len(tess_words))  # sentinel

    # Flat Gemini token list + per-line boundary indices (same sentinel pattern).
    gemini_tokens: list[tuple[int, str]] = []
    gem_line_starts: list[int] = []
    for li, gline in enumerate(gemini_lines):
        gem_line_starts.append(len(gemini_tokens))
        gemini_tokens.extend((li, w) for w in gline.split())
    gem_line_starts.append(len(gemini_tokens))  # sentinel

    if not tess_words or not gemini_tokens:
        return [], list(gemini_lines)

    def _nw_slice(tw_start: int, tw_end: int, gt_start: int, gt_end: int):
        """Run NW on flat-array slices; return pairs with original indices."""
        if tw_start >= tw_end or gt_start >= gt_end:
            return []
        pairs = needleman_wunsch(
            [w["text"] for w in tess_words[tw_start:tw_end]],
            [tok[1] for tok in gemini_tokens[gt_start:gt_end]],
            _text_sim,
        )
        return [
            (ti + tw_start if ti is not None else None,
             gi + gt_start if gi is not None else None)
            for ti, gi in pairs
        ]

    # Anchored segmented NW
    tess_line_texts = [" ".join(w["text"] for w in tl["words"]) for tl in sorted_tess_lines]
    anchors = _find_anchors(tess_line_texts, gemini_lines)

    all_pairs: list[tuple[int | None, int | None]] = []
    prev_tl = 0  # next Tess line to process (line index, inclusive)
    prev_gl = 0  # next Gem line to process (line index, inclusive)

    for tess_li, gem_li in anchors:
        # Segment before this anchor
        all_pairs.extend(_nw_slice(
            tess_line_starts[prev_tl], tess_line_starts[tess_li],
            gem_line_starts[prev_gl], gem_line_starts[gem_li],
        ))
        # Anchor line itself (tiny NW to handle minor OCR noise in the heading)
        all_pairs.extend(_nw_slice(
            tess_line_starts[tess_li], tess_line_starts[tess_li + 1],
            gem_line_starts[gem_li],  gem_line_starts[gem_li + 1],
        ))
        prev_tl = tess_li + 1
        prev_gl = gem_li + 1

    # Final segment after the last anchor
    all_pairs.extend(_nw_slice(
        tess_line_starts[prev_tl], len(tess_words),
        gem_line_starts[prev_gl],  len(gemini_tokens),
    ))

    # Group matched Tesseract words by Gemini line index
    line_matches: dict[int, list[tuple[str, list[int]]]] = {
        li: [] for li in range(len(gemini_lines))
    }
    for ti, gi in all_pairs:
        if ti is None or gi is None:
            continue
        li, gword = gemini_tokens[gi]
        line_matches[li].append((gword, tess_words[ti]["bbox"]))

    result_lines: list[dict] = []
    unmatched_gemini: list[str] = []

    for li, gline in enumerate(gemini_lines):
        matches = line_matches[li]
        if not matches:
            unmatched_gemini.append(gline)
            continue

        # Intra-line y-consistency filter
        if len(matches) > 1:
            y_centers = [_y_center(bb) for _, bb in matches]
            med_y = _median(y_centers)
            matches = [
                (gw, bb) for gw, bb in matches
                if abs(_y_center(bb) - med_y) <= _INTRA_LINE_Y_TOLERANCE
            ]

        if not matches:
            unmatched_gemini.append(gline)
            continue

        bboxes = [bb for _, bb in matches]
        line_bbox = _union_bbox(bboxes)

        output_words = [
            {
                "bbox": bb,
                "canvas_fragment": fragment_fn(bb),
                "confidence": "word",
                "text": gw,
            }
            for gw, bb in matches
        ]

        result_lines.append({
            "bbox": line_bbox,
            "canvas_fragment": fragment_fn(line_bbox),
            "confidence": "word",
            "gemini_text": gline,
            "words": output_words,
        })

    return result_lines, unmatched_gemini


# ---------------------------------------------------------------------------
# Line-level alignment (Surya backend)
# ---------------------------------------------------------------------------

def _build_line_aligned_lines(
    sorted_surya_lines: list[dict],
    gemini_lines: list[str],
    fragment_fn,
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
    anchors = _find_anchors(surya_texts, gemini_lines)

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

    # Any Gemini line not matched by a Surya line is unmatched
    unmatched_gemini = [
        gemini_lines[gi]
        for gi in range(len(gemini_lines))
        if gi not in matched_gi
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
            if gi2 not in rem_matched_gi
        ]

        if not any_matched:
            break

    return result_lines, unmatched_gemini


# ---------------------------------------------------------------------------
# Per-image alignment
# ---------------------------------------------------------------------------

def align_image(
    image_path: Path,
    model: str,
    force: bool = False,
    quiet: bool = False,
    min_surya_confidence: float = 0.35,
) -> str:
    """
    Align Gemini OCR text and OCR bounding boxes for one image.

    Prefers Surya (_surya.json, line-level) over Tesseract (_tesseract.hocr,
    word-level) when both are present.

    Returns 'ok', 'skipped', 'missing', or 'failed'.
    """
    slug = model_slug(model)
    stem = image_path.stem
    gemini_txt  = image_path.parent / f"{stem}_{slug}.txt"
    surya_path  = image_path.parent / f"{stem}_surya.json"
    hocr_path   = image_path.parent / f"{stem}_tesseract.hocr"
    out_path    = image_path.parent / f"{stem}_{slug}_aligned.json"

    if out_path.exists() and not force:
        return "skipped"

    use_surya = surya_path.exists()
    use_hocr  = hocr_path.exists()

    if not gemini_txt.exists() or (not use_surya and not use_hocr):
        return "missing"

    try:
        # Parse OCR source ----------------------------------------------------
        surya_median_conf: "float | None" = None
        if use_surya:
            page_bbox, lines, surya_median_conf = parse_surya(
                surya_path, min_confidence=min_surya_confidence
            )
        else:
            page_bbox, lines = parse_hocr(hocr_path)

        img_w, img_h = page_bbox[2], page_bbox[3]
        page_w = img_w - page_bbox[0]

        lines = sort_by_reading_order(lines, page_w)
        lines = filter_short_lines(lines)

        # Gemini text ---------------------------------------------------------
        gemini_text = gemini_txt.read_text(encoding="utf-8")
        gemini_lines = [
            ln for ln in (ln.strip() for ln in gemini_text.splitlines()) if ln
        ]

        # IIIF canvas info ----------------------------------------------------
        image_id = _extract_image_id(image_path.name)
        manifest_path = image_path.parent / "manifest.json"
        canvas_uri, canvas_w, canvas_h = load_canvas_info(manifest_path, image_id)

        # Fetch natural image dimensions so canvas_fragment outputs native pixel
        # coords (Mirador maps xywh directly to image pixel space).
        nat_w, nat_h = (0, 0)
        if canvas_uri:
            nat_w, nat_h = _get_natural_dims(canvas_uri)
        if not nat_w:
            nat_w, nat_h = canvas_w, canvas_h

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

        # NW alignment --------------------------------------------------------
        if use_surya:
            result_lines, unmatched_gemini = _build_line_aligned_lines(
                lines, gemini_lines, fragment
            )
        else:
            result_lines, unmatched_gemini = _build_word_aligned_lines(
                lines, gemini_lines, fragment
            )

        needs_review = (
            surya_median_conf is not None
            and surya_median_conf < _LOW_CONFIDENCE_PAGE_THRESHOLD
        )
        result = {
            "image": image_path.name,
            "model": model,
            "canvas_uri": canvas_uri or None,
            "canvas_width": nat_w or None,
            "canvas_height": nat_h or None,
            "surya_median_confidence": round(surya_median_conf, 4) if surya_median_conf is not None else None,
            "needs_review": needs_review,
            "lines": result_lines,
            "unmatched_gemini": unmatched_gemini,
        }
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return "ok"

    except Exception as exc:  # noqa: BLE001
        if not quiet:
            _log(f"  Error aligning {image_path.name}: {exc}")
        return "failed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Align Gemini OCR text with Surya line bboxes (primary) "
            "or Tesseract hOCR bboxes (legacy fallback)."
        ),
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
            f"(default: {DEFAULT_MODEL}, auto-detected from output files if omitted)."
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
        args.model = _discover_ocr_slug(output_root) or DEFAULT_MODEL
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

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(align_image, img, args.model, args.force, args.quiet): img
            for img in images
        }
        for future in as_completed(futures):
            image_path = futures[future]
            completed += 1
            try:
                status = future.result()
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                _log(f"Warning: exception for {image_path.name}: {exc}")

            counts[status] += 1

            if not args.quiet:
                slug = model_slug(args.model)
                out_name = f"{image_path.stem}_{slug}_aligned.json"
                if status == "skipped":
                    _log(f"[{completed:04d}/{total}] Skipped (exists): {out_name}")
                elif status == "ok":
                    _log(f"[{completed:04d}/{total}] Done:    {out_name}")
                elif status == "missing":
                    _log(
                        f"[{completed:04d}/{total}] Missing: Surya JSON (or hOCR) "
                        f"and/or Gemini txt for {image_path.name}"
                    )
                else:
                    _log(f"[{completed:04d}/{total}] FAILED:  {image_path.name}")

    if not args.quiet:
        parts = [f"{counts['ok']} aligned", f"{counts['skipped']} skipped"]
        if counts["missing"]:
            parts.append(
                f"{counts['missing']} missing (need both hOCR and Gemini txt)"
            )
        if counts["failed"]:
            parts.append(f"{counts['failed']} failed")
        print(f"\nDone. {total} image(s): {', '.join(parts)}.", file=sys.stderr)


if __name__ == "__main__":
    main()
