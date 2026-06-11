#!/usr/bin/env python3
"""
Annotation tool for reviewing and correcting Surya/Gemini alignment results.

Opens a local web UI where you can:
  - Browse pages sorted by number of unmatched Gemini entries
  - Draw a bounding box around an unmatched region
  - Re-run Surya OCR on just that crop
  - Review proposed matches and save accepted ones back to the _aligned.json

Usage
-----
    uv run review_alignment.py output/
    uv run review_alignment.py output/ --model gemini-3.1-flash-lite --port 5001

Then open http://localhost:5000 in your browser.
"""

import argparse
import difflib
import json
import os
import re
import sys
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, send_file
from PIL import Image

load_dotenv()

app = Flask(__name__)

# Project root is one level up from this script (pipeline/)
PROJECT_ROOT: Path = Path(__file__).parent.parent

from utils.models import DEFAULT_OCR_MODEL

# Set at startup
OUTPUT_ROOT: Path = Path("output")
MODEL: str = DEFAULT_OCR_MODEL

# Surya models – pre-loaded at startup (see main())
_det = None
_rec = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _text_sim(a: str, b: str) -> int:
    """0–100 similarity matching align_ocr._text_sim exactly."""
    an, bn = _normalize(a), _normalize(b)
    if not an and not bn:
        return 50
    if not an or not bn:
        return -80
    return round(difflib.SequenceMatcher(None, an, bn, autojunk=False).ratio() * 100)


def _canvas_fragment(
    canvas_uri: str, bbox: list[int],
    img_w: int, img_h: int, canvas_w: int, canvas_h: int,
) -> str:
    x1, y1, x2, y2 = bbox
    if canvas_w and img_w and canvas_w != img_w:
        sx = canvas_w / img_w
        sy = canvas_h / img_h if img_h else 1.0
        x1, y1, x2, y2 = round(x1*sx), round(y1*sy), round(x2*sx), round(y2*sy)
    return f"{canvas_uri}#xywh={x1},{y1},{max(1, x2-x1)},{max(1, y2-y1)}"


def _load_surya() -> None:
    global _det, _rec
    if _det is not None:
        return
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    _det = DetectionPredictor()
    _rec = RecognitionPredictor(FoundationPredictor())


def _json_suffix() -> str:
    return f"_{MODEL}_aligned.json"


def _stem_from(json_path: Path) -> str:
    return json_path.name[: -len(_json_suffix())]


def _img_path(json_path: Path) -> Path:
    return json_path.parent / f"{_stem_from(json_path)}.jpg"


def _find_pages() -> list[dict]:
    pages = []
    for p in sorted(OUTPUT_ROOT.rglob(f"*{_json_suffix()}")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ug = data.get("unmatched_gemini", [])
        img = _img_path(p)
        if not img.exists():
            continue
        # Volume = first path component below OUTPUT_ROOT, if there are at
        # least two directory levels (volume/item_dir/file).  Single-level
        # runs (item_dir/file) have an empty volume string.
        rel_parts = p.relative_to(OUTPUT_ROOT).parts
        volume = rel_parts[0] if len(rel_parts) >= 3 else ""
        median_conf = data.get("surya_median_confidence")
        pages.append({
            "json_path": str(p),
            "img_path": str(img),
            "stem": _stem_from(p),
            "item_dir": p.parent.name,
            "volume": volume,
            "unmatched_count": len(ug),
            "aligned_count": len(data.get("lines", [])),
            "needs_review": data.get("needs_review", False),
            "surya_median_confidence": median_conf,
        })
    # Sort: pages flagged needs_review (lowest median confidence first) come
    # before un-flagged pages; within each group, highest unmatched count first.
    pages.sort(key=lambda x: (
        not x["needs_review"],               # flagged pages first (False < True)
        x["surya_median_confidence"] if x["needs_review"] and x["surya_median_confidence"] is not None else 1.0,
        -x["unmatched_count"],
    ))
    return pages


# ---------------------------------------------------------------------------
# NW alignment (mirrors align_ocr.needleman_wunsch)
# ---------------------------------------------------------------------------

def _nw_align(
    surya_lines: list[dict],
    gemini_texts: list[str],
    gap: int = -40,
) -> list[dict]:
    """
    Align Surya crop lines against the unmatched Gemini text list.
    Returns a list of pair dicts:
      {surya: line_dict | None, gemini: str | None, sim: int}
    """
    seq_a = [ln["text"] for ln in surya_lines]
    seq_b = gemini_texts
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return []

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ch = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap
        ch[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap
        ch[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores = {
                "D": dp[i-1][j-1] + _text_sim(seq_a[i-1], seq_b[j-1]),
                "U": dp[i-1][j] + gap,
                "L": dp[i][j-1] + gap,
            }
            best = max(scores, key=scores.__getitem__)
            dp[i][j] = scores[best]
            ch[i][j] = best

    pairs: list[dict] = []
    i, j = n, m
    while i > 0 or j > 0:
        c = ch[i][j]
        if c == "D":
            sim = _text_sim(seq_a[i-1], seq_b[j-1])
            pairs.append({"surya": surya_lines[i-1], "gemini": seq_b[j-1], "sim": sim})
            i -= 1; j -= 1
        elif c == "U":
            pairs.append({"surya": surya_lines[i-1], "gemini": None, "sim": 0})
            i -= 1
        else:
            pairs.append({"surya": None, "gemini": seq_b[j-1], "sim": 0})
            j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    pages = _find_pages()
    volumes = sorted({p["volume"] for p in pages if p["volume"]})
    return render_template_string(_HTML, pages=pages, volumes=volumes, model=MODEL)


@app.route("/image")
def serve_image():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return "Not found", 404
    return send_file(p)


@app.route("/page_data")
def page_data():
    json_path = Path(request.args.get("json_path", ""))
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    img = _img_path(json_path)
    img_w, img_h = Image.open(img).size  # actual jpg pixel dimensions
    return jsonify({
        "img_path": str(img),
        "img_width": img_w,
        "img_height": img_h,
        "canvas_uri": data.get("canvas_uri", ""),
        "canvas_width": data.get("canvas_width", img_w),
        "canvas_height": data.get("canvas_height", img_h),
        "lines": data.get("lines", []),
        "unmatched_gemini": data.get("unmatched_gemini", []),
        "needs_review": data.get("needs_review", False),
        "surya_median_confidence": data.get("surya_median_confidence"),
    })


@app.route("/annotate", methods=["POST"])
def annotate():
    """Crop the image, run Surya OCR on each box, NW-align combined lines with unmatched_gemini."""
    try:
        body = request.json
        json_path = Path(body["json_path"])
        # Accept either 'bboxes' (array) or legacy 'bbox' (single)
        bboxes = body.get("bboxes") or [body["bbox"]]

        _load_surya()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Client may supply an override list (e.g. during reset-alignment mode where
        # existing matched lines have been moved back to unmatched client-side).
        unmatched = body.get("unmatched_override") or data.get("unmatched_gemini", [])

        img = Image.open(_img_path(json_path)).convert("RGB")
        iw, ih = img.size

        all_surya_lines: list[dict] = []
        for bbox in bboxes:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(iw, int(bbox[2]))
            y2 = min(ih, int(bbox[3]))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = img.crop((x1, y1, x2, y2))
            results = _rec([crop], det_predictor=_det, sort_lines=True)
            for ln in results[0].text_lines:
                all_surya_lines.append({
                    "bbox": [
                        int(ln.bbox[0]) + x1, int(ln.bbox[1]) + y1,
                        int(ln.bbox[2]) + x1, int(ln.bbox[3]) + y1,
                    ],
                    "text": ln.text,
                    "confidence": round(float(getattr(ln, "confidence", 1.0)), 4),
                })

        if not all_surya_lines:
            return jsonify({"error": "No lines detected (boxes may be too small)"}), 400

        # Sort all lines top-to-bottom across all crops before alignment
        all_surya_lines.sort(key=lambda ln: ln["bbox"][1])

        pairs = _nw_align(all_surya_lines, unmatched)
        return jsonify({"surya_lines": all_surya_lines, "pairs": pairs})

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        return jsonify({"error": f"{type(exc).__name__}: {exc}", "traceback": tb}), 500


@app.route("/save", methods=["POST"])
def save():
    """Merge accepted pairs into the _aligned.json."""
    body = request.json
    json_path = Path(body["json_path"])
    accepted = body["accepted"]  # [{surya_bbox, gemini_text}, ...]
    clear_existing = body.get("clear_existing", False)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if clear_existing:
        # Move all previously-aligned gemini_text back to unmatched before
        # merging the new manual pairs.
        old_texts = [ln["gemini_text"] for ln in data.get("lines", []) if ln.get("gemini_text")]
        lines = []
        unmatched = old_texts + list(data.get("unmatched_gemini", []))
    else:
        lines = data.get("lines", [])
        unmatched = list(data.get("unmatched_gemini", []))

    canvas_uri = data.get("canvas_uri", "")
    cw = data.get("canvas_width", 0)
    ch = data.get("canvas_height", 0)
    img_path = _img_path(json_path)
    img = Image.open(img_path)
    iw, ih = img.size

    # For split images (_left / _right), read the sidecar to get the x_offset
    # that maps split-image coordinates back to full-spread canvas coordinates.
    # Mirrors the same logic in align_ocr.align_image().
    stem = img_path.stem
    split_x_offset = 0
    split_y_offset = 0
    full_img_w = iw
    full_img_h = ih
    for _suffix in ("_left", "_right"):
        if stem.endswith(_suffix):
            _split_json = img_path.with_name(f"{stem[:-len(_suffix)]}_split.json")
            if _split_json.exists():
                try:
                    _sidecar = json.loads(_split_json.read_text(encoding="utf-8"))
                    full_img_w = _sidecar.get("original_width", iw)
                    full_img_h = _sidecar.get("original_height", ih)
                    _side = _suffix.lstrip("_")
                    for _page in _sidecar.get("pages", []):
                        if _page.get("side") == _side:
                            split_x_offset = _page.get("x_offset", 0)
                            split_y_offset = _page.get("y_offset", 0)
                            break
                except Exception:
                    pass
            break

    for pair in accepted:
        bbox = pair["surya_bbox"]
        gem = pair["gemini_text"]
        offset_bbox = [
            bbox[0] + split_x_offset,
            bbox[1] + split_y_offset,
            bbox[2] + split_x_offset,
            bbox[3] + split_y_offset,
        ]
        lines.append({
            "bbox": bbox,
            "canvas_fragment": _canvas_fragment(canvas_uri, offset_bbox, full_img_w, full_img_h, cw, ch),
            "confidence": "manual",
            "gemini_text": gem,
        })
        if gem in unmatched:
            unmatched.remove(gem)

    data["lines"] = lines
    data["unmatched_gemini"] = unmatched
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return jsonify({"ok": True, "remaining_unmatched": len(unmatched)})


@app.route("/delete_lines", methods=["POST"])
def delete_lines():
    """Remove matched lines by bbox and return their gemini_text to unmatched."""
    body = request.json
    json_path = Path(body["json_path"])
    bboxes_to_delete = {tuple(b) for b in body["bboxes"]}

    data = json.loads(json_path.read_text(encoding="utf-8"))
    removed_texts = []
    kept = []
    for ln in data.get("lines", []):
        if tuple(ln["bbox"]) in bboxes_to_delete:
            if ln.get("gemini_text"):
                removed_texts.append(ln["gemini_text"])
        else:
            kept.append(ln)
    data["lines"] = kept
    data["unmatched_gemini"] = removed_texts + list(data.get("unmatched_gemini", []))
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonify({"ok": True, "removed": len(removed_texts),
                    "remaining_unmatched": len(data["unmatched_gemini"])})


@app.route("/done", methods=["POST"])
def done():
    """Shut the server down so the pipeline can continue to the next stage."""
    def _shutdown() -> None:
        import time
        import signal
        time.sleep(0.4)   # let the JSON response reach the browser first
        # Send SIGTERM so Python's normal shutdown sequence runs, which lets
        # PyTorch DataLoader workers terminate cleanly (avoids "leaked
        # semaphore" warning that os._exit() triggers by skipping cleanup).
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------

# The page template lives in review_alignment.html next to this file — extracted from an
# inline string so it gets HTML/JS syntax highlighting and reviewable diffs.
# Generated output is unaffected; the template still ships inside the package.
_HTML = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global OUTPUT_ROOT, MODEL

    parser = argparse.ArgumentParser(
        description="Annotation tool for reviewing Surya/Gemini alignment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Root images directory (e.g. output/ or output/greenbooks)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_OCR_MODEL,
        help="Gemini model name used in aligned JSON filenames (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to listen on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open the review UI in a browser automatically",
    )
    args = parser.parse_args()

    OUTPUT_ROOT = Path(args.output_dir)
    if not OUTPUT_ROOT.exists():
        print(f"Error: directory not found: {OUTPUT_ROOT}", file=sys.stderr)
        sys.exit(1)

    MODEL = args.model

    print(f"  images: {OUTPUT_ROOT.resolve()}", flush=True)
    print(f"  model:  {MODEL}", flush=True)
    print("Loading Surya models… (this takes ~30 s)", flush=True)
    _load_surya()
    print("Models ready.", flush=True)
    print(f"Alignment review server — http://{args.host}:{args.port}", flush=True)
    if not args.no_open:
        # 0.0.0.0 binds all interfaces but isn't a browsable address itself.
        open_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        url = f"http://{open_host}:{args.port}"
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
