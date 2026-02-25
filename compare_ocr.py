#!/usr/bin/env python3
"""Compare OCR output from multiple models, including Tesseract and Gemini.

Accepts any mix of Gemini model names and the special token "tesseract" via
--models.  For Gemini models the script calls the API (or loads an existing
.txt file).  For Tesseract it reads the {stem}_tesseract.txt file produced by
run_ocr.py — no API call is made.

Outputs (placed alongside each image):
  {stem}_comparison.html     — side-by-side model comparison
  ocr_comparison_stats.csv   — summary stats (in the root images directory)

Requires GEMINI_API_KEY only when Gemini models are listed.
The system prompt (for Gemini API calls) is read from ocr_prompt.md.

Usage
-----
    python compare_ocr.py images/greenbooks/item_uuid \\
        --models gemini-2.0-flash tesseract

    python compare_ocr.py images/greenbooks/item_uuid \\
        --models gemini-2.0-flash gemini-2.5-pro tesseract --workers 6
"""

import argparse
import csv
import difflib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT_FILE = Path(__file__).parent / "ocr_prompt.md"
TESSERACT = "tesseract"

# Background colours for model panels in the HTML report
PANEL_COLORS = ["#e8f4fd", "#e8fde8", "#fdf5e8", "#fde8f4", "#f5e8fd"]

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr)


def model_slug(model: str) -> str:
    return model.replace("/", "_")


def txt_path_for(image_path: Path, model: str) -> Path:
    if model == TESSERACT:
        return image_path.parent / f"{image_path.stem}_tesseract.txt"
    return image_path.parent / f"{image_path.stem}_{model_slug(model)}.txt"


def get_text(
    image_path: Path,
    model: str,
    client,           # genai.Client | None
    system_prompt: str,
) -> tuple[str, str, str]:
    """
    Return (model, status, text) for one (image, model) pair.
    status is one of: 'ok', 'skipped', 'missing', 'empty', 'failed'.
    """
    path = txt_path_for(image_path, model)

    # ── Tesseract: read existing file, no API call ──────────────────────────
    if model == TESSERACT:
        if not path.exists():
            return model, "missing", ""
        text = path.read_text(encoding="utf-8")
        return model, "skipped", text

    # ── Gemini: load from cache or call API ─────────────────────────────────
    if path.exists():
        if path.stat().st_size > 0:
            return model, "skipped", path.read_text(encoding="utf-8")
        return model, "empty", ""

    if client is None:
        return model, "failed", ""

    try:
        from google.genai.types import GenerateContentConfig, Part  # noqa: PLC0415

        with open(image_path, "rb") as f:
            img_bytes = f.read()
        response = client.models.generate_content(
            model=model,
            config=GenerateContentConfig(system_instruction=system_prompt),
            contents=[Part.from_bytes(data=img_bytes, mime_type="image/jpeg")],
        )
        text = response.text or ""
        path.write_text(text, encoding="utf-8")
        return model, "ok", text
    except Exception as exc:  # noqa: BLE001
        _log(f"  API error {model} / {image_path.name}: {exc}")
        return model, "failed", ""


def build_comparison_html(image_path: Path, results: dict[str, str]) -> str:
    models = list(results.keys())

    panels_html = ""
    for i, model in enumerate(models):
        color = PANEL_COLORS[i % len(PANEL_COLORS)]
        text = results[model]
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.splitlines())
        panels_html += f"""
        <div class="panel" style="background:{color}">
            <h3>{model}</h3>
            <div class="stats">{word_count} words &nbsp;·&nbsp; {char_count} chars &nbsp;·&nbsp; {line_count} lines</div>
            <pre>{escaped}</pre>
        </div>"""

    comparison_section = ""
    if len(models) == 2:
        a_lines = results[models[0]].splitlines(keepends=True)
        b_lines = results[models[1]].splitlines(keepends=True)
        diff_table = difflib.HtmlDiff(wrapcolumn=80).make_table(
            a_lines, b_lines,
            fromdesc=models[0], todesc=models[1],
            context=True, numlines=3,
        )
        ratio = difflib.SequenceMatcher(None, results[models[0]], results[models[1]]).ratio()
        comparison_section = f"""
        <h2>Side-by-side diff &nbsp;<span class="sim">({ratio * 100:.1f}% similar)</span></h2>
        <div class="diff-wrap">{diff_table}</div>"""

    elif len(models) > 2:
        header_cells = "".join(f"<th>{m}</th>" for m in models)
        matrix_rows = ""
        for a in models:
            cells = ""
            for b in models:
                if a == b:
                    cells += "<td class='self'>—</td>"
                else:
                    ratio = difflib.SequenceMatcher(None, results[a], results[b]).ratio()
                    cells += f"<td>{ratio * 100:.1f}%</td>"
            matrix_rows += f"<tr><th>{a}</th>{cells}</tr>"
        comparison_section = f"""
        <h2>Pairwise similarity</h2>
        <table class="matrix">
            <tr><th></th>{header_cells}</tr>
            {matrix_rows}
        </table>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Comparison — {image_path.name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1rem 2rem; color: #222; max-width: 1600px; }}
  h1 {{ font-size: 1rem; color: #666; margin-bottom: 1rem; }}
  h2 {{ font-size: 1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
  h3 {{ margin: 0 0 0.25rem; font-size: 0.9rem; }}
  .panels {{ display: flex; gap: 1rem; flex-wrap: wrap; align-items: flex-start; }}
  .panel {{ flex: 1; min-width: 260px; border-radius: 6px; padding: 0.75rem 1rem; border: 1px solid #ddd; }}
  .stats {{ font-size: 0.75rem; color: #666; margin-bottom: 0.5rem; }}
  pre {{ white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; margin: 0; line-height: 1.5; }}
  .sim {{ font-weight: normal; color: #555; }}
  .diff-wrap {{ overflow-x: auto; margin-top: 0.5rem; }}
  table.diff {{ border-collapse: collapse; font-size: 0.78rem; width: 100%; }}
  .diff td, .diff th {{ padding: 1px 6px; }}
  .diff_header {{ background: #e0e0e0; }}
  .diff_next {{ background: #c0c0c0; }}
  td.diff_add {{ background: #c6efce; }}
  td.diff_chg {{ background: #ffeb9c; }}
  td.diff_sub {{ background: #ffc7ce; }}
  .matrix {{ border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }}
  .matrix td, .matrix th {{ border: 1px solid #ddd; padding: 5px 12px; text-align: center; }}
  .matrix th {{ background: #f5f5f5; font-weight: 600; }}
  td.self {{ color: #bbb; }}
</style>
</head>
<body>
<h1>OCR Comparison — {image_path.name}</h1>
<h2>Model outputs</h2>
<div class="panels">{panels_html}
</div>
{comparison_section}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OCR output from Gemini and/or Tesseract on the same images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images_dir",
        help="Directory of images to process (e.g. images/greenbooks/item_uuid)",
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        metavar="MODEL",
        help='Two or more model names to compare. Use "tesseract" for Tesseract output.',
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers for Gemini API calls (default: 4)",
    )
    parser.add_argument(
        "--skip-empty-rerun",
        action="store_true",
        help="Skip re-running Gemini API calls for empty output files; proceed directly to comparison.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-task progress output",
    )
    args = parser.parse_args()

    if len(args.models) < 2:
        parser.error("at least two --models are required for comparison.")

    gemini_models = [m for m in args.models if m != TESSERACT]

    # Only require API key if Gemini models are requested
    client = None
    system_prompt = ""
    if gemini_models:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
            sys.exit(1)
        if not PROMPT_FILE.exists():
            print(f"Error: prompt file not found: {PROMPT_FILE}", file=sys.stderr)
            sys.exit(1)
        system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
        from google import genai as _genai  # noqa: PLC0415
        client = _genai.Client(api_key=api_key)

    images_root = Path(args.images_dir)
    if not images_root.exists():
        print(f"Error: directory not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    all_jpgs = sorted(images_root.rglob("*.jpg"))
    images = []
    for p in all_jpgs:
        if p.stem.endswith("_left") or p.stem.endswith("_right"):
            images.append(p)
            continue
        left = p.with_name(f"{p.stem}_left.jpg")
        right = p.with_name(f"{p.stem}_right.jpg")
        if left.exists() and right.exists():
            continue
        images.append(p)
    if not images:
        print(f"No .jpg files found under {images_root}", file=sys.stderr)
        sys.exit(0)

    tasks = [(img, model) for img in images for model in args.models]
    total = len(tasks)

    if not args.quiet:
        print(
            f"{len(images)} image(s) × {len(args.models)} model(s) = {total} task(s) "
            f"({args.workers} workers)",
            file=sys.stderr,
        )

    image_results: dict[Path, dict[str, str]] = {img: {} for img in images}
    image_statuses: dict[Path, dict[str, str]] = {img: {} for img in images}
    counts: dict[str, int] = {"ok": 0, "skipped": 0, "missing": 0, "empty": 0, "failed": 0}
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(get_text, img, model, client, system_prompt): (img, model)
            for img, model in tasks
        }
        for future in as_completed(futures):
            img, model = futures[future]
            completed += 1
            try:
                _, status, text = future.result()
                image_results[img][model] = text
                image_statuses[img][model] = status
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                image_results[img][model] = ""
                image_statuses[img][model] = "failed"
                _log(f"Warning: {model} / {img.name}: {exc}")

            counts[status] += 1
            if not args.quiet:
                label = f"{img.name} / {model_slug(model)}"
                if status == "skipped":
                    _log(f"[{completed:04d}/{total}] Skipped: {label}")
                elif status == "ok":
                    _log(f"[{completed:04d}/{total}] Done:    {label}")
                elif status == "missing":
                    _log(f"[{completed:04d}/{total}] Missing: {label}")
                elif status == "empty":
                    _log(f"[{completed:04d}/{total}] Empty:   {label}")
                else:
                    _log(f"[{completed:04d}/{total}] FAILED:  {label}")

    # Smart re-run for Gemini empty files (not applicable to tesseract)
    rerun_tasks = []
    if not args.skip_empty_rerun:
        for img in images:
            has_content = any(image_results[img].get(m, "").strip() for m in args.models)
            if has_content:
                for m in gemini_models:
                    if image_statuses[img].get(m) == "empty":
                        rerun_tasks.append((img, m))

    if rerun_tasks:
        if not args.quiet:
            print(
                f"\nRe-running {len(rerun_tasks)} empty Gemini file(s)…",
                file=sys.stderr,
            )
        for img, model in rerun_tasks:
            txt = txt_path_for(img, model)
            if txt.exists():
                txt.unlink()
            if not args.quiet:
                _log(f"  Re-running: {img.name} / {model_slug(model)}")
            try:
                _, status, text = get_text(img, model, client, system_prompt)
                image_results[img][model] = text
                counts["ok" if status == "ok" else "failed"] += 1
                counts["empty"] -= 1
            except Exception as exc:  # noqa: BLE001
                _log(f"  Re-run failed: {img.name} / {model_slug(model)}: {exc}")

    # Write per-image HTML comparisons
    if not args.quiet:
        print("\nWriting HTML comparisons…", file=sys.stderr)
    for img in images:
        # Only include models that have some result (skip "missing" tesseract)
        present_models = [
            m for m in args.models
            if image_statuses[img].get(m) not in ("missing", "failed")
               or image_results[img].get(m, "")
        ]
        if len(present_models) < 1:
            continue
        results = {m: image_results[img].get(m, "") for m in present_models}
        html = build_comparison_html(img, results)
        html_path = img.parent / f"{img.stem}_comparison.html"
        html_path.write_text(html, encoding="utf-8")
        if not args.quiet:
            _log(f"  → {html_path}")

    # Write summary stats CSV
    stats_path = images_root / "ocr_comparison_stats.csv"
    fieldnames = ["image", "model", "char_count", "word_count", "line_count", "status"]
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for img in images:
            for model in args.models:
                text = image_results[img].get(model, "")
                status = image_statuses[img].get(model, "")
                writer.writerow({
                    "image": img.name,
                    "model": model,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "line_count": len(text.splitlines()),
                    "status": status,
                })

    if not args.quiet:
        print(f"Stats CSV → {stats_path}", file=sys.stderr)
        parts = [f"{counts['ok']} processed", f"{counts['skipped']} skipped"]
        if counts["missing"]:
            parts.append(f"{counts['missing']} missing (tesseract not yet run)")
        if counts["failed"]:
            parts.append(f"{counts['failed']} failed")
        print(f"\nDone. {total} task(s): {', '.join(parts)}.", file=sys.stderr)


if __name__ == "__main__":
    main()
