#!/usr/bin/env python3
"""Interactive browser UI for page selection — two modes in one interface.

Tab 1 — Sample pages: pick 4–10 representative pages for OCR/NER prompt
  calibration.  Saves selection.txt, consumed by generate_prompt.py.

Tab 2 — Scope pages: mark which pages in the volume should be processed by
  subsequent pipeline steps (OCR, extraction).  All pages start selected;
  deselect frontmatter/junk.  Saves included_pages.txt, consumed by
  run_gemini_ocr.py and extract_entries.py.  If included_pages.txt is absent,
  all pages are processed (backward compatible).

Usage
-----
    python pipeline/select_pages.py output/the_travelers_guide_e088efa0/
    python pipeline/select_pages.py output/the_travelers_guide_e088efa0/uuid/
    python pipeline/select_pages.py output/the_travelers_guide_e088efa0/ --no-open
"""

import argparse
import functools
import http.server
import json
import re
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add project root to path so utils can be imported

# ---------------------------------------------------------------------------
# Image discovery helpers
# ---------------------------------------------------------------------------

_EXCLUDE = re.compile(
    r"_(?:viz|surya|gemini|aligned|entries|annotations)"
)
_SPLIT = re.compile(r"_(left|right)$", re.IGNORECASE)


def _source_images(item_dir: Path) -> list[Path]:
    """Return sorted source JPGs.

    When split pages (*_left.jpg / *_right.jpg) exist for a spread, those are
    returned in place of the original double-page spread file.
    """
    all_jpgs = set(item_dir.glob("*.jpg"))

    # Partition into base images and split pages.
    base_jpgs = sorted(
        p for p in all_jpgs
        if not _EXCLUDE.search(p.stem) and not _SPLIT.search(p.stem)
    )
    split_jpgs = {p for p in all_jpgs if _SPLIT.search(p.stem)}

    result: list[Path] = []
    for base in base_jpgs:
        left = item_dir / f"{base.stem}_left.jpg"
        right = item_dir / f"{base.stem}_right.jpg"
        splits = [p for p in (left, right) if p in split_jpgs]
        if splits:
            result.extend(splits)
        else:
            result.append(base)

    return result


def _find_item_dirs(root: Path) -> list[Path]:
    """Given a slug dir (or item dir), return the list of item directories
    that contain source images."""
    # If root itself has images, use it directly.
    if _source_images(root):
        return [root]
    # Otherwise descend one level (slug → item_id sub-dirs).
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    found = [d for d in candidates if _source_images(d)]
    return found


def _load_txt(path: Path) -> list[str]:
    """Return non-empty, non-comment lines from a text file, or [] if absent."""
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


# ---------------------------------------------------------------------------
# Local HTTP server for direct save
# ---------------------------------------------------------------------------

class _SaveHandler(http.server.SimpleHTTPRequestHandler):
    """Serve images from item_dir and accept POSTs to write selection files."""
    save_path: Path   # selection.txt
    scope_path: Path  # included_pages.txt

    def do_POST(self):
        if self.path in ("/save", "/save-scope"):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length))
                text = body.get("selection", "")
                if self.path == "/save":
                    out = type(self).save_path
                    label = "selection.txt"
                else:
                    out = type(self).scope_path
                    label = "included_pages.txt"
                out.write_text(text, encoding="utf-8")
                payload = json.dumps({"path": str(out)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
                print(f"\n  Saved {label} → {out}", file=sys.stderr)
            except Exception as exc:
                self.send_error(500, str(exc))
        else:
            self.send_error(404)

    def log_message(self, *_):
        pass  # suppress per-request logging


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

# The page template lives in select_pages.html next to this file — extracted from an
# inline string so it gets HTML/JS syntax highlighting and reviewable diffs.
# Generated output is unaffected; the template still ships inside the package.
_HTML_TEMPLATE = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML generator
# ---------------------------------------------------------------------------

def _read_scope_excluded(item_dir: Path, images: list[Path]) -> list[str]:
    """Return filenames to mark as excluded in the scope tab on load.

    If included_pages.txt exists, excluded = all images NOT in that file.
    If it doesn't exist, return [] (all included = no pre-exclusions).
    """
    included_path = item_dir / "included_pages.txt"
    if not included_path.exists():
        return []
    included = set(_load_txt(included_path))
    all_names = {p.name for p in images}
    return [n for n in all_names if n not in included]


def generate_html(
    item_dir: Path,
    images: list[Path],
    save_url: str | None = None,
    save_scope_url: str | None = None,
) -> Path:
    """Write select_pages.html into item_dir and return its path."""
    fnames = [p.name for p in images]
    volume = item_dir.parent.name + "/" + item_dir.name

    # Pre-populate from existing saved files
    preselected = _load_txt(item_dir / "selection.txt")
    prescoped_excl = _read_scope_excluded(item_dir, images)

    save_sample_label = "Save to output folder" if save_url else "Download selection.txt"
    save_scope_label  = "Save scope"            if save_scope_url else "Download included_pages.txt"

    html = _HTML_TEMPLATE.format(
        volume             = volume,
        images_json        = json.dumps(fnames),
        save_url_json      = json.dumps(save_url),
        save_scope_url_json = json.dumps(save_scope_url),
        preselected_json   = json.dumps(preselected),
        prescoped_excl_json = json.dumps(prescoped_excl),
        save_sample_label  = save_sample_label,
        save_scope_label   = save_scope_label,
    )
    out_path = item_dir / "select_pages.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _seed_selection_from_sections(
    sections_path: Path,
    item_dir: Path,
    images: list[Path],
    selection_path: Path,
    n_per_section: int = 3,
) -> None:
    """Write an initial selection.txt with sample pages from each section.

    Only called when sections.txt exists and selection.txt does not yet exist.
    Samples n_per_section pages spread evenly through each section's range.
    """
    from utils.section_utils import load_sections

    all_names = [p.name for p in images]
    try:
        sections = load_sections(sections_path, all_names)
    except ValueError as e:
        print(f"  Warning: could not parse sections.txt: {e}", file=sys.stderr)
        return
    if not sections:
        return

    selected: list[str] = []
    for sec in sections:
        indices = sec["page_indices"]
        if not indices:
            continue
        if len(indices) <= n_per_section:
            picks = indices
        else:
            step = (len(indices) - 1) / (n_per_section - 1) if n_per_section > 1 else 0
            picks = [indices[round(i * step)] for i in range(n_per_section)]
        for idx in picks:
            fname = all_names[idx]
            if (item_dir / fname).exists() and fname not in selected:
                selected.append(fname)

    if selected:
        selection_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        print(
            f"  Auto-seeded selection.txt with {len(selected)} pages "
            f"from {len(sections)} section(s): {', '.join(s['label'] for s in sections)}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a browser-based page-selector (two-tab: sample + scope).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help="Slug directory (output/{slug}/) or item directory containing .jpg files",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Generate the HTML but do not open it in a browser",
    )
    parser.add_argument(
        "--sections",
        metavar="PATH",
        default=None,
        help=(
            "Path to sections.txt. When provided (or when sections.txt is found "
            "automatically in output_dir), seeds selection.txt with representative "
            "pages from each section if selection.txt does not already exist."
        ),
    )
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    item_dirs = _find_item_dirs(root)
    if not item_dirs:
        print(f"Error: no source .jpg images found under {root}", file=sys.stderr)
        sys.exit(1)

    # Detect sections.txt automatically if not explicitly supplied
    sections_path: Path | None = None
    if args.sections:
        sections_path = Path(args.sections)
        if not sections_path.exists():
            print(f"Warning: --sections file not found: {sections_path}", file=sys.stderr)
            sections_path = None
    else:
        candidate = root / "sections.txt"
        if candidate.exists():
            sections_path = candidate

    for item_dir in item_dirs:
        images = _source_images(item_dir)

        # Auto-seed selection.txt from section samples when sections.txt is present
        selection_path = item_dir / "selection.txt"
        if sections_path and not selection_path.exists():
            _seed_selection_from_sections(sections_path, item_dir, images, selection_path)

        if args.no_open:
            out_path = generate_html(item_dir, images)
            print(f"Generated: {out_path}  ({len(images)} pages)", file=sys.stderr)
        else:
            _SaveHandler.save_path  = item_dir / "selection.txt"
            _SaveHandler.scope_path = item_dir / "included_pages.txt"
            handler = functools.partial(_SaveHandler, directory=str(item_dir))
            with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
                server.allow_reuse_address = True
                port = server.server_address[1]
                save_url       = f"http://127.0.0.1:{port}/save"
                save_scope_url = f"http://127.0.0.1:{port}/save-scope"

                out_path = generate_html(item_dir, images, save_url, save_scope_url)
                url = f"http://127.0.0.1:{port}/select_pages.html"
                print(f"Serving: {url}  ({len(images)} pages)", file=sys.stderr)
                print("Press Ctrl+C when done.", file=sys.stderr)

                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                webbrowser.open(url)

                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    server.shutdown()
                    print("", file=sys.stderr)


if __name__ == "__main__":
    main()
