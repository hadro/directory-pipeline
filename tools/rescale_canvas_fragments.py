#!/usr/bin/env python3
"""Rescale canvas_fragment xywh coordinates in aligned JSON files.

When --align-ocr runs and _get_natural_dims() fails (network timeout, etc.),
it falls back to the IIIF manifest canvas size (often square, e.g. 2560x2560)
instead of the true image pixel dimensions.  The resulting xywh coordinates
are in the wrong space, causing thumbnail crops to land in the wrong region.

This script corrects the stored coordinates by:
  1. Reading canvas_width/canvas_height from each aligned JSON
  2. Fetching the actual natural dimensions from the IIIF info.json
  3. If they differ, rescaling every #xywh= fragment by (nat_w/cw, nat_h/ch)
  4. Updating canvas_width/canvas_height to the natural values
  5. Writing the corrected file back in place

All other content is left completely untouched: gemini_text, confidence
values, manual review edits, surya_confidence, bbox, unmatched_gemini.

Usage:
    python pipeline/rescale_canvas_fragments.py output/my_volume/ \\
        --aligned-model gemini-2.0-flash

    python pipeline/rescale_canvas_fragments.py output/collection/ \\
        --aligned-model gemini-2.0-flash --dry-run
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


_XYWH_RE = re.compile(r"#xywh=(\d+),(\d+),(\d+),(\d+)$")


def _fetch_natural_dims(uri: str) -> tuple[int, int]:
    """Fetch (width, height) from info.json for the IIIF image in *uri*."""
    base_url = uri.split("#")[0]
    try:
        parts = urlparse(base_url)
        path_parts = parts.path.split("/")
        # IIIF Image API path: /{prefix}/{identifier}/{region}/…
        # Service base uses the first 4 path segments.
        if len(path_parts) < 5:
            return 0, 0
        service_base = f"{parts.scheme}://{parts.netloc}" + "/".join(path_parts[:4])
        info_url = f"{service_base}/info.json"
        with urllib.request.urlopen(info_url, timeout=15) as resp:
            info = json.loads(resp.read())
        return int(info.get("width") or 0), int(info.get("height") or 0)
    except Exception as exc:
        print(f"  Warning: could not fetch info.json from {base_url[:70]}: {exc}",
              file=sys.stderr)
        return 0, 0


def _rescale_fragment(fragment: str, sx: float, sy: float) -> str:
    m = _XYWH_RE.search(fragment)
    if not m:
        return fragment
    x, y, w, h = (int(m.group(i)) for i in range(1, 5))
    x2 = round(x * sx)
    y2 = round(y * sy)
    w2 = max(1, round(w * sx))
    h2 = max(1, round(h * sy))
    return fragment[: m.start()] + f"#xywh={x2},{y2},{w2},{h2}"


def rescale_file(path: Path, dry_run: bool = False) -> dict:
    """Rescale one aligned JSON file in place. Returns a stats dict."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": f"read_error: {e}"}

    cw = data.get("canvas_width") or 0
    ch = data.get("canvas_height") or 0
    if not cw or not ch:
        return {"status": "no_canvas_dims"}

    # Get a sample URI to derive the info.json endpoint from
    sample_uri = data.get("canvas_uri") or ""
    if not sample_uri:
        for ln in data.get("lines", []):
            cf = ln.get("canvas_fragment", "")
            if cf:
                sample_uri = cf
                break
    if not sample_uri:
        return {"status": "no_uri_found"}

    nat_w, nat_h = _fetch_natural_dims(sample_uri)
    if not nat_w:
        return {"status": "info_fetch_failed"}

    if nat_w == cw and nat_h == ch:
        return {"status": "already_correct", "dims": f"{nat_w}x{nat_h}"}

    sx = nat_w / cw
    sy = nat_h / ch
    rescaled = 0

    for ln in data.get("lines", []):
        cf = ln.get("canvas_fragment", "")
        if cf and "#xywh=" in cf:
            ln["canvas_fragment"] = _rescale_fragment(cf, sx, sy)
            rescaled += 1

    data["canvas_width"] = nat_w
    data["canvas_height"] = nat_h

    if not dry_run:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "rescaled",
        "from": f"{cw}x{ch}",
        "to": f"{nat_w}x{nat_h}",
        "sx": round(sx, 4),
        "sy": round(sy, 4),
        "lines": rescaled,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir",
                   help="Item directory or collection directory containing aligned JSON files")
    p.add_argument("--aligned-model", default="gemini-2.0-flash",
                   help="Model slug used in *_aligned.json filenames (default: gemini-2.0-flash)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing files")
    p.add_argument("--check", action="store_true",
                   help="Exit with code 1 if any mismatches are found (implies --dry-run); "
                        "suitable for use as a post-alignment audit step")
    args = p.parse_args()
    if args.check:
        args.dry_run = True

    root = Path(args.output_dir)
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    suffix = f"_{args.aligned_model}_aligned.json"
    aligned_files = sorted(root.rglob(f"*{suffix}"))
    if not aligned_files:
        print(f"No *{suffix} files found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(aligned_files)} aligned JSON file(s)"
          + (" — dry run, no files written" if args.dry_run else ""))

    total_rescaled = total_ok = total_skipped = 0
    for af in aligned_files:
        stats = rescale_file(af, dry_run=args.dry_run)
        status = stats["status"]
        if status == "rescaled":
            total_rescaled += 1
            print(f"  {af.name}: {stats['from']} → {stats['to']} "
                  f"(sx={stats['sx']}, sy={stats['sy']}, {stats['lines']} lines rescaled)")
        elif status == "already_correct":
            total_ok += 1
            print(f"  {af.name}: already correct ({stats['dims']})")
        else:
            total_skipped += 1
            print(f"  {af.name}: skipped — {status}")

    verb = "would be " if args.dry_run else ""
    print(f"\nDone: {total_rescaled} {verb}rescaled, "
          f"{total_ok} already correct, {total_skipped} skipped.")

    if args.check and total_rescaled > 0:
        print(f"\n*** {total_rescaled} file(s) have mismatched canvas coordinates."
              f" Run without --check to fix them in place.")
        sys.exit(1)


if __name__ == "__main__":
    main()
