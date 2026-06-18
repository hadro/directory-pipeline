#!/usr/bin/env python3
"""
Sample K listing pages from each directory in a master CSV and download ONLY those pages
(never whole volumes) into the pipeline's per-item dir, ready for `--gemini-ocr --extract-entries`.

Multi-source: each master row says where the volume lives; we resolve it to a IIIF manifest, pick
K canvases from the listing range, TRIM the manifest to just those canvases, and let the existing
`download_images.py --manifest <trimmed.json>` fetch exactly them (so naming + manifest.json match
the normal pipeline contract and downstream stages just work).

Master CSV schema (see city-directory-extraction/data_prep/master_directories.README.md):
    source,id,publisher,city,borough,year,start_page,end_page,column_count,sample_page,...
  source -> id:
    nypl  -> item UUID           -> https://api-collections.nypl.org/manifests/{id}
    ia    -> IA identifier       -> https://iiif.archive.org/iiif/{id}/manifest.json
    loc   -> loc.gov item id/URL -> fetch_manifest handles the Cloudflare-blocked manifest
    iiif  -> a full manifest URL -> used directly

Usage
-----
    # sample 10 pages from every volume in the master list:
    python sources/sample_directories.py ../city-directory-extraction/data_prep/master_directories.csv -k 10
    # just one source / publisher, fewer pages (cheap test):
    python sources/sample_directories.py <master.csv> -k 3 --source nypl --publisher Trow --limit 1

After this, run the (paid) OCR/NER on just the sampled pages:
    pipeline ocr output/<slug>/ && pipeline extract output/<slug>/      # or --gemini-ocr --extract-entries
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent          # directory-pipeline/
sys.path.insert(0, str(REPO))
from pipeline.api import iter_canvases                  # noqa: E402
from pipeline.download_images import fetch_manifest     # noqa: E402
from utils.iiif_utils import manifest_version           # noqa: E402

_SESSION = requests.Session()


def manifest_url(source: str, ident: str) -> str:
    source = source.strip().lower()
    ident = ident.strip()
    if source == "nypl":
        return f"https://api-collections.nypl.org/manifests/{ident}"
    if source == "ia":
        return f"https://iiif.archive.org/iiif/{ident}/manifest.json"
    if source == "loc":
        return ident if ident.startswith("http") else f"https://www.loc.gov/item/{ident}/manifest.json"
    if source == "iiif":
        return ident                                    # already a manifest URL
    raise ValueError(f"unknown source {source!r} (want nypl|ia|loc|iiif)")


def pick_indices(n: int, start_page, end_page, k: int) -> list[int]:
    """K canvas indices evenly spaced across the listing window. Use the printed start/end page as
    the window when it fits the canvas count; otherwise fall back to the middle 80% (skips front/
    back matter). Exact pages don't matter for name harvesting — we just want dense listing pages."""
    lo, hi = int(n * 0.10), int(n * 0.90)               # default: middle 80%
    try:
        s, e = int(start_page) - 1, int(end_page)
        if 0 <= s < e <= n:
            lo, hi = s, e
    except (TypeError, ValueError):
        pass
    span = max(1, hi - lo)
    k = min(k, span)
    step = span / k
    return sorted({min(hi - 1, lo + int(i * step + step / 2)) for i in range(k)})


def trim_manifest(manifest: dict, idxs: list[int]) -> dict:
    """Keep only the selected canvases (IIIF v2 or v3)."""
    ver = str(manifest_version(manifest))
    if ver.startswith("2"):
        canv = manifest["sequences"][0]["canvases"]
        manifest["sequences"][0]["canvases"] = [canv[i] for i in idxs]
    else:                                               # v3
        manifest["items"] = [manifest["items"][i] for i in idxs]
    return manifest


def slugify(row: dict) -> str:
    raw = f"{row.get('source','')}_{row.get('publisher','')}_{row.get('year','')}_{row['id'][:8]}"
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("master_csv", help="path to master_directories.csv")
    ap.add_argument("-k", "--pages", type=int, default=10, help="sample pages per volume")
    ap.add_argument("--width", type=int, default=2000, help="image width px (download_images --width)")
    ap.add_argument("--out-root", default=str(REPO / "output"), help="root for output/<slug>/ dirs")
    ap.add_argument("--source", help="only rows with this source")
    ap.add_argument("--publisher", help="only rows whose publisher contains this (case-insensitive)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of volumes (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="resolve + pick pages; skip download")
    args = ap.parse_args(argv)

    rows = [r for r in csv.DictReader(open(args.master_csv, encoding="utf-8")) if r.get("id")]
    if args.source:
        rows = [r for r in rows if r.get("source", "").lower() == args.source.lower()]
    if args.publisher:
        rows = [r for r in rows if args.publisher.lower() in (r.get("publisher", "").lower())]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("no matching rows in master list")

    done = []
    for r in rows:
        slug = slugify(r)
        out_dir = Path(args.out_root) / slug
        try:
            url = manifest_url(r["source"], r["id"])
            manifest = fetch_manifest(_SESSION, url)
            n = len(list(iter_canvases(manifest)))
            idxs = pick_indices(n, r.get("start_page"), r.get("end_page"), args.pages)
            print(f"[{slug}] {n} canvases -> sample {len(idxs)} pages: {idxs}", file=sys.stderr)
            if args.dry_run:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            trimmed = trim_manifest(manifest, idxs)
            mpath = out_dir / "_sample_manifest.json"
            mpath.write_text(json.dumps(trimmed), encoding="utf-8")
            subprocess.run(
                [sys.executable, str(REPO / "pipeline" / "download_images.py"),
                 "--manifest", str(mpath), "--output-dir", str(out_dir),
                 "--width", str(args.width), "--resume"],
                check=True,
            )
            done.append(str(out_dir))
        except Exception as e:                           # keep going across volumes
            print(f"[{slug}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    if done:
        print("\nsampled dirs ready for OCR/extract:", file=sys.stderr)
        for d in done:
            print(f"  {d}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
