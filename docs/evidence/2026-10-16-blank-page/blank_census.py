"""Which pages does the blank-page check skip, and are any of them real?

For every page scan under output/ (NNNN_*.jpg|png, not *_viz), records:

  legacy  the rule before 2026-10-16: 512 px thumbnail, pixels darker than
          0.5 x p90 background, blank below 0.2% (inlined below)
  live    utils.image_utils.is_blank_page as it stands
  dark    the live rule's dark fraction (1600 px, darker than 0.75 x p90)
  ocr     length of the longest Gemini .txt beside the page

and prints the summary in results.txt. A page the check skips writes no .txt,
so "legacy-blank, no .txt, in a volume whose other pages have one" is the
signature of a page a past run silently dropped.

Run from the repo root (20-30 minutes on all cores; needs output/):
    ./.venv/bin/python docs/evidence/2026-10-16-blank-page/blank_census.py [--csv PATH]

The live arm imports utils.image_utils, so after a change it measures the change.
"""
import argparse
import datetime as dt
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils.image_utils import is_blank_page  # noqa: E402

PAGE_RE = re.compile(r"^\d{4}_.*\.(jpe?g|png)$", re.I)


def _crop(gray, margin=0.08):
    w, h = gray.size
    mx, my = int(w * margin), int(h * margin)
    return np.asarray(gray.crop((mx, my, w - mx, h - my)), dtype=np.uint8)


def legacy_blank(path):
    """is_blank_page as it was at commit 8740c4a."""
    with Image.open(path) as im:
        gray = ImageOps.grayscale(im)
        gray.thumbnail((512, 512))
    a = _crop(gray)
    bg = float(np.percentile(a, 90))
    if bg < 100:
        return False
    return float((a < bg * 0.5).mean()) < 0.002


def live_dark(path):
    with Image.open(path) as im:
        im.thumbnail((1600, 1600))
        a = _crop(ImageOps.grayscale(im))
    return float((a < np.percentile(a, 90) * 0.75).mean())


def ocr(path):
    """(longest Gemini .txt in chars, ISO date of the newest one or "")."""
    best, newest = 0, ""
    for t in path.parent.glob(f"{path.stem}_gemini-*.txt"):
        best = max(best, len(t.read_text(errors="replace").strip()))
        newest = max(newest, dt.date.fromtimestamp(t.stat().st_mtime).isoformat())
    return best, newest


def one(path_str):
    p = Path(path_str)
    try:
        return (path_str, legacy_blank(p), is_blank_page(p), live_dark(p), *ocr(p), "")
    except Exception as e:  # noqa: BLE001 — unreadable scans are counted, not fatal
        return (path_str, None, None, None, *ocr(p), type(e).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, help="also write one row per page here")
    args = ap.parse_args()

    paths = sorted(
        str(p) for p in (ROOT / "output").rglob("*")
        if p.is_file() and PAGE_RE.match(p.name) and not p.stem.endswith("_viz")
    )
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one, paths, chunksize=32))

    if args.csv:
        import csv
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "legacy_blank", "live_blank", "live_dark",
                        "ocr_chars", "ocr_date", "error"])
            w.writerows(rows)

    ok = [r for r in rows if not r[6]]
    legacy = [r for r in ok if r[1]]
    live = [r for r in ok if r[2]]
    print(f"page scans: {len(rows):,}   unreadable: {len(rows) - len(ok)}")
    print(f"legacy rule blank: {len(legacy):,}")
    print(f"live rule blank:   {len(live):,}   of them not legacy-blank: "
          f"{sum(1 for r in live if not r[1])}")
    released = [r for r in legacy if not r[2]]
    print(f"legacy-blank pages the live rule sends to OCR: {len(released):,}"
          f"   (with >=1,000 chars of Gemini OCR from an earlier run: "
          f"{sum(1 for r in released if r[4] >= 1000)})")
    print(f"live-blank pages with >=1,000 chars of Gemini OCR from an earlier run: "
          f"{sum(1 for r in live if r[4] >= 1000)}")

    print("\nlive dark fraction, legacy-blank pages:")
    for label, grp in (("  live blank   ", [r for r in legacy if r[2]]),
                       ("  live not     ", released)):
        if grp:
            q = np.percentile([r[3] for r in grp], [0, 10, 50, 90, 100])
            print(label + "  ".join(f"{v:.5f}" for v in q) + "   (min p10 p50 p90 max)")

    # Per volume: pages, pages with OCR, legacy-blank, live-blank, legacy-blank
    # pages with no OCR (the possible silent skips), OCR coverage of the pages
    # the legacy rule passed, and the newest OCR date. Sample probes OCR a few
    # pages on purpose, so only volumes whose passed pages are >=80% OCR'd are
    # listed. The check landed 2026-06-11; a volume last OCR'd before then
    # cannot have lost pages to it.
    vols = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, ""])
    for path, lg, lv, _, chars, date, err in rows:
        if err:
            continue
        v = vols[str(Path(path).parent.relative_to(ROOT / "output"))]
        v[0] += 1
        v[1] += chars > 0
        v[2] += bool(lg)
        v[3] += bool(lv)
        v[4] += bool(lg) and chars == 0
        v[5] += not lg
        v[6] += (not lg) and chars > 0
        v[7] = max(v[7], date)
    print("\nvolumes with legacy-blank pages missing OCR, where >=80% of the other pages have OCR:")
    print(f"  {'pages':>5} {'ocr':>5} {'legacy':>6} {'live':>5} {'no-ocr':>6}  {'last OCR':<10}  volume")
    for name, (n, o, lg, lv, gap, passed, passed_ocr, date) in sorted(
            vols.items(), key=lambda kv: -kv[1][4]):
        if gap and passed and passed_ocr >= 0.8 * passed:
            print(f"  {n:>5} {o:>5} {lg:>6} {lv:>5} {gap:>6}  {date:<10}  {name}")


if __name__ == "__main__":
    main()
