#!/usr/bin/env python3
"""Post-pipeline workflow: fix entries → combine volumes → build explorer.

Runs the three standard post-extraction steps in sequence against an output
directory. The model used during extraction is read from pipeline_state.json
automatically — no --model flag needed.

Usage
-----
    # Collection directory (runs combine step):
    python pipeline/postprocess.py output/green_books_and_related/

    # Single-volume directory (skips combine step):
    python pipeline/postprocess.py output/ldpd_11290437_000/ --no-combine

    # Dry run — print what would happen without writing:
    python pipeline/postprocess.py output/green_books_and_related/ --dry-run
"""

import argparse
import glob
import sys
from pathlib import Path


from pipeline.state import get_ner_model, get_ocr_model
from utils.models import DEFAULT_OCR_MODEL


def _find_model(directory: Path) -> str:
    model = get_ner_model(directory) or get_ocr_model(directory)
    if model:
        return model
    # Fallback: infer from existing entries CSV filenames.
    from collections import Counter
    import re
    counts: Counter = Counter()
    for p in directory.rglob("entries_*.csv"):
        m = re.search(r"entries_(gemini-[^_/]+)\.csv$", p.name)
        if m:
            counts[m.group(1)] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return DEFAULT_OCR_MODEL


def run_fix(directory: Path, model: str, dry_run: bool = False) -> list[Path]:
    from analysis.fix_entries import process_csv, _print_stats

    pattern = str(directory / f"*/entries_{model}.csv")
    # Try item-level subdirectory pattern first; fall back to top-level.
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        pattern2 = str(directory / f"entries_{model}.csv")
        paths = sorted(Path(p) for p in glob.glob(pattern2))

    if not paths:
        print(f"  fix: no entries_{model}.csv found under {directory}", file=sys.stderr)
        return []

    fixed: list[Path] = []
    for src in paths:
        dst = src.with_stem(src.stem + "_fixed")
        if dry_run:
            print(f"  [dry-run] fix: {src} → {dst}", file=sys.stderr)
            fixed.append(dst)
            continue
        stats = process_csv(src, dst)
        _print_stats(src, stats)
        fixed.append(dst)
    return fixed


def run_combine(directory: Path, fixed_paths: list[Path], dry_run: bool = False) -> Path | None:
    from analysis.combine_volumes import combine

    output_path = directory / "combined.csv"
    if dry_run:
        print(f"  [dry-run] combine: {directory} → {output_path}", file=sys.stderr)
        return output_path
    if not fixed_paths:
        print("  combine: no fixed CSVs to merge", file=sys.stderr)
        return None
    combine(directory, output_path)
    return output_path


def run_explore(csv_path: Path, dry_run: bool = False) -> Path | None:
    import subprocess, sys as _sys

    out_path = csv_path.with_stem(csv_path.stem + "_explorer").with_suffix(".html")
    if dry_run:
        print(f"  [dry-run] explore: {csv_path} → {out_path}", file=sys.stderr)
        return out_path
    result = subprocess.run(
        [_sys.executable, "pipeline/explore_entries.py", str(csv_path), "--out", str(out_path)],
        check=False,
    )
    if result.returncode != 0:
        print(f"  explore: failed (exit {result.returncode})", file=sys.stderr)
        return None
    return out_path


def postprocess(
    directory: Path,
    no_combine: bool = False,
    dry_run: bool = False,
    model: str | None = None,
) -> None:
    directory = directory.resolve()
    if not directory.is_dir():
        print(f"Error: not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    if model is None:
        model = _find_model(directory)

    print(f"\nPostprocessing {directory}", file=sys.stderr)
    print(f"  model: {model}", file=sys.stderr)

    # 1. Fix entries
    print("\n[1/3] Fixing entries…", file=sys.stderr)
    fixed_paths = run_fix(directory, model, dry_run=dry_run)

    # 2. Combine volumes (optional)
    combined_csv: Path | None = None
    if not no_combine:
        print("\n[2/3] Combining volumes…", file=sys.stderr)
        combined_csv = run_combine(directory, fixed_paths, dry_run=dry_run)
    else:
        print("\n[2/3] Skipping combine (--no-combine).", file=sys.stderr)

    # 3. Build explorer
    print("\n[3/3] Building explorer…", file=sys.stderr)
    explore_input = combined_csv or (fixed_paths[0] if fixed_paths else None)
    if explore_input is None:
        print("  explore: nothing to build explorer from", file=sys.stderr)
        return
    out = run_explore(explore_input, dry_run=dry_run)
    if out and not dry_run:
        print(f"\nDone. Explorer: {out}", file=sys.stderr)
    elif dry_run:
        print("\nDone (dry run).", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run fix → combine → explore on an output directory. "
            "Model is auto-detected from pipeline_state.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Output slug directory (e.g. output/green_books_and_related/).",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Skip the combine step (useful for single-volume directories).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing any files.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the auto-detected model slug.",
    )
    args = parser.parse_args()
    postprocess(
        directory=args.directory,
        no_combine=args.no_combine,
        dry_run=args.dry_run,
        model=args.model,
    )


if __name__ == "__main__":
    main()
