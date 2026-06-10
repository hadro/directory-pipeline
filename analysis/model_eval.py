#!/usr/bin/env python3
"""Compare NER extraction quality across models and modes.

Discovery: finds entries_*.csv files in the given volume directories.
  - entries_*_text_only.csv  → mode = text-only (baseline, backed-up before new runs)
  - entries_*.csv            → mode = multimodal (new run)
  - multimodal/<vol>/entries_*.csv  → mode = multimodal (existing partial subdir runs)

Metrics computed per (volume, model, mode):
  fill_rate, state_consistency, drift_window_rate, drift_geonames_rate,
  entries_per_page, cost_per_entry, cost_per_clean_entry

Output: analysis/model_eval_report.md

Usage:
    python analysis/model_eval.py
    python analysis/model_eval.py --vol-dirs path/to/vol1 path/to/vol2
    python analysis/model_eval.py --no-geonames   # skip GeoNames download
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Reuse helpers from detect_geographic_drift
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from detect_geographic_drift import (  # noqa: E402
    apply_geonames,
    apply_window,
    build_geonames_lookup,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
GB_BASE = REPO_ROOT / "output/green_books_and_related/the_green_book_9ea5d5b0"

DEFAULT_VOL_IDS = [
    "4bea2040-1118-0132-673e-58d385a7b928",   # 1947 Layout A
    "6c5ccbe0-1a73-0132-b81d-58d385a7bbd0",   # 1951 Layout A
    "1f0653d0-1a74-0132-69ff-58d385a7bbd0",   # 1960 Layout A→B
    "fdb5a8d0-1a75-0132-d947-58d385a7bbd0",   # 1966 Layout B
]

VOLUME_LABELS = {
    "4bea2040": "1947 (Layout A)",
    "6c5ccbe0": "1951 Railroad (Layout A)",
    "1f0653d0": "1960 (A→B)",
    "fdb5a8d0": "1966-67 (Layout B)",
}

# Cost per page (multimodal + --flex); text-only is slightly less but we use
# the same rate as a conservative estimate — the image adds <$0.0001/page.
COST_PER_PAGE: dict[str, float] = {
    "gemini-3.1-flash-lite": 0.0028,
    "gemini-3.1-flash-lite-preview": 0.0028,  # deprecated name; kept for historical CSV lookups
    "gemini-3-flash-preview": 0.0056,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_col(df: pd.DataFrame) -> str | None:
    """Return the state column name, handling schema variations across volume years."""
    for col in (
        "state", "location_primary", "location_state", "location_state_country",
        "state_country", "state_region", "state_or_region", "city_state",
    ):
        if col in df.columns:
            return col
    return None


def _name_col(df: pd.DataFrame) -> str | None:
    for col in ("name", "business_name", "establishment_name"):
        if col in df.columns:
            return col
    return None


def _city_col_safe(df: pd.DataFrame) -> str | None:
    for col in ("city", "location_city", "town", "city_state"):
        if col in df.columns:
            return col
    return None


def _gemini_txt_count(vol_dir: Path) -> int:
    return len(list(vol_dir.glob("*_gemini*.txt")))


def _model_from_stem(stem: str) -> tuple[str, str]:
    """Parse (model_name, mode) from a CSV stem like 'entries_gemini-3-flash-preview'.

    Returns (model, mode) where mode is 'multimodal' or 'text-only'.
    """
    stem = stem.removeprefix("entries_")
    if stem.endswith("_text_only"):
        return stem.removesuffix("_text_only"), "text-only"
    # Strip backup suffixes like _backup_20260424 or _fixed
    stem = re.sub(r"_(backup_\d+|fixed|fixed_backup_\d+)$", "", stem)
    return stem, "multimodal"


def _discover_csvs(vol_dir: Path) -> list[tuple[Path, str, str]]:
    """Return list of (csv_path, model_name, mode) for a volume directory."""
    found = []
    seen: set[tuple[str, str]] = set()

    for p in sorted(vol_dir.glob("entries_gemini-*.csv")):
        stem = p.stem
        # Skip intermediate files (fixed, backup) except _text_only
        if re.search(r"_(backup_\d+|fixed|fixed_backup_\d+)\.csv$", p.name):
            continue
        model, mode = _model_from_stem(stem)
        key = (model, mode)
        if key not in seen:
            seen.add(key)
            found.append((p, model, mode))

    return found


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    csv_path: Path,
    vol_dir: Path,
    model: str,
    mode: str,
    geonames_lookup: dict | None,
) -> dict:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    n_rows = len(df)
    if n_rows == 0:
        return {}

    pages = _gemini_txt_count(vol_dir)
    cost_rate = COST_PER_PAGE.get(model, 0.0)

    # Add synthetic volume_id for apply_window (requires the column)
    df["volume_id"] = vol_dir.name

    # --- fill_rate ---
    name_col = _name_col(df)
    city_col = _city_col_safe(df)
    state_col = _state_col(df)

    name_filled = df[name_col].str.strip().ne("") if name_col else pd.Series(False, index=df.index)
    state_filled = df[state_col].str.strip().ne("") if state_col else pd.Series(False, index=df.index)
    # city_col may resolve to the same column as state_col (e.g. city_state) or be absent
    if city_col and city_col != state_col:
        city_filled = df[city_col].str.strip().ne("")
        fill_rate = (name_filled & city_filled & state_filled).mean() * 100
    else:
        # No separate city column: fill_rate = name + state/location only
        fill_rate = (name_filled & state_filled).mean() * 100

    # --- state_consistency: % of pages where all entries share one state ---
    state_consistency = float("nan")
    if state_col and "canvas_fragment" in df.columns:
        # Extract image ID as a page proxy
        page_id = df["canvas_fragment"].str.extract(r"/iiif/3/(\d+)/")[0].fillna("unknown")
        groups = df.groupby(page_id)[state_col].apply(
            lambda s: s.str.strip().str.upper().nunique()
        )
        state_consistency = (groups == 1).mean() * 100

    # --- drift_window_rate ---
    drift_window_rate = float("nan")
    window_flags = pd.Series(False, index=df.index)
    if state_col and "canvas_fragment" in df.columns:
        working = df.copy()
        if state_col != "state":
            working = working.rename(columns={state_col: "state"})
        try:
            raw_flags = apply_window(working)
            window_flags = raw_flags == "suspicious"
            drift_window_rate = window_flags.mean() * 100
        except Exception:
            pass

    # --- drift_geonames_rate ---
    drift_geonames_rate = float("nan")
    if geonames_lookup is not None and state_col and "canvas_fragment" in df.columns:
        working = df.copy()
        if state_col != "state":
            working = working.rename(columns={state_col: "state"})
        city_col_safe = _city_col_safe(working)
        if city_col_safe and city_col_safe != "city":
            working = working.rename(columns={city_col_safe: "city"})
        try:
            geo_flags = apply_geonames(working, geonames_lookup)
            drift_geonames_rate = (geo_flags == "impossible").mean() * 100
        except Exception:
            pass

    # --- cost estimates ---
    total_cost = pages * cost_rate if pages and cost_rate else float("nan")
    n_clean = int((~window_flags).sum())
    cost_per_entry = total_cost / n_rows if n_rows and not pd.isna(total_cost) else float("nan")
    cost_per_clean = total_cost / n_clean if n_clean and not pd.isna(total_cost) else float("nan")

    return {
        "pages": pages,
        "entries": n_rows,
        "entries_per_page": round(n_rows / pages, 1) if pages else float("nan"),
        "fill_rate": round(fill_rate, 1),
        "state_consistency": round(state_consistency, 1) if not pd.isna(state_consistency) else "n/a",
        "drift_window_rate": round(drift_window_rate, 1) if not pd.isna(drift_window_rate) else "n/a",
        "drift_geonames_rate": round(drift_geonames_rate, 1) if not pd.isna(drift_geonames_rate) else "n/a",
        "total_cost_usd": round(total_cost, 2) if not pd.isna(total_cost) else "n/a",
        "cost_per_entry_cents": round(cost_per_entry * 100, 3) if not pd.isna(cost_per_entry) else "n/a",
        "cost_per_clean_entry_cents": round(cost_per_clean * 100, 3) if not pd.isna(cost_per_clean) else "n/a",
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, float) and pd.isna(v):
        return "—"
    return str(v)


def build_report(rows: list[dict]) -> str:
    lines = ["# Model Evaluation Report\n"]
    lines.append(
        "Metrics per (volume, model, mode). All API runs used `--flex`.\n"
        "Cost estimates assume multimodal rates (≈$0.0028/page for flash-lite, ≈$0.0056/page for flash-preview).\n"
    )

    headers = [
        "Volume", "Model", "Mode", "Pages", "Entries", "Entries/pg",
        "Fill%", "State cons%", "Drift win%", "Drift geo%",
        "Cost $", "¢/entry", "¢/clean",
    ]
    col_keys = [
        "volume_label", "model", "mode", "pages", "entries", "entries_per_page",
        "fill_rate", "state_consistency", "drift_window_rate", "drift_geonames_rate",
        "total_cost_usd", "cost_per_entry_cents", "cost_per_clean_entry_cents",
    ]

    sep = "| " + " | ".join("-" * max(len(h), 4) for h in headers) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    lines.append(header_row)
    lines.append(sep)

    for row in rows:
        cells = [_fmt(row.get(k, "—")) for k in col_keys]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Key: Primary metric is `¢/clean` (cost per drift-clean entry).")
    lines.append(
        "Decision threshold: if flash-lite multimodal `drift_window_rate` ≤ flash-preview + 5pp, "
        "flash-lite wins."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--vol-dirs", nargs="+",
        help="Volume directories to evaluate (default: 4 test volumes)",
    )
    p.add_argument(
        "--no-geonames", action="store_true",
        help="Skip GeoNames download/lookup (drift_geonames_rate will be n/a)",
    )
    p.add_argument(
        "--output", default="analysis/model_eval_report.md",
        help="Output report path (default: analysis/model_eval_report.md)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.vol_dirs:
        vol_dirs = [Path(d) for d in args.vol_dirs]
    else:
        vol_dirs = [GB_BASE / vid for vid in DEFAULT_VOL_IDS]

    missing = [d for d in vol_dirs if not d.exists()]
    if missing:
        sys.exit("Volume dirs not found:\n" + "\n".join(str(d) for d in missing))

    geonames_lookup = None
    if not args.no_geonames:
        print("Loading GeoNames lookup ...")
        geonames_lookup = build_geonames_lookup()
        print(f"  {len(geonames_lookup):,} city entries")

    report_rows: list[dict] = []

    for vol_dir in vol_dirs:
        short = vol_dir.name[:8]
        label = VOLUME_LABELS.get(short, short)
        csvs = _discover_csvs(vol_dir)

        if not csvs:
            print(f"\n[{label}] No entries CSVs found — skipping")
            continue

        for csv_path, model, mode in csvs:
            print(f"\n[{label}] {model} / {mode} → {csv_path.name}")
            metrics = compute_metrics(csv_path, vol_dir, model, mode, geonames_lookup)
            if not metrics:
                print("  (empty CSV — skipped)")
                continue
            print(
                f"  entries={metrics['entries']}  fill={metrics['fill_rate']}%"
                f"  drift_win={metrics['drift_window_rate']}%"
                f"  ¢/clean={metrics['cost_per_clean_entry_cents']}"
            )
            report_rows.append({"volume_label": label, "model": model, "mode": mode, **metrics})

    if not report_rows:
        print("No data collected — nothing to report.")
        return

    # Sort: volume → model → mode
    report_rows.sort(key=lambda r: (r["volume_label"], r["model"], r["mode"]))

    report = build_report(report_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")

    # Print summary table to stdout
    print("\n" + report)


if __name__ == "__main__":
    main()
