"""Build the cached U.S. confirmation artifact without external data access.

The verified July U.S. candidate artifact is the authoritative source for the
target and candidate features. Missing all-input comparator columns are joined
from two existing local artifacts. The output remains under this experiment so
none of the source artifacts is overwritten.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/build_cached_us_confirmation_artifact.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"

VERIFIED_BASE = (
    REPO_ROOT
    / "experiments/2026-07-us-equity-etf-features/artifacts/"
    "us_equity_first_round_feature_candidates_20260705T145015Z.parquet"
)
JULY_FULL_CACHE = (
    REPO_ROOT
    / "experiments/2026-07-us-equity-etf-features/artifacts/"
    "us_equity_first_round_feature_candidates_20260705T144719Z.parquet"
)
PRODUCTION_CACHE = Path("/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_06.parquet")
OUTPUT_PATH = ARTIFACT_DIR / "us_confirmation_cached_inputs_20261012.parquet"
MANIFEST_PATH = ARTIFACT_DIR / "us_confirmation_cached_inputs_20261012.manifest.json"

TICKERS = ["SPY", "MGK", "VTV", "IWM"]
ALL_INPUT_ETF_COLUMNS = [
    "lag_y_excess_lead",
    "baspread",
    "ret_kurt",
    "chmom",
    "mom12m",
    "mom36m",
    "cs_mom_rank",
    "max_dd_6m",
    "ma_signal",
    "ret_autocorr",
    "vol_z",
]
ALL_INPUT_MACRO_COLUMNS = [
    "hy_spread",
    "hy_spread_chg_1p",
    "hy_spread_z_12p",
    "vix_slope",
    "vix_ts_z_12p",
    "vix",
    "spy_flow_z_12p",
    "spy_ret",
    "erp",
    "cpi_chg_12p",
    "cpi_chg_1p",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
]
REQUIRED_COLUMNS = ["t_index", "y_excess_lead", *ALL_INPUT_ETF_COLUMNS, *ALL_INPUT_MACRO_COLUMNS]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Join cached inputs, validate the four-task panel, and write a manifest."""
    source_paths = [VERIFIED_BASE, JULY_FULL_CACHE, PRODUCTION_CACHE]
    missing_paths = [str(path) for path in source_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Required cached artifacts are missing: {missing_paths}")

    base = pd.read_parquet(VERIFIED_BASE)
    july_full = pd.read_parquet(JULY_FULL_CACHE)
    production = pd.read_parquet(PRODUCTION_CACHE)
    keys = ["date", "asset_id"]

    if sorted(base["asset_id"].dropna().astype(str).unique().tolist()) != sorted(TICKERS):
        raise ValueError("Verified base artifact does not contain exactly the approved four U.S. ETF tasks.")

    missing_from_base = [column for column in REQUIRED_COLUMNS if column not in base.columns]
    from_production = [column for column in missing_from_base if column in production.columns]
    from_july_full = [
        column for column in missing_from_base if column not in production.columns and column in july_full.columns
    ]
    unresolved = [
        column for column in missing_from_base if column not in from_production and column not in from_july_full
    ]
    if unresolved:
        raise ValueError(f"Cached sources cannot resolve required columns: {unresolved}")

    production_four = production.loc[production["asset_id"].isin(TICKERS), [*keys, *from_production]]
    july_four = july_full.loc[july_full["asset_id"].isin(TICKERS), [*keys, *from_july_full]]
    output = base.merge(production_four, on=keys, how="left", validate="one_to_one")
    output = output.merge(july_four, on=keys, how="left", validate="one_to_one")
    output = output.sort_values(keys).reset_index(drop=True)

    complete_by_date = output.groupby("date", observed=True)[REQUIRED_COLUMNS].apply(
        lambda frame: bool(frame.notna().all().all())
    )
    complete_dates = pd.DatetimeIndex(complete_by_date[complete_by_date].index)
    if len(complete_dates) < 24:
        raise ValueError(f"Expected at least 24 complete cached dates; found {len(complete_dates)}")
    approved_dates = complete_dates[-24:]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    manifest = {
        "schema": "bayesfolio.etf_family_cached_confirmation_artifact.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_path": str(OUTPUT_PATH),
        "artifact_sha256": sha256_file(OUTPUT_PATH),
        "source_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_paths
        ],
        "authoritative_base": str(VERIFIED_BASE),
        "joined_columns": {
            "production_cache": from_production,
            "july_full_cache": from_july_full,
        },
        "tickers": TICKERS,
        "ijr_excluded": True,
        "time_index_included": True,
        "row_count": int(len(output)),
        "column_count": int(len(output.columns)),
        "required_columns": REQUIRED_COLUMNS,
        "complete_date_count": int(len(complete_dates)),
        "complete_date_min": complete_dates.min().date().isoformat(),
        "complete_date_max": complete_dates.max().date().isoformat(),
        "approved_scored_date_min": approved_dates.min().date().isoformat(),
        "approved_scored_date_max": approved_dates.max().date().isoformat(),
        "approved_scored_dates": [date.date().isoformat() for date in approved_dates],
        "cached_data_only": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
