"""Build a cached rate-aware U.S. fixed-income experiment panel.

The long-history fixed-income artifact supplies ETF-local predictors and the
target. A separately cached July international-equity artifact supplies
date-level Treasury, real-rate, breakeven, and rate-volatility predictors.
No external data is fetched or refreshed.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/build_cached_us_fixed_income_rate_artifact.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
FIXED_INCOME_SOURCE = ARTIFACT_DIR / "us_fixed_income_cached_inputs_20261016.parquet"
RATE_SOURCE = (
    EXPERIMENT_DIR.parents[0]
    / "2026-07-us-equity-etf-features/artifacts/"
    "international_equity_custom_feature_candidates_20260707T182133Z.parquet"
)
OUTPUT_PATH = ARTIFACT_DIR / "us_fixed_income_rate_cached_inputs_20261016.parquet"
MANIFEST_PATH = ARTIFACT_DIR / "us_fixed_income_rate_cached_inputs_20261016.manifest.json"
DATES_PATH = ARTIFACT_DIR / "us_fixed_income_rate_scored_dates_24.txt"
TICKERS = ["BND", "IEF", "LQD", "HYG"]
RATE_COLUMNS = [
    "tnote10y",
    "tbill3m",
    "term_spread",
    "credit_spread",
    "credit_spread_chg_1p",
    "move_proxy",
    "y10_nominal",
    "schp_ret",
    "y10_real_proxy",
    "breakeven_proxy",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Merge cached date-level rate state into the U.S. fixed-income panel."""
    missing = [path for path in (FIXED_INCOME_SOURCE, RATE_SOURCE) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required cached artifacts are missing: {missing}")

    fixed_income = pd.read_parquet(FIXED_INCOME_SOURCE)
    rate_source = pd.read_parquet(RATE_SOURCE, columns=["date", *RATE_COLUMNS])
    rate_nunique = rate_source.groupby("date", observed=True)[RATE_COLUMNS].nunique(dropna=False)
    if bool((rate_nunique > 1).any().any()):
        raise ValueError("Rate-source predictors are not uniquely date-level.")
    rates = rate_source.groupby("date", observed=True, as_index=False)[RATE_COLUMNS].first()

    output = fixed_income.merge(rates, on="date", how="left", validate="many_to_one")
    value_columns = [column for column in output.columns if column not in {"date", "asset_id"}]
    complete_by_date = output.groupby("date", observed=True)[value_columns].apply(
        lambda frame: len(frame) == len(TICKERS) and bool(frame.notna().all().all())
    )
    complete_dates = pd.DatetimeIndex(complete_by_date[complete_by_date].index)
    if len(complete_dates) < 24:
        raise ValueError(f"Expected at least 24 complete rate-aware dates; found {len(complete_dates)}.")
    approved_dates = complete_dates[-24:]
    output = output.loc[output["date"].isin(complete_dates)].sort_values(["date", "asset_id"]).reset_index(drop=True)
    if sorted(output["asset_id"].unique().tolist()) != sorted(TICKERS):
        raise ValueError("Rate-aware panel does not contain the exact U.S. fixed-income universe.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    DATES_PATH.write_text(
        "# Frozen rate-aware U.S. fixed-income score panel: 24 business month-end dates.\n"
        + "\n".join(date.date().isoformat() for date in approved_dates)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "bayesfolio.etf_family_cached_confirmation_artifact.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "family": "us_fixed_income",
        "artifact_path": str(OUTPUT_PATH),
        "artifact_sha256": sha256_file(OUTPUT_PATH),
        "source_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in (FIXED_INCOME_SOURCE, RATE_SOURCE)
        ],
        "tickers": TICKERS,
        "time_index_included": True,
        "row_count": int(len(output)),
        "column_count": int(len(output.columns)),
        "required_columns": value_columns,
        "rate_columns": RATE_COLUMNS,
        "complete_date_count": int(len(complete_dates)),
        "complete_date_min": complete_dates.min().date().isoformat(),
        "complete_date_max": complete_dates.max().date().isoformat(),
        "approved_scored_date_min": approved_dates.min().date().isoformat(),
        "approved_scored_date_max": approved_dates.max().date().isoformat(),
        "approved_scored_dates": [date.date().isoformat() for date in approved_dates],
        "scored_dates_path": str(DATES_PATH),
        "cached_data_only": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
