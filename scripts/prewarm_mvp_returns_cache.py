"""Prewarm local monthly market-data caches for MVP chat.

Usage:
    poetry run python scripts/prewarm_mvp_returns_cache.py

This script fetches monthly return labels, ETF features, and macro features
for a fixed ETF universe and stores them in local parquet caches used by the
MVP providers.
"""

from __future__ import annotations

from bayesfolio.core.settings import Horizon
from bayesfolio.engine.features import make_default_feature_providers

ETF_TICKERS: list[str] = [
    "SPY",
    "IWM",
    "VNQ",
    "VNQI",
    "VEA",
    "VWO",
    "VSS",
    "BND",
    "IEF",
    "BNDX",
    "LQD",
    "HYG",
    "EWX",
    "VWOB",
    "HYEM",
]

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
RETURNS_CACHE_DIR = "artifacts/cache/returns"
ETF_FEATURES_CACHE_DIR = "artifacts/cache/etf_features"
MACRO_CACHE_DIR = "artifacts/cache/macro_enhanced"


def main() -> None:
    """Fetch monthly market data and persist it to local caches."""

    providers = make_default_feature_providers()

    returns_frame = providers.returns_provider.get_y_excess_lead_long(
        tickers=ETF_TICKERS,
        start=START_DATE,
        end=END_DATE,
        horizon=Horizon.MONTHLY,
    )

    etf_frame = providers.etf_features_provider.get_etf_features_long(
        tickers=ETF_TICKERS,
        start=START_DATE,
        end=END_DATE,
        horizon=Horizon.MONTHLY,
    )

    macro_frame = providers.macro_provider.get_macro_features(
        start=START_DATE,
        end=END_DATE,
        horizon=Horizon.MONTHLY,
    )

    assets = sorted(returns_frame["asset_id"].astype(str).unique().tolist()) if not returns_frame.empty else []
    date_min = str(returns_frame["date"].min()) if not returns_frame.empty else "N/A"
    date_max = str(returns_frame["date"].max()) if not returns_frame.empty else "N/A"

    print("Prewarm complete.")
    print(f"Returns rows: {len(returns_frame)}")
    print(f"Assets: {len(assets)} -> {assets}")
    print(f"Date range in fetched frame: {date_min} to {date_max}")
    print(f"ETF feature rows: {len(etf_frame)}")
    print(f"Macro rows: {len(macro_frame)}")
    print(f"Returns cache directory: {RETURNS_CACHE_DIR}")
    print(f"ETF features cache directory: {ETF_FEATURES_CACHE_DIR}")
    print(f"Macro cache directory: {MACRO_CACHE_DIR}")


if __name__ == "__main__":
    main()
