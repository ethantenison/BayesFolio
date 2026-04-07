from __future__ import annotations

from datetime import date

import pandas as pd

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine.features import build_features_dataset, make_default_feature_providers
from bayesfolio.io import ParquetArtifactStore

ETF_TICKERS = [
    "SPY",  # total US market big cap
    "MGK",  # US growth
    "VTV",  # US value
    "IJR",  # US small cap S&P index 600, more stable
    "IWM",  # US small cap Russel index, more volile than IJR
    "VNQ",  # REIT ETF US centric
    "VNQI",  # international REIT ETF
    "VEA",  # developed international equity
    "VWO",  # AVEM actually is better than VWO but not enough history
    "VSS",  # forein small/mid cap
    "BND",  # total bond market ETF US centric
    "IEF",  # 7-10 year treasury bond ETF US centric
    "BNDX",  # total international bond market ETF, USD hedged, but actually developed markets only
    "LQD",  # investment grade bond ETF US centric
    "HYG",  # High yield bond ETF US centric
    "EWX",  # emerging market small cap ETF
    "VWOB",  # Emerging Market Goverment bond
    "HYEM",  # emerging market high yield corporate bond ETF USD hedged
]

DROP_ASSETS: list[str] = []

LOOKBACK_DATE = date(2019, 1, 1)
START_DATE = date(2021, 1, 31)
END_DATE = date(2026, 4, 1)

SELECTED_ETF_COLS = [
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

SELECTED_MACRO_COLS = [
    "hy_spread",
    "hy_spread_chg_1m",
    "hy_spread_z_12m",
    "vix_slope",
    "vix_ts_z_12m",
    "vix",
    "spy_flow_z_12m",
    "spy_ret",
    "erp",
    "cpi_yoy",
    "cpi_mom",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
]


command = BuildFeaturesDatasetCommand.model_validate(
    {
        "schema": "bayesfolio.features_dataset.command",
        "tickers": ETF_TICKERS,
        "drop_assets": DROP_ASSETS,
        "lookback_date": LOOKBACK_DATE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "interval": Interval.DAILY,
        "horizon": Horizon.MONTHLY,
        "etf_cols": SELECTED_ETF_COLS,
        "macro_cols": SELECTED_MACRO_COLS,
        "drop_etf_cols": [],
        "drop_macro_cols": [],
        "clip_quantile": 0.99,
        "seed": 27,
        "artifact_name": "etf_macro_features_2026_04.parquet",
        "include_unlabeled_tail": True,
    }
)

providers = make_default_feature_providers(
    cache_root="artifacts/cache",
)

artifact_store = ParquetArtifactStore(
    base_dir="artifacts/features",
)

result = build_features_dataset(
    command=command,
    providers=providers,
    artifact_store=artifact_store,
)

print("Artifact URI:", result.artifact.uri)
print("Rows:", result.artifact.row_count)
print("Columns:", result.artifact.column_count)
print("Diagnostics:")
for note in result.diagnostics:
    print(" -", note)

features_df = pd.read_parquet(result.artifact.uri)
print(features_df.shape)
print(features_df.columns.tolist())
print(features_df.head())


features_df = pd.read_parquet(result.artifact.uri)

KEEP_COLS = [
    "t_index",
    "date",
    "asset_id",
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
    "hy_spread",
    "hy_spread_chg_1m",
    "hy_spread_z_12m",
    "vix_slope",
    "vix_ts_z_12m",
    "vix",
    "spy_flow_z_12m",
    "spy_ret",
    "erp",
    "cpi_yoy",
    "cpi_mom",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
    "y_excess_lead",
]

features_df = features_df.loc[:, KEEP_COLS].copy()

# Verify that 2026-03-31 is included in the final dataframe
features_df["date"] = pd.to_datetime(features_df["date"])
max_date = features_df["date"].max()
target_date = pd.Timestamp("2026-03-31")

print(f"\nDataset date range: {features_df['date'].min()} to {max_date}")
print(f"Target date (2026-03-31) present: {target_date in features_df['date'].values}")
print(f"Rows for 2026-03-31: {len(features_df[features_df['date'] == target_date])}")

# Show last few rows to inspect the unlabeled tail
print("\nLast 5 rows (should include unlabeled tail with NaN y_excess_lead):")
print(features_df[["date", "asset_id", "y_excess_lead"]].tail(20))
