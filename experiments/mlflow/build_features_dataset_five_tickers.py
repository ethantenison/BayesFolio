from __future__ import annotations

from datetime import date

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine.features.asset_prices import build_long_panel, fetch_etf_features
from bayesfolio.engine.features.dataset_builder import FeatureProviders, build_features_dataset
from bayesfolio.engine.features.market_fundamentals import fetch_macro_features
from bayesfolio.io.artifact_store import ParquetArtifactStore
from bayesfolio.io.providers.etf_features_provider import EtfFeaturesProvider
from bayesfolio.io.providers.macro_provider import MacroProvider
from bayesfolio.io.providers.returns_provider import ReturnsProvider


def main() -> None:
    """Build and persist the long-format features dataset for five ETFs.

    This script wires the feature dataset pipeline with concrete provider
    fetchers and stores the result as parquet under `artifacts/features/`.
    All return columns are in decimal units (for example, `0.02` = `2%`).
    """

    tickers = [
        "SPY",
        "MGK",
        "VTV",
        "IJR",
        "IWM",
    ]

    command = BuildFeaturesDatasetCommand(
        tickers=tickers,
        drop_assets=[],
        lookback_date=date(2022, 7, 1),
        start_date=date(2024, 11, 29),
        end_date=date(2026, 2, 28),
        interval=Interval.DAILY,
        horizon=Horizon.MONTHLY,
        clip_quantile=0.99,
        artifact_name="features_dataset_five_tickers_20241129_20260228",
    )

    providers = FeatureProviders(
        returns_provider=ReturnsProvider(fetcher=build_long_panel),
        macro_provider=MacroProvider(fetcher=fetch_macro_features),
        etf_features_provider=EtfFeaturesProvider(fetcher=fetch_etf_features),
    )
    artifact_store = ParquetArtifactStore(base_dir="artifacts/features")

    result = build_features_dataset(
        command=command,
        providers=providers,
        artifact_store=artifact_store,
    )

    print(f"Saved dataset: {result.artifact.uri}")
    print(f"Rows: {result.artifact.row_count}, Columns: {result.artifact.column_count}")


if __name__ == "__main__":
    main()