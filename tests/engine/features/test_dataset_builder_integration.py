from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine.features.dataset_builder import FeatureProviders, build_features_dataset
from bayesfolio.io.artifact_store import ParquetArtifactStore
from bayesfolio.io.providers.etf_features_provider import EtfFeaturesProvider
from bayesfolio.io.providers.macro_provider import MacroProvider
from bayesfolio.io.providers.returns_provider import ReturnsProvider


def _returns_fetcher(
    tickers: list[str],
    start: str,
    end: str,
    horizon: Horizon,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        rows.extend(
            [
                {"date": "2020-01-31", "asset_id": ticker, "y_excess_lead": 0.01},
                {"date": "2020-02-29", "asset_id": ticker, "y_excess_lead": 0.02},
                {"date": "2020-03-31", "asset_id": ticker, "y_excess_lead": 0.03},
            ]
        )
    return pd.DataFrame(rows)


def _macro_fetcher(start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "hy_spread": [3.0, 3.2, 3.1],
            "vix": [15.0, 20.0, 18.0],
        }
    )


def _etf_fetcher(
    tickers: list[str],
    start: str,
    end: str,
    horizon: Horizon,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        rows.extend(
            [
                {
                    "date": "2020-01-31",
                    "asset_id": ticker,
                    "mom12m": 1.0,
                    "chmom": 0.1,
                    "vol_z": -0.2,
                    "ill": 1.0e-9,
                    "dolvol": 1.0e-7,
                },
                {
                    "date": "2020-02-29",
                    "asset_id": ticker,
                    "mom12m": 2.0,
                    "chmom": 0.2,
                    "vol_z": 0.0,
                    "ill": 2.0e-9,
                    "dolvol": 2.0e-7,
                },
                {
                    "date": "2020-03-31",
                    "asset_id": ticker,
                    "mom12m": 3.0,
                    "chmom": 0.3,
                    "vol_z": 0.2,
                    "ill": 1.0e-2,
                    "dolvol": 1.0e-1,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_feature_preparation_workflow_via_new_pipeline(tmp_path: Path) -> None:
    """Recreate legacy feature-preparation behavior using the new workflow.

    This test wires the real pipeline components:
    - IO providers (with injected transitional fetchers)
    - Engine dataset builder
    - Parquet artifact store

    It validates core outputs expected from the legacy "Feature Preparation"
    section: merged long panel with lag features, liquidity transforms,
    cross-sectional momentum rank, and `t_index`.
    """

    providers = FeatureProviders(
        returns_provider=ReturnsProvider(fetcher=_returns_fetcher),
        macro_provider=MacroProvider(fetcher=_macro_fetcher),
        etf_features_provider=EtfFeaturesProvider(fetcher=_etf_fetcher),
    )
    artifact_store = ParquetArtifactStore(base_dir=tmp_path)

    command = BuildFeaturesDatasetCommand(
        tickers=["SPY", "VTV"],
        drop_assets=[],
        lookback_date=date(2019, 12, 1),
        start_date=date(2020, 1, 31),
        end_date=date(2020, 3, 31),
        interval=Interval.DAILY,
        horizon=Horizon.MONTHLY,
        macro_cols=["hy_spread", "vix"],
        etf_cols=["mom12m", "chmom", "vol_z", "ill_log", "dolvol_log", "cs_mom_rank"],
        clip_quantile=0.99,
        artifact_name="features_workflow_test",
    )

    result = build_features_dataset(command=command, providers=providers, artifact_store=artifact_store)

    assert result.return_unit == "decimal"
    assert result.artifact.format == "parquet"
    assert result.artifact.row_count == 6
    assert result.artifact.column_count > 0

    saved_path = tmp_path / "features_workflow_test.parquet"
    assert saved_path.exists()

    frame = pd.read_parquet(saved_path)
    assert "t_index" in frame.columns
    assert "y_excess_lead" in frame.columns
    assert "lag_y_excess_lead" in frame.columns
    assert "lag2_y_excess_lead" in frame.columns
    assert "ill_log" in frame.columns
    assert "dolvol_log" in frame.columns
    assert "cs_mom_rank" in frame.columns

    assert frame["t_index"].tolist() == [0, 0, 1, 1, 2, 2]
    assert any("Look-ahead policy applied" in message for message in result.diagnostics)
