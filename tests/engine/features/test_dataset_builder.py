from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.contracts.results.features import ArtifactPointer
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine.features.dataset_builder import FeatureProviders, build_features_dataset


class _FakeReturnsProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_y_excess_lead_long(
        self,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
    ) -> pd.DataFrame:
        return self._frame.copy()


class _FakeMacroProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_macro_features(self, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        return self._frame.copy()


class _FakeEtfProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_etf_features_long(
        self,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
    ) -> pd.DataFrame:
        return self._frame.copy()


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.saved_frame: pd.DataFrame | None = None
        self.saved_artifact_name: str | None = None
        self.saved_metadata: dict[str, object] | None = None

    def save_parquet(
        self,
        frame: pd.DataFrame,
        artifact_name: str,
        metadata: dict[str, object],
    ) -> ArtifactPointer:
        self.saved_frame = frame.copy()
        self.saved_artifact_name = artifact_name
        self.saved_metadata = metadata
        return ArtifactPointer(
            uri=f"memory://{artifact_name}",
            fingerprint="fake-sha256",
            row_count=int(frame.shape[0]),
            column_count=int(frame.shape[1]),
        )


@pytest.fixture
def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = ["2020-01-31", "2020-02-29", "2020-03-31"]
    assets = ["AAA", "BBB"]

    returns_rows: list[dict[str, object]] = []
    etf_rows: list[dict[str, object]] = []
    for asset in assets:
        for index, dt in enumerate(dates):
            returns_rows.append(
                {
                    "date": dt,
                    "asset_id": asset,
                    "y_excess_lead": 0.01 * (index + 1) if asset == "AAA" else 0.02 * (index + 1),
                }
            )
            etf_rows.append(
                {
                    "date": dt,
                    "asset_id": asset,
                    "mom12m": float(10 * (index + 1) + (0 if asset == "AAA" else 1)),
                    "ill": [1e-9, 2e-9, 1e-2][index],
                    "dolvol": [1e-7, 2e-7, 1e-1][index],
                }
            )

    macro_frame = pd.DataFrame(
        {
            "date": dates,
            "macro_signal": [100.0, 200.0, 300.0],
        }
    )

    return pd.DataFrame(returns_rows), macro_frame, pd.DataFrame(etf_rows)


def _make_command(**overrides: object) -> BuildFeaturesDatasetCommand:
    command_kwargs: dict[str, object] = {
        "tickers": ["AAA", "BBB"],
        "drop_assets": [],
        "lookback_date": date(2019, 12, 1),
        "start_date": date(2020, 1, 31),
        "end_date": date(2020, 3, 31),
        "interval": Interval.DAILY,
        "horizon": Horizon.MONTHLY,
        "macro_cols": ["macro_signal"],
        "etf_cols": ["mom12m", "ill_log", "dolvol_log", "cs_mom_rank"],
        "drop_macro_cols": [],
        "drop_etf_cols": [],
        "clip_quantile": 0.5,
        "seed": 7,
        "artifact_name": "unit_test_features",
    }
    command_kwargs.update(overrides)
    return BuildFeaturesDatasetCommand(**command_kwargs)


def test_build_features_dataset_happy_path(sample_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    returns_frame, macro_frame, etf_frame = sample_frames
    providers = FeatureProviders(
        returns_provider=_FakeReturnsProvider(returns_frame),
        macro_provider=_FakeMacroProvider(macro_frame),
        etf_features_provider=_FakeEtfProvider(etf_frame),
    )
    artifact_store = _FakeArtifactStore()

    result = build_features_dataset(
        command=_make_command(),
        providers=providers,
        artifact_store=artifact_store,
    )

    assert result.return_unit == "decimal"
    assert result.artifact.uri == "memory://unit_test_features"
    assert result.artifact.row_count == 6
    assert result.index_info.start_date == date(2020, 1, 31)
    assert result.index_info.end_date == date(2020, 3, 31)
    assert result.market_structure is not None
    assert result.market_structure.asset_count == 2
    assert result.market_structure.target_summary.count == 6
    assert artifact_store.saved_frame is not None
    assert artifact_store.saved_metadata is not None
    assert "market_structure" in artifact_store.saved_metadata
    market_structure_metadata = artifact_store.saved_metadata["market_structure"]
    assert isinstance(market_structure_metadata, dict)
    assert market_structure_metadata["asset_count"] == 2

    saved = artifact_store.saved_frame
    assert saved.columns[0] == "t_index"
    assert "lag_y_excess_lead" in saved.columns
    assert "lag2_y_excess_lead" in saved.columns
    assert saved["t_index"].tolist() == [0, 0, 1, 1, 2, 2]
    assert any("Look-ahead policy applied" in message for message in result.diagnostics)


def test_build_features_dataset_applies_t_minus_1_alignment(
    sample_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    returns_frame, macro_frame, etf_frame = sample_frames
    providers = FeatureProviders(
        returns_provider=_FakeReturnsProvider(returns_frame),
        macro_provider=_FakeMacroProvider(macro_frame),
        etf_features_provider=_FakeEtfProvider(etf_frame),
    )
    artifact_store = _FakeArtifactStore()

    build_features_dataset(
        command=_make_command(etf_cols=["mom12m"], macro_cols=["macro_signal"]),
        providers=providers,
        artifact_store=artifact_store,
    )

    assert artifact_store.saved_frame is not None
    saved = artifact_store.saved_frame
    saved["date"] = pd.to_datetime(saved["date"])

    row_aaa_first = saved[(saved["asset_id"] == "AAA") & (saved["date"] == pd.Timestamp("2020-01-31"))].iloc[0]
    row_aaa_second = saved[(saved["asset_id"] == "AAA") & (saved["date"] == pd.Timestamp("2020-02-29"))].iloc[0]

    assert pd.isna(row_aaa_first["mom12m"])
    assert pd.isna(row_aaa_first["macro_signal"])
    assert row_aaa_second["mom12m"] == pytest.approx(10.0)
    assert row_aaa_second["macro_signal"] == pytest.approx(100.0)


def test_build_features_dataset_invalid_date_bounds_raises(
    sample_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    returns_frame, macro_frame, etf_frame = sample_frames
    providers = FeatureProviders(
        returns_provider=_FakeReturnsProvider(returns_frame),
        macro_provider=_FakeMacroProvider(macro_frame),
        etf_features_provider=_FakeEtfProvider(etf_frame),
    )

    with pytest.raises(ValueError, match="start_date must be on or before end_date"):
        build_features_dataset(
            command=_make_command(start_date=date(2020, 4, 1), end_date=date(2020, 3, 31)),
            providers=providers,
            artifact_store=_FakeArtifactStore(),
        )


def test_build_features_dataset_missing_returns_column_raises(
    sample_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    returns_frame, macro_frame, etf_frame = sample_frames
    bad_returns = returns_frame.drop(columns=["y_excess_lead"])
    providers = FeatureProviders(
        returns_provider=_FakeReturnsProvider(bad_returns),
        macro_provider=_FakeMacroProvider(macro_frame),
        etf_features_provider=_FakeEtfProvider(etf_frame),
    )

    with pytest.raises(ValueError, match="Returns provider missing columns"):
        build_features_dataset(
            command=_make_command(),
            providers=providers,
            artifact_store=_FakeArtifactStore(),
        )


def test_build_features_dataset_unknown_selected_columns_raises(
    sample_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    returns_frame, macro_frame, etf_frame = sample_frames
    providers = FeatureProviders(
        returns_provider=_FakeReturnsProvider(returns_frame),
        macro_provider=_FakeMacroProvider(macro_frame),
        etf_features_provider=_FakeEtfProvider(etf_frame),
    )

    with pytest.raises(ValueError, match="Requested etf columns are unavailable"):
        build_features_dataset(
            command=_make_command(etf_cols=["nonexistent_col"]),
            providers=providers,
            artifact_store=_FakeArtifactStore(),
        )
