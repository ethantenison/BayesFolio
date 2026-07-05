"""Tests for IO returns provider cache and partial-miss behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bayesfolio.core.settings import Horizon
from bayesfolio.io.providers.returns_provider import ReturnsProvider


def _returns_frame(asset: str, dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "asset_id": [asset] * len(dates),
            "y_excess_lead": values,
        }
    )


def test_returns_provider_writes_cache_on_miss(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fetcher(*, tickers: list[str], start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        calls.append(tickers)
        return _returns_frame("SPY", ["2024-01-31", "2024-02-29"], [0.01, 0.02])

    provider = ReturnsProvider(fetcher=fetcher, cache_dir=tmp_path)

    first = provider.get_y_excess_lead_long(
        tickers=["SPY"],
        start="2024-01-01",
        end="2024-02-29",
        horizon=Horizon.MONTHLY,
    )
    second = provider.get_y_excess_lead_long(
        tickers=["SPY"],
        start="2024-01-01",
        end="2024-02-29",
        horizon=Horizon.MONTHLY,
    )

    assert len(calls) == 1
    assert len(first) == 2
    assert len(second) == 2


def test_returns_provider_fetches_only_missing_tickers(tmp_path: Path) -> None:
    cached = _returns_frame("SPY", ["2024-01-31", "2024-02-29"], [0.01, 0.02])
    cached.to_parquet(tmp_path / "returns_bme.parquet", index=False)

    calls: list[list[str]] = []

    def fetcher(*, tickers: list[str], start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        calls.append(tickers)
        return _returns_frame("QQQ", ["2024-01-31", "2024-02-29"], [0.03, 0.04])

    provider = ReturnsProvider(fetcher=fetcher, cache_dir=tmp_path)

    frame = provider.get_y_excess_lead_long(
        tickers=["SPY", "QQQ"],
        start="2024-01-01",
        end="2024-02-29",
        horizon=Horizon.MONTHLY,
    )

    assert calls == [["QQQ"]]
    assert sorted(frame["asset_id"].unique().tolist()) == ["QQQ", "SPY"]
    assert len(frame) == 4


def test_returns_provider_filters_cached_interleaved_three_week_anchors(tmp_path: Path) -> None:
    cached = _returns_frame(
        "SPY",
        ["2026-01-02", "2026-01-16", "2026-01-23", "2026-02-06", "2026-02-13"],
        [0.01, 0.99, 0.02, 0.98, 0.03],
    )
    cached.to_parquet(tmp_path / "returns_3w_fri.parquet", index=False)

    def fetcher(*, tickers: list[str], start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        raise AssertionError("cache should satisfy the requested anchor grid")

    provider = ReturnsProvider(fetcher=fetcher, cache_dir=tmp_path)

    frame = provider.get_y_excess_lead_long(
        tickers=["SPY"],
        start="2026-01-01",
        end="2026-02-13",
        horizon=Horizon.THREE_WEEK,
    )

    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-23", "2026-02-13"]
    assert frame["y_excess_lead"].tolist() == [0.01, 0.02, 0.03]
