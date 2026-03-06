"""Tests for IO macro provider retry, fallback, and cache behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bayesfolio.core.settings import Horizon
from bayesfolio.io.providers.macro_provider import MacroProvider


def _sample_macro_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-12-31", "2026-01-31", "2026-02-28"],
            "hy_spread": [0.03, 0.031, 0.029],
        }
    )


def test_macro_provider_retries_then_succeeds() -> None:
    calls = {"count": 0}

    def flaky_fetcher(*, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        calls["count"] += 1
        if calls["count"] < 3:
            raise TimeoutError("temporary timeout")
        return _sample_macro_frame()

    provider = MacroProvider(fetcher=flaky_fetcher, max_retries=3, retry_backoff_seconds=0.0)

    frame = provider.get_macro_features(start="2025-12-01", end="2026-02-28", horizon=Horizon.MONTHLY)

    assert calls["count"] == 3
    assert not frame.empty
    assert "hy_spread" in frame.columns


def test_macro_provider_uses_fallback_csv_after_failures(tmp_path: Path) -> None:
    fallback = _sample_macro_frame()
    fallback_path = tmp_path / "macro_fallback.csv"
    fallback.to_csv(fallback_path, index=False)

    def failing_fetcher(*, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        raise TimeoutError("always failing")

    provider = MacroProvider(
        fetcher=failing_fetcher,
        max_retries=1,
        retry_backoff_seconds=0.0,
        fallback_csv_path=fallback_path,
    )

    frame = provider.get_macro_features(start="2026-01-01", end="2026-02-28", horizon=Horizon.MONTHLY)

    assert len(frame) == 2
    assert frame["date"].min() >= pd.Timestamp("2026-01-01")
    assert frame["date"].max() <= pd.Timestamp("2026-02-28")


def test_macro_provider_raises_when_no_fallback_available() -> None:
    def failing_fetcher(*, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        raise TimeoutError("still failing")

    provider = MacroProvider(fetcher=failing_fetcher, max_retries=0, retry_backoff_seconds=0.0)

    with pytest.raises(TimeoutError):
        provider.get_macro_features(start="2026-01-01", end="2026-02-28", horizon=Horizon.MONTHLY)


def test_macro_provider_writes_cache_on_miss(tmp_path: Path) -> None:
    calls = {"count": 0}

    def fetcher(*, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-02-29"],
                "hy_spread": [0.02, 0.025],
            }
        )

    provider = MacroProvider(fetcher=fetcher, max_retries=0, retry_backoff_seconds=0.0, cache_dir=tmp_path)

    first = provider.get_macro_features(start="2024-01-01", end="2024-02-29", horizon=Horizon.MONTHLY)
    second = provider.get_macro_features(start="2024-01-01", end="2024-02-29", horizon=Horizon.MONTHLY)

    assert calls["count"] == 1
    assert len(first) == 2
    assert len(second) == 2


def test_macro_provider_reads_cache_for_subset_window(tmp_path: Path) -> None:
    cached = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"]),
            "hy_spread": [0.02, 0.025, 0.03],
        }
    )
    cached.to_parquet(tmp_path / "macro_bme.parquet", index=False)

    calls = {"count": 0}

    def fetcher(*, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        calls["count"] += 1
        return _sample_macro_frame()

    provider = MacroProvider(fetcher=fetcher, max_retries=0, retry_backoff_seconds=0.0, cache_dir=tmp_path)

    frame = provider.get_macro_features(start="2024-01-01", end="2024-02-29", horizon=Horizon.MONTHLY)

    assert calls["count"] == 0
    assert len(frame) == 2
    assert frame["date"].max() <= pd.Timestamp("2024-02-29")
