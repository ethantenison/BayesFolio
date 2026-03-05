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
