from __future__ import annotations

import pandas as pd
import pytest

from bayesfolio.core.settings import Horizon
from bayesfolio.engine.features import market_fundamentals as mf


def test_fetch_yield_curve_pcs_handles_partial_fred_timeouts(monkeypatch) -> None:
    monkeypatch.setattr(mf, "_download_frame", lambda *args, **kwargs: pd.DataFrame())

    dates = pd.date_range("2020-01-01", periods=80, freq="D")

    def fake_read_fred(symbols: str | list[str], start: str, end: str | None) -> pd.DataFrame:
        if symbols == "DGS30":
            raise TimeoutError("simulated timeout")

        values = pd.Series(range(len(dates)), index=dates, dtype=float)
        values = values + float(len(str(symbols)))
        return values.to_frame(name=str(symbols))

    monkeypatch.setattr(mf, "_read_fred", fake_read_fred)

    result = mf.fetch_yield_curve_pcs(
        start="2020-01-01",
        end="2020-03-31",
        horizon=Horizon.MONTHLY,
        n_components=3,
    )

    assert not result.empty
    assert "yc_pc1" in result.columns
    assert "date" in result.columns


def test_fetch_yield_curve_pcs_raises_when_all_sources_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(mf, "_download_frame", lambda *args, **kwargs: pd.DataFrame())

    def always_timeout(symbols: str | list[str], start: str, end: str | None) -> pd.DataFrame:
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(mf, "_read_fred", always_timeout)

    with pytest.raises(RuntimeError, match="No yield-curve series available"):
        mf.fetch_yield_curve_pcs(
            start="2020-01-01",
            end="2020-03-31",
            horizon=Horizon.MONTHLY,
            n_components=3,
        )


def test_fetch_macro_features_continues_when_yield_curve_times_out(monkeypatch) -> None:
    base_dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])

    def _frame(name: str) -> pd.DataFrame:
        return pd.DataFrame({"date": base_dates, name: [0.1, 0.2, 0.3]})

    monkeypatch.setattr(mf, "fetch_vix_term_structure", lambda **kwargs: _frame("vix_signal"))
    monkeypatch.setattr(mf, "fetch_term_spread", lambda **kwargs: _frame("term_spread"))
    monkeypatch.setattr(mf, "fetch_credit_spread", lambda **kwargs: _frame("credit_spread"))
    monkeypatch.setattr(mf, "fetch_dxy", lambda **kwargs: _frame("dxy"))
    monkeypatch.setattr(mf, "fetch_yield_curve_pcs", lambda **kwargs: (_ for _ in ()).throw(TimeoutError("timeout")))

    result = mf.fetch_macro_features(start="2020-01-01", end="2020-03-31", horizon=Horizon.MONTHLY)

    assert not result.empty
    assert "term_spread" in result.columns
    assert "credit_spread" in result.columns
    assert "dxy" in result.columns
    assert "yc_pc1" in result.columns
