from __future__ import annotations

import pandas as pd

from bayesfolio.core.settings import Horizon
from bayesfolio.engine.features import market_fundamentals as mf


def test_fetch_macro_features_continues_when_yield_curve_times_out(monkeypatch) -> None:
    base_dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])

    def _frame(name: str) -> pd.DataFrame:
        return pd.DataFrame({"date": base_dates, name: [0.1, 0.2, 0.3]})

    monkeypatch.setattr(mf, "fetch_vix_term_structure", lambda **kwargs: _frame("vix_signal"))
    monkeypatch.setattr(mf, "fetch_term_spread", lambda **kwargs: _frame("term_spread"))
    monkeypatch.setattr(mf, "fetch_credit_spread", lambda **kwargs: _frame("credit_spread"))
    monkeypatch.setattr(mf, "fetch_dxy", lambda **kwargs: _frame("dxy"))

    result = mf.fetch_macro_features(start="2020-01-01", end="2020-03-31", horizon=Horizon.MONTHLY)

    assert not result.empty
    assert "term_spread" in result.columns
    assert "credit_spread" in result.columns
    assert "dxy" in result.columns
    assert "yc_pc1" in result.columns
