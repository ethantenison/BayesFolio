"""Synthetic checks of independent joint objective arithmetic and allocation validation.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_joint_audit.py
"""

from __future__ import annotations

import math

import audit_joint_objectives as audit
import numpy as np
import pandas as pd
import pytest


def fixture_files(tmp_path) -> tuple[list[str], list[str], pd.Series]:
    """Generate known return/forecast paths for a small deterministic joint universe."""
    dates = [str(date.date()) for date in pd.date_range("2024-05-31", periods=24, freq="BME")]
    tickers = ["SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX",
               "BND", "IEF", "LQD", "HYG", "BNDX", "VWOB", "HYEM"]
    rows = [{"date": date, "asset_id": ticker, "y_true": (month % 2 * 2 - 1) * 0.01 + task * 0.001,
             "y_pred": (month % 2 * 2 - 1) * 0.01 + task * 0.001 - 0.002, "y_std": 0.01}
            for month, date in enumerate(dates) for task, ticker in enumerate(tickers)]
    frame = pd.DataFrame(rows)
    frame.to_csv(tmp_path / "gp_window_predictions.csv", index=False)
    weights = frame[["date", "asset_id"]].assign(strategy="gp_scenarios_riskfolio", weight=1 / 15)
    weights.to_csv(tmp_path / "portfolio_weights.csv", index=False)
    returns = frame.groupby("date").y_true.mean()
    returns.rename("return").reset_index().assign(strategy="gp_scenarios_riskfolio").to_csv(
        tmp_path / "portfolio_returns.csv", index=False)
    return tickers, dates, returns


def test_rank_ties_use_average_ranks() -> None:
    """Tied tasks use conventional Spearman ranks, not arbitrary asset ordering."""
    assert audit.spearman(np.array([1, 1, 3]), np.array([2, 2, 5])) == pytest.approx(1)
    assert np.isnan(audit.spearman(np.ones(3), np.arange(3)))


def test_independent_decimal_nll_and_conventional_sharpe(tmp_path) -> None:
    """No CAGR Sharpe, second RF subtraction, or joint covariance determinant is used."""
    tickers, dates, returns = fixture_files(tmp_path)
    result = audit.reconstruct(tmp_path, tickers, dates)
    assert result["mean_ic"] == pytest.approx(1)
    assert result["mean_gaussian_nll"] == pytest.approx(0.5 * math.log(2 * math.pi * 0.01**2) + 0.5 * 0.2**2)
    assert result["sharpe"] == pytest.approx(float(returns.mean() / returns.std(ddof=1) * math.sqrt(12)))
    assert result["rmse_diagnostic_only"] == pytest.approx(0.002)
    assert result["families"]["us_fixed_income"]["within_family_ic"] == pytest.approx(1)


def test_missing_task_is_not_accepted_as_a_complete_month(tmp_path) -> None:
    """A missing asset cannot silently alter pooled IC or density weights."""
    tickers, dates, _ = fixture_files(tmp_path)
    path = tmp_path / "gp_window_predictions.csv"
    pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="approved joint panel"):
        audit.reconstruct(tmp_path, tickers, dates)


def test_return_path_must_match_saved_allocations(tmp_path) -> None:
    """A portfolio result cannot be trusted if it differs from weights times true returns."""
    tickers, dates, _ = fixture_files(tmp_path)
    path = tmp_path / "portfolio_returns.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "return"] += 0.001
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        audit.reconstruct(tmp_path, tickers, dates)
