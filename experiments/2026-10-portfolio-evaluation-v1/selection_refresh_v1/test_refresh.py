"""Deterministic screening, allocation, shortlist and covariance adapter tests.

Usage: PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/pytest -q
experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1/test_refresh.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from refresh import _configs, _load_component, _workers
from refresh_math import CommonAllocator, _density, _portfolio, _ranking, _screen_one, _shortlist


def test_allocator_equivalence_and_no_warm_start() -> None:
    """DPP compilation must not change the existing strict allocator decisions."""
    sys.path.append(str(Path(__file__).resolve().parents[1] / "parallel_pipeline_v1"))
    from workers import solve_weights

    rng = np.random.default_rng(27)
    root = rng.normal(size=(15, 15)) / 100
    matrix = root @ root.T + np.eye(15) * 0.0001
    covariance = np.broadcast_to(matrix, (24, 15, 15))
    means = rng.normal(size=(24, 15)) / 100
    allocator = CommonAllocator(covariance)
    first = allocator.solve(means)
    for index in [0, 12, 23]:
        expected, _ = solve_weights(means[index], matrix, "risk_rank_cap35")
        np.testing.assert_allclose(first[index], expected, atol=1e-6, rtol=0)
    allocator.solve(means[::-1])
    np.testing.assert_array_equal(first, allocator.solve(means))


def test_undefined_ic_rejected() -> None:
    """Partial ranking cannot silently become a promotable mean IC."""
    with pytest.raises(ValueError, match="Incomplete ranking"):
        _ranking(np.ones((24, 4)), np.ones((24, 4)))


def test_shortlist_exclusion_dedup_and_compact() -> None:
    """Select distinct deterministic roles without admitting the HY-change mask."""
    rows = [
        {
            "configuration": f"c{i}",
            "family": family,
            "eligible": i != 0,
            "input_count": 10 - i,
            "mean_ic": 1 - i / 10,
            "equal_family_ic": 1 - i / 10,
            "conventional_sharpe": i,
            "mean_marginal_nll": abs(i - 2),
        }
        for family in ["joint", "us_equity"]
        for i in range(8)
    ]
    selected = _shortlist(rows, "c3")
    for family in ["joint", "us_equity"]:
        keys = [r["configuration"] for r in selected if r["family"] == family]
        assert len(keys) == len(set(keys)) == 4
        assert "c0" not in keys
        assert {"c1", "c2", "c7"}.issubset(keys)
    assert "c5" in [r["configuration"] for r in selected if r["family"] == "us_equity"]


def test_raw_target_conversion_and_portfolio_risk() -> None:
    """Use arithmetic excess Sharpe and full covariance in portfolio calibration."""
    actual = np.tile(np.linspace(-0.03, 0.04, 24)[:, None], (1, 15))
    weights = np.ones((24, 15)) / 15
    rf = np.full(24, 0.004)
    result = _portfolio({"actual": actual, "risk_free": rf}, weights)
    expected = actual[:, 0] * (1 + rf)
    np.testing.assert_allclose(result["arithmetic_excess"], expected, atol=1e-15)
    assert result["conventional_sharpe"] == pytest.approx(expected.mean() / expected.std(ddof=1) * np.sqrt(12))
    covariance = np.broadcast_to(np.eye(3) * 0.002 + np.ones((3, 3)) * 0.001, (24, 3, 3))
    density = _density(actual[:, :3], actual[:, :3] * 0, covariance)
    assert density["equal_weight_reference"]["mean_predicted_sd"] == pytest.approx(np.sqrt(0.002 / 3 + 0.001))
    with pytest.raises(ValueError):
        _density(actual[:, :3], actual[:, :3] * 0, covariance * -1)


def test_matched_architecture_changes_only_tasks() -> None:
    """Keep the baseline feature mask and fitting controls across task groups."""
    tickers = [
        "SPY",
        "MGK",
        "VTV",
        "IWM",
        "VEA",
        "VWO",
        "VSS",
        "EWX",
        "BND",
        "IEF",
        "LQD",
        "HYG",
        "BNDX",
        "VWOB",
        "HYEM",
    ]
    context = {
        "families": {
            "us_equity": tickers[:4],
            "international_equity": tickers[4:8],
            "us_fixed_income": tickers[8:12],
            "international_fixed_income": tickers[12:],
        }
    }
    candidate = {
        "configuration": "global_e6c41f2020fd",
        "config": {
            "command": ["python", "wrapper.py", "--tickers", ",".join(tickers), "--run-id", "old"],
            "tickers": tickers,
            "inputs": ["t_index", "oil_ret"],
            "old_run_root": "original",
        },
    }
    configs = _configs([], [candidate], context)
    assert len(configs) == 6
    for config in configs.values():
        assert config["inputs"] == candidate["config"]["inputs"]
        assert "old_run_root" not in config
        assert config["command"][3].split(",") == config["tickers"]
    assert candidate["config"]["command"][-1] == "old"


def test_exact_component_grid_validation(tmp_path, monkeypatch) -> None:
    """Analytic artifacts must match registered covariance and asset order."""
    import refresh

    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    folder = tmp_path / "runs/test_s27/analytic_months"
    folder.mkdir(parents=True)
    covariance = np.eye(3) * 0.01
    np.savez(
        folder / "covariance_2024-05-31.npz",
        date=np.array("2024-05-31"),
        assets=np.array(["A", "B", "C"]),
        mean=np.zeros(3),
        std=np.full(3, 0.1),
        covariance=covariance,
    )
    result = _load_component("test", 27, {"tickers": ["A", "B", "C"]}, {"dates": ["2024-05-31"]})
    assert result["covariance"].shape == (1, 3, 3)
    with pytest.raises(ValueError, match="metadata"):
        _load_component("test", 27, {"tickers": ["B", "A", "C"]}, {"dates": ["2024-05-31"]})


def test_worker_memory_and_core_ceiling(monkeypatch) -> None:
    """Use one pool, cores minus two, and the aggregate research memory budget."""
    import refresh

    monkeypatch.setattr(refresh.os, "cpu_count", lambda: 16)
    assert _workers() == 14
    assert _workers(4 * 1024**3) == 9
    monkeypatch.setattr(refresh.os, "cpu_count", lambda: 8)
    assert _workers() == 6


def test_strict_failure_keeps_forecasts_but_never_creates_weights() -> None:
    """Do not substitute past weights or silently drop a failed portfolio month."""

    class Failed:
        def solve(self, mean):
            raise RuntimeError("Strict allocator failed: optimal_inaccurate; no fallback")

    rng = np.random.default_rng(27)
    actual = rng.normal(0, 0.02, size=(24, 15))
    assets = [str(i) for i in range(15)]
    context = {
        "actual": actual,
        "assets": assets,
        "baseline_mean": actual * 0,
        "risk_free": np.full(24, 0.004),
        "families": {"first": assets[:4], "rest": assets[4:]},
    }
    candidate = {
        "configuration": "test",
        "family": "first",
        "tickers": assets[:4],
        "mean": np.stack([actual[:, :4]] * 3),
        "std": np.full((3, 24, 4), 0.02),
        "inputs": ["t_index"],
        "eligible": True,
    }
    result = _screen_one(candidate, context, np.stack([actual] * 3), Failed())
    for row in result["results"]:
        assert row["mean_ic"] == pytest.approx(1)
        assert row["portfolio"]["weights"] is None
        assert row["portfolio"]["conventional_sharpe"] is None
        assert row["portfolio"]["fallback_used"] is False
