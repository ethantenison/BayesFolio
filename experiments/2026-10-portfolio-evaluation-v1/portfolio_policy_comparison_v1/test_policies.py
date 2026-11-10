"""Paired-policy correctness, strict failures and source-preservation tests.

Usage: .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from full_math import NoCapAllocator
from policy_math import PolicyAllocator, _policies, _score_allocations, _summary


def test_upstream_requires_completed_status_and_released_writer(tmp_path, monkeypatch) -> None:
    """A stale completion seal or unfinished final audit must not start scoring."""
    import run_comparison

    monkeypatch.setattr(run_comparison, "FITS", tmp_path)
    assert not run_comparison._ready()
    (tmp_path / "COMPLETED.json").write_text('{"passed": true}')
    (tmp_path / "status.json").write_text('{"stage": "failed"}')
    (tmp_path / "run.lock").touch()
    assert not run_comparison._ready()
    (tmp_path / "status.json").write_text('{"stage": "completed"}')
    with (tmp_path / "run.lock").open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not run_comparison._ready()
    assert run_comparison._ready()


def _context() -> dict:
    rng = np.random.default_rng(27)
    return {
        "actual": rng.normal(0, 0.02, (24, 15)),
        "risk_free": np.full(24, 0.004),
        "dates": pd.date_range("2024-05-31", periods=24, freq="BME").strftime("%Y-%m-%d").tolist(),
        "assets": [f"ETF{i}" for i in range(15)],
        "risk_budget": np.full(24, 0.01),
        "historical_covariance": np.broadcast_to(np.eye(15) * 0.0004, (24, 15, 15)),
    }


def test_policy_registration_joint10_family3_none_other_variant() -> None:
    """Do not apply a ten-asset floor inside a three/four-ETF family model."""
    assert _policies("joint") == {"risk_budget_only": None, "risk_budget_plus_effective": 10}
    assert _policies("us_equity") == {"risk_budget_only": None, "risk_budget_plus_effective": 3}


def test_risk_only_can_hold_100_percent_and_joint10_cannot() -> None:
    """No hidden ETF cap or effective-count minimum survives in risk-only mode."""
    context = _context()
    mean = np.tile(np.arange(15), (24, 1)).astype(float)
    risk_only = PolicyAllocator(context["historical_covariance"], context["risk_budget"], None).allocate(mean)
    diversified = PolicyAllocator(context["historical_covariance"], context["risk_budget"], 10).allocate(mean)
    assert all(record["status"] == "solved" for record in risk_only + diversified)
    assert risk_only[0]["largest_weight"] > 0.999
    assert risk_only[0]["effective_count"] < 1.001
    assert min(record["effective_count"] for record in diversified) >= 10 - 1e-5
    assert all(record["ETF_position_cap"] is None for record in risk_only + diversified)


def test_minimum_three_matches_frozen_allocator() -> None:
    """The legacy control remains equivalent without modifying its sealed source."""
    context = _context()
    mean = context["actual"] * 0.5
    expected = NoCapAllocator(context["historical_covariance"], context["risk_budget"]).solve(mean)
    actual = PolicyAllocator(context["historical_covariance"], context["risk_budget"], 3).allocate(mean)
    np.testing.assert_allclose(np.array([r["weights"] for r in actual]), expected, atol=1e-7, rtol=0)


def test_one_failed_month_recorded_and_no_annual_performance() -> None:
    """Do not skip a failed month, copy previous weights, or stop later solves."""
    context = _context()
    covariance = context["historical_covariance"].copy()
    covariance[5] = np.eye(15)
    records = PolicyAllocator(covariance, context["risk_budget"], 10).allocate(context["actual"])
    assert len(records) == 24
    assert records[5]["status"] != "solved" and records[5]["weights"] is None
    assert records[6]["status"] == "solved"
    scored = _score_allocations(context, context["actual"], records, {"control": covariance})
    assert scored["performance"] is None
    assert scored["solved_months"] == 23
    assert all(record["fallback_used"] is False for record in records)


def test_missing_months_rejected() -> None:
    """An incomplete input grid is not an allocation failure to drop silently."""
    context = _context()
    with pytest.raises(ValueError, match="Every original month"):
        _score_allocations(context, context["actual"], [], {})


def test_positive_budget_not_widened_for_uncertain_GP() -> None:
    """Both variants retain the common absolute budget, including infeasibility."""
    context = _context()
    covariance = np.broadcast_to(np.eye(15), (24, 15, 15))
    for minimum in [None, 10]:
        records = PolicyAllocator(covariance, context["risk_budget"], minimum).allocate(context["actual"])
        assert all(record["status"] != "solved" for record in records)
        assert all(record["weights"] is None and record["variance_budget"] == 0.01 for record in records)


@pytest.mark.parametrize("minimum", [0, 16, -1])
def test_invalid_minimum_rejected(minimum: int) -> None:
    """Count controls cannot be silently clipped into a different policy."""
    context = _context()
    with pytest.raises(ValueError):
        PolicyAllocator(context["historical_covariance"], context["risk_budget"], minimum)


def test_joint_worker_records_four_policies_without_writing_upstream(tmp_path, monkeypatch) -> None:
    """All seeds/months/assets are saved for both covariance/policy combinations."""
    import policy_workers

    context = _context()
    upstream, output = tmp_path / "upstream", tmp_path / "comparison"
    candidate = {"configuration": "joint", "family": "joint", "tickers": context["assets"], "inputs": ["t_index"]}
    expected = {}
    for seed in [27, 127, 227]:
        path = upstream / f"analytic/joint/s{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            dates=np.array(context["dates"]),
            assets=np.array(context["assets"]),
            observation_noise=np.array(True),
            units=np.array("raw_discounted_excess"),
            mean=context["actual"] * 0.5,
            std=np.full((24, 15), 0.02),
            covariance=context["historical_covariance"],
            likelihood_noise_diagonal=np.full((24, 15), 0.0001),
        )
        expected[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    policy_workers.initialize(context, np.stack([context["actual"] * 0.5] * 3))
    result = policy_workers.evaluate(
        {"candidate": candidate, "upstream": str(upstream), "output": str(output), "expected_analytic_sha": expected}
    )
    assert len(result["summaries"]) == 4
    for row in result["summaries"]:
        assert row["complete_seeds"] == 3
        assert row["minimum_effective"] == (10 if row["variant"] == "risk_budget_plus_effective" else None)
    weights = pd.read_csv(output / "monthly_weights/joint.csv")
    assert len(weights) == 3 * 24 * 15 * 4
    assert weights.groupby(["seed", "covariance_policy", "variant", "date"]).size().eq(15).all()
    for path, digest in expected.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    scored = json.loads((output / "portfolios/joint.json").read_text())
    assert scored["CVaR"] is False and scored["ETF_position_cap"] is None
    assert len(scored["seeds"]) == 3


def test_family_summary_has_both_historical_policies_only() -> None:
    """Do not manufacture a GP-native full-portfolio covariance for a family."""
    context = _context()
    portfolios = {}
    for variant, minimum in _policies("us_equity").items():
        records = PolicyAllocator(context["historical_covariance"], context["risk_budget"], minimum).allocate(
            context["actual"]
        )
        portfolios[f"historical__{variant}"] = _score_allocations(
            context, context["actual"], records, {"historical": context["historical_covariance"]}
        )
    summaries = _summary({"configuration": "family", "family": "us_equity"}, [{"portfolios": portfolios}] * 3)
    assert len(summaries) == 2
    assert {r["covariance_policy"] for r in summaries} == {"historical"}
    assert {r["minimum_effective"] for r in summaries} == {None, 3}
