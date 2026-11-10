"""Synthetic local tests of static blending, complete scoring and immutable cases."""

from __future__ import annotations

import json
from pathlib import Path

import case_evaluator as evaluator
import numpy as np
import pytest
from cvxpy.error import SolverError


@pytest.fixture
def campaign(tmp_path: Path) -> tuple[Path, dict]:
    """Create three small source panels with a fully feasible cached baseline."""
    rng = np.random.default_rng(20261025)
    dates = np.array([
        "2024-05-31", "2024-06-28", "2024-07-31", "2024-08-30", "2024-09-30", "2024-10-31",
        "2024-11-29", "2024-12-31", "2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30",
        "2025-05-30", "2025-06-30", "2025-07-31", "2025-08-29", "2025-09-30", "2025-10-31",
        "2025-11-28", "2025-12-31", "2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30",
        "2026-05-29", "2026-06-30", "2026-07-31", "2026-08-31",
    ])
    axes = {"dates": dates, "assets": np.array(evaluator._ASSETS), "seeds": np.array([27, 127, 227])}
    broad = rng.normal(0.008, 0.012, (3, 28, 15))
    compact = rng.normal(0.012, 0.009, (3, 28, 15))
    third = rng.normal(0.004, 0.02, (3, 28, 15))
    covariance = np.broadcast_to(np.eye(15) * 0.001, (3, 28, 15, 15)).copy()
    components = tmp_path / "components"
    components.mkdir()
    for name, mean, multiplier in [
        ("broad_r2_i15_exp", broad, 1), ("compact_r2_i15_exp", compact, 3), ("third", third, 5),
    ]:
        np.savez_compressed(components / f"{name}.npz", **axes, mean=mean, covariance=covariance * multiplier)
    alpha = np.repeat([0.5, 0.0, 0.375, 0.25], [4, 4, 4, 3])
    mean = (1 - alpha) * broad + alpha * compact
    weights = np.full((3, 28, 15), 1 / 15)
    actual = rng.normal(0.009, 0.03, (28, 15))
    np.savez_compressed(
        tmp_path / "context.npz", **axes, actual=actual, risk_free=np.full(28, 0.002),
        risk_budget=np.full(28, 0.001), baseline_mean=mean, baseline_covariance=covariance * 2,
        baseline_hard_weights=weights, baseline_scenario_weights=weights,
        benchmark_schwab_total=np.full(28, 0.004), benchmark_momentum_total=np.full(28, 0.005),
    )
    case = {"id": "baseline", "track": 1, "stage": "test", "allocator": "hard_rank",
            "broad_id": "broad_r2_i15_exp", "compact_id": "compact_r2_i15_exp",
            "alpha": [0.5, 0.0, 0.375, 0.25], "cov_compact_fraction": 0.5,
            "third_id": None, "third_fraction": 0.0, "third_cov_fraction": 0.0}
    return tmp_path, case


def test_baseline_reuses_weights_scores_seeds_and_immutable_replay(campaign: tuple, monkeypatch) -> None:
    """Exact baseline moments reuse weights and preserve all means/returns/NAV."""
    root, case = campaign

    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline must not call an allocation solver")

    monkeypatch.setattr(evaluator, "_solve", forbidden)
    record = evaluator.evaluate_case(case, str(root))
    assert record["status"] == "complete"
    assert record["successful_allocations"] == 84
    assert record["numerical_route_counts"] == {"exact_baseline_cache": 84}
    assert set(record["periods"]) == {"tuning_first16", "validation_last12", "all28"}
    period = record["periods"]["all28"]
    assert len(period["per_seed"]) == 3
    assert period["months"] == 28
    with np.load(root / "cases/baseline.npz") as panel, np.load(root / "context.npz") as context:
        expected = (1 + np.mean(context["actual"], axis=1)) * (1 + context["risk_free"]) - 1
        np.testing.assert_allclose(panel["total_returns"], np.broadcast_to(expected, (3, 28)), atol=1e-15)
        np.testing.assert_allclose(panel["wealth"][:, -1], 10000 * np.prod(1 + expected), atol=1e-9)
        excess = expected - context["risk_free"]
        assert period["sharpe"] == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(12))
        assert period["total_return"] == pytest.approx(np.prod(1 + expected) - 1)
    assert evaluator.evaluate_case(case, str(root)) == record
    assert json.loads((root / "cases/baseline.json").read_text()) == record


def test_family_and_third_mean_covariance_fractions_are_independent(campaign: tuple, monkeypatch) -> None:
    """Third-model mean/covariance coefficients cannot silently become matched."""
    root, case = campaign
    case.update(id="third_blend", third_id="third", third_fraction=0.25, third_cov_fraction=0.75)
    calls = []

    def solve(allocator, mean, covariance, budget, seed, origin, assets):
        calls.append((seed, origin))
        return {"weights": np.full(15, 1 / 15), "solver_route": "synthetic_feasible"}

    monkeypatch.setattr(evaluator, "_solve", solve)
    record = evaluator.evaluate_case(case, str(root))
    assert record["status"] == "complete"
    assert len(calls) == len(set(calls)) == 84
    with (np.load(root / "cases/third_blend.npz") as panel,
          np.load(root / "context.npz") as context, np.load(root / "components/third.npz") as third):
        np.testing.assert_allclose(panel["mean"], 0.75 * context["baseline_mean"] + 0.25 * third["mean"])
        np.testing.assert_allclose(panel["covariance"], 0.25 * context["baseline_covariance"]
                                   + 0.75 * third["covariance"])


def test_failed_months_are_explicit_and_no_partial_case_is_scored(campaign: tuple, monkeypatch) -> None:
    """Infeasible risk budgets remain failed observations, not fallback weights."""
    root, case = campaign
    case.update(id="infeasible", alpha=[0, 0, 0, 0])
    count = 0

    def solve(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("Strict optimal hard-rank solve required: status infeasible")
        return {"weights": np.full(15, 1 / 15), "solver_route": "synthetic_feasible"}

    monkeypatch.setattr(evaluator, "_solve", solve)
    record = evaluator.evaluate_case(case, str(root))
    assert count == 84
    assert record["status"] == "failed"
    assert record["periods"] == {}
    assert record["successful_allocations"] == 83
    assert record["failure_kind_counts"] == {"risk_budget_infeasible_under_frozen_constraints": 1}
    with np.load(root / "cases/infeasible.npz") as arrays:
        assert np.isnan(arrays["weights"][0, 1]).all()
        assert np.isnan(arrays["wealth"]).all()
        assert not arrays["allocated"][0, 1]


def test_exhausted_numerical_solver_does_not_abort_other_cases(campaign: tuple, monkeypatch) -> None:
    """CVXPY's non-RuntimeError exception remains explicit numerical failure data."""
    root, case = campaign
    case.update(id="numeric_failure", alpha=[0, 0, 0, 0])

    def solve(*args, **kwargs):
        raise SolverError("Exhausted unchanged-problem numerical retries")

    monkeypatch.setattr(evaluator, "_solve", solve)
    record = evaluator.evaluate_case(case, str(root))
    assert record["status"] == "failed" and record["successful_allocations"] == 0
    assert record["failed_allocation_pairs"] == 84
    assert record["failure_kind_counts"] == {"numerical_or_weight_validation_failure": 84}
    assert record["periods"] == {}


def test_missing_component_is_a_failed_case_not_a_campaign_exception(campaign: tuple) -> None:
    """Unavailable source panels are recorded without inventing forecasts."""
    root, case = campaign
    case.update(id="missing_source", broad_id="not_available")
    record = evaluator.evaluate_case(case, str(root))
    assert record["status"] == "failed" and record["periods"] == {}
    assert record["successful_allocations"] == 0
    assert "FileNotFoundError" in record["failures"][0]["error"]


@pytest.mark.parametrize("update", [
    {"alpha": [[0.5] * 28] * 4}, {"cov_compact_fraction": [0.5] * 28},
    {"third_fraction": 1.1}, {"third_fraction": 0.2}, {"alpha": [0, 0, 0]},
    {"id": "../escape"}, {"broad_id": "../escape"}, {"allocator": "CVaR"},
])
def test_dynamic_or_invalid_weights_and_paths_are_rejected(campaign: tuple, update: dict) -> None:
    """Only static valid blending recipes and safe artifact identifiers are allowed."""
    root, case = campaign
    case.update(update)
    with pytest.raises((ValueError, TypeError)):
        evaluator.evaluate_case(case, str(root))
    assert not (root / "cases").exists()


def test_cache_detects_changed_inputs_and_changed_case_settings(campaign: tuple) -> None:
    """A case ID cannot be rebound to new settings or new numerical inputs."""
    root, case = campaign
    evaluator.evaluate_case(case, str(root))
    changed = {**case, "alpha": [0, 0, 0, 0]}
    with pytest.raises(ValueError, match="different settings"):
        evaluator.evaluate_case(changed, str(root))
    component = root / "components/broad_r2_i15_exp.npz"
    component.write_bytes(component.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="input changed"):
        evaluator.evaluate_case(case, str(root))


def test_cache_detects_altered_arrays(campaign: tuple) -> None:
    """Cached results require their sealed NPZ digest, not just a JSON label."""
    root, case = campaign
    evaluator.evaluate_case(case, str(root))
    path = root / "cases/baseline.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="arrays changed"):
        evaluator.evaluate_case(case, str(root))


def test_baseline_moment_mismatch_fails_without_using_cache(campaign: tuple) -> None:
    """A nominal baseline with changed predictions cannot inherit its old weights."""
    root, case = campaign
    path = root / "components/broad_r2_i15_exp.npz"
    with np.load(path) as panel:
        arrays = {key: panel[key].copy() for key in panel.files}
    arrays["mean"] += 0.001
    np.savez_compressed(path, **arrays)
    record = evaluator.evaluate_case(case, str(root))
    assert record["status"] == "failed"
    assert record["successful_allocations"] == 0
    assert "Cached baseline mean identity failed" in record["failures"][0]["error"]


def test_real_frozen_hard_rank_economics_one_month() -> None:
    """The shared adapter invokes the unchanged allocator with its raw constraints."""
    mean, covariance, budget = np.linspace(-0.02, 0.02, 15), np.eye(15) * 0.001, 0.001
    result = evaluator._solve("hard_rank", mean, covariance, budget, 27, "2024-05-31", evaluator._ASSETS)
    audit = evaluator._audit(np.asarray(result["weights"]), covariance, budget, "hard_rank")
    assert audit["effective_count"] >= 7 - 1e-5
    assert result["fallback_used"] is False


def test_real_scenarios_use_common_key_draws_and_signed_solver(monkeypatch) -> None:
    """Scenario adaptation preserves all 5,000 common-normal draws and seed keys."""
    sampling = evaluator._dependency("sampling")
    solver = evaluator._dependency("scenarios")
    recorded = {}

    def allocate(scenarios, assets, *, seed):
        recorded.update(scenarios=scenarios.copy(), assets=assets, seed=seed)
        return {"weights": np.full(15, 1 / 15), "solver_route": "test_signed_entry"}

    monkeypatch.setattr(solver, "allocate_scenarios", allocate)
    mean, covariance = np.linspace(-0.02, 0.02, 15), np.eye(15) * 0.001
    evaluator._solve("scenarios", mean, covariance, 0.001, 127, "2025-04-30", evaluator._ASSETS)
    np.testing.assert_array_equal(recorded["scenarios"], sampling.sample_joint(
        mean, covariance, seed=127, month="2025-04-30", count=5000))
    assert recorded["seed"] == 20261024 + 127 + 20250430
    assert recorded["assets"] == evaluator._ASSETS
