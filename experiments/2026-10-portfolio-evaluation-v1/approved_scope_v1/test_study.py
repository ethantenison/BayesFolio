"""Synthetic gates for strict allocations, chronology and approved study scope.

Usage: .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1
No GP fitting, network calls or writes to existing study artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from policy_math import PolicyAllocator
from study_common import _groups, _portfolio_cases
from study_math import ScoreAllocator, _online_scale, _risk, _score, _scores


def _context() -> dict:
    rng = np.random.default_rng(27)
    return {
        "assets": [f"ETF{i}" for i in range(15)],
        "dates": [str(i) for i in range(24)],
        "families": {
            "a": [f"ETF{i}" for i in range(4)],
            "b": [f"ETF{i}" for i in range(4, 8)],
            "c": [f"ETF{i}" for i in range(8, 12)],
            "d": [f"ETF{i}" for i in range(12, 15)],
        },
        "actual": rng.normal(0, 0.02, (24, 15)),
        "risk_free": np.full(24, 0.004),
        "risk_budget": np.full(24, 0.01),
        "baseline_mean": np.zeros((24, 15)),
        "historical_covariance": np.broadcast_to(np.eye(15) * 0.0004, (24, 15, 15)).copy(),
    }


@pytest.mark.parametrize("minimum", [None, 3, 10])
def test_rank_allocator_matches_frozen_policy(minimum) -> None:
    """Generalized objectives do not change the protected rank allocator."""
    context = _context()
    mean = context["actual"] * 0.5
    scores, penalty = _scores(mean, context["baseline_mean"], "rank")
    actual = ScoreAllocator(context["historical_covariance"], context["risk_budget"], minimum, penalty).allocate(scores)
    expected = PolicyAllocator(context["historical_covariance"], context["risk_budget"], minimum).allocate(mean)
    np.testing.assert_allclose([r["weights"] for r in actual], [r["weights"] for r in expected], atol=1e-6, rtol=0)


def test_failure_does_not_reuse_weights_or_skip_later_months() -> None:
    """Risk-budget infeasibility is recorded, never relaxed or averaged away."""
    context = _context()
    covariance = context["historical_covariance"]
    covariance[3] = np.eye(15)
    scores, penalty = _scores(context["actual"], context["baseline_mean"], "smooth_rank")
    records = ScoreAllocator(covariance, context["risk_budget"], 10, penalty).allocate(scores)
    assert len(records) == 24 and records[3]["weights"] is None
    assert records[4]["status"] == "solved"
    result = _score(context, context["actual"], covariance, records)
    assert result["performance"] is None and result["solved_months"] == 23
    assert all(not row["fallback_used"] and not row["constraint_relaxation"] for row in records)


def test_prequential_scale_cannot_see_current_or_future_outcomes() -> None:
    """Changing outcome t first changes the multiplier at t+1, never at t."""
    context = _context()
    mean = np.zeros((24, 15))
    original = _online_scale(context["actual"], mean, context["historical_covariance"])
    revised = context["actual"].copy()
    revised[10:] += 0.5
    changed = _online_scale(revised, mean, context["historical_covariance"])
    np.testing.assert_allclose(changed[:11], original[:11], rtol=0, atol=1e-14)
    assert changed[11] > original[11]
    assert original[0] == pytest.approx(1)


def test_covariance_blends_are_declared_arms_not_hidden_repairs() -> None:
    """Positive-definite blending preserves exact declared coefficients."""
    context = _context()
    native = 2 * context["historical_covariance"]
    mixed, factors = _risk(
        "GP_blend_0.5", context["actual"], context["baseline_mean"], native, context["historical_covariance"]
    )
    np.testing.assert_allclose(mixed, 1.5 * context["historical_covariance"], rtol=1e-14)
    np.testing.assert_allclose(factors, 1)
    np.linalg.cholesky(mixed)
    with pytest.raises(ValueError, match="cross-family"):
        _risk("GP", context["actual"], context["baseline_mean"], None, context["historical_covariance"])


def test_task_groups_and_policy_grid_match_approved_scope() -> None:
    """Only task partitions, existing masks and declared allocation grids vary."""
    groups = _groups(_context())
    assert len(groups) == 15
    native = _portfolio_cases({"name": "joint", "native_covariance": True})
    assembly = _portfolio_cases({"name": "assembly", "native_covariance": False})
    assert len(native) == 54 and len(assembly) == 2
    assert {row["minimum"] for row in native} == {None, 3, 5, 7, 10, 12}
    assert {row["budget_factor"] for row in native} == {0.5, 1, 1.5}
    assert all(row["risk_source"] == "historical" for row in assembly)


def test_complete_family_assembly_does_not_fabricate_GP_covariance(tmp_path, monkeypatch) -> None:
    """Whole-family substitutions cover 15 ETFs exactly once without GP blocks."""
    import plan

    context = _context()
    monkeypatch.setattr(plan, "ROOT", tmp_path)
    monkeypatch.setattr(plan, "_context", lambda: context)
    folder = tmp_path / "components"
    folder.mkdir()
    names = []
    for index, (name, assets) in enumerate(context["families"].items()):
        names.append(name)
        np.savez(
            folder / f"{name}.npz",
            mean=np.full((3, 24, len(assets)), index * 0.01),
            assets=np.array(assets),
            covariance=np.broadcast_to(np.eye(len(assets)) * 0.001, (3, 24, len(assets), len(assets))),
        )
    contract = {"models": [{"name": "whole", "components": names, "native_covariance": False}]}
    assert len(plan._assemble(contract)) == 1
    with np.load(tmp_path / "panels/whole.npz", allow_pickle=False) as values:
        assert "covariance" not in values.files
        assert values["mean"].shape == (3, 24, 15)
        np.testing.assert_allclose(values["mean"][0, :, -1], 0.03)


def test_effective_floor_and_no_hidden_cap() -> None:
    """No-floor may fully concentrate; floor-ten requires inverse concentration ten."""
    context = _context()
    scores = np.tile(np.arange(15) / 15, (24, 1))
    free = ScoreAllocator(context["historical_covariance"], context["risk_budget"], None, 1e-8).allocate(scores)
    constrained = ScoreAllocator(context["historical_covariance"], context["risk_budget"], 10, 1e-8).allocate(scores)
    assert free[0]["largest_weight"] > 0.999
    assert min(row["effective_count"] for row in constrained) >= 10 - 1e-5
    assert all(row["ETF_position_cap"] is None for row in free + constrained)


def test_portfolio_worker_retains_all_seed_months_and_source(tmp_path, monkeypatch) -> None:
    """The real worker records weights, calibration and hashes for all seeds."""
    import json

    import study_common
    import study_workers

    context = _context()
    context["fixed_probes"] = {"equal_weight": np.ones(15) / 15}
    monkeypatch.setattr(study_workers, "ROOT", tmp_path)
    monkeypatch.setattr(study_common, "ROOT", tmp_path)
    panel = tmp_path / "panels"
    panel.mkdir()
    path = panel / "joint.npz"
    np.savez(
        path, mean=np.stack([context["actual"] * 0.5] * 3), covariance=np.stack([context["historical_covariance"]] * 3)
    )
    before = path.read_bytes()
    study_workers.initialize_portfolios(context)
    case = {
        "configuration": "joint",
        "risk_source": "GP_prequential_scale",
        "objective": "rank",
        "minimum": 10,
        "budget_factor": 1.0,
        "case_id": "synthetic",
        "job_id": "synthetic",
    }
    result = study_workers.portfolio_job(case)
    assert result["summary"]["complete_seeds"] == 3
    artifact = json.loads((tmp_path / "portfolio_cases/synthetic.json").read_text())
    assert len(artifact["seeds"]) == 3
    assert all(len(row["months"]) == 24 for row in artifact["seeds"])
    assert all(row["prequential_variance_scale"][0] == pytest.approx(1) for row in artifact["seeds"])
    assert path.read_bytes() == before
