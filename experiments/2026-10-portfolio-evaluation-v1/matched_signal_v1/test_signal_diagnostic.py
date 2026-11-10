"""Synthetic score-scale, strict-constraint and allocator-wiring regression tests.

Usage: .venv/bin/pytest -q <this_directory>. No fitting, data refresh or writes to
real study artifacts; temporary fixtures and immutable synthetic records only.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import run_matched_signal as runner
from plot_diagnostic import render_diagnostic
from signal_math import audit_weights, normalized_scores


def test_average_ties_and_gp_score_normalization_are_preserved() -> None:
    """Both signals use average ranks, not top-k weights or alphabetical ties."""
    signal = np.array([[1, 4, 4, 2], [-5, -4, -4, -3]], dtype=float)
    expected = np.array([[1, 3.5, 3.5, 2], [1, 2.5, 2.5, 4]]) / 4
    np.testing.assert_allclose(normalized_scores(signal), expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(normalized_scores(signal), normalized_scores(3 * signal + 9))


@pytest.mark.parametrize("signal", [np.array([1, 2]), np.array([[np.nan, 2]]), np.ones((2, 1))])
def test_incomplete_signals_fail(signal: np.ndarray) -> None:
    """Missing scores do not get filled or silently dropped."""
    with pytest.raises(ValueError):
        normalized_scores(signal)


def _allocation_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.full((3, 24, 15), 1 / 15)
    covariance = np.broadcast_to(np.eye(15) * 0.001, (3, 24, 15, 15)).copy()
    budget = np.full(24, 0.001)
    return weights, covariance, budget


def test_independent_constraint_audit_preserves_raw_weights() -> None:
    """Auditing leaves inputs unchanged and checks original decimal-squared risk."""
    weights, covariance, budget = _allocation_fixture()
    before = weights.copy()
    result = audit_weights(weights, covariance, budget)
    np.testing.assert_allclose(result["effective_count"], 15, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["variance_ratio"], 1 / 15, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(weights, before)


@pytest.mark.parametrize("violation", ["short", "budget_sum", "effective_count", "risk", "missing"])
def test_constraint_violations_fail_without_repairs(violation: str) -> None:
    """Each original guardrail remains binding in the audit, with no fixes."""
    weights, covariance, budget = _allocation_fixture()
    if violation == "short":
        weights[0, 0, 0], weights[0, 0, 1] = -0.01, 2 / 15 + 0.01
    elif violation == "budget_sum":
        weights[0, 0, 0] += 0.1
    elif violation == "effective_count":
        weights[0, 0] = np.r_[1.0, np.zeros(14)]
    elif violation == "risk":
        budget[:] = 1e-8
    else:
        weights[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        audit_weights(weights, covariance, budget)


def test_worker_passes_same_risk_and_constraints_and_only_selected_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual outcomes and means are not arguments to allocation; only scores switch."""
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    policy = {"minimum_effective_assets": 7, "concentration_tie_breaker": 1e-8}
    (tmp_path / "contract.json").write_text(json.dumps({"policy": policy}))
    weights, covariance, budget = _allocation_fixture()
    gp = np.broadcast_to(np.arange(1, 16) / 15, (3, 24, 15)).copy()
    momentum = gp[..., ::-1].copy()
    np.savez(
        tmp_path / "input_panel.npz",
        covariance=covariance,
        risk_budget=budget,
        gp_scores=gp,
        momentum_scores=momentum,
        actual=np.full((24, 15), 999999),
    )
    calls = []

    def fake_allocator(cov: np.ndarray, limit: np.ndarray, minimum: int, penalty: float, scores: np.ndarray) -> dict:
        """Capture unchanged allocator inputs and return synthetic optimal records."""
        calls.append((cov.copy(), limit.copy(), minimum, penalty, scores.copy()))
        return {
            "status": "allocated",
            "preflight": [],
            "allocations": [
                {"status": "solved", "weights": row.tolist(), "numerical_attempts": [{"status": "optimal"}]}
                for row in weights[0]
            ],
        }

    monkeypatch.setattr(runner, "allocate_checked", fake_allocator)
    runner._allocation_job(("gp", 0, 27))
    runner._allocation_job(("momentum", 0, 27))
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    assert calls[0][2:4] == calls[1][2:4] == (7, 1e-8)
    np.testing.assert_array_equal(calls[0][4], gp[0])
    np.testing.assert_array_equal(calls[1][4], momentum[0])


def test_failed_allocator_is_not_saved_or_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocked solve cannot create a successful checkpoint or fallback weights."""
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    (tmp_path / "contract.json").write_text(
        json.dumps({"policy": {"minimum_effective_assets": 7, "concentration_tie_breaker": 1e-8}})
    )
    _, covariance, budget = _allocation_fixture()
    np.savez(
        tmp_path / "input_panel.npz", covariance=covariance, risk_budget=budget, momentum_scores=np.zeros((3, 24, 15))
    )
    monkeypatch.setattr(runner, "allocate_checked", lambda *args: {"status": "blocked_preflight"})
    with pytest.raises(ValueError, match="Incomplete"):
        runner._allocation_job(("momentum", 0, 27))
    assert not (tmp_path / "cases/momentum_s27.json").exists()


def test_original_allocator_accepts_matched_synthetic_scores() -> None:
    """A real synthetic solve exercises the unchanged cone allocator and raw audit."""
    _, covariance, budget = _allocation_fixture()
    scores = np.broadcast_to(np.arange(1, 16) / 15, (24, 15)).copy()
    allocated = runner.allocate_checked(covariance[0], budget, 7, 1e-8, scores)
    assert allocated["status"] == "allocated"
    weights = np.array([row["weights"] for row in allocated["allocations"]])[None, ...]
    audit_weights(weights, covariance[:1], budget)
    assert weights[0, 0, -1] > weights[0, 0, 0]


def test_plot_smoke_and_input_immutability() -> None:
    """A complete synthetic two-panel chart renders without changing results."""
    months = np.arange(np.datetime64("2024-05", "M"), np.datetime64("2026-05", "M"))
    arrays = {
        "dates": ((months + 1).astype("datetime64[D]") - 1).astype(str),
        "gp_wealth": np.tile(np.linspace(10000, 16000, 25), (3, 1)),
        "momentum_wealth": np.tile(np.linspace(10000, 14000, 25), (3, 1)),
        "simple_momentum_wealth": np.linspace(10000, 13500, 25),
        "gp_ic_pooled": np.tile(np.sin(np.arange(24)), (3, 1)),
        "momentum_ic_pooled": np.tile(np.cos(np.arange(24)), (3, 1)),
    }
    before = {key: value.copy() for key, value in arrays.items()}
    assert render_diagnostic(arrays).startswith(b"\x89PNG\r\n\x1a\n")
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


def test_modules_and_public_symbols_documented() -> None:
    """Validate doc coverage for new experiment files outside the package gate."""
    for path in Path(__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        assert ast.get_docstring(tree), path
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                assert ast.get_docstring(node), (path, node.name)
