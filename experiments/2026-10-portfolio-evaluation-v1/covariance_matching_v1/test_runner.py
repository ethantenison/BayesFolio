"""Coverage and worker-adapter tests for frozen-prediction covariance research.

Usage: pytest this directory. No fits, allocations, network or financial writes.
"""

from __future__ import annotations

import numpy as np
import run_covariances as study
from covariance_math import METHODS, matched_covariance


def test_complete_cases_and_diagnostic_roles() -> None:
    """Both frozen mean recipes use every method with explicit independence labels."""
    cases = study._cases()
    assert len(cases) == len({case["id"] for case in cases}) == 20
    assert sum(case["role"] == "primary" for case in cases) == 14
    assert sum(case["role"] == "dependence_sensitivity" for case in cases) == 4
    assert sum(case["role"] == "independence_diagnostic_not_winner_eligible" for case in cases) == 2
    for recipe in study.RECIPES:
        assert {case["covariance"] for case in cases if case["prediction"] == recipe} == set(METHODS)


def test_worker_adapters_preserve_frozen_implementation(monkeypatch) -> None:
    """Only worker-local output and panel adapters are replaced."""
    original = study.base.OUTPUT
    monkeypatch.setattr(study.base, "OUTPUT", original)
    monkeypatch.setattr(study.base, "_panel", study.base._panel)
    monkeypatch.setattr(
        study.base,
        "_job",
        lambda case: {
            "case": case,
            "output": study.base.OUTPUT,
            "panel": study.base._panel,
        },
    )
    case = study._cases()[0]
    result = study._worker(case)
    assert result["output"] == study.OUTPUT and result["output"] != original
    assert result["panel"] is study._frozen_panel and result["case"] == case


def test_batched_covariance_axes() -> None:
    """Every method supports explicit leading seed and month dimensions."""
    broad = np.broadcast_to(np.array([[0.04, 0.01], [0.01, 0.02]]), (3, 4, 2, 2))
    compact = np.broadcast_to(np.array([[0.02, -0.005], [-0.005, 0.03]]), (3, 4, 2, 2))
    for method in METHODS:
        result = matched_covariance(broad, compact, np.array([0.5, 0.25]), method)
        assert result.shape == broad.shape
        np.linalg.cholesky(result)


def test_matched_method_endpoints() -> None:
    """All family-matched methods recover a single selected source at endpoints."""
    broad = np.array([[0.04, 0.01], [0.01, 0.02]])
    compact = np.array([[0.02, -0.005], [-0.005, 0.03]])
    for method in [
        "family_gate",
        "family_variance_only",
        "root_shared_100",
        "root_shared_90",
        "root_shared_50",
        "root_independent",
    ]:
        # Variance-only preserves half-blend correlations by design, so only its diagonal is an endpoint.
        for alpha, source in [(np.zeros(2), broad), (np.ones(2), compact)]:
            result = matched_covariance(broad, compact, alpha, method)
            if method == "family_variance_only":
                np.testing.assert_allclose(np.diag(result), np.diag(source), atol=1e-14)
            else:
                np.testing.assert_allclose(result, source, atol=1e-14)
