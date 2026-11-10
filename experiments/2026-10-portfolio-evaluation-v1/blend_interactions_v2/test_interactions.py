"""Deterministic coverage and worker-isolation tests for cached interactions.

Research tests check recipes, not live fitting or trading. Usage: pytest here.
"""

from __future__ import annotations

import itertools

import run_interactions as study


def test_full_factorial_is_complete_and_unique() -> None:
    """Every declared family-weight combination exists exactly once."""
    cases = study._cases()
    assert len(cases) == len({case["id"] for case in cases}) == 768
    expected = set(itertools.product(*(study.GRID[family] for family in study.base.FAMILIES)))
    assert {tuple(case["alpha"]) for case in cases} == expected
    assert all(case["beta"] == 0.5 and case["decision"] == "mean" for case in cases)
    assert all(case["covariance"] == "arithmetic" for case in cases)


def test_previously_missing_international_interactions_included() -> None:
    """Both U.S. leaders are crossed with small and large international blends."""
    alphas = {tuple(case["alpha"]) for case in study._cases()}
    for us in [0.375, 0.5]:
        for international in [0.0625, 0.125, 0.25, 0.5, 1]:
            assert (us, 0, 0.25, international) in alphas
        assert (us, 0.125, 0.25, 0.125) in alphas


def test_worker_only_redirects_output(monkeypatch) -> None:
    """Frozen solver logic is reused with a new output destination, not copied."""
    original = study.base.OUTPUT
    monkeypatch.setattr(study.base, "OUTPUT", original)
    monkeypatch.setattr(study.base, "_job", lambda case: {"destination": study.base.OUTPUT, "case": case})
    case = study._cases()[0]
    result = study._worker(case)
    assert result["destination"] == study.OUTPUT
    assert study.OUTPUT != original and result["case"] == case
