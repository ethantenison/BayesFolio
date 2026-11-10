"""Joint-simplex, first16-only and analytical-allocator parity regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "track1_scenarios")]
from case_evaluator import _dependency  # noqa: E402
from joint_math import (  # noqa: E402
    BASE,
    RankAllocator,
    blend,
    coarse_recipes,
    local_recipes,
    recipe,
    tuning_scores,
)


def test_families_have_independent_three_way_weights() -> None:
    """International compact/third weights are free, not fixed at baseline zero."""
    cases = coarse_recipes("third")
    assert len(cases) > 2300
    assert any(c["mean_counts"][1] == [0, 32, 32] for c in cases)
    assert any(len(set(tuple(row) for row in c["mean_counts"])) == 4 for c in cases)
    assert cases == coarse_recipes("third")
    for c in cases:
        np.testing.assert_array_equal(np.array(c["mean_counts"]).sum(1), 64)


def test_direct_blend_matches_explicit_family_formula() -> None:
    """All three terms enter directly; each family can use a different third share."""
    means = np.stack([np.full((3, 16, 15), x) for x in [.01, .03, -.02]])
    covariance = np.stack([np.broadcast_to(np.eye(15) * x, (3, 16, 15, 15)) for x in [.01, .02, .04]])
    counts = np.array([[16, 16, 32], [0, 48, 16], [40, 8, 16], [24, 16, 24]])
    case = recipe("third", counts, [16, 32, 16])
    mean, cov = blend(means, covariance, case)
    expected = np.repeat((counts @ np.array([.01, .03, -.02])) / 64, [4, 4, 4, 3])
    np.testing.assert_allclose(mean, np.broadcast_to(expected, mean.shape), rtol=0, atol=1e-15)
    np.testing.assert_allclose(cov, np.broadcast_to(np.eye(15) * .0225, cov.shape), rtol=0, atol=1e-15)


def test_joint_local_proposals_change_multiple_families() -> None:
    """Refinement is not limited to a one-family or fixed-ratio path."""
    center = recipe("third", np.full((4, 3), [24, 24, 16]))
    proposals = local_recipes([center], 8, seed=27)
    assert any(np.count_nonzero(np.any(np.array(p["mean_counts"]) != center["mean_counts"], axis=1)) > 1
               for p in proposals)


def test_analytical_hard_rank_matches_frozen_baseline() -> None:
    """KKT acceleration preserves the existing objective/constraints and metrics."""
    source = HERE.parent / "scenario_numerics_v2/outputs"
    with np.load(source / "cases/control_hard_rank.npz") as a, np.load(source / "context.npz") as context:
        allocator = RankAllocator()
        w, constrained = allocator.allocate(a["mean"], a["covariance"], context["risk_budget"])
        assert constrained == 1
        np.testing.assert_allclose(w, a["weights"], rtol=0, atol=3e-7)
        scores = tuning_scores(context["actual"][:16], context["risk_free"][:16], a["mean"][:, :16], w[:, :16])
        old = tuning_scores(context["actual"][:16], context["risk_free"][:16], a["mean"][:, :16], a["weights"][:, :16])
        assert scores["sharpe"] == pytest.approx(old["sharpe"], abs=3e-6)
        assert scores["total_return"] == pytest.approx(old["total_return"], abs=1e-7)
        assert scores["rank_ic"] == old["rank_ic"]


def test_random_case_parity_with_original_conic_allocator() -> None:
    """Both slack-risk and genuinely binding-risk solutions match frozen code."""
    rng = np.random.default_rng(127)
    mean = rng.normal(.01, .02, size=(3, 4, 15))
    f = rng.normal(size=(3, 4, 15, 15)) * .01
    covariance = f @ f.swapaxes(-1, -2) + np.eye(15) * .001
    budget = np.max(np.einsum("a,stab,b->st", np.full(15, 1 / 15), covariance, np.full(15, 1 / 15)), axis=0) * 1.1
    new, count = RankAllocator().allocate(mean, covariance, budget)
    assert count > 0
    for s in range(3):
        for t in range(4):
            old = _dependency("hard").hard_rank(mean[s, t], covariance[s, t], budget[t])
            np.testing.assert_allclose(new[s, t], old["weights"], rtol=0, atol=5e-6)


def test_no_later_months_accepted_for_tuning() -> None:
    """A full28 panel is rejected by the tuning-scoring API."""
    with pytest.raises(ValueError, match="sixteen"):
        tuning_scores(np.ones((28, 15)), np.ones(28), np.ones((3, 28, 15)), np.ones((3, 28, 15)) / 15)


def test_two_model_boundary_and_invalid_simplexes() -> None:
    """Controls remain reachable but no component ratio is imposed on other cases."""
    r = recipe("third", BASE)
    assert np.array(r["mean_counts"])[:, 2].sum() == 0
    with pytest.raises(ValueError):
        recipe("third", np.ones((4, 3)))
