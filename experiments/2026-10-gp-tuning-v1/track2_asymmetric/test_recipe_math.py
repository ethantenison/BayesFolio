"""Synthetic tests for static, temporally separated asymmetric-GP recipes."""

from __future__ import annotations

import numpy as np
import pytest
import recipe_math as recipes


def _component_ids() -> list[str]:
    return [
        f"{family}_r{rank}_i{maxiter}_exp"
        for family in ("broad", "compact")
        for rank in recipes.RANKS
        for maxiter in recipes.MAXITERS
    ]


def _record(identifier: str, sharpe: float, total_return: float, rank_ic: float) -> dict[str, str | float | None]:
    return {
        "id": identifier,
        "period": recipes.SELECTION_PERIOD,
        "sharpe": sharpe,
        "total_return": total_return,
        "rank_ic": rank_ic,
    }


def test_screen_has_all_component_pairs_and_unchanged_recipe() -> None:
    """The 144-cell screen changes only model components, not blending."""
    cases = recipes.screen_cases(_component_ids() + ["third_optional_component"])
    assert len(cases) == 144
    assert len({case["id"] for case in cases}) == 144
    assert len({(case["broad_id"], case["compact_id"]) for case in cases}) == 144
    for case in cases:
        np.testing.assert_allclose(case["alpha"], recipes.BASELINE_ALPHA, rtol=0, atol=1e-15)
        assert case["cov_compact_fraction"] == pytest.approx(0.5)
        assert case["third_id"] is None
    with pytest.raises(ValueError, match="Missing required"):
        recipes.screen_cases(_component_ids()[1:])


def test_mean_grid_is_bounded_static_and_contains_controls() -> None:
    """Coarse and local menus are bounded and contain both approved recipes."""
    pair = recipes.screen_cases(_component_ids())[0]
    cases = recipes.mean_retuning_cases(pair)
    alphas = {tuple(case["alpha"]) for case in cases}
    assert len(cases) == 97
    assert len(alphas) == len(cases)
    assert len({case["id"] for case in cases}) == len(cases)
    assert recipes.BASELINE_ALPHA in alphas
    assert recipes.COMPROMISE_ALPHA in alphas
    assert (0.0, 0.0, 0.0, 0.0) in alphas
    assert (1.0, 1.0, 1.0, 1.0) in alphas
    for case in cases:
        assert len(case["alpha"]) == 4
        assert case["cov_compact_fraction"] == pytest.approx(0.5)
    np.testing.assert_allclose(pair["alpha"], recipes.BASELINE_ALPHA, rtol=0, atol=1e-15)


def test_covariance_stage_does_not_rewrite_mean_weights() -> None:
    """Covariance changes are isolated and add at most 12 cases per pair."""
    means = recipes.mean_retuning_cases(recipes.screen_cases(_component_ids())[0])[:3]
    cases = recipes.covariance_retuning_cases(means)
    assert len(cases) == 12
    for case in cases:
        assert tuple(case["alpha"]) in {tuple(mean["alpha"]) for mean in means}
        assert case["cov_compact_fraction"] in (0.0, 0.25, 0.75, 1.0)
    with pytest.raises(ValueError, match="At most three"):
        recipes.covariance_retuning_cases(means + [means[0]])
    with pytest.raises(ValueError, match="distinct means"):
        recipes.covariance_retuning_cases([means[0], means[0]])


def test_blend_uses_same_family_fractions_across_all_dates_and_seeds() -> None:
    """A recipe is a static family vector, never a monthly weight schedule."""
    broad = np.zeros((3, 28, 15))
    compact = np.full_like(broad, 0.02)
    covariance = np.broadcast_to(np.eye(15) * 0.0004, (3, 28, 15, 15)).copy()
    mean, blended_covariance = recipes.blend_outputs(
        broad, compact, covariance, 4 * covariance, recipes.baseline_recipe()
    )
    expected = np.array([0.01] * 4 + [0.0] * 4 + [0.0075] * 4 + [0.005] * 3)
    np.testing.assert_allclose(mean, np.broadcast_to(expected, mean.shape), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(blended_covariance, 2.5 * covariance, rtol=1e-14, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(blended_covariance) >= 0)
    np.testing.assert_allclose(broad, np.zeros_like(broad), rtol=0, atol=1e-15)


def test_blend_rejects_invalid_inputs_without_repair() -> None:
    """Invalid fractions, shapes, covariance symmetry, and NaNs fail loudly."""
    mean = np.zeros((2, 15))
    covariance = np.broadcast_to(np.eye(15) * 0.001, (2, 15, 15)).copy()
    with pytest.raises(ValueError, match="fractions"):
        recipes.blend_outputs(mean, mean, covariance, covariance, {"alpha": [0, 0, 0, 2], "cov_compact_fraction": 0.5})
    with pytest.raises(ValueError, match="aligned"):
        recipes.blend_outputs(mean[:, :14], mean, covariance, covariance, recipes.baseline_recipe())
    covariance[0, 0, 1] = 1
    with pytest.raises(ValueError, match="symmetric"):
        recipes.blend_outputs(mean, mean, covariance, covariance, recipes.baseline_recipe())
    mean[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        recipes.blend_outputs(mean, mean, covariance, covariance, recipes.baseline_recipe())


def test_pareto_is_multiobjective_and_preserves_equal_vectors() -> None:
    """No objective is a guardrail or silently substituted for another."""
    records = [_record("a", 2, 0.1, 0.1), _record("b", 1, 0.2, 0.2), _record("c", 1, 0.1, 0.1)]
    records.append(_record("a_equal", 2, 0.1, 0.1))
    assert recipes.pareto_ids(records) == ["a", "a_equal", "b"]
    assert recipes.pareto_ids([]) == []


def test_selection_is_deterministic_bounded_and_keeps_control() -> None:
    """A dominated baseline remains an explicit control, not a hidden winner."""
    records = [
        _record("baseline", 0, 0, 0),
        _record("sharpe", 2, 0.1, 0.1),
        _record("return", 1, 0.3, 0.2),
        _record("rank", 1, 0.1, 0.5),
        _record("balanced", 1.8, 0.25, 0.4),
    ]
    selected = recipes.select_candidates(records, limit=3, always_include=("baseline",))
    assert selected[0] == "baseline"
    assert len(selected) == 3
    assert "balanced" in selected
    assert selected == recipes.select_candidates(list(reversed(records)), limit=3, always_include=("baseline",))
    assert set(selected[1:]).issubset(recipes.pareto_ids(records))


def test_selection_rejects_validation_and_full_window_scores() -> None:
    """Last-12 or all-28 outcomes cannot choose the claimed validation model."""
    for period in ("validation_last12", "exploratory_all28", None):
        record = _record("candidate", 1, 0.2, 0.1)
        record["period"] = period
        with pytest.raises(ValueError, match="tuning_first16"):
            recipes.select_candidates([record])


def test_selection_reports_incomplete_metrics_instead_of_silently_dropping() -> None:
    """Failures cannot quietly improve the candidate pool or objective means."""
    with pytest.raises(ValueError, match="nonfinite"):
        recipes.select_candidates([_record("failed", np.nan, 0, 0)])
    with pytest.raises(ValueError, match="unique"):
        recipes.pareto_ids([_record("same", 1, 0.2, 0.1), _record("same", 2, 0.3, 0.2)])
    with pytest.raises(ValueError, match="retained control"):
        recipes.select_candidates([_record("ok", 1, 0.2, 0.1)], always_include=("missing",))


def test_degenerate_frontier_is_stable_without_division_by_zero() -> None:
    """Equal and single-objective frontiers use deterministic lexical ties."""
    records = [_record("b", 1, 0.2, 0.1), _record("a", 1, 0.2, 0.1)]
    assert recipes.select_candidates(records) == ["a", "b"]
    assert recipes.select_candidates([]) == []
