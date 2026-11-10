"""Pure, bounded static recipes for the asymmetric-GP research track.

This experiment-only module performs no fits, allocation, file I/O, or selection
on validation outcomes. Means are next-month discounted excess returns in
decimal units; covariance entries have squared-decimal-return units. Every
generated recipe uses the same family fractions for every seed and month.

Usage: import ``screen_cases`` to enumerate the fixed-blend component screen,
then ``mean_retuning_cases`` and ``covariance_retuning_cases`` after selecting
on first-16-month metrics with ``select_candidates``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

FAMILIES = ("us_equity", "international_equity", "us_fixed_income", "international_fixed_income")
ASSETS = ("SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX", "BND", "IEF", "LQD", "HYG", "BNDX", "VWOB", "HYEM")
FAMILY_INDEX = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
RANKS = (1, 2, 5)
MAXITERS = (5, 15, 36, 75)
BASELINE_ALPHA = (0.5, 0.0, 0.375, 0.25)
COMPROMISE_ALPHA = (0.375, 0.0, 0.25, 0.0625)
OBJECTIVES = ("sharpe", "total_return", "rank_ic")
SELECTION_PERIOD = "tuning_first16"


def baseline_recipe() -> dict[str, Any]:
    """Return the approved, month-invariant family and covariance fractions.

    Returns:
        Recipe fractions; ``alpha`` is compact's weight in each of four means,
        and ``cov_compact_fraction`` is compact's covariance weight.
    """
    return {"alpha": list(BASELINE_ALPHA), "cov_compact_fraction": 0.5}


def _case(
    broad_id: str, compact_id: str, alpha: Sequence[float], covariance_fraction: float, stage: str
) -> dict[str, Any]:
    fractions = _fractions(alpha, 4)
    covariance_fraction = _fractions((covariance_fraction,), 1)[0]
    fraction_id = "_".join(f"{round(value * 10000):05d}" for value in fractions)
    return {
        "id": f"track2_{stage}__{broad_id}__{compact_id}__a{fraction_id}__c{round(covariance_fraction * 10000):05d}",
        "track": "track2_asymmetric",
        "stage": stage,
        "broad_id": broad_id,
        "compact_id": compact_id,
        "alpha": list(fractions),
        "cov_compact_fraction": covariance_fraction,
        "third_id": None,
        "third_fraction": 0.0,
        "third_cov_fraction": 0.0,
    }


def _fractions(values: Sequence[float], count: int) -> tuple[float, ...]:
    fractions = tuple(float(value) for value in values)
    if len(fractions) != count or any(not isfinite(value) or not 0 <= value <= 1 for value in fractions):
        raise ValueError(f"Expected {count} finite fractions in [0, 1].")
    return fractions


def screen_cases(component_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Enumerate all 144 component pairs at the unchanged approved blend.

    Args:
        component_ids: Available shared-registry component IDs. Additional
            third-model IDs are permitted but not used by this track.

    Returns:
        Case dictionaries in deterministic broad-rank/iteration, compact order.

    Raises:
        ValueError: Any of the 24 required components is absent.
    """
    expected = {
        family: [f"{family}_r{rank}_i{maxiter}_exp" for rank, maxiter in product(RANKS, MAXITERS)]
        for family in ("broad", "compact")
    }
    missing = sorted(set(expected["broad"] + expected["compact"]) - set(component_ids))
    if missing:
        raise ValueError(f"Missing required registry components: {missing}")
    return [
        _case(broad_id, compact_id, BASELINE_ALPHA, 0.5, "screen")
        for broad_id, compact_id in product(expected["broad"], expected["compact"])
    ]


def mean_retuning_cases(pair_case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generate 97 fixed family-mean recipes while holding covariance 50/50.

    The bounded menu includes the 81 combinations of 0, 1/2, and 1 for all
    families; the approved leader and compromise; and one-family perturbations
    of the approved leader by +/-1/16 and +/-1/8. Clipped duplicate recipes are
    removed. This is a prescribed menu, not an exhaustive continuous optimum.

    Args:
        pair_case: A screened case with the selected broad and compact IDs.

    Returns:
        Deterministically ordered case dictionaries, all month-invariant.
    """
    alpha_values = set(product((0.0, 0.5, 1.0), repeat=4))
    alpha_values.update((BASELINE_ALPHA, COMPROMISE_ALPHA))
    for family_index, delta in product(range(4), (-0.125, -0.0625, 0.0625, 0.125)):
        candidate = list(BASELINE_ALPHA)
        candidate[family_index] = min(1.0, max(0.0, candidate[family_index] + delta))
        alpha_values.add(tuple(candidate))
    return [
        _case(str(pair_case["broad_id"]), str(pair_case["compact_id"]), alpha, 0.5, "mean")
        for alpha in sorted(alpha_values)
    ]


def covariance_retuning_cases(mean_cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Generate four additional covariance recipes for each selected mean.

    Args:
        mean_cases: At most three first-16-month mean finalists from one
            selected component pair. Each has an already evaluated 50/50 case.

    Returns:
        Cases at compact covariance fractions 0, 1/4, 3/4, and 1. Means are
        unchanged, isolating covariance from prediction-blending changes.

    Raises:
        ValueError: More than three means, repeated means, or multiple pairs.
    """
    if len(mean_cases) > 3:
        raise ValueError("At most three mean finalists may be covariance-retuned per component pair.")
    pairs = {(case["broad_id"], case["compact_id"]) for case in mean_cases}
    alphas = {_fractions(case["alpha"], 4) for case in mean_cases}
    if len(pairs) > 1 or len(alphas) != len(mean_cases):
        raise ValueError("Covariance retuning requires distinct means from the same component pair.")
    return [
        _case(str(case["broad_id"]), str(case["compact_id"]), case["alpha"], fraction, "covariance")
        for case, fraction in product(mean_cases, (0.0, 0.25, 0.75, 1.0))
    ]


def blend_outputs(
    broad_mean: NDArray[np.float64],
    compact_mean: NDArray[np.float64],
    broad_covariance: NDArray[np.float64],
    compact_covariance: NDArray[np.float64],
    recipe: Mapping[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one fixed recipe to all seed/month forecasts without repair.

    Args:
        broad_mean: Decimal excess-return means, shape (..., 15).
        compact_mean: Matching decimal excess-return means, shape (..., 15).
        broad_covariance: Observed covariance including likelihood noise,
            shape (..., 15, 15), in squared-decimal-return units.
        compact_covariance: Matching compact observed covariance.
        recipe: Fixed ``alpha`` and ``cov_compact_fraction`` recipe fractions.

    Returns:
        Blended means and an arithmetic covariance pool in original units.
        No ensemble-disagreement term, independence assumption, jitter, or
        recalibration is introduced. Convex pooling preserves input PSD.

    Raises:
        ValueError: Shapes, fractions, symmetry, or finiteness are invalid.
    """
    broad_mean = np.asarray(broad_mean, dtype=np.float64)
    compact_mean = np.asarray(compact_mean, dtype=np.float64)
    broad_covariance = np.asarray(broad_covariance, dtype=np.float64)
    compact_covariance = np.asarray(compact_covariance, dtype=np.float64)
    covariance_shape = (*broad_mean.shape, 15)
    if (
        broad_mean.ndim < 1
        or broad_mean.shape[-1] != 15
        or compact_mean.shape != broad_mean.shape
        or broad_covariance.shape != covariance_shape
        or compact_covariance.shape != covariance_shape
    ):
        raise ValueError("Means/covariances must have aligned (..., 15) and (..., 15, 15) shapes.")
    arrays = (broad_mean, compact_mean, broad_covariance, compact_covariance)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Nonfinite model outputs cannot be blended.")
    for covariance in (broad_covariance, compact_covariance):
        if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-12):
            raise ValueError("Covariance must be symmetric; no automatic repair is permitted.")
    alpha = np.asarray(_fractions(recipe["alpha"], 4))[FAMILY_INDEX]
    fraction = _fractions((recipe["cov_compact_fraction"],), 1)[0]
    return (
        (1 - alpha) * broad_mean + alpha * compact_mean,
        (1 - fraction) * broad_covariance + fraction * compact_covariance,
    )


def _score_matrix(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], NDArray[np.float64]]:
    identifiers = [str(record["id"]) for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Candidate IDs must be unique.")
    values = np.array([[float(record[key]) for key in OBJECTIVES] for record in records], dtype=np.float64)
    values = values.reshape(len(records), len(OBJECTIVES))
    if not np.isfinite(values).all():
        raise ValueError("Incomplete or nonfinite candidates must be reported separately, not silently ranked.")
    return identifiers, values


def pareto_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return nondominated IDs, maximizing Sharpe, total return, and rank IC.

    Args:
        records: Unique-ID metric dictionaries with ``sharpe``,
            ``total_return`` (decimal), and ``rank_ic`` (unitless). Three-seed
            averaging must precede this function. Any period may be described.

    Returns:
        Nondominated IDs in lexical order. Equal vectors remain nondominated.

    Raises:
        ValueError: Duplicate IDs or nonfinite objective values.
    """
    identifiers, values = _score_matrix(records)
    return sorted(
        identifier
        for index, identifier in enumerate(identifiers)
        if not np.any(np.all(values >= values[index], axis=1) & np.any(values > values[index], axis=1))
    )


def select_candidates(
    records: Sequence[Mapping[str, Any]], *, limit: int = 3, always_include: Sequence[str] = ()
) -> list[str]:
    """Select a bounded diverse Pareto shortlist using first-16 months only.

    Args:
        records: Three-seed mean metrics with ``id``, ``sharpe``,
            ``total_return``, ``rank_ic``, and ``period='tuning_first16'``.
        limit: Maximum selected IDs, including explicitly retained controls.
        always_include: Existing IDs retained as controls, even if dominated.
            Use the original rank-2/maxiter-15 pair for the pair-selection step.

    Returns:
        Selected IDs. After controls, first admit the normalized-frontier point
        closest to the ideal, then use farthest-first coverage among frontier
        points. Break ties by ideal distance and lexical ID. Constant objective
        dimensions are ignored. No validation/full-window outcomes are read.

    Raises:
        ValueError: Records are not marked first-16-month tuning, IDs/metrics
            are invalid, or the control/limit arguments are inconsistent.
    """
    if limit < 1 or len(set(always_include)) != len(always_include) or len(always_include) > limit:
        raise ValueError("Positive limit and distinct controls fitting within that limit are required.")
    if any(record.get("period") != SELECTION_PERIOD for record in records):
        raise ValueError("Selection is restricted to metrics labeled tuning_first16.")
    identifiers, values = _score_matrix(records)
    if not set(always_include).issubset(identifiers):
        raise ValueError("Every retained control must have complete first-16-month metrics.")
    frontier = pareto_ids(records)
    selected = list(always_include)
    if not frontier:
        return selected
    indices = {identifier: index for index, identifier in enumerate(identifiers)}
    frontier_values = values[[indices[identifier] for identifier in frontier]]
    minimum = frontier_values.min(axis=0)
    span = np.ptp(frontier_values, axis=0)
    normalized = np.divide(frontier_values - minimum, span, out=np.zeros_like(frontier_values), where=span > 0)
    normalized_by_id = dict(zip(frontier, normalized, strict=True))
    ideal = np.where(span > 0, 1.0, 0.0)
    ideal_distance = {
        identifier: float(np.linalg.norm(point - ideal)) for identifier, point in normalized_by_id.items()
    }
    while len(selected) < limit:
        remaining = [identifier for identifier in frontier if identifier not in selected]
        if not remaining:
            break
        anchors = [normalized_by_id[identifier] for identifier in selected if identifier in normalized_by_id]
        if not anchors:
            chosen = min(remaining, key=lambda identifier: (ideal_distance[identifier], identifier))
        else:
            coverage = {
                identifier: min(float(np.linalg.norm(normalized_by_id[identifier] - anchor)) for anchor in anchors)
                for identifier in remaining
            }
            chosen = min(
                remaining, key=lambda identifier: (-coverage[identifier], ideal_distance[identifier], identifier)
            )
        selected.append(chosen)
    return selected
