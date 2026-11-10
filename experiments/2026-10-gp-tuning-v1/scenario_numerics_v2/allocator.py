"""Reusable sign-safe scenario allocator with no positive-return requirement.

Research numerical adapter: consumes unchanged decimal-return scenarios and
returns fraction weights. Negative observations, negative asset means and
negative optimal portfolio means are supported. Pure numerical code performs
no GP fits, downloads, outcome-dependent selection or portfolio fallback.
The versioned bridge retains the already-tested signed-negative solver and
replaces the unstable positive-Sharpe normalization at its source.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from case_evaluator import _dependency
from positive_sharpe import solve_positive


def allocate_scenarios(scenarios: np.ndarray, assets: list[str], *, seed: int) -> dict[str, Any]:
    """Maximize signed scenario Sharpe without requiring positive returns.

    Args:
        scenarios: (samples,15) finite decimal discounted-excess returns,
            including any number of negative returns or negative ETF means.
        assets: Fifteen unique ETF names in the exact scenario column order.
        seed: Deterministic optimizer-start seed for the nonpositive branch.

    Returns:
        Unmodified weights, numerical route, signed Sharpe, and raw audits.
        Constraints remain long-only/full-investment, effective assets >=10
        and cap25%. If any feasible portfolio has positive mean, its global
        optimum uses the scale-stable conic ratio formulation. Otherwise the
        original direct signed-ratio multistart solver optimizes even when
        every feasible portfolio has a negative mean. No return is clipped,
        shifted or filtered, and no minimum portfolio return is imposed.

    Raises:
        ValueError: Nonfinite or misaligned input, or failed raw audit.
        RuntimeError: The genuine optimization fails; no replacement weights.
    """
    scenarios = np.asarray(scenarios, dtype=float)
    if (scenarios.ndim != 2 or scenarios.shape[0] < 2 or scenarios.shape[1] != 15
            or len(assets) != 15 or len(set(assets)) != 15 or not np.isfinite(scenarios).all()):
        raise ValueError("Finite sample matrix and fifteen unique aligned ETF names required")
    mean, covariance = scenarios.mean(0), np.cov(scenarios, rowvar=False)
    signed = _dependency("scenarios")
    _, maximum_mean = signed._maximum_mean(mean)
    if maximum_mean <= 1e-9:
        result = signed.solve_signed(mean, covariance, seed=seed)
    else:
        result = solve_positive(mean, covariance)
    result["audit"] = _dependency("sampling").audit_weights(result["weights"])
    result["maximum_feasible_mean"] = maximum_mean
    result["negative_scenario_observations"] = int((scenarios < 0).sum())
    result["negative_asset_means"] = int((mean < 0).sum())
    result["minimum_portfolio_return_constraint"] = False
    return result


def evaluate_case(case: dict, output_root: str) -> dict:
    """Evaluate a NEW research case using the repaired allocator by default.

    Args:
        case: The frozen three-seed/28-origin evaluator's static recipe schema.
        output_root: New research directory containing context/component panels;
            never use a completed campaign's sealed directory.

    Returns:
        Immutable case metrics/arrays using the existing deterministic scoring
        and constraints, but with this allocator for all scenario allocations.
        Existing cases remain immutable and are not silently overwritten.
        Use one case per process, not shared threads (as in the campaign pool).
    """
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    import case_evaluator

    if (Path(output_root) / "COMPLETED.json").exists():
        raise ValueError("Completed campaign is sealed; use a new versioned output directory")
    original = case_evaluator._dependency
    original_inputs = case_evaluator._inputs
    bridge = SimpleNamespace(allocate_scenarios=allocate_scenarios)

    def _route(kind: str) -> Any:
        return bridge if kind == "scenarios" else original(kind)

    def _inputs(recipe: dict, root: Path) -> tuple:
        context, components, hashes = original_inputs(recipe, root)
        for path in [Path(__file__), Path(__file__).with_name("positive_sharpe.py")]:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return context, components, hashes

    # The allocator's captured original dependency function cannot recurse
    # through this process-local bridge. Hash the new numerical code as well.
    with patch.object(case_evaluator, "_dependency", _route), patch.object(case_evaluator, "_inputs", _inputs):
        return case_evaluator.evaluate_case(case, output_root)
