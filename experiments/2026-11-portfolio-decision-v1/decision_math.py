"""Pure static family blends and predeclared November promotion rules.

No I/O or fitting. Predictions/realized returns are decimals, covariance
squared decimals, and weights fractions. The live policy pools seed moments
before one portfolio allocation, never averages seed-specific allocations.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

FAMILY = np.repeat(np.arange(4), [4, 4, 4, 3])
BLOCKS = [[0, 12], [12, 24], [24, 31], [31, 38], [38, 45], [45, 52]]
CONTROL_COUNTS = {
    "incumbent": [[32, 32, 0], [64, 0, 0], [40, 24, 0], [48, 16, 0]],
    "compromise": [[40, 24, 0], [64, 0, 0], [48, 16, 0], [60, 4, 0]],
    "prior48_sharpe": [[24, 38, 2], [63, 0, 1], [0, 4, 60], [1, 3, 60]],
    "prior48_return": [[19, 33, 12], [64, 0, 0], [0, 4, 60], [4, 1, 59]],
    "prior48_ranking": [[39, 25, 0], [54, 10, 0], [0, 6, 58], [0, 0, 64]],
}


def recipe(label: str, fractions: np.ndarray, *, control: bool = False) -> dict:
    """Validate four static three-expert family simplexes.

    Args:
        label: Human-readable hypothesis identifier.
        fractions: (4,3) nonnegative broad/compact/broad48 fractions summing to1.
        control: Whether this is one of the five protected comparison controls.

    Returns:
        Content-addressed recipe; identical coefficients share an identity.
    """
    values = np.asarray(fractions, dtype=float)
    if (values.shape != (4, 3) or not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Static four-family three-expert simplexes required")
    encoded = json.dumps(values.tolist(), separators=(",", ":")).encode()
    token = hashlib.sha256(encoded).hexdigest()[:20]
    return {"id": "recipe_" + token, "label": label, "fractions": values.tolist(), "control": control}


def candidates() -> list[dict]:
    """Return the bounded predefined controls and stable batch1 hypotheses.

    Returns:
        At most five controls plus48 new coefficient recipes after deduplication.
        This menu cannot adapt to historical scores or October outcomes.
    """
    controls = {name: np.array(counts, dtype=float) / 64 for name, counts in CONTROL_COUNTS.items()}
    result = [recipe(name, values, control=True) for name, values in controls.items()]
    base = controls["incumbent"]
    third_names = ["prior48_sharpe", "prior48_return", "prior48_ranking"]
    for name in third_names:
        for amount in [.25, .5, .75]:
            result.append(recipe(f"shrink_{name}_{amount}", (1 - amount) * base + amount * controls[name]))
    averages = []
    for size in [2, 3]:
        for names in combinations(third_names, size):
            mean = np.mean([controls[name] for name in names], axis=0)
            averages.append(mean)
            result.append(recipe("average_" + "_".join(names), mean))
    for amount in [.25, .5, .75]:
        result.append(recipe(f"shrink_average48_{amount}", (1 - amount) * base + amount * averages[-1]))
    result.append(recipe("average_incumbent_compromise", .5 * (base + controls["compromise"])))
    result.append(recipe("uniform_three_experts", np.full((4, 3), 1 / 3)))
    result.append(recipe("uniform_broad_compact", np.tile([.5, .5, 0], (4, 1))))
    for families in [[2], [3], [2, 3]]:
        for amount in [.25, .5]:
            updated = base.copy()
            updated[families] *= 1 - amount
            updated[families, 2] += amount
            result.append(recipe(f"FI48_{families}_{amount}", updated))
    unique = {}
    for item in result:
        unique.setdefault(item["id"], item)
    output = list(unique.values())
    if sum(not item["control"] for item in output) > 48:
        raise ValueError("Batch1 exceeds its registered limit")
    return output


def pool(means: np.ndarray, covariances: np.ndarray, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply family blending, then average seed moments before allocation.

    Args:
        means: (3 experts,3 seeds,months,15 assets) decimal forecasts.
        covariances: Matching expert/seed/month/15/15 observed covariance.
        fractions: (4,3) static prediction fractions.

    Returns:
        (months,15) means and (months,15,15) fixed50/50 pooled covariance.
    """
    recipe("validation", fractions)
    if means.shape[:2] != (3, 3) or means.shape[-1] != 15 or covariances.shape != (*means.shape, 15):
        raise ValueError("Exact three-expert/three-seed/15-ETF moments required")
    if not np.isfinite(means).all() or not np.isfinite(covariances).all():
        raise ValueError("Missing forecasts cannot be filled")
    mixed = np.einsum("ksta,ak->sta", means, fractions[FAMILY])
    covariance = (.5 * covariances[0] + .5 * covariances[1]).mean(0)
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
        raise ValueError("Covariance is not symmetric; no repair")
    np.linalg.cholesky(covariance)
    return mixed.mean(0), covariance


def metrics(actual: np.ndarray, rf: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> tuple[dict, dict]:
    """Score one pooled portfolio path without averaging seed portfolio statistics.

    Args:
        actual: (months,15) realized discounted excess returns, decimals.
        rf: (months,) monthly risk-free returns, decimals.
        mean: Matching pooled forecasts.
        weights: Matching raw allocator fractions.

    Returns:
        Summary metrics and complete monthly numerical evidence.
    """
    if actual.shape != mean.shape or actual.shape != weights.shape or rf.shape != (len(actual),):
        raise ValueError("Complete aligned portfolio arrays required")
    if not all(np.isfinite(x).all() for x in [actual, rf, mean, weights]):
        raise ValueError("Missing months cannot be dropped")
    total = (1 + np.sum(weights * actual, axis=1)) * (1 + rf) - 1
    excess = total - rf
    std = excess.std(ddof=1)
    if std <= 0 or (total <= -1).any():
        raise ValueError("Invalid return path")
    predicted, realized = rankdata(mean, axis=-1), rankdata(actual, axis=-1)
    predicted -= predicted.mean(-1, keepdims=True)
    realized -= realized.mean(-1, keepdims=True)
    denominator = np.sqrt(np.sum(predicted**2, -1) * np.sum(realized**2, -1))
    if (denominator <= 0).any():
        raise ValueError("Undefined ranking IC")
    ic = np.sum(predicted * realized, axis=1) / denominator
    wealth = 10000 * np.r_[1., np.cumprod(1 + total)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    family_ic = []
    for family in range(4):
        p, y = rankdata(mean[:, FAMILY == family], axis=-1), rankdata(actual[:, FAMILY == family], axis=-1)
        p -= p.mean(-1, keepdims=True)
        y -= y.mean(-1, keepdims=True)
        norm = np.sqrt(np.sum(p**2, -1) * np.sum(y**2, -1))
        if (norm <= 0).any():
            raise ValueError("Undefined family ranking IC")
        family_ic.append(float((np.sum(p * y, axis=1) / norm).mean()))
    return {"sharpe": float(np.sqrt(12) * excess.mean() / std), "rank_ic": float(ic.mean()),
            "total_return": float(wealth[-1] / 10000 - 1), "max_drawdown": float(drawdown.min()),
            "annualized_excess_vol": float(std * np.sqrt(12)), "family_ic": family_ic,
            "effective_assets": float(np.mean(1 / np.sum(weights**2, -1))),
            "equity_weight": float(weights[:, :8].sum(-1).mean())}, {
                "total_returns": total, "excess_returns": excess, "rank_ic": ic, "wealth": wealth,
                "family_weights": np.stack([weights[:, FAMILY == f].sum(-1) for f in range(4)], axis=-1)}


def promotion(candidate: dict, incumbent: dict) -> dict:
    """Apply every preregistered operational gate without altering thresholds.

    Args:
        candidate: Complete all52/later28/block/leave-one-out scored evidence.
        incumbent: Identically evaluated pooled incumbent evidence.

    Returns:
        Individual gate decisions and conservative eligibility, not significance.
    """
    a, b = candidate["all52"], incumbent["all52"]
    recent, reference = candidate["later28"], incumbent["later28"]
    loo = np.asarray(candidate["leave_one_out_sharpe"]) - incumbent["leave_one_out_sharpe"]
    block_advantage = np.asarray(candidate["block_mean_excess"]) - incumbent["block_mean_excess"]
    gates = {
        "all52_sharpe_improvement_0p05": a["sharpe"] >= b["sharpe"] + .05,
        "all52_IC_improvement_0p01": a["rank_ic"] >= b["rank_ic"] + .01,
        "all52_return_noninferior": a["total_return"] >= b["total_return"] - 1e-10,
        "later28_sharpe_noninferior_0p05": recent["sharpe"] >= reference["sharpe"] - .05,
        "later28_IC_noninferior_0p01": recent["rank_ic"] >= reference["rank_ic"] - .01,
        "later28_return_noninferior_1pp": recent["total_return"] >= reference["total_return"] - .01,
        "four_of_six_positive_blocks": int(np.count_nonzero(block_advantage >= -1e-12)) >= 4,
        "every_leave_one_month_out_sharpe_positive": bool(np.all(loo > 1e-10)),
        "drawdown_noninferior_1pp": a["max_drawdown"] >= b["max_drawdown"] - .01,
        "no_family_IC_loss_over_0p05": bool(np.all(np.array(a["family_ic"]) >= np.array(b["family_ic"]) - .05)),
    }
    return {"eligible": all(gates.values()), "gates": gates,
            "positive_blocks": int(np.count_nonzero(block_advantage >= -1e-12)),
            "minimum_leave_one_out_sharpe_advantage": float(loo.min())}
