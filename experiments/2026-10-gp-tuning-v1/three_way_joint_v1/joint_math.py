"""Pure fixed three-component blend search and fast certified hard-rank math.

Mean weights are four independent broad/compact/third simplexes; covariance
weights are one independent three-component simplex. All weights are static.
Returns are decimals, covariance decimal squared and portfolio weights fractions.
No I/O, fitting, later-period selection, portfolio fallback or covariance repair.
The analytical hard-rank solution is accepted only if feasible for the unchanged
variance ceiling; otherwise the original conic objective is solved exactly.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from itertools import permutations, product

import cvxpy as cp
import numpy as np
from scipy.stats import qmc, rankdata

FAMILY = np.repeat(np.arange(4), [4, 4, 4, 3])
DENOMINATOR = 64
BASE = np.array([[32, 32, 0], [64, 0, 0], [40, 24, 0], [48, 16, 0]])
REFINED = np.array([[32, 32, 0], [64, 0, 0], [40, 24, 0], [52, 12, 0]])
COMPROMISE = np.array([[40, 24, 0], [64, 0, 0], [48, 16, 0], [60, 4, 0]])
FIXED_COV = [32, 32, 0]


def _simplex_counts(points: np.ndarray) -> np.ndarray:
    raw = points / points.sum(axis=-1, keepdims=True) * DENOMINATOR
    counts = np.floor(raw).astype(int)
    flat, fractions = counts.reshape(-1, 3), (raw - counts).reshape(-1, 3)
    for row, fraction in zip(flat, fractions, strict=True):
        remainder = DENOMINATOR - int(row.sum())
        row[np.argsort(-fraction, kind="stable")[:remainder]] += 1
    return counts


def recipe(model: str, counts: np.ndarray, covariance: list[int] | np.ndarray = FIXED_COV) -> dict:
    """Build a canonical static recipe with exact binary-fraction coefficients.

    Args:
        model: Frozen third-component identifier, never a date or seed rule.
        counts: (4,3) nonnegative integers summing to 64 in every family.
        covariance: Three nonnegative integers summing to 64, independent of means.

    Returns:
        Content-addressed JSON recipe. Zero weights are allowed, never forced.
    """
    counts, covariance = np.asarray(counts), np.asarray(covariance)
    if (counts.shape != (4, 3) or covariance.shape != (3,) or (counts < 0).any() or (covariance < 0).any()
            or not np.all(counts == counts.astype(int)) or not np.all(covariance == covariance.astype(int))
            or not np.all(counts.sum(-1) == 64) or covariance.sum() != 64):
        raise ValueError("Four independent three-component simplexes and one covariance simplex required")
    data = {"third_id": model, "mean_counts": counts.astype(int).tolist(),
            "covariance_counts": covariance.astype(int).tolist()}
    identity = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:20]
    return {"id": f"joint_{identity}", **data}


def coarse_recipes(model: str) -> list[dict]:
    """Cover the joint four-family product simplex, not a fixed two-model ratio.

    Args:
        model: One of the six frozen third GP candidates.

    Returns:
        1296 half-grid combinations, 1024 deterministic Sobol interior recipes,
        controls and former nested blends, deduplicated by exact coefficients.
    """
    simplex = [(a, b, 64 - a - b) for a in [0, 32, 64] for b in [0, 32, 64] if a + b <= 64]
    all_counts = [np.array(rows) for rows in product(simplex, repeat=4)]
    uniform = qmc.Sobol(12, scramble=True, seed=20261025).random_base2(10).reshape(-1, 4, 3)
    all_counts.extend(_simplex_counts(-np.log(uniform)))
    for control in [BASE, REFINED, COMPROMISE]:
        all_counts.append(control)
        for gamma in [.125, .25, .5]:
            point = np.rint((1 - gamma) * control).astype(int)
            point[:, 2] = 64 - point[:, :2].sum(-1)
            all_counts.append(point)
    return sorted({r["id"]: r for r in [recipe(model, c) for c in all_counts]}.values(), key=lambda r: r["id"])


def local_recipes(centers: list[dict], step: int, *, seed: int) -> list[dict]:
    """Refine all component/family weights with mass transfers and joint probes.

    Args:
        centers: First16-only diverse multiobjective beam, at most six recipes.
        step: Transfer size in 1/64 units; use 8,4,2,1 in successive rounds.
        seed: Fixed proposal RNG seed; independent of later outcomes.

    Returns:
        Feasible same-family transfers plus 32 multi-family simultaneous probes
        per center. Broad:compact ratios and third fractions can all change.
    """
    if step not in [8, 4, 2, 1] or len(centers) > 6:
        raise ValueError("Bounded six-center refinement with declared step required")
    rng = np.random.default_rng(seed)
    result = {}
    for center in centers:
        counts = np.array(center["mean_counts"])
        proposals = []
        for family, (donor, receiver) in product(range(4), permutations(range(3), 2)):
            if counts[family, donor] >= step:
                updated = counts.copy()
                updated[family, donor] -= step
                updated[family, receiver] += step
                proposals.append(updated)
        for _ in range(32):
            updated = counts.copy()
            for family in range(4):
                donor, receiver = rng.choice(3, 2, replace=False)
                amount = min(step, int(updated[family, donor]))
                updated[family, donor] -= amount
                updated[family, receiver] += amount
            proposals.append(updated)
        for counts in proposals:
            r = recipe(center["third_id"], counts)
            result[r["id"]] = r
    return sorted(result.values(), key=lambda r: r["id"])


def covariance_recipes(centers: list[dict]) -> list[dict]:
    """Jointly vary the three PSD covariance-pooling weights, holding means fixed.

    Args:
        centers: At most three first16-selected mean recipes per third model.

    Returns:
        Complete eighth-grid triangle plus 64 deterministic Sobol covariance
        simplexes per mean. Weights are global/static, not ETF-specific matrices.
    """
    if len(centers) > 3:
        raise ValueError("At most three first16 prediction finalists per model")
    choices = [[a, b, 64 - a - b] for a in range(0, 65, 8) for b in range(0, 65 - a, 8)]
    uniform = qmc.Sobol(3, scramble=True, seed=20261026).random_base2(6)
    choices.extend(_simplex_counts(-np.log(uniform)).tolist())
    result = {}
    for center, covariance in product(centers, choices):
        r = recipe(center["third_id"], np.array(center["mean_counts"]), covariance)
        result[r["id"]] = r
    return sorted(result.values(), key=lambda r: r["id"])


def blend(means: np.ndarray, covariances: np.ndarray, case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply independent family prediction and global covariance simplexes.

    Args:
        means: (3 components,3 seeds,months,15 ETFs), decimal returns.
        covariances: (3 components,3 seeds,months,15,15), observed covariance.
        case: Canonical static recipe; no date/seed-indexed weights accepted.

    Returns:
        Blended means and convex arithmetic observed covariance. This pooling
        recipe does not claim known cross-model forecast-error covariance.
    """
    recipe(case["third_id"], np.array(case["mean_counts"]), case["covariance_counts"])
    if means.shape[0:2] != (3, 3) or means.shape[-1] != 15 or covariances.shape != (*means.shape, 15):
        raise ValueError("Three aligned component/seed/month/ETF panels required")
    fractions = np.array(case["mean_counts"], dtype=float)[FAMILY] / 64
    mean = np.einsum("ksta,ak->sta", means, fractions)
    covariance = np.einsum("kstab,k->stab", covariances, np.array(case["covariance_counts"]) / 64)
    return mean, covariance


def _linear_rank_optimum(scores: np.ndarray) -> np.ndarray:
    # KKT optimum on simplex intersected with ||w||^2<=1/7. The 1e-8
    # concentration tie-break is constant on that sphere. If >=7 top scores
    # tie, their equal-weight portfolio is the exact minimum-concentration tie.
    order = np.argsort(-scores, kind="stable")
    for count in range(15, 6, -1):
        values = scores[order[:count]]
        centered = values - values.mean()
        squared = float(centered @ centered)
        if squared < 1e-26:
            if count < 15 and scores[order[count]] > values.mean():
                continue
            local = np.full(count, 1 / count)
        else:
            if count == 7:
                continue
            slope = np.sqrt((1 / 7 - 1 / count) / squared)
            local = 1 / count + slope * centered
            threshold = values.mean() - 1 / (count * slope)
            if local.min() < -1e-12 or (count < 15 and scores[order[count]] > threshold + 1e-12):
                continue
        weights = np.zeros(15)
        weights[order[:count]] = local
        return weights
    raise ValueError("Analytical hard-rank KKT certificate failed")


class RankAllocator:
    """Cached conic template plus exact analytical solution when risk is slack.

    Attributes:
        constrained_solves: Number of genuinely risk-constrained conic solves.
        analytical_solves: Number of KKT-certified feasible analytical solves.
    """

    def __init__(self) -> None:
        """Build the original objective and constraints once per worker."""
        self.constrained_solves = 0
        self.analytical_solves = 0
        self._canonical = _linear_rank_optimum(np.arange(1, 16) / 15)
        self._w = cp.Variable(15)
        self._scores = cp.Parameter(15)
        self._factor = cp.Parameter((15, 15))
        self._problem = cp.Problem(cp.Maximize(self._scores @ self._w - 1e-8 * cp.sum_squares(self._w)), [
            self._w >= 0, cp.sum(self._w) == 1, cp.norm(self._factor @ self._w) <= 1,
            cp.norm(self._w) <= 1 / np.sqrt(7),
        ])

    def allocate(self, mean: np.ndarray, covariance: np.ndarray, budget: np.ndarray) -> tuple[np.ndarray, int]:
        """Allocate seed/month panels under exactly the original hard-rank policy.

        Args:
            mean: (3,months,15) finite decimal predictions.
            covariance: Matching (3,months,15,15) observed covariance, SPD.
            budget: (months,) positive training-only variance ceilings.

        Returns:
            Unmodified weights and the count of binding-risk conic solves.
            No previous-month weights, clipping or constraint relaxation.
        """
        if (mean.ndim != 3 or mean.shape[0] != 3 or mean.shape[-1] != 15
                or covariance.shape != (*mean.shape, 15) or budget.shape != (mean.shape[1],)
                or not np.isfinite(mean).all() or not np.isfinite(covariance).all() or (budget <= 0).any()):
            raise ValueError("Complete aligned seed/month panels and positive budgets required")
        ranks = rankdata(mean, axis=-1)
        weights = self._canonical[(ranks - 1).astype(int)]
        tied = np.any(np.sort(ranks, axis=-1) != np.arange(1, 16), axis=-1)
        for s, t in np.argwhere(tied):
            weights[s, t] = _linear_rank_optimum(ranks[s, t] / 15)
        variances = np.einsum("sta,stab,stb->st", weights, covariance, weights)
        constrained = variances > budget[None, :] * (1 + 1e-12)
        for s, t in np.argwhere(constrained):
            self._scores.value = (ranks[s, t] - ranks[s, t].mean()) / 15
            self._factor.value = np.linalg.cholesky(covariance[s, t] / budget[t]).T
            succeeded = False
            for step, regularize in [(.9, True), (.8, False), (.95, True), (.99, True)]:
                settings = {"max_step_fraction": step, "static_regularization_enable": regularize}
                if regularize:
                    settings["static_regularization_constant"] = 1e-12
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        self._problem.solve(solver="CLARABEL", enforce_dpp=True, warm_start=False,
                                            tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, **settings)
                    except cp.error.SolverError:
                        continue
                if self._problem.status == cp.OPTIMAL and self._w.value is not None:
                    weights[s, t] = np.asarray(self._w.value).copy()
                    succeeded = True
                    break
            if not succeeded:
                if self._problem.status not in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
                    self._problem.solve(solver="SCS", enforce_dpp=True, warm_start=False,
                                        eps_abs=1e-10, eps_rel=1e-10, max_iters=100000)
                    if self._problem.status == cp.OPTIMAL and self._w.value is not None:
                        weights[s, t] = np.asarray(self._w.value).copy()
                        succeeded = True
                if not succeeded:
                    raise RuntimeError(f"Strict hard-rank solve failed: {self._problem.status}; no fallback")
        variance = np.einsum("sta,stab,stb->st", weights, covariance, weights)
        if (weights.min() < -1e-8 or np.max(abs(weights.sum(-1) - 1)) > 1e-7
                or np.max(np.sum(weights**2, axis=-1)) > 1 / 7 + 1e-7
                or np.max(variance / budget[None, :]) > 1 + 1e-6):
            raise ValueError("Raw portfolio weights violate unchanged policy")
        count = int(constrained.sum())
        self.constrained_solves += count
        self.analytical_solves += constrained.size - count
        return weights, count


def tuning_scores(actual: np.ndarray, rf: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> dict:
    """Score first16 only using the framework's conventional-Sharpe convention.

    Args:
        actual: (16,15) decimal realized discounted-excess returns.
        rf: (16,) monthly risk-free returns, decimals.
        mean: (3,16,15) predictions, decimals.
        weights: Matching portfolio fractions.

    Returns:
        Three-seed mean Sharpe, compounded total return and cross-sectional IC.
        No later-period data are accepted, even incidentally.
    """
    if actual.shape != (16, 15) or rf.shape != (16,) or mean.shape != (3, 16, 15) or weights.shape != mean.shape:
        raise ValueError("Tuning may only access the first sixteen months")
    total = (1 + np.einsum("sta,ta->st", weights, actual)) * (1 + rf) - 1
    excess = total - rf
    std = excess.std(axis=1, ddof=1)
    if not np.isfinite(total).all() or (std <= 0).any() or (total <= -1).any():
        raise ValueError("Undefined primary tuning objectives")
    predicted = rankdata(mean, axis=-1)
    predicted -= predicted.mean(-1, keepdims=True)
    realized = rankdata(actual, axis=-1)
    realized -= realized.mean(-1, keepdims=True)
    ic = np.einsum("sta,ta->st", predicted, realized) / np.sqrt(
        np.sum(predicted**2, axis=-1) * np.sum(realized**2, axis=-1))
    return {"period": "tuning_first16", "sharpe": float((np.sqrt(12) * excess.mean(1) / std).mean()),
            "total_return": float((np.prod(1 + total, axis=1) - 1).mean()), "rank_ic": float(ic.mean())}
