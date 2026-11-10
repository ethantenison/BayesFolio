"""Pure, unit-explicit portfolio and ranking metrics for November v2 research.

This experiment-only numerical module has no I/O, fitting, allocation, selection,
or date-block policy. Callers pass only the permitted period. Realized GP targets
are discounted excess decimals: total=(1+target)*(1+rf)-1. Weights are fractions,
wealth is dollars, and cost rates are basis points per gross traded dollar.
Ranking excludes MGK explicitly; all fifteen delivered weight columns remain.
Raw optimizer weights are accepted only at minimum >=-1e-8 and absolute sum
error <=1e-7, matching the independent validator. No clipping or normalization
is performed; residuals are recorded and the original fractions drive returns.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import rankdata

FAMILY_ASSETS = {
    "us_equity": ["SPY", "VTV", "IWM"],
    "international_equity": ["VEA", "VWO", "VSS", "EWX"],
    "us_fixed_income": ["BND", "IEF", "LQD", "HYG"],
    "international_fixed_income": ["BNDX", "VWOB", "HYEM"],
}
COST_BPS = (0, 5, 10, 20)
MIN_RAW_WEIGHT = -1e-8
MAX_WEIGHT_SUM_ERROR = 1e-7


def _finite_array(value: np.ndarray, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array; months cannot be dropped")
    return result


def _drawdown_path(wealth: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    high = np.maximum.accumulate(wealth)
    drawdown = wealth / high - 1
    episodes = []
    peak = 0
    active = None
    for index in range(1, len(wealth)):
        if wealth[index] >= high[index - 1] * (1 - 1e-13):
            if active is not None:
                active.update(recovery_index=index, duration_months=index - active["peak_index"], censored=False)
                episodes.append(active)
                active = None
            peak = index
        else:
            if active is None:
                active = {"peak_index": peak, "trough_index": index, "depth": float(drawdown[index])}
            if drawdown[index] < active["depth"]:
                active.update(trough_index=index, depth=float(drawdown[index]))
    if active is not None:
        active.update(recovery_index=None, duration_months=len(wealth) - 1 - active["peak_index"], censored=True)
        episodes.append(active)
    return drawdown, episodes


def score_returns(total_returns: np.ndarray, rf: np.ndarray, *, initial_wealth: float = 10000) -> tuple[dict, dict]:
    """Score one complete monthly return path, including its initial investment.

    Args:
        total_returns: (months,) decimal total returns, each greater than -1.
        rf: Matching monthly risk-free decimal returns. Conventional excess is
            total_returns-rf, not the discounted GP target.
        initial_wealth: Positive dollar value before the first observed return.

    Returns:
        Summary and complete wealth/drawdown/excess-return evidence. Undefined
        Sharpe at zero variance and unavailable twelve-month metrics are None.
        Recovery durations count months from prior peak to recovery or censoring.

    Raises:
        ValueError: Nonfinite, misaligned, empty or bankrupt return paths.
    """
    total = _finite_array(total_returns, 1, "Total returns")
    risk_free = _finite_array(rf, 1, "Risk-free returns")
    if not len(total) or risk_free.shape != total.shape or (total <= -1).any() or (risk_free <= -1).any():
        raise ValueError("Complete nonempty monthly paths with returns greater than -1 required")
    if not np.isfinite(initial_wealth) or initial_wealth <= 0:
        raise ValueError("Initial wealth must be positive")
    excess = total - risk_free
    wealth = initial_wealth * np.r_[1.0, np.cumprod(1 + total)]
    drawdown, episodes = _drawdown_path(wealth)
    sd = (0.0 if np.ptp(excess) == 0 else float(excess.std(ddof=1))) if len(total) > 1 else None
    total_sd = (0.0 if np.ptp(total) == 0 else float(total.std(ddof=1))) if len(total) > 1 else None
    deepest = min(episodes, key=lambda episode: episode["depth"]) if episodes else None
    rolling = np.lib.stride_tricks.sliding_window_view(total, 12) if len(total) >= 12 else np.empty((0, 12))
    rolling_returns = np.prod(1 + rolling, axis=1) - 1
    rolling_drawdowns = [float(_drawdown_path(np.r_[1., np.cumprod(1 + block)])[0].min()) for block in rolling]
    summary = {
        "months": len(total), "total_return": float(wealth[-1] / initial_wealth - 1),
        "cagr": float((wealth[-1] / initial_wealth) ** (12 / len(total)) - 1),
        "sharpe": float(np.sqrt(12) * excess.mean() / sd) if sd is not None and sd > 0 else None,
        "annualized_excess_vol": float(sd * np.sqrt(12)) if sd is not None else None,
        "annualized_total_vol": float(total_sd * np.sqrt(12)) if total_sd is not None else None,
        "max_drawdown": float(drawdown.min()), "worst_month": float(total.min()),
        "worst_12_month_return": float(rolling_returns.min()) if len(rolling) else None,
        "worst_12_month_drawdown": min(rolling_drawdowns) if len(rolling) else None,
        "max_drawdown_recovery_months": deepest["duration_months"] if deepest else 0,
        "max_drawdown_recovery_censored": deepest["censored"] if deepest else False,
        "longest_underwater_months": max((episode["duration_months"] for episode in episodes), default=0),
        "underwater_at_end": bool(episodes and episodes[-1]["censored"]),
        "ongoing_underwater_months": episodes[-1]["duration_months"] if episodes and episodes[-1]["censored"] else 0,
    }
    return summary, {"total_returns": total.copy(), "excess_returns": excess, "wealth": wealth,
                     "drawdown": drawdown, "drawdown_episodes": episodes,
                     "rolling_12_month_returns": rolling_returns,
                     "rolling_12_month_drawdowns": np.asarray(rolling_drawdowns)}


def _spearman_rows(actual: np.ndarray, mean: np.ndarray) -> np.ndarray:
    predicted, realized = rankdata(mean, axis=1), rankdata(actual, axis=1)
    predicted -= predicted.mean(1, keepdims=True)
    realized -= realized.mean(1, keepdims=True)
    denominator = np.sqrt(np.sum(predicted**2, 1) * np.sum(realized**2, 1))
    if (denominator == 0).any():
        raise ValueError("Ranking IC is undefined for a constant monthly vector; no zero-fill or month dropping")
    return np.sum(predicted * realized, 1) / denominator


def ranking_diagnostics(actual: np.ndarray, mean: np.ndarray, *, assets: list[str],
                        families: dict[str, list[str]] | None = None) -> tuple[dict, dict]:
    """Compute cross-sectional and within-family Spearman IC excluding MGK.

    Args:
        actual: (months,15) discounted excess decimal realized targets.
        mean: Matching pooled decimal forecasts.
        assets: Unique aligned delivered asset names, including MGK exactly once.
        families: Explicit family mapping; MGK, if present in the U.S. equity
            family, is removed before ranking. Eligible sizes must be 3/4/4/3.
            Defaults to the approved named mapping; families cannot overlap.

    Returns:
        Mean monthly pooled/family IC, membership and the complete monthly IC
        arrays. Ties use average ranks; constant rows raise instead of being hidden.

    Raises:
        ValueError: Invalid shapes, membership, nonfinite values or undefined IC.
    """
    observed = _finite_array(actual, 2, "Realized targets")
    predicted = _finite_array(mean, 2, "Pooled forecasts")
    supplied = FAMILY_ASSETS if families is None else families
    if (set(supplied) != set(FAMILY_ASSETS)
            or sum(members.count("MGK") for members in supplied.values()) > 1
            or any("MGK" in members for name, members in supplied.items() if name != "us_equity")):
        raise ValueError("The four named ETF families and MGK membership only in U.S. equity are required")
    groups = {name: [asset for asset in supplied[name] if asset != "MGK"] for name in FAMILY_ASSETS}
    eligible = [asset for members in groups.values() for asset in members]
    if (observed.shape != predicted.shape or observed.shape[1] != len(assets) or len(assets) != 15
            or len(set(assets)) != 15 or assets.count("MGK") != 1 or len(set(eligible)) != 14
            or len(eligible) != 14 or "MGK" in eligible or set(eligible) != set(assets) - {"MGK"}
            or list(map(len, groups.values())) != [3, 4, 4, 3]):
        raise ValueError("Explicit aligned 15-ETF columns and disjoint eligible 3/4/4/3 family membership required")
    indices = [assets.index(asset) for asset in eligible]
    pooled_ic = _spearman_rows(observed[:, indices], predicted[:, indices])
    family_ic = {name: _spearman_rows(observed[:, [assets.index(asset) for asset in members]],
                                    predicted[:, [assets.index(asset) for asset in members]])
                 for name, members in groups.items()}
    return {"rank_ic": float(pooled_ic.mean()), "family_ic": {key: float(value.mean())
                                                            for key, value in family_ic.items()},
            "ranking_assets": eligible.copy(), "family_assets": {key: list(value) for key, value in groups.items()}}, {
                "rank_ic": pooled_ic, "family_ic": family_ic}


def _trading_cost_path(asset_returns: np.ndarray, weights: np.ndarray, fee: float) -> dict:
    months = len(weights)
    net = np.empty(months)
    notional = np.empty(months)
    fraction = np.empty(months)
    drift = np.zeros_like(weights)
    for month in range(months):
        if month:
            holdings = weights[month - 1] * (1 + asset_returns[month - 1])
            drift[month] = holdings / holdings.sum()
        if fee == 0:
            fraction[month] = 1
        else:
            fraction[month] = brentq(
                lambda value, m=month: value + fee * np.abs(value * weights[m] - drift[m]).sum() - 1,
                0, 1, xtol=5e-15, rtol=1e-14)
        notional[month] = np.abs(fraction[month] * weights[month] - drift[month]).sum()
        net[month] = fraction[month] * (1 + np.dot(weights[month], asset_returns[month])) - 1
    return {"total_returns": net, "post_cost_wealth_fraction": fraction,
            "traded_notional_fraction": notional, "cost_fraction": fee * notional,
            "pretrade_asset_fractions": drift}


def score_portfolio(actual: np.ndarray, rf: np.ndarray, mean: np.ndarray, weights: np.ndarray, *,
                    assets: list[str], families: dict[str, list[str]] | None = None,
                    cost_bps: tuple[int, ...] = COST_BPS) -> tuple[dict, dict]:
    """Score a pooled, single-allocation portfolio and predeclared cost diagnostics.

    Args:
        actual: (months,15) discounted excess decimal realized targets.
        rf: (months,) decimal risk-free returns.
        mean: Matching pooled forecasts, scored on fourteen eligible ETFs only.
        weights: (months,15) raw fractions; minimum >=-1e-8 and absolute sum error
            <=1e-7, matching the independent validator. MGK is exactly zero.
            Accepted numerical residuals remain unmodified in every calculation.
        assets: Aligned unique asset labels including MGK; names determine families.
        families: Explicit approved mapping, with MGK removed from rankings,
            or the named default. Remaining family sizes are 3/4/4/3.
        cost_bps: Diagnostic cost cases per gross traded dollar; costs are never
            model-selection objectives. The default menu is 0/5/10/20 basis points.

    Returns:
        Gross return/ranking/diversification/family summaries, cost-case summaries,
        and complete monthly numerical paths. Family arithmetic return
        contributions sum to gross portfolio return, not to compounded return.
        Costs charge the initial cash-to-assets purchase and subsequent drift-aware
        purchases plus sales, with no half-turnover factor. A scalar self-financing
        solve pays costs before investing in the unmodified target fractions,
        retaining the same stated raw numerical acceptance tolerances.

    Raises:
        ValueError: Missing/alignment errors, invalid weights/returns or membership.
    """
    observed = _finite_array(actual, 2, "Realized targets")
    risk_free = _finite_array(rf, 1, "Risk-free returns")
    target = _finite_array(weights, 2, "Portfolio weights")
    if (observed.shape != target.shape or risk_free.shape != (len(observed),) or not len(observed)
            or (observed < -1).any() or (risk_free <= -1).any() or (target < MIN_RAW_WEIGHT).any()
            or (np.abs(target.sum(1) - 1) > MAX_WEIGHT_SUM_ERROR).any()):
        raise ValueError("Complete valid returns and long-only fully invested weights required")
    rank_summary, rank_paths = ranking_diagnostics(observed, mean, assets=assets, families=families)
    if np.any(target[:, assets.index("MGK")] != 0):
        raise ValueError("MGK must have exactly zero delivered portfolio weight")
    if (len(set(cost_bps)) != len(cost_bps) or not cost_bps
            or any(not np.isfinite(value) or value < 0 or value >= 10000 for value in cost_bps)):
        raise ValueError("Unique cost cases in [0,10000) basis points required")
    asset_returns = (1 + observed) * (1 + risk_free[:, None]) - 1
    total = np.sum(target * asset_returns, 1)
    summary, paths = score_returns(total, risk_free)
    summary.update(rank_summary)
    effective = 1 / np.sum(target**2, 1)
    group_indices = {name: [assets.index(asset) for asset in members]
                     for name, members in rank_summary["family_assets"].items()}
    family_weights = {name: target[:, indices].sum(1) for name, indices in group_indices.items()}
    family_contributions = {name: (target[:, indices] * asset_returns[:, indices]).sum(1)
                            for name, indices in group_indices.items()}
    np.testing.assert_allclose(np.sum(list(family_contributions.values()), axis=0), total, rtol=0, atol=1e-12)
    summary.update({"effective_assets": float(effective.mean()), "minimum_effective_assets": float(effective.min()),
                    "raw_minimum_weight": float(target.min()),
                    "raw_maximum_weight_sum_error": float(np.max(np.abs(target.sum(1) - 1))),
                    "raw_weight_tolerances": {"minimum_weight": MIN_RAW_WEIGHT,
                                              "maximum_weight_sum_error": MAX_WEIGHT_SUM_ERROR},
                    "family_weights": {key: float(value.mean()) for key, value in family_weights.items()},
                    "family_mean_monthly_return_contribution": {key: float(value.mean())
                                                                for key, value in family_contributions.items()}})
    costs = {}
    cost_paths = {}
    for bps in cost_bps:
        details = _trading_cost_path(asset_returns, target, bps / 10000)
        scored, _ = score_returns(details["total_returns"], risk_free)
        key = str(bps)
        costs[key] = {**scored, "bps_per_traded_dollar": bps,
                      "mean_traded_notional_fraction": float(details["traded_notional_fraction"].mean()),
                      "sum_cost_fractions_of_pretrade_wealth": float(details["cost_fraction"].sum())}
        cost_paths[key] = details
    summary["cost_cases"] = costs
    paths.update(rank_paths)
    paths.update({"asset_total_returns": asset_returns, "effective_assets": effective,
                  "family_weights": family_weights, "family_arithmetic_return_contributions": family_contributions,
                  "cost_cases": cost_paths})
    return summary, paths


def paired_block_bootstrap(candidate_returns: np.ndarray, reference_returns: np.ndarray, rf: np.ndarray, *,
                           candidate_ic: np.ndarray, reference_ic: np.ndarray,
                           block_length: int, draws: int, seed: int) -> tuple[dict, dict]:
    """Describe paired policy differences under a fixed circular block resampling.

    Args:
        candidate_returns: (months,) complete candidate decimal total returns.
        reference_returns: Matching reference total returns for the same dates.
        rf: Matching monthly risk-free decimal returns.
        candidate_ic: Matching monthly fourteen-asset candidate ranking IC.
        reference_ic: Matching reference monthly ranking IC.
        block_length: Caller-preregistered consecutive months per circular block.
        draws: Caller-preregistered positive number of bootstrap replicates.
        seed: Explicit deterministic local NumPy seed.

    Returns:
        Point differences and 2.5/50/97.5 percentiles, positive-difference fractions
        and undefined-Sharpe counts, plus sampled indices and difference arrays.
        Positive differences favor the candidate for every reported metric,
        including negative-valued drawdown. These are retrospective resampling
        diagnostics, not calibrated future probabilities or a pristine holdout.

    Raises:
        ValueError: Incomplete, invalid, unpaired or undersized arrays/settings.
    """
    candidate = _finite_array(candidate_returns, 1, "Candidate returns")
    reference = _finite_array(reference_returns, 1, "Reference returns")
    risk_free = _finite_array(rf, 1, "Risk-free returns")
    candidate_rank = _finite_array(candidate_ic, 1, "Candidate monthly IC")
    reference_rank = _finite_array(reference_ic, 1, "Reference monthly IC")
    months = len(candidate)
    if (months < 2 or any(value.shape != candidate.shape for value in [reference, risk_free,
                                                                    candidate_rank, reference_rank])
            or isinstance(block_length, bool) or not isinstance(block_length, int) or not 1 <= block_length <= months
            or isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0):
        raise ValueError("Complete paired arrays, positive draws and a block length within the panel required")
    point_candidate, _ = score_returns(candidate, risk_free)
    point_reference, _ = score_returns(reference, risk_free)
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, months, size=(draws, (months + block_length - 1) // block_length))
    indices = ((starts[..., None] + np.arange(block_length)) % months).reshape(draws, -1)[:, :months]

    def replicates(returns: np.ndarray, ranks: np.ndarray) -> dict[str, np.ndarray]:
        sampled = returns[indices]
        excess = sampled - risk_free[indices]
        gross = np.prod(1 + sampled, axis=1)
        wealth = np.column_stack([np.ones(draws), np.cumprod(1 + sampled, axis=1)])
        drawdown = wealth / np.maximum.accumulate(wealth, axis=1) - 1
        sd = excess.std(axis=1, ddof=1)
        sd[np.ptp(excess, axis=1) == 0] = 0
        sharpe = np.full(draws, np.nan)
        np.divide(np.sqrt(12) * excess.mean(1), sd, out=sharpe, where=sd > 0)
        return {"cagr": gross ** (12 / months) - 1, "total_return": gross - 1,
                "max_drawdown": drawdown.min(1), "sharpe": sharpe, "rank_ic": ranks[indices].mean(1)}

    candidate_draws = replicates(candidate, candidate_rank)
    reference_draws = replicates(reference, reference_rank)
    differences = {key: candidate_draws[key] - reference_draws[key] for key in candidate_draws}
    summary = {"method": "paired_circular_moving_block", "months": months, "block_length": block_length,
               "draws": draws, "seed": seed,
               "difference_direction": "candidate_minus_reference; positive favors candidate",
               "interpretation": "Retrospective sensitivity; empirical fractions are not future win probabilities",
               "metrics": {}}
    for key, values in differences.items():
        valid = np.isfinite(values)
        quantiles = np.quantile(values[valid], [.025, .5, .975]).tolist() if valid.any() else [None, None, None]
        if key == "rank_ic":
            point = float((candidate_rank - reference_rank).mean())
        else:
            left, right = point_candidate[key], point_reference[key]
            point = left - right if left is not None and right is not None else None
        summary["metrics"][key] = {
            "point_difference": point, "p025": quantiles[0], "median": quantiles[1], "p975": quantiles[2],
            "fraction_above_zero": float(np.mean(values[valid] > 0)) if valid.any() else None,
            "undefined_replicates": int((~valid).sum()), "defined_replicates": int(valid.sum())}
    return summary, {"sampled_indices": indices, "differences": differences}
