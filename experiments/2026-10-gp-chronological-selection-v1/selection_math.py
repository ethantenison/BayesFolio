"""Pure bounded GP catalog and earlier-only static blend selection arithmetic.

No I/O or model fitting. Inputs are causal GP forecast ledgers in decimal
discounted-excess returns; covariance/variance use squared-decimal units.
Selection accepts only the earlier ledger supplied by the caller. All family
coefficients are constant across dates/seeds within a declared test block.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

FAMILY = np.repeat(np.arange(4), [4, 4, 4, 3])
OBJECTIVES = ["sharpe", "total_return", "rank_ic"]
RICH_ETF = ["lag_y_excess_lead", "baspread", "ret_kurt", "chmom", "mom12m", "mom36m",
            "cs_mom_rank", "max_dd_6m", "ma_signal", "ret_autocorr", "vol_z"]
RICH_MACRO = ["hy_spread_z_12p", "vix_slope", "vix_ts_z_12p", "vix", "spy_flow_z_12p", "spy_ret",
              "erp", "cpi_chg_12p", "cpi_chg_1p", "copper_ret", "oil_ret", "gold_crude_ratio",
              "pct_above_50dma", "em_fx_ret"]
REMOVALS = {
    "risk_liquidity": ["baspread", "ret_kurt", "max_dd_6m", "ret_autocorr", "vol_z"],
    "long_momentum": ["mom36m"],
    "breadth_flow": ["cs_mom_rank", "spy_flow_z_12p", "pct_above_50dma"],
    "macro_regime": ["vix", "vix_slope", "vix_ts_z_12p", "erp", "cpi_chg_12p", "cpi_chg_1p"],
    "commodities_fx": ["copper_ret", "oil_ret", "gold_crude_ratio", "em_fx_ret"],
    "lag": ["lag_y_excess_lead"],
}


def catalog() -> list[dict]:
    """Return the preregistered nineteen feature/setting/window hypotheses.

    Returns:
        Specs with source mask, explicit optional columns, rank/iteration cap,
        rolling count and an exact reusable panel ID when available.
    """
    result = []
    for source in ["broad", "compact"]:
        for rank in [1, 2]:
            for iterations in [5, 15]:
                name = f"{source}_r{rank}_i{iterations}_exp"
                result.append(dict(id=name, source=source, rank=rank, maxiter=iterations,
                                   rolling_months=None, reuse=name))
    result.append(dict(id="broad_roll48_r2_i15", source="broad", rank=2, maxiter=15,
                       rolling_months=48, reuse="broad_roll48_r2_i15"))
    for name, rank, iterations, rolling in [("rich_r2_i15_exp", 2, 15, None),
                                          ("rich_r1_i15_exp", 1, 15, None),
                                          ("rich_r2_i5_exp", 2, 5, None),
                                          ("rich_roll48_r2_i15", 2, 15, 48)]:
        result.append(dict(id=name, source="broad", rank=rank, maxiter=iterations,
                           rolling_months=rolling, reuse=None, etf=RICH_ETF, macro=RICH_MACRO))
    for label, removed in REMOVALS.items():
        result.append(dict(id=f"rich_without_{label}_r2_i15_exp", source="broad", rank=2, maxiter=15,
                           rolling_months=None, reuse=None,
                           etf=[x for x in RICH_ETF if x not in removed],
                           macro=[x for x in RICH_MACRO if x not in removed]))
    return result


def training_panel(frame: pd.DataFrame, date: str, index: int, tickers: list[str],
                   rolling_months: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice complete causal training panels while preserving original seed indexes.

    Args:
        frame: Frozen 112-month panel; targets are decimal next-month excess.
        date: Forecast origin, May2022 through August2026.
        index: Original May2024-relative index, -24 through27.
        tickers: Exact approved fifteen-task ordering.
        rolling_months: None for 60--111 expanding dates, otherwise48.

    Returns:
        Train/evaluation frames; neither future nor origin targets enter train.
    """
    if index not in range(-24, 28) or rolling_months not in {None, 48}:
        raise ValueError("Unregistered chronology/window")
    train = frame[frame.date.lt(pd.Timestamp(date)) & frame.y_excess_lead.notna()].copy()
    evaluation = frame[frame.date.eq(pd.Timestamp(date))].copy()
    if train.date.nunique() != 84 + index:
        raise ValueError("Unexpected pre-origin training coverage")
    count = rolling_months or 84 + index
    if rolling_months:
        retained = sorted(train.date.unique())[-count:]
        train = train[train.date.isin(retained)].copy()
    counts = train.groupby("asset_id", observed=True).size()
    if (len(train) != count * 15 or set(counts.index) != set(tickers) or not counts.eq(count).all()
            or len(evaluation) != 15 or set(evaluation.asset_id) != set(tickers)
            or not np.isfinite(train.y_excess_lead.to_numpy()).all()):
        raise ValueError("Incomplete causal training/evaluation panel")
    return train, evaluation


def recipe(counts: np.ndarray) -> dict:
    """Validate static family simplexes with at most three experts per family.

    Args:
        counts: (4,experts) integer nonnegative mass; each family sums to64.

    Returns:
        Content-addressed fixed recipe; no date- or seed-specific coefficients.
    """
    counts = np.asarray(counts)
    if (counts.ndim != 2 or counts.shape[0] != 4 or counts.shape[1] < 3
            or not np.isfinite(counts).all() or (counts < 0).any()
            or not np.all(counts == counts.astype(int)) or not np.all(counts.sum(1) == 64)
            or (np.count_nonzero(counts, axis=1) > 3).any()):
        raise ValueError("Four static integer simplexes with at most three experts required")
    values = counts.astype(int).tolist()
    token = hashlib.sha256(json.dumps(values).encode()).hexdigest()[:20]
    return {"id": "blend_" + token, "counts": values}


def blend(means: np.ndarray, case: dict) -> np.ndarray:
    """Pool expert predictions with the same family recipe for all dates/seeds.

    Args:
        means: (experts,3,months,15) finite decimal predictions.
        case: Validated (4,experts) coefficient recipe.

    Returns:
        (3,months,15) decimal predictions; no covariance assumption is added.
    """
    canonical = recipe(np.asarray(case["counts"]))
    if canonical != case or means.ndim != 4 or means.shape[1] != 3 or means.shape[-1] != 15:
        raise ValueError("Aligned three-seed expert panel required")
    counts = np.asarray(case["counts"])
    if counts.shape[1] != means.shape[0] or not np.isfinite(means).all():
        raise ValueError("Complete finite expert predictions required")
    return np.einsum("ksta,ak->sta", means, counts[FAMILY] / 64)


def scores(actual: np.ndarray, risk_free: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> dict:
    """Compute complete-period objectives and two temporal-half diagnostics.

    Args:
        actual: (months,15) realized discounted excess returns, decimals.
        risk_free: (months,) realized monthly risk-free returns, decimals.
        mean: (3,months,15) predictions from causal base forecasts.
        weights: Matching raw portfolio fractions from the unchanged allocator.

    Returns:
        Mean seed Sharpe, compounded total return, rank IC and half-period scores.
    """
    months = len(actual)
    if (months < 12 or actual.shape != (months, 15) or risk_free.shape != (months,)
            or mean.shape != (3, months, 15) or weights.shape != mean.shape
            or not all(np.isfinite(x).all() for x in [actual, risk_free, mean, weights])):
        raise ValueError("At least twelve complete aligned earlier outcome months required")
    total = (1 + np.einsum("sta,ta->st", weights, actual)) * (1 + risk_free) - 1
    excess = total - risk_free
    std = excess.std(1, ddof=1)
    if (std <= 0).any() or (total <= -1).any():
        raise ValueError("Undefined portfolio objectives")
    predicted, realized = rankdata(mean, axis=-1), rankdata(actual, axis=-1)
    predicted -= predicted.mean(-1, keepdims=True)
    realized -= realized.mean(-1, keepdims=True)
    denominator = np.sqrt(np.sum(predicted**2, -1) * np.sum(realized**2, -1))
    if (denominator <= 0).any():
        raise ValueError("Undefined cross-sectional rank IC")
    ic = np.einsum("sta,ta->st", predicted, realized) / denominator
    halves = []
    for section in [slice(0, months // 2), slice(months // 2, months)]:
        half = excess[:, section]
        halves.append({"sharpe": float((np.sqrt(12) * half.mean(1) / half.std(1, ddof=1)).mean()),
                       "rank_ic": float(ic[:, section].mean())})
    return {"sharpe": float((np.sqrt(12) * excess.mean(1) / std).mean()),
            "total_return": float((np.prod(1 + total, axis=1) - 1).mean()),
            "rank_ic": float(ic.mean()), "temporal_halves": halves}


def choose(records: list[dict]) -> dict[str, dict]:
    """Return Sharpe, rank, and equal-objective-percentile compromise choices.

    Args:
        records: Complete earlier-only objectives with canonical recipes.

    Returns:
        Three transparent tradeoff representatives; none auto-promotes a model.
    """
    if not records:
        raise ValueError("No complete candidate recipes")
    ordered = sorted(records, key=lambda row: (np.count_nonzero(row["recipe"]["counts"]), row["id"]))
    matrix = np.array([[row[key] for key in OBJECTIVES] for row in ordered])
    if not np.isfinite(matrix).all():
        raise ValueError("Failed/partial recipes cannot enter selection")
    percentiles = rankdata(matrix, axis=0) / len(matrix)
    return {"sharpe": ordered[int(np.argmax(matrix[:, 0]))],
            "ranking": ordered[int(np.argmax(matrix[:, 2]))],
            "compromise": ordered[int(np.argmax(percentiles.mean(1)))]}


def select_static(means: np.ndarray, actual: np.ndarray, risk_free: np.ndarray,
                  covariance: np.ndarray, budget: np.ndarray, incumbent: dict,
                  allocate: Callable, *, random_recipes: int = 256) -> tuple[dict, list[dict]]:
    """Select features/settings/blends from only the provided earlier OOS ledger.

    Args:
        means: Earlier expert/seed/month/asset predictions, never test targets.
        actual: Earlier realized discounted excess returns only.
        risk_free: Earlier realized monthly risk-free returns only.
        covariance: Fixed50/50 broad/compact observed covariance for earlier dates.
        budget: Causal earlier monthly variance ceilings.
        incumbent: Existing family blend encoded on this expert catalog.
        allocate: Unchanged hard-rank allocator accepting moments and budgets.
        random_recipes: Bounded deterministic sparse-simplex exploration count.

    Returns:
        Three frozen recipes and all complete earlier-only search records.
        GP settings and input masks are chosen through expert inclusion.
    """
    if means.shape[2] != len(actual) or len(actual) < 24 or budget.shape != (len(actual),):
        raise ValueError("Only aligned earlier ledgers of at least24 months may select")
    experts = len(means)
    initial = [incumbent]
    for expert in range(experts):
        counts = np.zeros((4, experts), dtype=int)
        counts[:, expert] = 64
        initial.append(recipe(counts))
    rng = np.random.default_rng(20261025)
    for _ in range(random_recipes):
        counts = np.zeros((4, experts), dtype=int)
        for family in range(4):
            selected = rng.choice(experts, 3, replace=False)
            mass = rng.multinomial(64, rng.dirichlet(np.ones(3)))
            counts[family, selected] = mass
        initial.append(recipe(counts))
    records = {}
    cache = {}

    def evaluate(cases: list[dict]) -> None:
        for case in cases:
            if case["id"] in records:
                continue
            mean = blend(means, case)
            signature = rankdata(mean, axis=-1).tobytes()
            if signature not in cache:
                weights, constrained = allocate(mean, covariance, budget)
                cache[signature] = (scores(actual, risk_free, mean, weights), constrained)
            values, constrained = cache[signature]
            records[case["id"]] = {"id": case["id"], "recipe": case, **values,
                                    "risk_constrained_pairs": constrained}

    evaluate(initial)
    for step in [16, 8, 4]:
        centers = {row["id"]: row for row in choose(list(records.values())).values()}
        proposals = []
        for center in centers.values():
            original = np.array(center["recipe"]["counts"])
            for family in range(4):
                for donor in np.flatnonzero(original[family] >= step):
                    for receiver in range(experts):
                        if receiver == donor:
                            continue
                        counts = original.copy()
                        counts[family, donor] -= step
                        counts[family, receiver] += step
                        if np.count_nonzero(counts[family]) <= 3:
                            proposals.append(recipe(counts))
        evaluate(proposals)
    return choose(list(records.values())), sorted(records.values(), key=lambda row: row["id"])
