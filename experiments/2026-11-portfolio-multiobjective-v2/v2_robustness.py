"""Predeclared development-only sensitivity checks for fixed November policies.

This experiment analysis layer reads exactly41 development return months from
supplied immutable paths. It neither allocates nor fits, saves files, selects a
new reference, nor opens confirmation histories. Returns/aspirations are decimal;
drawdown-loss aspirations are positive magnitudes. Resampling completed strategy
returns is retrospective sensitivity, not causally refitted alternate history.
"""

from __future__ import annotations

import numpy as np
from v2_common import arrays
from v2_metrics import paired_block_bootstrap, score_returns

DEVELOPMENT_MONTHS = 41
ANNUAL_BLOCKS = [(5, 17), (17, 29), (29, 41)]


def _distance(cagr: np.ndarray | float, drawdown: np.ndarray | float, anchors: dict) -> np.ndarray:
    growth_gap = np.maximum(float(anchors["aspiration_CAGR"]) - cagr, 0) / float(anchors["CAGR_scale"])
    drawdown_gap = (np.maximum(-drawdown - float(anchors["aspiration_drawdown_loss"]), 0)
                    / float(anchors["drawdown_scale"]))
    return np.hypot(growth_gap, drawdown_gap)


def _feasible(record: dict) -> bool:
    rows = record.get("numerical", [])
    return (record.get("feasibility_complete") is True and record.get("complete") is True
            and len(rows) == DEVELOPMENT_MONTHS and all(row.get("passed") is True for row in rows))


def _source(record: dict) -> dict:
    source = arrays(record["artifact_path"])
    for key in ["total_returns", "rf", "rank_ic", "cost_10_total_returns"]:
        values = np.asarray(source[key], dtype=float)
        if values.shape != (DEVELOPMENT_MONTHS,) or not np.isfinite(values).all():
            raise ValueError("Robustness requires exactly41 finite development months; no confirmation slicing")
    if np.asarray(source["dates"]).shape != (DEVELOPMENT_MONTHS,):
        raise ValueError("Explicit complete development date axis required")
    months = np.asarray(source["dates"]).astype("datetime64[M]")
    expected = np.arange(np.datetime64("2022-05", "M"), np.datetime64("2025-10", "M"))
    if not np.array_equal(months, expected):
        raise ValueError("Development origins must be May2022–September2025; confirmation cannot enter")
    return source


def _batch_stats(returns: np.ndarray, rf: np.ndarray, ranks: np.ndarray, indices: np.ndarray) -> dict:
    sampled = returns[indices]
    gross = np.prod(1 + sampled, axis=1)
    wealth = np.column_stack([np.ones(len(indices)), np.cumprod(1 + sampled, axis=1)])
    drawdown = wealth / np.maximum.accumulate(wealth, axis=1) - 1
    excess = sampled - rf[indices]
    sd = excess.std(axis=1, ddof=1)
    sd[np.ptp(excess, axis=1) == 0] = 0
    sharpe = np.full(len(indices), np.nan)
    np.divide(np.sqrt(12) * excess.mean(1), sd, out=sharpe, where=sd > 0)
    return {"cagr": gross ** (12 / sampled.shape[1]) - 1, "max_drawdown": drawdown.min(1),
            "sharpe": sharpe, "rank_ic": ranks[indices].mean(1)}


def _percentiles(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    quantiles = np.quantile(values[finite], [.025, .5, .975]).tolist() if finite.any() else [None, None, None]
    return {"p025": quantiles[0], "median": quantiles[1], "p975": quantiles[2],
            "defined_replicates": int(finite.sum()), "undefined_replicates": int((~finite).sum())}


def _return_distance(returns: np.ndarray, rf: np.ndarray, anchors: dict) -> float:
    summary, _ = score_returns(returns, rf)
    return float(_distance(summary["cagr"], summary["max_drawdown"], anchors))


def assess(record: dict, reference: dict, anchors: dict, *, draws: int = 512, seed: int = 20261101) -> dict:
    """Apply fixed development robustness gates versus one supplied fixed control.

    Args:
        record: Complete candidate development record, explicit
            feasibility_complete=True,41 numerical passed rows and dev-only NPZ.
            NPZ must supply total_returns/rf/rank_ic/cost_10_total_returns/dates.
        reference: Identical schema for the caller-frozen development-best control.
            The function never searches or changes that reference identity.
        anchors: Frozen aspiration_CAGR/aspiration_drawdown_loss and positive
            CAGR_scale/drawdown_scale, both predeclared0.02.
        draws: Positive number of paired circular block replicates, default512.
        seed: Explicit deterministic local random seed, default20261101. The same
            seed is supplied independently to block lengths3 and6.

    Returns:
        Point/annual/leave-one-out/cost evidence, descriptive paired-bootstrap
        distributions, explicit fixed-reference identity and all frozen gates.
        Distance gains are reference minus candidate: positive favors candidate.
        Frozen selection fractions count nonnegative gains >=-1e-12; separately
        labeled positive descriptive fractions count only gains strictly >0.
        Resampled drawdown exceedance fractions are not future crash probabilities.

    Raises:
        ValueError: Invalid anchors, incomplete paired dates, wrong month count,
            or invalid return arrays. Infeasible candidates return a failed gate
            before attempting to read a missing artifact.
    """
    frozen_anchors = {key: float(anchors[key]) for key in ["aspiration_CAGR", "aspiration_drawdown_loss",
                                                         "CAGR_scale", "drawdown_scale"]}
    if (not np.isfinite(list(frozen_anchors.values())).all()
            or frozen_anchors["aspiration_drawdown_loss"] < 0
            or frozen_anchors["CAGR_scale"] != .02 or frozen_anchors["drawdown_scale"] != .02):
        raise ValueError("Frozen finite aspirations and exact0.02 CAGR/drawdown scales required")
    result = {"candidate_id": record["id"], "fixed_reference_id": reference["id"], "anchors": frozen_anchors,
              "months": DEVELOPMENT_MONTHS, "confirmation_used": False, "draws": draws, "seed": seed,
              "distance_gain_direction": "reference_minus_candidate; positive favors candidate",
              "selection_fraction_rule": "nonnegative gain >= -1e-12; descriptive positive fractions remain >0",
              "interpretation": "Resampled completed strategy returns are not refitted alternate histories; "
                                "frequencies are retrospective descriptions, not future crash probabilities"}
    feasible = _feasible(record)
    if not feasible:
        return {**result, "gates": {"feasibility_complete": False}, "eligible": False}
    if not _feasible(reference):
        raise ValueError("The fixed development reference must have complete passed numerical evidence")
    candidate, control = _source(record), _source(reference)
    np.testing.assert_array_equal(candidate["dates"].astype(str), control["dates"].astype(str))
    np.testing.assert_array_equal(candidate["rf"], control["rf"])
    point = _return_distance(candidate["total_returns"], candidate["rf"], frozen_anchors)
    reference_point = _return_distance(control["total_returns"], control["rf"], frozen_anchors)
    point_gain = reference_point - point
    bootstrap = {}
    for length in [3, 6]:
        paired, paths = paired_block_bootstrap(candidate["total_returns"], control["total_returns"], candidate["rf"],
            candidate_ic=candidate["rank_ic"], reference_ic=control["rank_ic"],
            block_length=length, draws=draws, seed=seed)
        indices = paths["sampled_indices"]
        candidate_stats = _batch_stats(candidate["total_returns"], candidate["rf"], candidate["rank_ic"], indices)
        control_stats = _batch_stats(control["total_returns"], control["rf"], control["rank_ic"], indices)
        gain = (_distance(control_stats["cagr"], control_stats["max_drawdown"], frozen_anchors)
                - _distance(candidate_stats["cagr"], candidate_stats["max_drawdown"], frozen_anchors))
        bootstrap[str(length)] = {
            "paired_differences": paired, "distance_gain": _percentiles(gain),
            "distance_gain_fraction": float(np.mean(gain >= -1e-12)),
            "positive_distance_gain_fraction": float(np.mean(gain > 0)),
            "candidate_percentiles": {key: _percentiles(value) for key, value in candidate_stats.items()},
            "reference_percentiles": {key: _percentiles(value) for key, value in control_stats.items()},
            "candidate_drawdown_exceedance_fractions": {str(threshold): float(np.mean(
                -candidate_stats["max_drawdown"] >= threshold)) for threshold in [.10, .15, .20]},
            "reference_drawdown_exceedance_fractions": {str(threshold): float(np.mean(
                -control_stats["max_drawdown"] >= threshold)) for threshold in [.10, .15, .20]}}
    leave_one_out = []
    for index in range(DEVELOPMENT_MONTHS):
        remaining = np.arange(DEVELOPMENT_MONTHS) != index
        candidate_distance = _return_distance(candidate["total_returns"][remaining], candidate["rf"][remaining],
                                              frozen_anchors)
        control_distance = _return_distance(control["total_returns"][remaining], control["rf"][remaining],
                                            frozen_anchors)
        leave_one_out.append(control_distance - candidate_distance)
    annual = []
    for start, end in ANNUAL_BLOCKS:
        left = _return_distance(candidate["total_returns"][start:end], candidate["rf"][start:end], frozen_anchors)
        right = _return_distance(control["total_returns"][start:end], control["rf"][start:end], frozen_anchors)
        annual.append({"start_index": start, "end_index_exclusive": end,
                       "candidate_distance": left, "reference_distance": right,
                       "within_reference_plus_0p25": bool(left <= right + .25)})
    cost_candidate = _return_distance(candidate["cost_10_total_returns"], candidate["rf"], frozen_anchors)
    cost_reference = _return_distance(control["cost_10_total_returns"], control["rf"], frozen_anchors)
    loo_fraction = float(np.mean(np.asarray(leave_one_out) >= -1e-12))
    annual_count = sum(block["within_reference_plus_0p25"] for block in annual)
    gates = {"feasibility_complete": feasible, "development_distance_gain_at_least_0p05": bool(point_gain >= .05),
             "block3_distance_gain_fraction_at_least_0p60": bootstrap["3"]["distance_gain_fraction"] >= .60,
             "leave_one_month_out_distance_gain_fraction_at_least_0p80": loo_fraction >= .80,
             "at_least_two_annual_blocks_within_distance_0p25": annual_count >= 2,
             "10bps_distance_gain_at_least_minus_0p10": bool(cost_reference - cost_candidate >= -.10)}
    result.update({"candidate_distance": point, "reference_distance": reference_point, "distance_gain": point_gain,
                   "bootstrap": bootstrap, "leave_one_month_out": {"distance_gains": leave_one_out,
                    "nonnegative_gain_fraction": loo_fraction,
                    "positive_gain_fraction": float(np.mean(np.asarray(leave_one_out) > 0)),
                    "minimum_gain": min(leave_one_out)},
                   "annual_development_blocks": annual, "annual_blocks_within_tolerance": annual_count,
                   "10bps_cost_check": {"candidate_distance": cost_candidate, "reference_distance": cost_reference,
                                        "distance_gain": cost_reference - cost_candidate},
                   "gates": gates, "eligible": all(gates.values())})
    return result
