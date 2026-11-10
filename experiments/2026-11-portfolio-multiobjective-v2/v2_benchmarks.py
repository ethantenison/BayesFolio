"""Frozen standing benchmarks and explicitly static stress diagnostics for v2.

This research I/O adapter starts no fits, optimizer pools, downloads or trades.
Realized benchmarking is gated on the frozen shortlist. Returns/weights are
decimals, NAV is dollars, and fees are basis points per gross ETF traded dollar.
Only Schwab aggressive and top-seven eligible14 simple momentum are standing
benchmarks. Static historical stresses apply today's fixed targets each month;
they are not historical GP-strategy forecasts or calibrated crash probabilities.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from v2_common import ASSETS, ELIGIBLE, ELIGIBLE_INDEX, HERE, V1, arrays, digest, read, save, snapshot
from v2_metrics import COST_BPS, MAX_WEIGHT_SUM_ERROR, MIN_RAW_WEIGHT, score_returns


def _require_unlocked() -> None:
    if not (HERE / "SHORTLIST_FROZEN.json").exists():
        raise ValueError("Freeze the final shortlist before benchmark or static-stress scoring")


def _frozen(path: Path, seal: dict) -> str:
    fingerprint = digest(path)
    expected = {**seal.get("artifact_sha256", {}), **seal.get("environment_sha256", {})}.get(str(path.resolve()))
    if expected != fingerprint:
        raise ValueError(f"Changed or unsealed protected source: {path}")
    return fingerprint


def _v1_benchmarks() -> tuple[ModuleType, dict[str, str]]:
    path = V1 / "benchmarks.py"
    fingerprint = _frozen(path, read(V1 / "COMPLETED.json"))
    spec = importlib.util.spec_from_file_location("_v2_frozen_standing_benchmarks", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen v1 benchmark definitions")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {str(path): fingerprint}


def _cost_path(asset_total: np.ndarray, weights: np.ndarray, fee_bps: int) -> dict:
    """Pay ETF transaction costs self-financing, with fee-free cash movements."""
    asset_total, weights = np.asarray(asset_total, dtype=float), np.asarray(weights, dtype=float)
    if (asset_total.ndim != 2 or weights.shape != asset_total.shape or not len(weights)
            or not np.isfinite(asset_total).all() or not np.isfinite(weights).all()
            or (asset_total <= -1).any() or (weights < 0).any()
            or np.max(abs(weights.sum(1) - 1)) > 1e-12 or not 0 <= fee_bps < 10000):
        raise ValueError("Aligned valid asset/cash returns, weights and fee required")
    fee = fee_bps / 10000
    total, notional, fraction = (np.empty(len(weights)) for _ in range(3))
    pretrade = np.zeros_like(weights)
    pretrade[0, -1] = 1
    for month in range(len(weights)):
        if month:
            holdings = weights[month - 1] * (1 + asset_total[month - 1])
            pretrade[month] = holdings / holdings.sum()
        fraction[month] = (1.0 if fee == 0 else brentq(
            lambda value, m=month: value + fee * np.abs(
                value * weights[m, :-1] - pretrade[m, :-1]).sum() - 1,
            0, 1, xtol=5e-15, rtol=1e-14))
        notional[month] = np.abs(fraction[month] * weights[month, :-1] - pretrade[month, :-1]).sum()
        total[month] = fraction[month] * (1 + np.dot(weights[month], asset_total[month])) - 1
    return {"total_returns": total, "ETF_traded_notional_fraction": notional,
            "cost_fraction": fee * notional, "post_cost_wealth_fraction": fraction,
            "pretrade_weights": pretrade}


def build_benchmarks(context: dict) -> dict:
    """Build and save matched standing benchmarks after final shortlist freezing.

    Args:
        context: Dates53 origins May2022--September2026, actual(53,15) discounted
            excess decimal targets, rf(53,) decimal RF and canonical15 assets.

    Returns:
        Rows of all53/latest12/development summaries, paths per benchmark,
        provenance and artifact paths. Each saved NPZ has weights(53,16), total
        returns53, wealth/drawdown54 and cost sensitivity paths. MGK is zero in
        both portfolios; momentum is reranked over fourteen eligible assets.
    """
    _require_unlocked()
    dates = np.asarray(context["dates"]).astype(str).tolist()
    assets = np.asarray(context["assets"]).astype(str).tolist()
    actual, rf = np.asarray(context["actual"], dtype=float), np.asarray(context["rf"], dtype=float)
    if (assets != ASSETS or len(dates) != 53 or dates[0] != "2022-05-31" or dates[-1] != "2026-09-30"
            or actual.shape != (53, 15) or rf.shape != (53,)
            or not np.isfinite(actual).all() or not np.isfinite(rf).all()):
        raise ValueError("Complete canonical53-month/15-task context required")
    old, source_hashes = _v1_benchmarks()
    seal = read(V1 / "COMPLETED.json")
    audit_path = V1 / "outputs/benchmarks/AUDIT.json"
    source_hashes[str(audit_path)] = _frozen(audit_path, seal)
    old_audit = read(audit_path)
    for name, expected in old_audit["input_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Protected prior benchmark dependency changed: {name}")
    source_hashes.update(old_audit["input_sha256"])
    prior_path = V1 / "outputs/benchmarks/schwab_aggressive.npz"
    source_hashes[str(prior_path)] = _frozen(prior_path, seal)
    prior = arrays(prior_path)
    np.testing.assert_array_equal(prior["dates"], dates)
    schedule = read(old.SCHEDULE)
    aggressive = schedule["profiles"]["aggressive"]
    scheduled = old._scheduled(dates, aggressive["initial_weights"], aggressive["verified_revisions"])
    schwab = np.zeros((53, 16))
    for position, asset in enumerate(schedule["proxy_order"]):
        schwab[:, -1 if asset == "CASH" else assets.index(asset)] = scheduled["weights"][:, position]
    np.testing.assert_array_equal(schwab, prior["weights"])
    prices_dates, prices, price_hashes = old._history(dates, ELIGIBLE)
    momentum = old.momentum_from_prices(prices_dates, prices, dates, ELIGIBLE)
    simple = np.zeros((53, 16))
    simple[:, ELIGIBLE_INDEX] = momentum["weights"]
    source_hashes.update(price_hashes)
    paths, rows, artifacts = {}, {}, {}
    asset_total = np.column_stack([(1 + actual) * (1 + rf[:, None]) - 1, rf])
    for name, weights in [("schwab_aggressive", schwab), ("momentum_top7", simple)]:
        if (np.any(weights[:, ASSETS.index("MGK")] != 0) or (weights < 0).any()
                or np.max(abs(weights.sum(1) - 1)) > 1e-12):
            raise ValueError("Benchmark weights violate MGK exclusion or accounting")
        total = old.total_returns(actual, rf, weights)
        summary, numeric = score_returns(total, rf)
        latest, _ = score_returns(total[41:53], rf[41:53])
        development, _ = score_returns(total[:41], rf[:41])
        if name == "schwab_aggressive":
            np.testing.assert_array_equal(total, prior["total_returns"])
            np.testing.assert_array_equal(numeric["wealth"], prior["wealth"])
        costs = {}
        evidence = {"dates": np.array(dates), "assets_with_cash": np.array([*ASSETS, "CASH"]),
                    "weights": weights, "total_returns": total, "wealth": numeric["wealth"],
                    "drawdown": numeric["drawdown"]}
        for bps in COST_BPS:
            cost = _cost_path(asset_total, weights, bps)
            costs[str(bps)], scored = score_returns(cost["total_returns"], rf)
            costs[str(bps)]["total_ETF_traded_notional_fraction"] = float(cost["ETF_traded_notional_fraction"].sum())
            evidence.update({f"cost_{bps}bps_{key}": value for key, value in cost.items()})
            evidence[f"cost_{bps}bps_wealth"] = scored["wealth"]
            evidence[f"cost_{bps}bps_drawdown"] = scored["drawdown"]
        if name == "momentum_top7":
            evidence.update(momentum_scores=momentum["scores"], eligible_assets=np.array(ELIGIBLE),
                            momentum_selected=momentum["selected"],
                            momentum_start_price_dates=momentum["start_price_dates"].astype(str),
                            momentum_end_price_dates=momentum["end_price_dates"].astype(str))
        path = HERE / "outputs/benchmarks" / f"{name}.npz"
        snapshot(path, evidence)
        artifacts[name] = str(path)
        rows[name] = {"summary": summary, "confirmation12": latest, "development41": development,
                      "cost_sensitivity": costs, "artifact_path": str(path)}
        paths[name] = evidence
    audit = {"standing_benchmarks": ["schwab_aggressive", "momentum_top7"],
             "schwab_exact_frozen53_weight_return_wealth_parity": True,
             "schwab_verified_revision_count": scheduled["applied_revision_count"],
             "momentum_definition": "P[t-1]/P[t-12]-1, eligible14 top7, alphabetical ties, equalweight monthly",
             "MGK_weight": 0, "momentum_cash_weight": 0,
             "cash_fee_policy": "Cash earns RF; cash movement has no fee. Fee charges gross ETF purchases+sales.",
             "cost_policy": "Initial ETF purchase charged; subsequent drift-aware self-financing monthly rebalance.",
             "calendar_proxy_conditions": "Month-end adjusted-price information and monthly reset targets; "
                 "historical close fills are retrospective proxies, not executable November2 fills.",
             "source_vintage_note": old_audit["vintage_note"], "source_sha256": source_hashes,
             "code_sha256": {str(Path(__file__)): digest(Path(__file__)),
                             str(HERE / "v2_metrics.py"): digest(HERE / "v2_metrics.py")}}
    result = {"passed": True, "rows": rows, "audit": audit, "artifacts": artifacts}
    save(HERE / "BENCHMARKS.json", result)
    return {**result, "paths": paths}


def _stress_history() -> tuple[np.ndarray, np.ndarray, dict]:
    seal = read(V1 / "data/COMPLETED.json")
    if not seal["passed"]:
        raise ValueError("Passed sealed price extension required")
    monthly, endpoints, sources = {}, {}, {}
    for asset in ELIGIBLE:
        path = V1 / "data/raw/combined" / f"{asset}.parquet"
        sources[str(path)] = _frozen(path, seal)
        daily = pd.read_parquet(path)["Adj Close"]
        if (daily.index.has_duplicates or not daily.index.is_monotonic_increasing or daily.empty
                or not np.isfinite(daily.to_numpy()).all() or (daily <= 0).any()
                or daily.index.max() > pd.Timestamp("2026-10-30")):
            raise ValueError("Invalid sealed static-stress adjusted-price observations")
        groups = daily.groupby(daily.index.to_period("M"))
        monthly[asset] = groups.last()
        endpoints[asset] = groups.apply(lambda rows: rows.index[-1])
    first = max(values.index[0] for values in monthly.values())
    last = min(values.index[-1] for values in monthly.values())
    months = pd.period_range(first, last, freq="M")
    if len(months) < 13:
        raise ValueError("Insufficient complete common history for12-month static stress")
    prices, observed = [], None
    for asset in ELIGIBLE:
        if not months.isin(monthly[asset].index).all():
            raise ValueError(f"Missing interior monthly price in static stress: {asset}")
        asset_dates = endpoints[asset].loc[months].to_numpy("datetime64[D]")
        if observed is not None:
            np.testing.assert_array_equal(asset_dates, observed)
        observed = asset_dates
        prices.append(monthly[asset].loc[months].to_numpy())
    if observed is None:
        raise ValueError("No common eligible-asset observation dates")
    prices = np.column_stack(prices)
    return observed[1:].astype(str), prices[1:] / prices[:-1] - 1, {
        "source_sha256": sources, "common_first_price_month": str(first), "common_last_price_month": str(last),
        "historical_months": len(months) - 1, "no_interior_months_dropped": True}


def stress(weights15: np.ndarray) -> dict:
    """Apply fixed November target weights to sealed historical/stylized shocks.

    Args:
        weights15: Canonical15 raw decimal targets, MGK exactlyzero, fully invested
            within approved numerical tolerances. No clipping or normalization.

    Returns:
        Worst1/3/12-month static-weight returns, static path drawdown, March2020
        when available, explicit two shock assumptions, provenance/artifact paths.
        This is a fixed-target sensitivity exercise, not a GP backtest/probability.
    """
    _require_unlocked()
    weights = np.asarray(weights15, dtype=float)
    if (weights.shape != (15,) or not np.isfinite(weights).all() or weights.min() < MIN_RAW_WEIGHT
            or abs(weights.sum() - 1) > MAX_WEIGHT_SUM_ERROR or weights[ASSETS.index("MGK")] != 0):
        raise ValueError("Validated canonical15 weights with exactlyzero MGK required")
    dates, returns, audit = _stress_history()
    total = returns @ weights[ELIGIBLE_INDEX]
    summary, numeric = score_returns(total, np.zeros(len(total)))
    result = {"historical_months": len(total), "historical_first_return_month": dates[0][:7],
              "historical_last_return_month": dates[-1][:7], "static_path_max_drawdown": summary["max_drawdown"],
              "fixed_weights": dict(zip(ASSETS, weights.tolist())), "worst_windows": {}}
    for count in [1, 3, 12]:
        rolling = np.prod(1 + np.lib.stride_tricks.sliding_window_view(total, count), axis=1) - 1
        index = int(np.argmin(rolling))
        result["worst_windows"][str(count)] = {"months": count, "return": float(rolling[index]),
                                               "start_month": dates[index][:7],
                                               "end_month": dates[index + count - 1][:7]}
    march = np.flatnonzero(np.char.startswith(dates.astype(str), "2020-03"))
    result["march2020"] = {"available": bool(len(march)), "return": float(total[march[0]]) if len(march) else None}
    equity = {"SPY", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX"}
    high_yield = {"HYG", "HYEM"}
    shocks = {
        "equity20_FI5": np.array([-.20 if asset in equity else -.05 for asset in ELIGIBLE]),
        "equity30_HY15_otherFI5": np.array([-.30 if asset in equity else -.15 if asset in high_yield else -.05
                                           for asset in ELIGIBLE]),
    }
    result["hypothetical_shocks"] = {name: {"portfolio_return": float(shock @ weights[ELIGIBLE_INDEX]),
        "asset_returns": dict(zip(ELIGIBLE, shock.tolist()))} for name, shock in shocks.items()}
    result["assumptions"] = [
        "Historical prices are dividend/split-adjusted total-return proxies from the sealed October31 vintage.",
        "The same November target is reset monthly throughout history; no historical GP forecasts are used.",
        "Static historical sensitivity is gross of fees/taxes; November2 execution at October30 closes is not assumed.",
        "Shock2: HYG/HYEM receive-15%; BND/IEF/LQD/BNDX/VWOB receive-5%. Scenario buckets are not ratings.",
        "No probability, crash-protection guarantee, currency hedging or recovery path is inferred from these shocks.",
    ]
    key = hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()[:16]
    path = HERE / "outputs/stress" / f"static_{key}.npz"
    snapshot(path, {"dates": dates, "eligible_assets": np.array(ELIGIBLE), "weights15": weights,
                    "asset_returns": returns, "total_returns": total, "wealth": numeric["wealth"],
                    "drawdown": numeric["drawdown"]})
    result.update(audit=audit, artifact_path=str(path), code_sha256=digest(Path(__file__)))
    save(HERE / "outputs/stress" / f"static_{key}.json", result)
    return result
