"""Build only the two approved standing benchmarks from frozen local evidence.

Experiment I/O adapter; no downloads, GP fits, optimizers or allocation changes.
Inputs/outputs use decimal returns and portfolio fractions; cash is last in each
weight matrix. Momentum is P[t-1]/P[t-12]-1 with alphabetical top-seven ties.
Usage: import build_benchmarks from the November decision reporting pipeline.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent / "2026-10-portfolio-evaluation-v1"
DATA = HERE / "data"
SCHEDULE = FRAMEWORK / "schwab_benchmark_v1/weights_schedule.json"
POLICY = FRAMEWORK / "CURRENT_BENCHMARKS.json"
BASELINES = FRAMEWORK / "CURRENT_BASELINES.json"
PRIOR_PATHS = FRAMEWORK / "monthly_extension_v1/outputs/paths.npz"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scheduled(dates: list[str], initial: list[float], revisions: list[dict]) -> dict:
    path = FRAMEWORK / "schwab_benchmark_v1/benchmark_math.py"
    spec = importlib.util.spec_from_file_location("_november_standing_benchmark_math", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot read the frozen benchmark schedule implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scheduled_weights(dates, initial, revisions)


def momentum_from_prices(price_dates: np.ndarray, prices: np.ndarray, origins: list[str],
                         assets: list[str]) -> dict[str, np.ndarray]:
    """Compute the fixed standalone momentum strategy without extra benchmarks.

    Args:
        price_dates: Consecutive monthly actual last-observation dates.
        prices: Matching positive finite adjusted-price matrix, months by assets.
        origins: Consecutive monthly forecast/rebalance origins.
        assets: Unique ETF identifiers in column order, at least seven.

    Returns:
        Momentum scores, top-seven equal weights and explicit lookback endpoints.
        Even when every score is negative the highest-ranked seven remain held.
    """
    observed = np.asarray(price_dates, dtype="datetime64[D]")
    decisions = np.asarray(origins, dtype="datetime64[D]")
    if (observed.ndim != 1 or decisions.ndim != 1 or not len(observed) or not len(decisions)
            or np.isnat(observed).any() or np.isnat(decisions).any()
            or len(assets) < 7 or len(set(assets)) != len(assets)
            or prices.shape != (len(observed), len(assets))
            or not np.isfinite(prices).all() or (prices <= 0).any()):
        raise ValueError("Complete positive monthly prices and unique asset/date axes required")
    months, decision_months = observed.astype("datetime64[M]"), decisions.astype("datetime64[M]")
    if (np.diff(months).astype(int) != 1).any() or (np.diff(decision_months).astype(int) != 1).any():
        raise ValueError("Consecutive monthly observations required; no gaps or duplicates")
    positions = np.searchsorted(months, decision_months)
    if ((positions >= len(months)).any() or not np.array_equal(months[positions], decision_months)
            or (positions < 12).any() or (observed[positions] > decisions).any()):
        raise ValueError("Complete causal twelve-month momentum lookbacks required")
    scores = prices[positions - 1] / prices[positions - 12] - 1
    selected = np.zeros_like(scores, dtype=bool)
    for row, values in enumerate(scores):
        chosen = np.lexsort((np.asarray(assets), -values))[:7]
        selected[row, chosen] = True
    return {"weights": selected.astype(float) / 7, "scores": scores, "selected": selected,
            "start_price_dates": observed[positions - 12], "end_price_dates": observed[positions - 1]}


def total_returns(actual: np.ndarray, risk_free: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Use the unchanged discounted-excess plus cash-RF accounting convention.

    Args:
        actual: (months,15) decimal discounted-excess ETF outcomes.
        risk_free: (months,) decimal arithmetic risk-free outcomes.
        weights: (months,16) ETF/cash fractions; cash last.

    Returns:
        Complete monthly total returns with no imputation or dropped dates.
    """
    if (actual.ndim != 2 or risk_free.shape != (len(actual),)
            or weights.shape != (len(actual), actual.shape[1] + 1)
            or not all(np.isfinite(value).all() for value in [actual, risk_free, weights])
            or (actual <= -1).any() or (risk_free <= -1).any()
            or (weights < 0).any() or np.max(abs(weights.sum(1) - 1)) > 1e-12):
        raise ValueError("Complete valid decimal outcomes and long-only fully invested weights required")
    return (1 + np.sum(actual * weights[:, :-1], axis=1)) * (1 + risk_free) - 1


def _history(dates: list[str], assets: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    first = pd.Period(dates[0], freq="M") - 12
    expected = pd.period_range(first, pd.Period(dates[-1], freq="M"), freq="M")
    seal = json.loads((DATA / "COMPLETED.json").read_text())
    if not seal["passed"]:
        raise ValueError("Completed November data extension required")
    columns, observed, hashes = [], None, {}
    for asset in assets:
        path = DATA / "raw/combined" / f"{asset}.parquet"
        fingerprint = _digest(path)
        if seal["artifact_sha256"].get(str(path.resolve())) != fingerprint:
            raise ValueError(f"Changed frozen benchmark price source: {asset}")
        daily = pd.read_parquet(path)["Adj Close"]
        daily = daily.loc[(daily.index >= first.start_time) & (daily.index <= pd.Timestamp(dates[-1]))]
        if (daily.index.has_duplicates or not daily.index.is_monotonic_increasing
                or not np.isfinite(daily.to_numpy()).all() or (daily <= 0).any()):
            raise ValueError(f"Invalid frozen daily adjusted prices: {asset}")
        grouped = daily.groupby(daily.index.to_period("M"))
        values = grouped.last()
        endpoints = pd.DatetimeIndex(grouped.apply(lambda row: row.index[-1])).to_numpy("datetime64[D]")
        if not values.index.equals(expected):
            raise ValueError(f"Missing monthly benchmark history: {asset}")
        if observed is not None:
            np.testing.assert_array_equal(endpoints, observed)
        observed = endpoints
        columns.append(values.to_numpy())
        hashes[str(path.resolve())] = fingerprint
    return observed, np.column_stack(columns), hashes


def build_benchmarks(dates: list[str], actual: np.ndarray, rf: np.ndarray, assets: list[str]) -> dict:
    """Return the two approved benchmarks on the exact53-month decision panel.

    Args:
        dates: Origins May2022--September2026, producing June2022--October2026 returns.
        actual: Matching (53,15) decimal discounted-excess realized ETF outcomes.
        rf: Matching (53,) decimal monthly risk-free returns.
        assets: The approved15 unique ETF identifiers, in the supplied column order.

    Returns:
        Cash-inclusive16-column weights, two53-month total-return arrays, momentum
        signals/lookbacks, axes, and provenance/old-benchmark parity audit metadata.
        Revisions to historical vendor snapshots are disclosed, not back-applied.
    """
    policy = json.loads(POLICY.read_text())
    approved_assets = json.loads(BASELINES.read_text())["assets_in_order"]
    if (policy["active_benchmark_ids"] != ["schwab_aggressive", "momentum_top7"]
            or len(assets) != 15 or len(set(assets)) != 15 or set(assets) != set(approved_assets)
            or len(dates) != 53 or dates[0] != "2022-05-31" or dates[-1] != "2026-09-30"
            or actual.shape != (53, 15) or rf.shape != (53,)):
        raise ValueError("The approved two benchmarks and complete53-month/15-ETF panel are required")
    price_dates, prices, source_hashes = _history(dates, assets)
    momentum = momentum_from_prices(price_dates, prices, dates, assets)
    schedule = json.loads(SCHEDULE.read_text())
    aggressive = schedule["profiles"]["aggressive"]
    scheduled = _scheduled(dates, aggressive["initial_weights"], aggressive["verified_revisions"])
    schwab_weights = np.zeros((53, 16))
    for position, proxy in enumerate(schedule["proxy_order"]):
        schwab_weights[:, assets.index(proxy) if proxy != "CASH" else -1] = scheduled["weights"][:, position]
    momentum_weights = np.column_stack([momentum["weights"], np.zeros(53)])
    outputs = {"schwab_aggressive_weights": schwab_weights, "momentum_top7_weights": momentum_weights}
    parity = {}
    with np.load(PRIOR_PATHS, allow_pickle=False) as prior:
        order = [assets.index(str(asset)) for asset in prior["assets"]]
        old_dates = prior["dates"].tolist()
        positions = np.array([dates.index(date) for date in old_dates])
        np.testing.assert_allclose(actual[positions][:, order], prior["actual"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(rf[positions], prior["risk_free"], rtol=0, atol=1e-12)
        for label in ["schwab_aggressive", "momentum_top7"]:
            key = label + "_weights"
            reconstructed = total_returns(actual, rf, outputs[key])
            difference = float(np.max(abs(reconstructed[positions] - prior[label + "_total_returns"])))
            changed = int(np.any(abs(outputs[key][positions][:, [*order, 15]] - prior[key]) > 1e-12, axis=1).sum())
            if changed:
                raise ValueError(f"Frozen {label} benchmark reconstruction changed {changed} historical weight months; "
                                 "investigate source-vintage or definition drift, never substitute saved weights")
            np.testing.assert_allclose(reconstructed[positions], prior[label + "_total_returns"],
                                       rtol=0, atol=1e-12)
            parity[label] = {"overlap_months": len(positions),
                             "reconstructed_max_absolute_return_difference": difference,
                             "changed_reconstructed_weight_months": changed,
                             "sealed_old_weight_months_preserved": 0,
                             "final_old_return_parity_passed": True}
    for label in ["schwab_aggressive", "momentum_top7"]:
        outputs[label + "_total_returns"] = total_returns(actual, rf, outputs[label + "_weights"])
    source_hashes.update({str(path.resolve()): _digest(path) for path in [POLICY, BASELINES, SCHEDULE, PRIOR_PATHS,
        FRAMEWORK / "schwab_benchmark_v1/benchmark_math.py", Path(__file__)]})
    outputs.update(dates=np.array(dates), assets=np.array(assets), momentum_score=momentum["scores"],
                   momentum_selected=momentum["selected"],
                   momentum_start_price_date=momentum["start_price_dates"].astype(str),
                   momentum_end_price_date=momentum["end_price_dates"].astype(str),
                   audit={"standing_benchmark_ids": policy["active_benchmark_ids"],
                          "schwab_verified_revision_count": scheduled["applied_revision_count"],
                          "old28_month_parity": parity, "input_sha256": source_hashes,
                          "price_snapshot_as_of": "2026-10-31",
                          "vintage_note": "New combined files preserve frozen daily history through September30; "
                                          "October-only prices append without rewriting prior signals. Retrospective "
                                          "adjusted-price vintages are not point-in-time vendor snapshots.",
                          "no_GP_covariance_or_extra_variance_budget": True})
    return outputs
