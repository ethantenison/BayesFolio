"""Extend the frozen four-strategy wealth comparison through September 2026.

Research I/O and reporting only. Requires the 24 checkpointed appended GP fits;
never fits or downloads. Reuses all prior months exactly and the same economic
allocation/benchmark rules. Returns are decimals and investment values dollars.
Usage: uv run --locked --no-sync python <this_file> --execute [--workers 14]
       uv run --locked --no-sync python <this_file> --verify
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-monthly-extension-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-monthly-extension-cache")

from run_fits import DATA, FRAMEWORK, HERE, OLD, OUTPUT, _hash, _verify_hashes, _write  # noqa: E402
from run_fits import _verify as _verify_fits  # noqa: E402

sys.path.extend([str(FRAMEWORK), str(FRAMEWORK / "scenario_riskfolio_v1"),
                 str(FRAMEWORK / "scenario_sharpe_v3"), str(FRAMEWORK / "schwab_benchmark_v1"),
                 str(FRAMEWORK / "benchmark_suite_v1")])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from benchmark_math import portfolio_returns, scheduled_weights, wealth_path  # noqa: E402
from extension_math import append_paths, blend, hard_rank  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from metrics import portfolio_performance  # noqa: E402
from scenario_allocator import audit_weights, sample_joint  # noqa: E402
from signed_sharpe import allocate_scenarios  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from strategy_math import benchmark_panel  # noqa: E402

LABELS = {
    "gp_hard_rank": "Current GP rank + hard-rank allocation",
    "gp_scenarios": "GP scenarios + max Sharpe (Riskfolio + sign-safe fix)",
    "schwab_aggressive": "Schwab aggressive ETF proxy",
    "momentum_top7": "Simple momentum: top 7, equal weight",
}


def _bytes(path: Path, value: bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only new versioned outputs may be written")
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Conflicting immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _snapshot(path: Path, arrays: dict) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _bytes(path, buffer.getvalue())


def _read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def _prepare() -> dict:
    _verify_fits()
    contract_path = OUTPUT / "extension_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        _verify_hashes(contract["input_sha256"])
        return contract
    fit = json.loads((OUTPUT / "fit_contract.json").read_text())
    protected = dict(fit["input_sha256"])
    for path in [OUTPUT / "fit_contract.json", OUTPUT / "FITS_COMPLETED.json",
                 *sorted((OUTPUT / "fit_checkpoints").glob("*.json")),
                 *sorted(HERE.glob("*.py")), HERE / "README.md",
                 FRAMEWORK / "schwab_benchmark_v1/weights_schedule.json"]:
        protected[str(path)] = _hash(path)
    for path in (OUTPUT / "fit_checkpoints").glob("*.json"):
        protected.update(json.loads(path.read_text())["artifact_sha256"])
    old = _read(OLD / "paths.npz")
    dates = fit["configs"]["broad14"]["dates"][24:]
    assets = old["assets"].tolist()
    panel = pd.read_parquet(DATA / "dataset/global.parquet")
    panel["date"] = pd.to_datetime(panel.date).dt.strftime("%Y-%m-%d")
    targets = panel.pivot(index="date", columns="asset_id", values="y_excess_lead").reindex(columns=assets)
    np.testing.assert_array_equal(targets.loc[old["dates"]].to_numpy(), old["actual"])
    actual = targets.loc[dates].to_numpy()
    budget = []
    for index, date in enumerate(dates, 24):
        train = targets.loc[targets.index < date].to_numpy()
        if train.shape != (84 + index, 15) or not np.isfinite(train).all():
            raise ValueError("Risk budget must use only complete expanding pre-origin targets")
        matrix = LedoitWolf().fit(train).covariance_
        equal = np.ones(15) / 15
        budget.append(float(equal @ matrix @ equal))
    models = {}
    for component in ["broad14", "compact5"]:
        means, covariances = [], []
        for seed in old["seeds"]:
            seed_means, seed_covariances = [], []
            for date in dates:
                path = OUTPUT / "runs" / f"{component}_s{seed}" / "analytic_months" / f"covariance_{date}.npz"
                values = _read(path)
                np.testing.assert_array_equal(values["assets"], old["assets"])
                predictions = pd.read_csv(path.with_name(f"predictions_{date}.csv"))
                np.testing.assert_array_equal(predictions.asset_id, old["assets"])
                np.testing.assert_allclose(predictions.y_true, actual[dates.index(date)], rtol=0, atol=1e-15)
                seed_means.append(values["mean"])
                seed_covariances.append(values["covariance"])
            means.append(seed_means)
            covariances.append(seed_covariances)
        models[component] = (np.asarray(means), np.asarray(covariances))
    mean, covariance = blend(models["broad14"][0], models["compact5"][0],
                             models["broad14"][1], models["compact5"][1])
    irx = pd.read_parquet(DATA / "raw/combined/INDEX_IRX.parquet")["Adj Close"]
    daily = (irx / 100 / 252).reindex(pd.bdate_range(irx.index.min(), irx.index.max())).ffill()
    risk_free = []
    for date in dates:
        point = pd.Timestamp(date)
        end = point + pd.offsets.BMonthEnd(1)
        interval = daily.loc[(daily.index > point) & (daily.index <= end)]
        if end > daily.index.max() or interval.empty or interval.isna().any():
            raise ValueError("Complete realized risk-free outcome required")
        risk_free.append(np.expm1(interval.sum()))
    _snapshot(OUTPUT / "new_forecasts.npz", {
        "dates": np.array(dates), "assets": old["assets"], "seeds": old["seeds"], "actual": actual,
        "risk_free": np.array(risk_free), "risk_budget": np.array(budget), "mean": mean, "covariance": covariance,
        "broad_mean": models["broad14"][0], "compact_mean": models["compact5"][0],
        "broad_covariance": models["broad14"][1], "compact_covariance": models["compact5"][1],
    })
    contract = {
        "schema": "bayesfolio.append_wealth_comparison.v1", "input_sha256": protected,
        "old_months": 24, "new_months": 4, "new_fits": 24, "old_refits": 0,
        "compact_mean_fractions": {"US_equity": 0.5, "international_equity": 0,
                                   "US_fixed_income": 0.375, "international_fixed_income": 0.25},
        "compact_covariance_fraction": 0.5, "CVaR": False, "fallback": False, "constraint_relaxation": False,
        "hard_rank": {"minimum_effective": 7, "cap": None, "risk_budget": "training_historical_EW_variance"},
        "scenario_max_sharpe": {"minimum_effective": 10, "cap": 0.25, "draws": 5000, "risk_measure": "MV"},
        "limitations": ["Gross of costs and taxes; Schwab is a simulated ETF proxy, not a managed account.",
                        "Earlier 24 months were used in research selection; the extension is not an untouched holdout.",
                        "Three GP seeds share one market history; seed range is not statistical confidence.",
                        "Nonpositive-mean signed-Sharpe multistart solutions are not global optimality certificates."],
    }
    _write(contract_path, contract)
    return contract


def _allocation(task: tuple[int, int]) -> dict:
    seed_index, month = task
    panel = _read(OUTPUT / "new_forecasts.npz")
    seed, date = int(panel["seeds"][seed_index]), str(panel["dates"][month])
    destination = OUTPUT / "allocations" / f"s{seed}_{date}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    mean, covariance = panel["mean"][seed_index, month], panel["covariance"][seed_index, month]
    rank = hard_rank(mean, covariance, float(panel["risk_budget"][month]))
    scenarios = sample_joint(mean, covariance, seed=seed, month=date)
    scenario_path = OUTPUT / "scenarios" / f"s{seed}_{date}.npz"
    _snapshot(scenario_path, {"scenarios": scenarios})
    result = allocate_scenarios(scenarios, panel["assets"].tolist(), seed=20261024 + seed + int(date.replace("-", "")))
    result["weights"], result["raw_weights"] = result["weights"].tolist(), result["raw_weights"].tolist()
    record = {"seed": seed, "date": date, "seed_index": seed_index, "month_index": month,
              "rank": rank, "scenarios": result, "scenario_sha256": _hash(scenario_path)}
    _write(destination, record)
    return record


def _benchmarks(panel: dict) -> tuple[np.ndarray, np.ndarray]:
    assets, dates = panel["assets"].tolist(), panel["dates"].tolist()
    columns, common_dates = [], None
    first = (pd.Period(dates[0], freq="M") - 36).start_time
    for asset in assets:
        price = pd.read_parquet(DATA / f"raw/combined/{asset}.parquet")["Adj Close"]
        price = price.loc[(price.index >= first) & (price.index <= pd.Timestamp(dates[-1]))]
        grouped = price.groupby(price.index.to_period("M"))
        observed_dates = pd.DatetimeIndex(grouped.apply(lambda series: series.index[-1])).to_numpy("datetime64[D]")
        if common_dates is not None:
            np.testing.assert_array_equal(observed_dates, common_dates)
        common_dates = observed_dates
        columns.append(grouped.last().to_numpy())
    signals = benchmark_panel(common_dates, np.column_stack(columns), dates, assets, assets[:8])
    schedule = json.loads((FRAMEWORK / "schwab_benchmark_v1/weights_schedule.json").read_text())
    profile = schedule["profiles"]["aggressive"]
    targets = scheduled_weights(dates, profile["initial_weights"], profile["verified_revisions"])["weights"]
    schwab = np.zeros((4, 16))
    for index, proxy in enumerate(schedule["proxy_order"]):
        schwab[:, assets.index(proxy) if proxy != "CASH" else -1] = targets[:, index]
    momentum = np.column_stack([signals["momentum_top7"], np.zeros(4)])
    _snapshot(OUTPUT / "benchmark_signals.npz", signals)
    return schwab, momentum


def _score() -> dict:
    old = _read(OLD / "paths.npz")
    new = _read(OUTPUT / "new_forecasts.npz")
    tail = {key: new[key] for key in ["dates", "actual", "risk_free", "mean", "covariance"]}
    records = [[json.loads((OUTPUT / "allocations" / f"s{seed}_{date}.json").read_text())
                for date in new["dates"]] for seed in new["seeds"]]
    for name, section in [("gp_hard_rank", "rank"), ("gp_scenarios", "scenarios")]:
        weights = np.array([[row[section]["weights"] for row in rows] for rows in records])
        if section == "scenarios":
            for rows in records:
                for row in rows:
                    audit_weights(np.asarray(row[section]["raw_weights"]))
        totals = np.array([portfolio_returns(new["actual"], new["risk_free"],
                                            np.column_stack([value, np.zeros(4)])) for value in weights])
        tail[f"{name}_weights"], tail[f"{name}_total_returns"] = weights, totals
        tail[f"{name}_wealth"] = old[f"{name}_wealth"][:, -1:] * np.cumprod(1 + totals, axis=1)
    for name, weights in zip(["schwab_aggressive", "momentum_top7"], _benchmarks(new), strict=True):
        totals = portfolio_returns(new["actual"], new["risk_free"], weights)
        tail[f"{name}_weights"], tail[f"{name}_total_returns"] = weights, totals
        tail[f"{name}_wealth"] = old[f"{name}_wealth"][-1] * np.cumprod(1 + totals)
    panel = append_paths(old, tail)
    _snapshot(OUTPUT / "paths.npz", panel)
    rows = []
    for name, label in LABELS.items():
        totals, paths = panel[f"{name}_total_returns"], panel[f"{name}_wealth"]
        runs = totals if totals.ndim == 2 else totals[None]
        stats = [portfolio_performance(value, panel["risk_free"]) for value in runs]
        rows.append({"strategy": name, "label": label, "ending_value": float(np.mean(paths[..., -1])),
                     "total_return": float(np.mean(paths[..., -1]) / 10000 - 1),
                     "conventional_sharpe": float(np.mean([value["conventional_sharpe"] for value in stats])),
                     "max_drawdown": float(np.mean([value["max_drawdown_total_nav"] for value in stats])),
                     "extension_return": float(np.mean(paths[..., -1] / paths[..., 24] - 1))})
        np.testing.assert_allclose(wealth_path(totals), paths, rtol=0, atol=1e-9)
    _write(OUTPUT / "results.json", {"months": 28, "strategies": rows})
    _bytes(OUTPUT / "summary.csv", pd.DataFrame(rows).to_csv(index=False).encode())
    _bytes(OUTPUT / "growth_comparison.png", _plot(panel))
    return {"passed": True, "new_GP_fits": 24, "old_GP_refits": 0, "new_month_seed_allocations_per_GP_arm": 12,
            "old_prefix_bitwise_unchanged": True, "complete_months": 28, "dropped_months": 0,
            "CVaR": False, "fallbacks": 0, "relaxed_constraints": 0,
            "new_scenario_solver_routes": [row["scenarios"]["solver_route"] for rows in records for row in rows]}


def _plot(panel: dict) -> bytes:
    origins = panel["dates"].astype("datetime64[D]")
    endings = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    timeline = np.r_[origins[:1], endings]
    figure, axis = plt.subplots(figsize=(14, 8.5), layout="constrained")
    axis.axvspan(timeline[24], timeline[-1], color="#eaf0f8", alpha=0.55)
    axis.axvline(timeline[24], color="#7a8491", linestyle="--", linewidth=1)
    for (name, label), color in zip(LABELS.items(), ["#2455a4", "#ba4536", "#b17c26", "#368275"], strict=True):
        paths = panel[f"{name}_wealth"]
        if paths.ndim == 2:
            axis.fill_between(timeline, paths.min(axis=0), paths.max(axis=0), color=color, alpha=0.1)
            paths = paths.mean(axis=0)
        axis.plot(timeline, paths, color=color, linewidth=2.7, label=f"{label} | ${paths[-1]:,.0f}")
    axis.set_title("Same current GP setup: extended portfolio-allocation comparison\n"
                   "Growth of $10,000 | June 2024–September 2026 returns | Monthly rebalancing", pad=14)
    axis.set_xlabel("Time — month-end investment value (shaded region: four newly added months)")
    axis.set_ylabel("Investment value ($)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.legend(loc="upper left", fontsize=10, framealpha=0.96)
    axis.grid(alpha=0.18)
    axis.set_xlim(timeline[0], timeline[-1] + np.timedelta64(12, "D"))
    figure.supxlabel(
        "24 new fits: two joint models × three seeds × four months; rank 2, maxiter 15. Earlier results unchanged.\n"
        "GP lines average three seeds; bands show their range, not confidence. Covariance blend remains 50/50.\n"
        "Scenarios: 5,000 joint draws/month; max signed Sharpe; effective assets ≥10; cap 25%; no CVaR or fallback.\n"
        "Hard rank: effective assets ≥7, no ETF cap, unchanged variance ceiling. "
        "Gross of explicit costs/taxes; proxy benchmarks.",
        fontsize=9,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    return buffer.getvalue()


def _verify() -> None:
    seal = json.loads((OUTPUT / "COMPLETED.json").read_text())
    _verify_hashes(seal["input_sha256"])
    _verify_hashes(seal["artifact_sha256"])
    old, panel = _read(OLD / "paths.npz"), _read(OUTPUT / "paths.npz")
    for key in old:
        axis = 1 if key in {"mean", "covariance"} or key.startswith("gp_") else 0
        np.testing.assert_array_equal(np.take(panel[key], range(old[key].shape[axis]), axis=axis), old[key])
    for name in LABELS:
        np.testing.assert_allclose(wealth_path(panel[f"{name}_total_returns"]), panel[f"{name}_wealth"],
                                   rtol=0, atol=1e-9)
    if panel["mean"].shape != (3, 28, 15) or str(panel["dates"][-1]) != "2026-08-31":
        raise ValueError("Complete 28-month evidence required")
    print("Verified 24 new fits, all four 28-month paths, unchanged old prefixes and protected evidence", flush=True)


def _execute(workers: int) -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    contract = _prepare()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_allocation, (seed, month)) for seed in range(3) for month in range(4)]
        for future in as_completed(futures):
            row = future.result()
            print(f"Allocated both arms: s{row['seed']} {row['date']}", flush=True)
    audit = _score()
    _write(OUTPUT / "audit.json", audit)
    _verify_hashes(contract["input_sha256"])
    artifacts = {str(path): _hash(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file()}
    _write(OUTPUT / "COMPLETED.json", {"passed": True, "input_sha256": contract["input_sha256"],
                                        "artifact_sha256": artifacts})
    _verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=min(15, max(1, (os.cpu_count() or 1) - 2)))
    args = parser.parse_args()
    if not 1 <= args.workers <= 15:
        parser.error("One through fifteen workers required")
    if args.execute:
        _execute(args.workers)
    else:
        _verify()
