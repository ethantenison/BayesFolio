"""Compare five frozen benchmark rules with saved monthly BayesFolio weights.

Research-only I/O orchestration: reads sealed evidence and frozen adjusted prices,
uses pure numerical strategies, and writes only a new versioned run directory.
Returns and weights are decimals; prices and investment values are dollars.
No GP fits, allocator optimization, data refresh, or production changes occur.

Usage from repository root:
    .venv/bin/python <this_file> --preview
    .venv/bin/python <this_file> --execute
    .venv/bin/python <this_file> --verify
For another saved portfolio on the identical evaluation grid, use --comparison-npz
<file> --label <name> --run-id <new_name>. NPZ requires weights, dates and assets;
weights have shape (months, assets) or (runs, months, assets), without cash.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-benchmark-suite-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-benchmark-suite-cache")
HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
PRIOR = FRAMEWORK / "schwab_benchmark_v1"
REPAIRED = FRAMEWORK.parent / "2026-10-repaired-global-feature-selection"
sys.path.append(str(PRIOR))

import compare_schwab as previous  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from benchmark_math import portfolio_returns, scheduled_weights, wealth_path  # noqa: E402
from metrics import paired_block_bootstrap, portfolio_performance  # noqa: E402
from plot_suite import render_growth  # noqa: E402
from strategy_math import benchmark_panel  # noqa: E402

from artifacts import Reader  # noqa: E402

LABELS = {
    "schwab_aggressive": "Schwab-style aggressive",
    "equal_weight": "Equal weight — 15 ETFs",
    "fixed_80_20": "Fixed 80/20 — equal within sleeves",
    "inverse_volatility": "Inverse volatility — trailing 36 months",
    "momentum_top7": "Momentum — top 7, 12-minus-1 month",
}
METRICS = [
    "ending_value",
    "compounded_total_return",
    "conventional_sharpe",
    "annualized_excess_vol",
    "max_drawdown_total_nav",
]


def _write(path: Path, value: bytes | str | dict) -> None:
    if not path.resolve().is_relative_to((HERE / "outputs").resolve()):
        raise ValueError("Only this suite's new outputs are writable")
    if isinstance(value, dict):
        value = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    payload = value.encode() if isinstance(value, str) else value
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Conflicting immutable artifact; choose a new --run-id: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _history(reader: Reader, assets: list[str], origins: list[str], lookback: int) -> tuple[np.ndarray, np.ndarray]:
    raw = REPAIRED / "artifacts/v1/raw"
    metadata = reader.read(raw / "sources.json")
    first = pd.Period(origins[0], freq="M") - lookback
    expected = pd.period_range(first, pd.Period(origins[-1], freq="M"), freq="M")
    columns, common_dates = [], None
    for ticker in assets:
        path = raw / f"{ticker}.parquet"
        frame = reader.read(path)
        if reader.hashes[str(path.resolve())] != metadata[ticker]["sha256"]:
            raise ValueError(f"Changed registered adjusted-price source: {ticker}")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Daily source chronology differs: {ticker}")
        selected = frame.loc[
            (frame.index >= first.start_time) & (frame.index <= pd.Timestamp(origins[-1])), "Adj Close"
        ]
        if selected.isna().any() or not np.isfinite(selected.to_numpy()).all() or (selected <= 0).any():
            raise ValueError(f"Invalid daily adjusted prices: {ticker}")
        grouped = selected.groupby(selected.index.to_period("M"))
        monthly = grouped.last()
        dates = pd.DatetimeIndex(grouped.apply(lambda series: series.index[-1])).to_numpy(dtype="datetime64[D]")
        if not monthly.index.equals(expected):
            raise ValueError(f"Incomplete monthly signal history: {ticker}")
        if common_dates is None:
            common_dates = dates
        else:
            np.testing.assert_array_equal(common_dates, dates)
        columns.append(monthly.to_numpy(dtype=float))
    if common_dates is None:
        raise ValueError("Nonempty ETF universe required")
    return common_dates, np.column_stack(columns)


def _audit_signals(dates: np.ndarray, prices: np.ndarray, origins: list[str], assets: list[str], panel: dict) -> None:
    for row, origin in enumerate(origins):
        end = np.flatnonzero(dates.astype("datetime64[M]") == np.datetime64(origin, "M")).item()
        observed = prices[end - 35 : end + 1] / prices[end - 36 : end] - 1
        sigma = np.array([np.std(observed[:, i], ddof=1) for i in range(len(assets))])
        np.testing.assert_allclose(panel["volatility"][row], sigma, rtol=0, atol=1e-14)
        score = np.array([prices[end - 1, i] / prices[end - 12, i] - 1 for i in range(len(assets))])
        np.testing.assert_allclose(panel["momentum_score"][row], score, rtol=0, atol=1e-14)
        ordered = sorted(range(len(assets)), key=lambda i: (-score[i], assets[i]))[:7]
        assert sorted(np.flatnonzero(panel["momentum_selected"][row]).tolist()) == sorted(ordered)


def _comparison(reader: Reader, options: argparse.Namespace, source: dict) -> tuple[np.ndarray, str]:
    if options.comparison_npz is None:
        return source["GP_weights"], options.label or "Sharpe leader — hard rank"
    if options.run_id == "current_sharpe_leader":
        raise ValueError("A custom comparator requires a distinct --run-id")
    custom = reader.read(Path(options.comparison_npz).resolve())
    for key in ["dates", "assets"]:
        np.testing.assert_array_equal(custom[key], source[key])
    weights = custom["weights"]
    if weights.ndim == 2:
        weights = weights[None, ...]
    return weights, options.label or "BayesFolio comparator"


def _calculate(options: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    reader = Reader()
    prior_path = PRIOR / "outputs/COMPLETED.json"
    seal = reader.read(prior_path)
    if seal["passed"] is not True:
        raise ValueError("Previous benchmark evidence incomplete")
    protected = {**seal["input_sha256"], **seal["artifact_sha256"]}
    previous.base._verify(protected)
    source = reader.read(PRIOR / "outputs/paths.npz")
    old_results = reader.read(PRIOR / "outputs/results.json")
    baselines = reader.read(FRAMEWORK / "CURRENT_BASELINES.json")
    context = reader.read(FRAMEWORK / "full_rerun_v1/outputs/context.json")
    policy = reader.read(HERE / "suite.json")
    schedule = reader.read(PRIOR / "weights_schedule.json")
    for path in [HERE / "README.md", *sorted(HERE.glob("*.py"))]:
        reader.read(path)
    origins, assets = source["dates"].tolist(), source["assets"].tolist()
    if context["dates"] != origins or context["assets"] != assets or list(LABELS) != policy["benchmark_ids"]:
        raise ValueError("Exact common axes and approved benchmark order required")
    history_dates, prices = _history(reader, assets, origins, policy["inverse_volatility"]["lookback_months"])
    equities = context["families"]["us_equity"] + context["families"]["international_equity"]
    panel = benchmark_panel(
        history_dates,
        prices,
        origins,
        assets,
        equities,
        volatility_months=policy["inverse_volatility"]["lookback_months"],
        momentum_months=policy["momentum_top7"]["lookback_months"],
        momentum_skip=policy["momentum_top7"]["skip_months"],
        top_k=policy["momentum_top7"]["top_k"],
        equity_fraction=policy["fixed_80_20"]["equity_fraction"],
    )
    _audit_signals(history_dates, prices, origins, assets, panel)
    count, dimension = source["actual"].shape
    arrays = {key: source[key] for key in ["dates", "assets", "actual", "risk_free"]}
    arrays.update(signal_price_dates=history_dates.astype(str), signal_adjusted_prices=prices)
    arrays.update(
        {
            f"signal_{key}": value.astype(str) if value.dtype.kind == "M" else value
            for key, value in panel.items()
            if key not in LABELS
        }
    )
    selected = scheduled_weights(
        origins,
        schedule["profiles"]["aggressive"]["initial_weights"],
        schedule["profiles"]["aggressive"]["verified_revisions"],
    )
    schwab = np.zeros((count, dimension + 1))
    for column, proxy in enumerate(schedule["proxy_order"]):
        schwab[:, dimension if proxy == "CASH" else assets.index(proxy)] = selected["weights"][:, column]
    equity_mask = np.isin(assets, equities)
    comparison_weights, label = _comparison(reader, options, source)
    if (
        comparison_weights.ndim != 3
        or not len(comparison_weights)
        or comparison_weights.shape[1:] != (count, dimension)
    ):
        raise ValueError("Comparator requires complete (runs, months, assets) weights")
    seed_results, seed_total, seed_wealth = [], [], []
    for index, weights in enumerate(comparison_weights):
        cash_inclusive = np.column_stack([weights, np.zeros(count)])
        total = portfolio_returns(source["actual"], source["risk_free"], cash_inclusive)
        wealth = wealth_path(total, policy["initial_dollars"])
        previous._audit_wealth(source["actual"], source["risk_free"], cash_inclusive, wealth)
        seed_results.append(
            {"run_index": index, "ending_value": float(wealth[-1]), **portfolio_performance(total, source["risk_free"])}
        )
        if options.comparison_npz is None:
            seed_results[-1]["seed"] = baselines["evaluation_reference"]["seeds"][index]
        seed_total.append(total)
        seed_wealth.append(wealth)
    arrays.update(
        comparison_weights=comparison_weights,
        comparison_total_returns=np.array(seed_total),
        comparison_wealth=np.array(seed_wealth),
    )
    summary = {key: float(np.mean([record[key] for record in seed_results])) for key in METRICS}
    if options.comparison_npz is None:
        for key, value in old_results["Sharpe_leader"]["summary"].items():
            np.testing.assert_allclose(summary[key], value, rtol=0, atol=1e-12)
        np.testing.assert_allclose(arrays["comparison_wealth"], source["GP_wealth"], rtol=0, atol=1e-10)
    results = {"comparison": {"label": label, "summary": summary, "runs": seed_results}, "benchmarks": {}}
    for name in LABELS:
        weights = schwab if name == "schwab_aggressive" else np.column_stack([panel[name], np.zeros(count)])
        total = portfolio_returns(source["actual"], source["risk_free"], weights)
        wealth = wealth_path(total, policy["initial_dollars"])
        previous._audit_wealth(source["actual"], source["risk_free"], weights, wealth)
        if name == "schwab_aggressive":
            np.testing.assert_allclose(total, source["aggressive_total_returns"], rtol=0, atol=1e-14)
        arrays.update({f"{name}_weights": weights, f"{name}_total_returns": total, f"{name}_wealth": wealth})
        results["benchmarks"][name] = {
            "label": LABELS[name],
            "ending_value": float(wealth[-1]),
            **portfolio_performance(total, source["risk_free"]),
            "mean_equity_fraction": float(weights[:, :dimension][:, equity_mask].sum(axis=1).mean()),
            "mean_effective_assets_including_cash": float(np.mean(1 / np.sum(weights**2, axis=1))),
            "comparison_ending_value_advantage": summary["ending_value"] - float(wealth[-1]),
            "paired_exploratory": paired_block_bootstrap(
                (arrays["comparison_total_returns"] - source["risk_free"]).T,
                np.broadcast_to((total - source["risk_free"])[:, None], (count, len(comparison_weights))),
            ),
        }
    reader.verify()
    for path, digest in reader.hashes.items():
        if path in protected and protected[path] != digest:
            raise ValueError(f"Previously sealed input changed while reading: {path}")
    protected.update(reader.hashes)
    audit = {
        "passed": True,
        "months": count,
        "benchmark_paths": len(LABELS),
        "comparison_paths": len(comparison_weights),
        "lookback_windows_independently_checked": count,
        "all_monthly_dollar_holdings_verified": True,
        "no_imputation_or_weight_fallback": True,
        "source_closes_available_by_each_decision": True,
        "old_Schwab_returns_reproduced": True,
        "old_Sharpe_leader_reproduced": options.comparison_npz is None,
        "schwab_verified_revisions_applied": selected["applied_revision_count"],
    }
    contract = {
        "schema": "bayesfolio.benchmark_suite_comparison.v1",
        "policy": policy,
        "comparison_label": label,
        "comparison_source": (
            str(Path(options.comparison_npz).resolve()) if options.comparison_npz else "sealed_Sharpe_leader"
        ),
        "GP_settings": baselines["GP_settings"] if options.comparison_npz is None else None,
        "comparison_seeds": baselines["evaluation_reference"]["seeds"] if options.comparison_npz is None else None,
        "new_GP_fits": 0,
        "new_optimizer_calls": 0,
        "network_refreshes": 0,
        "input_sha256": protected,
        "assets": assets,
        "dates": origins,
        "GP_display": "mean_of_independently_compounded_saved_runs; band_is_not_confidence",
        "Schwab_revision_limitation": schedule["revision_search_result"],
        "signal_months": len(history_dates),
        "signal_price_dates": history_dates.astype(str).tolist(),
    }
    return contract, arrays, results, audit


def _report(contract: dict, results: dict, audit: dict) -> str:
    lines = [
        "# BayesFolio and standing benchmark comparison",
        "",
        f"A $10,000 investment in {results['comparison']['label']} is compared with five approved simple "
        "strategies over June 2024 through May 2026. All rebalance monthly, reinvest distributions and "
        "have no external cash flows. Results are gross of explicit trading costs and taxes.",
        "",
        "| Strategy | Ending value | Total return | Conventional Sharpe | Annualized excess volatility | "
        "Maximum month-end drawdown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [(results["comparison"]["label"], results["comparison"]["summary"])]
    rows.extend((row["label"], row) for row in results["benchmarks"].values())
    for label, row in rows:
        lines.append(
            f"| {label} | ${row['ending_value']:,.2f} | {row['compounded_total_return']:.2%} | "
            f"{row['conventional_sharpe']:.3f} | {row['annualized_excess_vol']:.2%} | "
            f"{row['max_drawdown_total_nav']:.2%} |"
        )
    lines.extend(
        [
            "",
            "GP/comparator statistics average the individual saved-run metrics. The chart averages "
            "independently compounded wealth paths, not allocations; its band is the initialization range, "
            "not a confidence interval. Maximum drawdowns use monthly observations, not intramonth lows.",
            "",
            "## Rules and interpretation",
            "",
            "Schwab aggressive retains its approved 50% SPY, 20% IWM, 25% VEA and 5% cash proxy mix. "
            "No dated revision was verified; no annual change is invented. Cash earns the shared frozen RF "
            "proxy, not a verified Schwab sweep rate. The original Schwab return path is reproduced.",
            "",
            "Equal weight holds 1/15 per ETF, or 53.33% equities and 46.67% fixed income. Fixed 80/20 "
            "holds 10% per equity ETF and 2.857% per fixed-income ETF. Inverse volatility uses sample "
            "standard deviations of 36 trailing monthly adjusted total returns available through decision t. "
            "It ignores correlations and is not full risk parity. Momentum ranks P[t-1]/P[t-12]-1 "
            "and equally weights the top seven, breaking ties alphabetically. This is 11 monthly returns "
            "inside the 12-minus-1 convention; even if all scores are negative it still holds the best seven.",
            "",
            "The 36-month lookback and top-seven count were approved before these benchmark results, "
            "not optimized on this evaluation window. No extra variance budget, effective-count floor, "
            "ETF cap or CVaR rule is imposed on the benchmarks. The GP comparator remains unchanged.",
            "",
            "Signals use frozen retrospective adjusted prices, not an archived point-in-time vendor feed. "
            "The frictionless month-end allocation convention matches the GP evaluation; this is not a "
            "verified next-open execution backtest. Different risk exposures and asset mixes prevent "
            "attributing performance differences solely to the GP forecasts or covariance. Repeated "
            "selection of the GP strategy on these 24 months limits out-of-sample conclusions.",
            "",
            "Paired block-bootstrap diagnostics are in results.json: 2,000 circular draws of three-month "
            "blocks with identical sampled months across strategies and model runs. They are exploratory "
            "and not selection-adjusted; model seeds are not additional market histories.",
            "",
            "## Reproducibility",
            "",
            f"The audit passed for all {audit['months']} months, {audit['benchmark_paths']} benchmark paths "
            f"and {audit['comparison_paths']} saved comparator paths. Every monthly dollar balance and "
            "every signal window was independently checked. There were no GP refits, optimizer calls, "
            "network refreshes or changes to previous sealed studies.",
            "",
            "paths.npz contains monthly weights, total returns and wealth for every strategy, plus exact "
            "signal inputs, scores and lookback dates. contract.json fingerprints all sources. The "
            "completion record seals this report, code, configuration, prior evidence and numerical outputs.",
            "",
            "![Investment growth](growth_10000.png)",
            "",
            "Verification and future-comparator commands are in the suite README. A changed rule, "
            "data source or comparator requires a new run/version; existing results are not overwritten.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify(output: Path) -> None:
    completed = json.loads((output / "COMPLETED.json").read_text())
    if completed["passed"] is not True:
        raise ValueError("Suite did not complete")
    previous.base._verify(completed["input_sha256"])
    previous.base._verify(completed["artifact_sha256"])
    print(f"Verified benchmark suite, saved comparator and unchanged prior evidence: {output}")


def _run(options: argparse.Namespace) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", options.run_id):
        raise ValueError("A short alphanumeric run ID without path separators is required")
    output = HERE / "outputs" / options.run_id
    if options.comparison_npz is not None and options.run_id == "current_sharpe_leader":
        raise ValueError("A custom comparator requires a distinct --run-id")
    if options.verify:
        _verify(output)
        return
    if options.execute and (output / "COMPLETED.json").exists():
        completed_contract = json.loads((output / "contract.json").read_text())
        requested_source = (
            str(Path(options.comparison_npz).resolve()) if options.comparison_npz else "sealed_Sharpe_leader"
        )
        requested_label = options.label or (
            "BayesFolio comparator" if options.comparison_npz else "Sharpe leader — hard rank"
        )
        if (
            completed_contract["comparison_source"] != requested_source
            or completed_contract["comparison_label"] != requested_label
        ):
            raise ValueError("Completed run has a different comparator or label; choose a new --run-id")
        _verify(output)
        return
    contract, arrays, results, audit = _calculate(options)
    chart = render_growth(arrays, contract["comparison_label"], LABELS)
    if options.preview:
        temporary = Path(tempfile.mkdtemp(prefix="bayesfolio-benchmark-suite-")) / "preview.png"
        temporary.write_bytes(chart)
        print(
            json.dumps(
                {
                    "preview": str(temporary),
                    "summary": results["comparison"]["summary"],
                    "benchmarks": {key: {m: row[m] for m in METRICS} for key, row in results["benchmarks"].items()},
                }
            )
        )
        return
    previous.base._verify(contract["input_sha256"])
    _write(output / "contract.json", contract)
    previous.base._snapshot(output / "paths.npz", arrays)
    _write(output / "results.json", results)
    _write(output / "audit.json", audit)
    _write(output / "growth_10000.png", chart)
    _write(output / "FINDINGS.md", _report(contract, results, audit))
    _write(
        output / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": contract["input_sha256"],
            "artifact_sha256": {
                str(path): previous.base._hash(path)
                for path in output.iterdir()
                if path.is_file() and path.name != "COMPLETED.json"
            },
        },
    )
    _verify(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "comparison": results["comparison"]["summary"],
                "benchmarks": {key: {m: row[m] for m in METRICS} for key, row in results["benchmarks"].items()},
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preview", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--comparison-npz")
    parser.add_argument("--label")
    parser.add_argument("--run-id", default="current_sharpe_leader")
    _run(parser.parse_args())
