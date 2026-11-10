"""Run a matched GP-ranking versus momentum-ranking allocation diagnostic.

Research I/O orchestration only: reuse frozen scores/covariance and the identical
strict allocator; write only this new folder's outputs. No GP fits, calibration,
source refreshes, parameter searches or changes to protected studies. Weights are
fractions, returns decimals, covariance decimal squared and account values dollars.

Usage from repository root:
    .venv/bin/python <this_file> --execute
    .venv/bin/python <this_file> --verify
Optional --workers limits six independent signal/seed jobs, never above 15 or
available CPUs minus two. Individual successful jobs checkpoint for safe resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-matched-signal-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-matched-signal-cache")
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
FRAMEWORK = HERE.parent
BENCHMARK = FRAMEWORK / "benchmark_suite_v1/outputs/current_sharpe_leader"
sys.path.append(str(FRAMEWORK / "benchmark_suite_v1"))

import numpy as np  # noqa: E402
import run_benchmarks as benchmark  # noqa: E402
from benchmark_math import portfolio_returns, wealth_path  # noqa: E402
from feasibility import allocate_checked  # noqa: E402
from metrics import paired_block_bootstrap, portfolio_performance, rank_ic  # noqa: E402
from plot_diagnostic import render_diagnostic  # noqa: E402
from signal_math import audit_weights, normalized_scores  # noqa: E402

from artifacts import Reader  # noqa: E402

LABELS = {"gp": "GP ranking + hard-rank allocator", "momentum": "Momentum ranking + same allocator"}
PERFORMANCE = [
    "ending_value",
    "compounded_total_return",
    "conventional_sharpe",
    "annualized_excess_vol",
    "max_drawdown_total_nav",
    "mean_effective_assets",
    "mean_equity_fraction",
    "variance_ceiling_binding_months",
    "decision_IC",
]


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict | str | bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new matched-signal outputs may be written")
    if isinstance(value, dict):
        value = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    payload = value.encode() if isinstance(value, str) else value
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Conflicting immutable evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _snapshot(path: Path, arrays: dict) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new matched-signal arrays may be written")
    benchmark.previous.base._snapshot(path, arrays)


def _prepare() -> dict:
    reader = Reader()
    seal = reader.read(BENCHMARK / "COMPLETED.json")
    if seal["passed"] is not True:
        raise ValueError("Benchmark suite did not complete")
    protected = {**seal["input_sha256"], **seal["artifact_sha256"]}
    benchmark.previous.base._verify(protected)
    baseline = reader.read(FRAMEWORK / "CURRENT_BASELINES.json")
    reference = next(row for row in baseline["baselines"] if row["id"] == "sharpe_leader")
    control_path = FRAMEWORK / reference["evidence"]["npz"]
    control = reader.read(control_path)
    if reader.hashes[str(control_path.resolve())] != reference["evidence"]["npz_sha256"]:
        raise ValueError("Baseline control identity changed")
    frozen = reader.read(FRAMEWORK / "uncertainty_rank_v1/outputs/input_panel.npz")
    previous = reader.read(BENCHMARK / "paths.npz")
    context = reader.read(FRAMEWORK / "full_rerun_v1/outputs/context.json")
    policy = reader.read(HERE / "policy.json")
    for path in [HERE / "README.md", *sorted(HERE.glob("*.py"))]:
        reader.read(path)
    allocator = baseline["allocator"]
    if (
        allocator["method"] != "hard_rank"
        or allocator["minimum_effective_assets"] != 7
        or allocator["concentration_tie_breaker"] != 1e-8
        or allocator["continuous_risk_penalty"] != 0
        or allocator["ETF_position_cap"] is not None
        or allocator["fixed_equity_allocation"] is not None
        or allocator["CVaR"]
        or allocator["fallback_weights"]
        or allocator["constraint_relaxation"]
    ):
        raise ValueError("Current allocator differs from the approved matched diagnostic")
    for key in ["dates", "assets", "actual", "risk_free"]:
        np.testing.assert_array_equal(previous[key], frozen[key])
    if context["dates"] != frozen["dates"].tolist() or context["assets"] != frozen["assets"].tolist():
        raise ValueError("Original case/context axes differ")
    np.testing.assert_array_equal(frozen["seeds"], policy["seeds"])
    np.testing.assert_array_equal(control["mean"], frozen["mean_sharpe"])
    np.testing.assert_array_equal(control["scores"], normalized_scores(control["mean"]))
    np.testing.assert_array_equal(control["weights"], previous["comparison_weights"])
    np.testing.assert_array_equal(
        control["covariance"], (frozen["broad_covariance"] + frozen["compact_covariance"]) / 2
    )
    # Recompute the frozen monthly momentum ratio independently, not from model inputs.
    prices = previous["signal_adjusted_prices"]
    months = previous["signal_price_dates"].astype("datetime64[M]")
    positions = np.searchsorted(months, frozen["dates"].astype("datetime64[M]"))
    raw_momentum = prices[positions - 1] / prices[positions - 12] - 1
    np.testing.assert_array_equal(raw_momentum, previous["signal_momentum_score"])
    momentum_scores = np.broadcast_to(normalized_scores(raw_momentum), control["scores"].shape).copy()
    arrays = {key: frozen[key] for key in ["dates", "assets", "seeds", "actual", "risk_free", "risk_budget"]}
    arrays.update(
        covariance=control["covariance"],
        gp_scores=control["scores"],
        gp_mean=control["mean"],
        momentum_scores=momentum_scores,
        raw_momentum=raw_momentum,
        original_gp_weights=control["weights"],
        original_gp_total_returns=previous["comparison_total_returns"],
        original_gp_wealth=previous["comparison_wealth"],
        simple_momentum_wealth=previous["momentum_top7_wealth"],
        simple_momentum_total_returns=previous["momentum_top7_total_returns"],
    )
    reader.verify()
    for path, digest in reader.hashes.items():
        if path in protected and protected[path] != digest:
            raise ValueError(f"Previously sealed source changed while reading: {path}")
    protected.update(reader.hashes)
    contract = {
        "schema": "bayesfolio.matched_signal_diagnostic.v1",
        "policy": policy,
        "input_sha256": protected,
        "families": context["families"],
        "baseline": reference,
        "baseline_GP_settings": baseline["GP_settings"],
        "allocator": allocator,
        "covariance": baseline["covariance"],
        "bootstrap": policy["bootstrap"],
        "only_changed_input": "normalized_ranking_scores",
        "notes": "No top-k selection in matched momentum. Both arms retain GP covariance. "
        "No momentum NLL or monthly expected-return interpretation is invented.",
    }
    _write(OUTPUT / "contract.json", contract)
    _snapshot(OUTPUT / "input_panel.npz", arrays)
    return contract


def _allocation_job(task: tuple[str, int, int]) -> dict:
    signal, offset, seed = task
    path = OUTPUT / "cases" / f"{signal}_s{seed}.json"
    if path.exists():
        record = _read(path)
        if record["signal"] != signal or record["seed"] != seed or record["offset"] != offset:
            raise ValueError("Cached job identity differs")
        return record
    contract = _read(OUTPUT / "contract.json")
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as panel:
        covariance, budget = panel["covariance"][offset], panel["risk_budget"]
        scores = panel[f"{signal}_scores"][offset]
    result = allocate_checked(
        covariance,
        budget,
        contract["policy"]["minimum_effective_assets"],
        contract["policy"]["concentration_tie_breaker"],
        scores,
    )
    if result["status"] != "allocated" or len(result["allocations"]) != 24:
        raise ValueError(f"Incomplete strict allocation for {signal}, seed {seed}: {result['status']}")
    for month in result["allocations"]:
        if (
            month["status"] != "solved"
            or month["weights"] is None
            or month["numerical_attempts"][-1]["status"] != "optimal"
        ):
            raise ValueError(f"Failed monthly allocation for {signal}, seed {seed}: {month}")
    record = {"signal": signal, "seed": seed, "offset": offset, **result}
    _write(path, record)
    return record


def _score(contract: dict) -> tuple[dict, dict, dict]:
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source:
        panel = {key: source[key].copy() for key in source.files}
    arrays = {
        key: panel[key]
        for key in [
            "dates",
            "assets",
            "seeds",
            "actual",
            "risk_free",
            "risk_budget",
            "covariance",
            "gp_scores",
            "momentum_scores",
            "simple_momentum_wealth",
        ]
    }
    actual, risk_free = panel["actual"], panel["risk_free"]
    scopes = {
        "pooled": list(range(15)),
        **{
            family: [panel["assets"].tolist().index(t) for t in tickers]
            for family, tickers in contract["families"].items()
        },
    }
    equity = scopes["us_equity"] + scopes["international_equity"]
    rows, replay = {}, {}
    for signal, label in LABELS.items():
        jobs = [_read(OUTPUT / "cases" / f"{signal}_s{seed}.json") for seed in panel["seeds"]]
        for offset, job in enumerate(jobs):
            if (
                job["signal"] != signal
                or job["offset"] != offset
                or job["seed"] != int(panel["seeds"][offset])
                or job["status"] != "allocated"
                or len(job["allocations"]) != 24
                or not all(row["status"].startswith("feasible_") for row in job["preflight"])
            ):
                raise ValueError("Incomplete cached allocation job")
            for index, month in enumerate(job["allocations"]):
                if (
                    month["index"] != index
                    or month["status"] != "solved"
                    or month["fallback_used"]
                    or month["constraint_relaxation"]
                    or month["ETF_position_cap"] is not None
                    or month["numerical_attempts"][-1]["status"] != "optimal"
                ):
                    raise ValueError("Cached solver record violates the unchanged acceptance rules")
        weights = np.array([[month["weights"] for month in job["allocations"]] for job in jobs])
        audit = audit_weights(weights, panel["covariance"], panel["risk_budget"])
        arrays[f"{signal}_weights"] = weights
        for key, value in audit.items():
            arrays[f"{signal}_{key}"] = value
        scoped = {}
        for name, indices in scopes.items():
            values = np.array([rank_ic(actual[:, indices], scores[:, indices]) for scores in panel[f"{signal}_scores"]])
            if not np.isfinite(values).all():
                raise ValueError(f"Undefined {signal} ranking IC in {name}; no months dropped")
            scoped[name] = {"mean": float(values.mean()), "by_seed_month": values.tolist()}
            arrays[f"{signal}_ic_{name}"] = values
        family_average = np.mean([arrays[f"{signal}_ic_{family}"] for family in contract["families"]], axis=0)
        arrays[f"{signal}_ic_equal_family_average"] = family_average
        scoped["equal_family_average"] = {
            "mean": float(family_average.mean()),
            "by_seed_month": family_average.tolist(),
        }
        outcomes, totals, wealths = [], [], []
        for offset, seed in enumerate(panel["seeds"]):
            cash_inclusive = np.column_stack([weights[offset], np.zeros(24)])
            total = portfolio_returns(actual, risk_free, cash_inclusive)
            wealth = wealth_path(total)
            benchmark.previous._audit_wealth(actual, risk_free, cash_inclusive, wealth)
            outcomes.append(
                {
                    "seed": int(seed),
                    **portfolio_performance(total, risk_free),
                    "ending_value": float(wealth[-1]),
                    "mean_effective_assets": float(audit["effective_count"][offset].mean()),
                    "mean_equity_fraction": float(weights[offset][:, equity].sum(axis=1).mean()),
                    "variance_ceiling_binding_months": int((audit["variance_ratio"][offset] >= 1 - 1e-5).sum()),
                    "decision_IC": float(arrays[f"{signal}_ic_pooled"][offset].mean()),
                }
            )
            totals.append(total)
            wealths.append(wealth)
        arrays.update({f"{signal}_total_returns": np.array(totals), f"{signal}_wealth": np.array(wealths)})
        summary = {key: float(np.mean([row[key] for row in outcomes])) for key in PERFORMANCE}
        rows[signal] = {"label": label, "summary": summary, "seeds": outcomes, "rank_IC": scoped}
        if signal == "gp":
            difference = float(np.max(np.abs(weights - panel["original_gp_weights"])))
            np.testing.assert_allclose(weights, panel["original_gp_weights"], rtol=0, atol=1e-7)
            for offset in range(3):
                old = portfolio_performance(panel["original_gp_total_returns"][offset], risk_free)
                for key in [
                    "conventional_sharpe",
                    "compounded_total_return",
                    "annualized_excess_vol",
                    "max_drawdown_total_nav",
                ]:
                    np.testing.assert_allclose(outcomes[offset][key], old[key], rtol=0, atol=1e-8)
            np.testing.assert_allclose(
                summary["decision_IC"],
                contract["baseline"]["recorded_three_seed_mean_metrics"]["decision_IC"],
                rtol=0,
                atol=1e-12,
            )
            replay = {
                "passed": True,
                "maximum_absolute_weight_difference": difference,
                "maximum_absolute_monthly_return_difference": float(
                    np.max(np.abs(arrays["gp_total_returns"] - panel["original_gp_total_returns"]))
                ),
            }
    bootstrap_options = {
        "samples": contract["bootstrap"]["samples"],
        "block": contract["bootstrap"]["block_months"],
        "seed": contract["bootstrap"]["seed"],
    }
    paired_returns = paired_block_bootstrap(
        (arrays["gp_total_returns"] - risk_free).T,
        (arrays["momentum_total_returns"] - risk_free).T,
        **bootstrap_options,
    )
    paired_ic = {}
    for scope in [*scopes, "equal_family_average"]:
        diagnostic = paired_block_bootstrap(
            arrays[f"gp_ic_{scope}"].T, arrays[f"momentum_ic_{scope}"].T, **bootstrap_options
        )
        paired_ic[scope] = {key: value for key, value in diagnostic.items() if "sharpe" not in key.lower()}
    difference = arrays["gp_weights"] - arrays["momentum_weights"]
    contributions = np.mean(difference * actual[None, ...] * (1 + risk_free)[None, :, None], axis=(0, 1))
    np.testing.assert_allclose(
        contributions.sum(), (arrays["gp_total_returns"] - arrays["momentum_total_returns"]).mean(), rtol=0, atol=1e-14
    )
    results = {
        "matched": rows,
        "GP_minus_momentum": {
            key: rows["gp"]["summary"][key] - rows["momentum"]["summary"][key] for key in PERFORMANCE
        },
        "paired_exploratory_returns": paired_returns,
        "paired_exploratory_rank_IC": paired_ic,
        "family_mean_monthly_return_gap_basis_points": {
            name: float(contributions[indices].sum() * 10000) for name, indices in scopes.items() if name != "pooled"
        },
        "mean_half_L1_between_strategy_weight_difference": float(np.abs(difference).sum(axis=-1).mean() / 2),
        "context_only_original_top7": portfolio_performance(panel["simple_momentum_total_returns"], risk_free),
    }
    results["context_only_original_top7"]["ending_value"] = float(panel["simple_momentum_wealth"][-1])
    audit = {
        "passed": True,
        "GP_control_replay": replay,
        "monthly_portfolios_solved_and_audited": 144,
        "same_covariance_in_both_arms": True,
        "same_original_allocator_and_constraints": True,
        "same_seed_month_axes": True,
        "only_ranking_input_changed": True,
        "independent_dollar_holdings_checks": True,
        "return_gap_attribution_identity_checked": True,
        "momentum_signal_recomputed_from_frozen_past_prices": True,
        "new_GP_fits": 0,
        "weight_repairs": 0,
        "fallback_weights": 0,
        "omitted_months": 0,
    }
    return arrays, results, audit


def _report(results: dict, audit: dict) -> str:
    gp, momentum = (results["matched"][key]["summary"] for key in LABELS)
    delta = results["GP_minus_momentum"]
    lines = [
        "# Matched GP and momentum ranking diagnostic",
        "",
        f"Replacing GP ranking with momentum changes the three-seed mean conventional Sharpe from "
        f"{gp['conventional_sharpe']:.3f} to {momentum['conventional_sharpe']:.3f}, cumulative return from "
        f"{gp['compounded_total_return']:.2%} to {momentum['compounded_total_return']:.2%}, and pooled "
        f"rank IC from {gp['decision_IC']:.4f} to {momentum['decision_IC']:.4f}. Only ranking scores "
        "change in the matched comparison; both arms retain the same seed/month GP covariance, "
        "allocator, risk budget and constraints.",
        "",
        "## Portfolio results",
        "",
        "| Ranking signal | Ending value | Cumulative return | Sharpe | Annualized excess volatility | "
        "Maximum month-end drawdown | Rank IC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results["matched"].items():
        data = row["summary"]
        lines.append(
            f"| {name} | ${data['ending_value']:,.2f} | {data['compounded_total_return']:.2%} | "
            f"{data['conventional_sharpe']:.3f} | {data['annualized_excess_vol']:.2%} | "
            f"{data['max_drawdown_total_nav']:.2%} | {data['decision_IC']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"The GP-minus-momentum ending-value gap is ${delta['ending_value']:,.2f} per initial $10,000. "
            "The 24 realized return months are June 2024 through May 2026. Distributions are reinvested, "
            "there are no external cash flows, and returns are gross of explicit costs and taxes. Metrics "
            "average individual seed statistics; seeds are not independent market histories.",
            "",
            "## Family ranking and return gaps",
            "",
            "| Scope | GP rank IC | Momentum rank IC | GP minus momentum mean monthly return contribution |",
            "|---|---:|---:|---:|",
        ]
    )
    for scope, row in results["matched"]["gp"]["rank_IC"].items():
        other = results["matched"]["momentum"]["rank_IC"][scope]
        contribution = results["family_mean_monthly_return_gap_basis_points"].get(scope)
        value = f"{contribution:.2f} basis points/month" if contribution is not None else "Not a family attribution"
        lines.append(f"| {scope} | {row['mean']:.4f} | {other['mean']:.4f} | {value} |")
    old = results["context_only_original_top7"]
    lines.extend(
        [
            "",
            "Family contributions sum to the average monthly arithmetic-return gap. They are an "
            "accounting attribution, not independent causal effects or an attribution of compound return/Sharpe.",
            "",
            "## What the diagnostic establishes",
            "",
            "This comparison isolates the effect of replacing this particular ranking signal conditional "
            "on the frozen GP risk inputs and allocator. It does not establish whether GP covariance is "
            "accurate or necessary: momentum still uses it. No model refits, blend changes, parameter search "
            "or production promotion occurred. Momentum is a long-horizon ranking signal, not a calibrated "
            "one-month distribution; no momentum NLL is invented.",
            "",
            f"The previously tested top-seven equal-weight momentum strategy ended at ${old['ending_value']:,.2f} "
            f"with Sharpe {old['conventional_sharpe']:.3f}. Its dashed chart line is context only: it uses "
            "a different allocation rule and is not a matched control. A difference from that line changes "
            "the allocation/risk handling package, not the momentum signal.",
            "",
            "The period was repeatedly reused to develop the GP strategy. Historical rankings and returns "
            "do not establish future superiority. The same frozen retrospective adjusted-price and "
            "frictionless month-end execution limitations apply. Maximum drawdowns are month-end, not intramonth.",
            "",
        ]
    )
    paired = results["paired_exploratory_returns"]
    lines.extend(
        [
            "## Paired exploratory uncertainty",
            "",
            f"The conditional GP-minus-momentum Sharpe interval is {paired['sharpe_difference_ci95']}; "
            f"the pooled mean-IC-difference interval is "
            f"{results['paired_exploratory_rank_IC']['pooled']['mean_difference_ci95']}. These use 2,000 "
            "paired circular bootstrap draws with three-month blocks and fixed seed 20261020. They are "
            "not selection-adjusted confidence guarantees; all model seeds share the same sampled months.",
            "",
            "## Verification and artifacts",
            "",
            f"All {audit['monthly_portfolios_solved_and_audited']} raw monthly allocations passed the "
            "original feasibility and acceptance checks. The GP control replay's maximum absolute weight "
            f"difference is {audit['GP_control_replay']['maximum_absolute_weight_difference']:.3g}. "
            "No weights were clipped, normalized, substituted or carried forward after a failure.",
            "",
            "input_panel.npz freezes the identical allocation risk and both score panels. paths.npz "
            "records weights, returns, wealth, IC, effective counts and variance ratios. Per-seed cases "
            "record raw solver attempts and preflight checks. GP and momentum weight exports have explicit "
            "date/asset axes for reuse in the standing benchmark suite. The completion seal protects "
            "source code, policy, prior evidence, numerical outputs, chart and this report.",
            "",
            "![Matched ranking diagnostic](matched_signal.png)",
            "",
            "Exact execution and verification commands are in the experiment README. Pylance diagnostics "
            "are unavailable in the current tool environment; no completed Pylance gate is asserted.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify() -> None:
    completed = _read(OUTPUT / "COMPLETED.json")
    if completed["passed"] is not True:
        raise ValueError("Matched diagnostic incomplete")
    benchmark.previous.base._verify(completed["input_sha256"])
    benchmark.previous.base._verify(completed["artifact_sha256"])
    print("Verified matched ranking-signal diagnostic and unchanged prior evidence")


def _execute(requested_workers: int) -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    started = time.monotonic()
    contract = _prepare()
    tasks = [(signal, offset, seed) for signal in LABELS for offset, seed in enumerate(contract["policy"]["seeds"])]
    workers = min(requested_workers, 15, max(1, (os.cpu_count() or 3) - 2), len(tasks))
    print(
        json.dumps({"jobs": len(tasks), "workers": workers, "monthly_allocation_solves": 144, "GP_fits": 0}), flush=True
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_allocation_job, task) for task in tasks]):
            result = future.result()
            print(f"Completed {result['signal']} seed {result['seed']}: 24 strict allocations", flush=True)
    arrays, results, audit = _score(contract)
    _write(OUTPUT / "results.json", results)
    _write(OUTPUT / "audit.json", audit)
    _snapshot(OUTPUT / "paths.npz", arrays)
    for signal in LABELS:
        _snapshot(
            OUTPUT / f"{signal}_weights.npz",
            {
                "weights": arrays[f"{signal}_weights"],
                "assets": arrays["assets"],
                "dates": arrays["dates"],
                "seeds": arrays["seeds"],
            },
        )
    _write(OUTPUT / "matched_signal.png", render_diagnostic(arrays))
    _write(OUTPUT / "FINDINGS.md", _report(results, audit))
    _write(OUTPUT / "runtime.json", {"seconds": time.monotonic() - started, "workers": workers, "jobs": len(tasks)})
    benchmark.previous.base._verify(contract["input_sha256"])
    _write(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": contract["input_sha256"],
            "artifact_sha256": {
                str(path): benchmark.previous.base._hash(path)
                for path in OUTPUT.rglob("*")
                if path.is_file() and path.name != "COMPLETED.json"
            },
        },
    )
    _verify()
    print(json.dumps({name: row["summary"] for name, row in results["matched"].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=15)
    options = parser.parse_args()
    if options.workers < 1:
        parser.error("--workers must be positive")
    if options.execute:
        _execute(options.workers)
    else:
        _verify()
