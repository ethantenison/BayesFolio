"""Re-evaluate the 36 frozen scenario recipes at NE7, with no GP refits.

Research-only orchestration: exact same 15 ETFs, 28 months, three seeds, 5000
frozen-key scenarios, signed-Sharpe objective and cap25%; only NE10 becomes NE7.
Writes a separate versioned outputs folder. Returns are decimals, weights
fractions, covariance decimal squared and NAV dollars. Never changes old evidence.

Usage:
    uv run --locked --no-sync python <this_file> --execute --workers 14
    uv run --locked --no-sync python <this_file> --verify
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
SOURCE = CAMPAIGN / "scenario_numerics_v2/outputs"
COMPONENTS = CAMPAIGN / "outputs/components"
OUTPUT = HERE / "outputs"
BASE_ID = "t1_r2_i15_blend"
sys.path[:0] = [str(HERE), str(CAMPAIGN / "track1_scenarios"), str(CAMPAIGN / "track2_asymmetric")]

import case_evaluator as evaluator  # noqa: E402
from ne7_math import Policy, allocate_scenarios, audit_weights  # noqa: E402
from recipe_math import pareto_ids, select_candidates  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _save(path: Path, content: dict | bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only versioned NE7 outputs are writable")
    payload = content if isinstance(content, bytes) else (
        json.dumps(evaluator._jsonable(content), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    evaluator._immutable_bytes(path, payload)


def _verify_hashes(hashes: dict) -> None:
    for name, expected in hashes.items():
        if _hash(Path(name)) != expected:
            raise ValueError(f"Protected evidence changed: {name}")


def _verify_source() -> None:
    seal = _read(SOURCE / "COMPLETED.json")
    if seal.get("passed") is not True:
        raise ValueError("Repaired campaign completion seal required")
    for key in ["implementation_sha256", "artifact_sha256"]:
        _verify_hashes(seal[key])
    original = CAMPAIGN / "outputs/COMPLETED.json"
    if _hash(original) != seal["original_seal_sha256"]:
        raise ValueError("Original campaign seal changed")
    for key in ["input_sha256", "implementation_sha256", "artifact_sha256"]:
        _verify_hashes(_read(original)[key])


def _evaluate(source_id: str) -> dict:
    destination = OUTPUT / "cases" / f"{source_id}.json"
    if destination.exists():
        previous = _read(destination)
        _verify_hashes(previous["input_sha256"])
        if _hash(destination.with_suffix(".npz")) != previous["artifact_sha256"]:
            raise ValueError("Cached NE7 case changed")
        return previous
    context = _arrays(SOURCE / "context.npz")
    source_path = SOURCE / "cases" / f"{source_id}.json"
    old = _read(source_path)
    original = _arrays(source_path.with_suffix(".npz"))
    policy = Policy()
    arrays = {key: original[key].copy() for key in
              ["mean", "covariance", "dates", "assets", "seeds", "actual", "risk_free"]}
    arrays.update(weights=np.full((3, 28, 15), np.nan), raw_weights=np.full((3, 28, 15), np.nan),
                  allocated=np.zeros((3, 28), dtype=bool), effective_count=np.full((3, 28), np.nan),
                  equity_weight=np.full((3, 28), np.nan), predicted_variance=np.full((3, 28), np.nan))
    numerical, failures = [], []
    sampling = evaluator._dependency("sampling")
    for s, seed in enumerate(context["seeds"]):
        for m, date in enumerate(context["dates"]):
            origin = str(date)
            identity = {"seed": int(seed), "origin": origin, "seed_index": s, "month_index": m}
            scenarios = sampling.sample_joint(arrays["mean"][s, m], arrays["covariance"][s, m],
                                               seed=int(seed), month=origin, count=5000)
            try:
                result = allocate_scenarios(scenarios, context["assets"].tolist(), policy=policy,
                                            seed=20261024 + int(seed) + int(origin.replace("-", "")))
                w, raw = result["weights"], result["raw_weights"]
                audit_weights(w, policy)
                audit_weights(raw, policy)
                np.testing.assert_array_equal(w, raw)
                arrays["weights"][s, m], arrays["raw_weights"][s, m] = w, raw
                arrays["allocated"][s, m] = True
                arrays["effective_count"][s, m] = 1 / (w @ w)
                arrays["equity_weight"][s, m] = w[:8].sum()
                arrays["predicted_variance"][s, m] = w @ arrays["covariance"][s, m] @ w
                numerical.append({**identity, "status": "allocated", **{
                    k: v for k, v in result.items() if k not in {"weights", "raw_weights"}},
                    "scenario_sha256": hashlib.sha256(scenarios.tobytes()).hexdigest()})
            except (ValueError, RuntimeError, ArithmeticError, AssertionError) as error:
                failures.append({**identity, "error": f"{type(error).__name__}: {error}"})
    record = {"schema": "bayesfolio.gp_tuning.scenario_NE7.v1", "id": source_id,
              "case": copy.deepcopy(old["case"]), "status": "failed", "periods": {}, "calibration": {},
              "successful_allocations": int(arrays["allocated"].sum()), "expected_allocations": 84,
              "policy": {"minimum_effective_assets": 7, "maximum_weight": .25, "long_only": True,
                         "fully_invested": True, "portfolio_variance_ceiling": False, "CVaR": False},
              "failures": failures, "numerical_records": numerical, "GP_fits": 0,
              "same_scenario_draw_keys_as_NE10": True, "static_blending_all_months_and_seeds": True,
              "input_sha256": {str(p): _hash(p) for p in
                               [source_path, source_path.with_suffix(".npz"), SOURCE / "context.npz",
                                HERE / "ne7_math.py", Path(__file__), Path(evaluator.__file__)]}}
    if arrays["allocated"].all():
        try:
            record["periods"], record["calibration"] = evaluator._scores(
                context, arrays["mean"], arrays["covariance"], arrays)
            record["status"] = "complete"
            for period in old["periods"]:
                for key in ["rank_ic", "joint_nll", "marginal_nll"]:
                    if record["periods"][period][key] != old["periods"][period][key]:
                        raise ValueError("Allocator change must not change forecast-quality metrics")
        except (ValueError, ArithmeticError, AssertionError) as error:
            record["status"] = "failed"
            record["failures"].append({"stage": "scoring", "error": str(error)})
    arrays["status"] = np.array(record["status"])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _save(destination.with_suffix(".npz"), buffer.getvalue())
    record["artifact_sha256"] = _hash(destination.with_suffix(".npz"))
    _save(destination, record)
    return record


def _three_way_audit() -> dict:
    checked, hashes = [], {}
    for path in sorted((SOURCE / "cases").glob("t3*.json")):
        case = _read(path)["case"]
        if case["third_id"] is None:
            continue
        parts = {}
        for key in ["broad_id", "compact_id", "third_id"]:
            panel = COMPONENTS / f"{case[key]}.npz"
            hashes[str(panel)] = _hash(panel)
            parts[key] = _arrays(panel)
        a = np.repeat(case["alpha"], [4, 4, 4, 3])
        g, b, c = case["third_fraction"], case["third_cov_fraction"], case["cov_compact_fraction"]
        mean = ((1 - g) * ((1 - a) * parts["broad_id"]["mean"] + a * parts["compact_id"]["mean"])
                + g * parts["third_id"]["mean"])
        covariance = ((1 - b) * ((1 - c) * parts["broad_id"]["covariance"]
                                + c * parts["compact_id"]["covariance"])
                      + b * parts["third_id"]["covariance"])
        saved = _arrays(path.with_suffix(".npz"))
        np.testing.assert_array_equal(mean, saved["mean"])
        np.testing.assert_array_equal(covariance, saved["covariance"])
        checked.append(case["id"])
        hashes.update({str(path): _hash(path), str(path.with_suffix(".npz")): _hash(path.with_suffix(".npz"))})
    if len(checked) != 96:
        raise ValueError("Expected all 96 actual three-component blend cases")
    return {"passed": True, "three_component_cases_verified": checked, "input_sha256": hashes,
            "standalone_third_model_portfolios_claimed": False,
            "prediction_formula": "(1-gamma)*((1-alpha_family)*broad + alpha_family*compact) + gamma*third",
            "covariance_formula": "(1-beta)*((1-c)*broad_cov + c*compact_cov) + beta*third_cov",
            "example_US_equity_gamma_0_125": {"broad": .4375, "compact": .4375, "third": .125},
            "international_equity_note": "Compact base fraction is zero; existing family recipe is preserved."}


def _report(records: list[dict]) -> None:
    complete = [r for r in records if r["status"] == "complete"]
    selection_rows = [{"id": r["id"], **{key: r["periods"]["tuning_first16"][key]
                                         for key in ["sharpe", "total_return", "rank_ic"]}} for r in complete]
    selected = select_candidates(selection_rows, limit=3)
    old_cases = {r["id"]: _read(SOURCE / "cases" / f"{r['id']}.json") for r in records}
    leaders = {period: {key: max(complete, key=lambda r: r["periods"][period][key])["id"]
                        for key in ["sharpe", "total_return", "rank_ic"]}
               for period in ["tuning_first16", "validation_last12", "all28"]}
    summary = {"passed": len(complete) == 36, "complete_cases": len(complete), "failed_cases": 36 - len(complete),
               "complete_allocations": sum(r["successful_allocations"] for r in records),
               "expected_allocations": 3024, "GP_fits": 0, "selected_first16_ids": selected,
               "descriptive_leaders_by_period": leaders, "first16_pareto_ids": pareto_ids(selection_rows),
               "only_policy_change": "minimum effective assets 10 -> 7; scenario cap25% unchanged",
               "hard_rank_comparison_limitations": ["Hard rank has no position cap and a variance ceiling.",
                                                     "Scenario cap is 25% and it has no variance ceiling."],
               "temporal_periods": {"tuning_first16": "June2024-September2025 returns",
                                    "validation_last12": "October2025-September2026 returns",
                                    "all28": "June2024-September2026 returns"},
               "pristine_holdout": False, "automatic_promotion": False,
               "same_scenarios_predictions_covariance_and_benchmark_data": True}
    _save(OUTPUT / "SUMMARY.json", summary)
    rows = ["id,period,NE7_Sharpe,NE10_Sharpe,NE7_return,NE10_return,rank_IC,NE7_equity,NE10_equity,NE7_effective"]
    for record in complete:
        old = old_cases[record["id"]]
        for period, v in record["periods"].items():
            prev = old["periods"][period]
            values = [record["id"], period, v["sharpe"], prev["sharpe"], v["total_return"], prev["total_return"],
                      v["rank_ic"], v["equity_weight"], prev["equity_weight"], v["effective_assets"]]
            rows.append(",".join(map(str, values)))
    _save(OUTPUT / "COMPARISONS.csv", ("\n".join(rows) + "\n").encode())
    old = old_cases[BASE_ID]
    new = next(r for r in records if r["id"] == BASE_ID)
    hard = _read(SOURCE / "cases/control_hard_rank.json")
    lines = ["# Scenario NE7 comparison and three-way blend clarification", "",
             "Only minimum effective assets changed (10 -> 7). Same GP fits, signed-Sharpe objective, "
             "5000 joint draws and draw keys, 15 ETFs, three seeds, 28 months and static blends. "
             "Cap25%, long-only, full investment and no variance ceiling remain unchanged for scenarios. "
             "Negative returns are valid; no CVaR, fallback weights, return floor or risk relaxation.", "",
             f"Complete cases: {len(complete)}/36. Complete allocations: {summary['complete_allocations']}/3024. "
             "New GP fits: 0.", "", "## Same blended GP: allocator comparison", "",
             "| Strategy | Period | Sharpe | Return | IC | Effective assets | Equity allocation |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, record in [("Hard rank NE7", hard), ("Scenarios NE10", old), ("Scenarios NE7", new)]:
        if record["status"] != "complete":
            continue
        for period in ["tuning_first16", "validation_last12", "all28"]:
            v = record["periods"][period]
            lines.append(f"| {label} | {period} | {v['sharpe']:.4f} | {v['total_return']:.2%} | "
                         f"{v['rank_ic']:.4f} | {v['effective_assets']:.3f} | {v['equity_weight']:.1%} |")
    lines.extend(["", "## First16-selected recipes; later12 is descriptive", "",
                  "| ID | Period | Sharpe | Return | IC |", "|---|---|---:|---:|---:|"])
    by_id = {r["id"]: r for r in complete}
    for identity in selected:
        for period in ["tuning_first16", "validation_last12", "all28"]:
            v = by_id[identity]["periods"][period]
            lines.append(f"| {identity} | {period} | {v['sharpe']:.4f} | "
                         f"{v['total_return']:.2%} | {v['rank_ic']:.4f} |")
    if new["status"] == "complete":
        a, b = new["periods"]["all28"], old["periods"]["all28"]
        lines.extend(["", "## Interpretation", "",
                      f"For unchanged blended forecasts, moving NE10 -> NE7 changed Sharpe "
                      f"{b['sharpe']:.3f} -> {a['sharpe']:.3f}, return {b['total_return']:.2%} -> "
                      f"{a['total_return']:.2%}, and equity allocation {b['equity_weight']:.1%} -> "
                      f"{a['equity_weight']:.1%}. This is an allocation-policy effect; "
                      "prediction IC and NLL are exactly unchanged. Relaxing a constraint can increase "
                      "forecast Sharpe but does not guarantee better realized Sharpe.",
                      "The comparison is closer, but still not fully policy-matched: scenarios retain "
                      "cap25% and no variance ceiling; hard rank has no cap and its unchanged variance ceiling."])
    lines.extend(["", "## Third model really was a three-way ensemble", "",
                  "All 96 third-component cases were verified against the exact saved arrays: "
                  "mu=(1-gamma)*[(1-alpha_family)*mu_broad+alpha_family*mu_compact]+gamma*mu_third. "
                  "They were not standalone third-model portfolio experiments. Standalone forecast/error "
                  "statistics were diagnostics only. Gamma was fixed at 12.5%,25%,50% across dates/seeds. "
                  "Covariance was independently blended, including unchanged covariance controls.",
                  "For U.S. equities, gamma=12.5% means 43.75% broad + 43.75% compact + 12.5% third. "
                  "International equities retain zero compact weight because the approved base recipe "
                  "has zero compact weight there; other families have their own inherited fractions.", "",
                  "All history was previously examined; later12 is not a pristine holdout. "
                  "Seeds share one market history. Returns are gross of costs/taxes. "
                  "Nonpositive-mean multistart agreement is not a proof of global optimality.", ""])
    _save(OUTPUT / "RESULTS.md", "\n".join(lines).encode())
    _plot(new, old, hard)


def _plot(new: dict, old: dict, hard: dict) -> None:
    if new["status"] != "complete":
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.patheffects as effects
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    context = _arrays(SOURCE / "context.npz")
    origin = context["dates"].astype("datetime64[D]")
    ends = (origin.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    dates = np.r_[origin[:1], ends]
    figure, axis = plt.subplots(figsize=(14, 8), layout="constrained")
    baseline = _arrays(SOURCE / "cases/control_hard_rank.npz")["wealth"].mean(0)
    for name, path, color in [
        ("GP scenarios NE7; cap25%", OUTPUT / "cases" / f"{new['id']}.npz", "#E69F00"),
        ("GP scenarios NE10; cap25%", SOURCE / "cases" / f"{old['id']}.npz", "#009E73"),
    ]:
        wealth = _arrays(path)["wealth"].mean(0)
        axis.plot(dates, wealth, linewidth=2.4, color=color, linestyle="--",
                  label=f"{name} | ${wealth[-1]:,.0f}")
    for label, key, color in [("Schwab aggressive", "benchmark_schwab_total", "#8E6BBE"),
                              ("Momentum top7", "benchmark_momentum_total", "#666666")]:
        wealth = 10000 * np.r_[1., np.cumprod(1 + context[key])]
        axis.plot(dates, wealth, color=color, linestyle=":", linewidth=2,
                  label=f"{label} | ${wealth[-1]:,.0f}")
    line, = axis.plot(dates, baseline, color="#1565C0", linewidth=2.8, marker="o", markersize=3,
                      markevery=2, zorder=20, label=f"Current hard rank NE7; no cap | ${baseline[-1]:,.0f}")
    line.set_path_effects([effects.Stroke(linewidth=4.4, foreground="white"), effects.Normal()])
    axis.axvline(dates[16], color="gray", linestyle="--", alpha=.7)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend([handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]], loc="upper left", fontsize=9)
    axis.set_title("Same blended GP forecasts: minimum effective assets 7 versus 10\n"
                   "Growth of $10,000 | June 2024–September 2026 | Monthly rebalancing")
    axis.set_xlabel("Month-end investment value")
    axis.set_ylabel("Investment value ($)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.grid(alpha=.2)
    figure.supxlabel("Three-seed mean NAV; no GP refits; static blending; gross of costs/taxes.\n"
                    "Scenarios retain cap25% and no variance ceiling; hard rank has no cap and a variance ceiling. "
                    "Later12 is not an untouched holdout.", fontsize=9)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    _save(OUTPUT / "COMPARISON.png", buffer.getvalue())


def _execute(workers: int) -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    _verify_source()
    cases = [_read(p) for p in sorted((SOURCE / "cases").glob("t1_*.json"))]
    cases = [r for r in cases if r["case"]["allocator"] == "scenarios"]
    if len(cases) != 36 or any(r["status"] != "complete" for r in cases):
        raise ValueError("36 completed frozen scenario recipes required")
    inputs = {str(SOURCE / "COMPLETED.json"): _hash(SOURCE / "COMPLETED.json")}
    inputs.update({str(p): _hash(p) for p in sorted(HERE.glob("*.py"))})
    _save(OUTPUT / "CONTRACT.json", {"minimum_effective_assets": 7, "maximum_weight": .25,
          "source_NE10_cases": [c["id"] for c in cases], "expected_allocations": 3024, "GP_fits": 0,
          "workers": workers, "input_sha256": inputs, "all_blending_fixed": True,
          "same_5000_draws_per_seed_month_as_NE10": True})
    _save(OUTPUT / "THREE_WAY_BLEND_AUDIT.json", _three_way_audit())
    print("Verified original evidence and all 96 actual three-component blends; starting NE7 allocations", flush=True)
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[key] = "1"
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        tasks = [pool.submit(_evaluate, case["id"]) for case in cases]
        records = []
        for task in as_completed(tasks):
            record = task.result()
            records.append(record)
            print(f"{len(records)}/36: {record['id']} {record['successful_allocations']}/84 {record['status']}",
                  flush=True)
    records.sort(key=lambda r: r["id"])
    _report(records)
    _verify_hashes(inputs)
    _verify_source()
    summary = _read(OUTPUT / "SUMMARY.json")
    if not summary["passed"]:
        raise ValueError("Explicit case failures remain; do not mark experiment complete")
    _save(OUTPUT / "COMPLETED.json", {"passed": True, "complete_cases": 36, "allocations": 3024,
          "GP_fits": 0, "input_sha256": inputs,
          "artifact_sha256": {str(p): _hash(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file()}})
    _verify()


def _verify() -> None:
    _verify_source()
    seal = _read(OUTPUT / "COMPLETED.json")
    if seal.get("passed") is not True:
        raise ValueError("Passed completion seal required")
    _verify_hashes(seal["input_sha256"])
    _verify_hashes(seal["artifact_sha256"])
    for path in sorted((OUTPUT / "cases").glob("*.json")):
        record = _read(path)
        _verify_hashes(record["input_sha256"])
        if record["status"] != "complete" or record["successful_allocations"] != 84:
            raise ValueError("Incomplete NE7 case")
        arrays = _arrays(path.with_suffix(".npz"))
        if _hash(path.with_suffix(".npz")) != record["artifact_sha256"]:
            raise ValueError("Case arrays changed")
        original = _arrays(SOURCE / "cases" / path.with_suffix(".npz").name)
        for key in ["mean", "covariance", "actual", "dates", "assets", "seeds", "risk_free", "rank_ic", "joint_nll"]:
            np.testing.assert_array_equal(arrays[key], original[key])
        for weights in arrays["weights"].reshape(-1, 15):
            audit_weights(weights, Policy())
    _verify_hashes(_read(OUTPUT / "THREE_WAY_BLEND_AUDIT.json")["input_sha256"])
    print("Verified 36/36 NE7 cases, 3024/3024 allocations; "
          "same GP forecasts/scenarios; 96 three-way blends", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    options = parser.parse_args()
    if not 1 <= options.workers <= 14:
        parser.error("One through fourteen shared workers required")
    _execute(options.workers) if options.execute else _verify()
