"""Versioned NE7 uncapped scenario comparison using frozen GP forecasts.

Report-only research orchestration. Reuses the existing deterministic case
evaluation adapter in process-local policy bindings; no old source or output is
edited. Writes decimal returns, fraction weights and dollar NAV to new outputs.
Repairs the single capped diagnostic allocation while preserving its failure
evidence and every valid weight. Only the new NE7/uncapped grid is the primary run.

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
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
SOURCE = CAMPAIGN / "scenario_numerics_v2/outputs"
CAPPED = CAMPAIGN / "scenario_ne7_v1/outputs"
OUTPUT = HERE / "outputs"
BASE = "t1_r2_i15_blend"
sys.path[:0] = [str(HERE), str(CAMPAIGN / "scenario_ne7_v1"),
               str(CAMPAIGN / "track1_scenarios"), str(CAMPAIGN / "track2_asymmetric")]

import case_evaluator as evaluator  # noqa: E402
import run_ne7 as previous  # noqa: E402
from recipe_math import select_candidates  # noqa: E402
from uncapped_math import Policy, allocate_scenarios, audit_weights  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, value: dict | bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only new uncapped outputs are writable")
    content = value if isinstance(value, bytes) else (
        json.dumps(evaluator._jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    evaluator._immutable_bytes(path, content)


def _new_implementation() -> dict:
    paths = [*HERE.glob("*.py"), CAMPAIGN / "scenario_ne7_v1/ne7_math.py", Path(previous.__file__)]
    return {str(p): _hash(p) for p in paths}


def _uncapped_policy() -> Policy:
    return Policy(7, 1)


def _save_evaluation(path: Path, value: dict | bytes) -> None:
    if isinstance(value, dict) and value.get("schema") == "bayesfolio.gp_tuning.scenario_NE7.v1":
        value = copy.deepcopy(value)
        value["schema"] = "bayesfolio.gp_tuning.scenario_NE7_uncapped.v1"
        value["policy"]["maximum_weight"] = None
        value["input_sha256"].update(_new_implementation())
    _save(path, value)


def _evaluate(identity: str) -> dict:
    # Reuse all scoring, chronology, scenario-key and artifact logic. These
    # bindings are process-local; each worker handles only one case at a time.
    with patch.multiple(previous, OUTPUT=OUTPUT, Policy=_uncapped_policy,
                        allocate_scenarios=allocate_scenarios, _save=_save_evaluation):
        return previous._evaluate(identity)


def _repair_capped_diagnostic() -> dict:
    identity = "t1_r5_i36_compact"
    destination = OUTPUT / "capped_diagnostic" / f"{identity}.json"
    if destination.exists():
        return _read(destination)
    source = CAPPED / "cases" / f"{identity}.json"
    record = copy.deepcopy(_read(source))
    arrays = previous._arrays(source.with_suffix(".npz"))
    original = arrays["weights"].copy()
    valid = arrays["allocated"].copy()
    context = previous._arrays(SOURCE / "context.npz")
    failures = copy.deepcopy(record["failures"])
    repairs = []
    for failure in failures:
        s, m = failure["seed_index"], failure["month_index"]
        scenarios = evaluator._dependency("sampling").sample_joint(arrays["mean"][s, m],
            arrays["covariance"][s, m], seed=failure["seed"], month=failure["origin"])
        result = allocate_scenarios(scenarios, arrays["assets"].tolist(), policy=Policy(7, .25),
            seed=20261024 + failure["seed"] + int(failure["origin"].replace("-", "")))
        w = result["weights"]
        arrays["weights"][s, m], arrays["raw_weights"][s, m] = w, result["raw_weights"]
        arrays["allocated"][s, m] = True
        arrays["effective_count"][s, m] = 1 / (w @ w)
        arrays["equity_weight"][s, m] = w[:8].sum()
        arrays["predicted_variance"][s, m] = w @ arrays["covariance"][s, m] @ w
        repairs.append({"original_failure": failure, "numerical": {
            k: v for k, v in result.items() if k not in {"weights", "raw_weights"}}})
    np.testing.assert_array_equal(original[valid], arrays["weights"][valid])
    if not arrays["allocated"].all():
        raise ValueError("Capped diagnostic remains incomplete")
    periods, calibration = evaluator._scores(context, arrays["mean"], arrays["covariance"], arrays)
    record.update(status="complete", successful_allocations=84, periods=periods, calibration=calibration,
                  failures=[], repair_history=repairs, original_failure_record=str(source))
    record["input_sha256"].update({str(source): _hash(source),
                                   str(source.with_suffix(".npz")): _hash(source.with_suffix(".npz")),
                                   **_new_implementation()})
    arrays["status"] = np.array("complete")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _save(destination.with_suffix(".npz"), buffer.getvalue())
    record["artifact_sha256"] = _hash(destination.with_suffix(".npz"))
    _save(destination, record)
    return record


def _comparators() -> list[tuple[str, dict, Path]]:
    return [
        ("Hard rank NE7, no cap", _read(SOURCE / "cases/control_hard_rank.json"),
         SOURCE / "cases/control_hard_rank.npz"),
        ("GP scenarios NE10, cap25%", _read(SOURCE / "cases" / f"{BASE}.json"),
         SOURCE / "cases" / f"{BASE}.npz"),
        ("GP scenarios NE7, cap25%", _read(CAPPED / "cases" / f"{BASE}.json"),
         CAPPED / "cases" / f"{BASE}.npz"),
        ("GP scenarios NE7, no cap", _read(OUTPUT / "cases" / f"{BASE}.json"),
         OUTPUT / "cases" / f"{BASE}.npz"),
    ]


def _report(records: list[dict]) -> None:
    if len(records) != 36 or any(r["status"] != "complete" for r in records):
        _save(OUTPUT / "INCOMPLETE.json", {"passed": False, "failures": [
            {"id": r["id"], "failures": r["failures"]} for r in records if r["status"] != "complete"]})
        raise ValueError("Every uncapped allocation must complete; no partial case ranking")
    rows = [{"id": r["id"], "period": "tuning_first16", **{
        k: r["periods"]["tuning_first16"][k] for k in ["sharpe", "total_return", "rank_ic"]}} for r in records]
    selected = select_candidates(rows, limit=3)
    leaders = {p: {k: max(records, key=lambda r: r["periods"][p][k])["id"]
                   for k in ["sharpe", "total_return", "rank_ic"]}
               for p in ["tuning_first16", "validation_last12", "all28"]}
    summary = {"passed": True, "complete_cases": 36, "complete_allocations": 3024, "GP_fits": 0,
               "minimum_effective_assets": 7, "position_cap": None, "CVaR": False,
               "fallback": False, "minimum_portfolio_return_constraint": False,
               "selected_first16_ids": selected, "descriptive_leaders_by_period": leaders,
               "unchanged_forecasts_scenarios_and_blends": True, "automatic_promotion": False,
               "pristine_holdout": False, "remaining_difference_from_hard_rank": "variance ceiling",
               "capped_diagnostic_repair": "capped_diagnostic/t1_r5_i36_compact.json"}
    _save(OUTPUT / "SUMMARY.json", summary)
    lines = ["# NE7 uncapped GP-scenario comparison", "",
             "Same frozen GP forecasts and 5000 scenarios, all 36 settings/blend recipes, three seeds, "
             "28 months. No refits. Primary scenario policy: long-only/full investment, NE>=7, no ETF cap, "
             "maximum signed Sharpe, no CVaR or variance ceiling, negative returns allowed. "
             "Hard rank retains its variance ceiling, so this is closer but not fully policy-matched.", "",
             "## Same blended GP, different allocation constraints", "",
             "| Strategy | Period | Sharpe | Return | Drawdown | Equity | Effective count | Ending $10k |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, record, _ in _comparators():
        for period in ["tuning_first16", "validation_last12", "all28"]:
            v = record["periods"][period]
            lines.append(f"| {label} | {period} | {v['sharpe']:.4f} | {v['total_return']:.2%} | "
                         f"{v['max_drawdown']:.2%} | {v['equity_weight']:.1%} | "
                         f"{v['effective_assets']:.3f} | ${v['ending_value']:,.0f} |")
    lines.extend(["", "IC and NLL are exactly unchanged by allocator policy for each GP recipe.", "",
                  "## First16-selected static recipes", "",
                  "| Recipe | Period | Sharpe | Return | IC |", "|---|---|---:|---:|---:|"])
    by_id = {r["id"]: r for r in records}
    for identity in selected:
        for p in ["tuning_first16", "validation_last12", "all28"]:
            v = by_id[identity]["periods"][p]
            lines.append(f"| {identity} | {p} | {v['sharpe']:.4f} | {v['total_return']:.2%} | {v['rank_ic']:.4f} |")
    lines.extend(["", "## Interpretation boundaries", "",
                  "The NE10/capped, NE7/capped and NE7/uncapped baseline rows separately isolate "
                  "diversification and cap effects. Constraint relaxation can improve forecast Sharpe "
                  "without improving realized Sharpe. The GP model and its forecast ranking did not change.",
                  "All36 uncapped configurations are complete. The capped NE7 diagnostic's one missing "
                  "mean-sign solve was repaired with identical objective/constraints and stricter audited "
                  "numerical retries. Original failure evidence and all successful weights were retained.", "",
                  "## Third-model clarification", "",
                  "All96 third-component portfolio cases already combined the existing two-model ensemble "
                  "with the third model. Exact saved means and covariance were checked. "
                  "For example, U.S. equity at gamma12.5% used broad43.75% + compact43.75% + third12.5%. "
                  "Standalone prediction-error statistics were diagnostics, not standalone portfolio trials. "
                  "International-equity compact weight stayed zero under the approved family base recipe.", "",
                  "Full28 descriptive winners are not validation-selected. All history was previously viewed; "
                  "later12 is not untouched. Seeds share one market path. Returns are gross of costs/taxes. "
                  "Negative-mean multistart agreement is not a global-optimality proof.", ""])
    _save(OUTPUT / "RESULTS.md", "\n".join(lines).encode())
    table = ["id,period,sharpe,total_return,rank_ic,max_drawdown,effective_assets,equity_weight"]
    for r in records:
        for p, v in r["periods"].items():
            table.append(",".join(map(str, [r["id"], p, *[v[k] for k in
                ["sharpe", "total_return", "rank_ic", "max_drawdown", "effective_assets", "equity_weight"]]])))
    _save(OUTPUT / "ALL_CASES.csv", ("\n".join(table) + "\n").encode())
    _plot()


def _plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.patheffects as effects
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    context = previous._arrays(SOURCE / "context.npz")
    origins = context["dates"].astype("datetime64[D]")
    ends = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    dates = np.r_[origins[:1], ends]
    figure, axis = plt.subplots(figsize=(14, 8), layout="constrained")
    comparators = _comparators()
    colors = ["#1565C0", "#009E73", "#999999", "#E69F00"]
    for index in [1, 2, 3, 0]:
        label, _, path = comparators[index]
        values = previous._arrays(path)["wealth"].mean(0)
        line, = axis.plot(dates, values, color=colors[index], linewidth=2.8 if index == 0 else 2.2,
                          linestyle="-" if index == 0 else "--", zorder=20 if index == 0 else 3,
                          label=f"{label} | ${values[-1]:,.0f}")
        if index == 0:
            line.set_path_effects([effects.Stroke(linewidth=4.4, foreground="white"), effects.Normal()])
    for label, key, color in [("Schwab aggressive", "benchmark_schwab_total", "#8E6BBE"),
                              ("Momentum top7", "benchmark_momentum_total", "#555555")]:
        values = 10000 * np.r_[1., np.cumprod(1 + context[key])]
        axis.plot(dates, values, color=color, linestyle=":", linewidth=2, label=f"{label} | ${values[-1]:,.0f}")
    axis.axvline(dates[16], color="gray", linestyle="--", alpha=.7)
    handles, labels = axis.get_legend_handles_labels()
    order = [3, 2, 1, 0, 4, 5]
    axis.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", fontsize=9)
    axis.set_title("Same blended multitask GP: scenario diversification and cap effects\n"
                   "Growth of $10,000 | June 2024–September 2026 | Monthly rebalancing")
    axis.set_xlabel("Month-end investment value")
    axis.set_ylabel("Investment value ($)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.grid(alpha=.2)
    figure.supxlabel("Three-seed mean NAV; no GP refits; fixed blending; gross of costs/taxes.\n"
                    "Hard rank retains a variance ceiling; scenarios do not. "
                    "Divider separates first16 from later12; not an untouched holdout.", fontsize=9)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    _save(OUTPUT / "COMPARISON.png", buffer.getvalue())


def _execute(workers: int) -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    previous._verify_source()
    identities = [_read(p)["id"] for p in sorted((SOURCE / "cases").glob("t1_*.json"))
                  if _read(p)["case"]["allocator"] == "scenarios"]
    if len(identities) != 36:
        raise ValueError("Exactly36 scenario configurations required")
    inputs = {str(SOURCE / "COMPLETED.json"): _hash(SOURCE / "COMPLETED.json"), **_new_implementation()}
    _save(OUTPUT / "CONTRACT.json", {"input_sha256": inputs, "GP_fits": 0, "allocations": 3024,
          "minimum_effective_assets": 7, "position_cap": None, "workers": workers,
          "cases": identities, "unchanged_scenarios_and_blends": True})
    _save(OUTPUT / "THREE_WAY_BLEND_AUDIT.json", previous._three_way_audit())
    _repair_capped_diagnostic()
    print("Verified three-way blends; repaired capped diagnostic; starting NE7 uncapped allocations", flush=True)
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[key] = "1"
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        tasks = [pool.submit(_evaluate, identity) for identity in identities]
        records = []
        for task in as_completed(tasks):
            r = task.result()
            records.append(r)
            print(f"{len(records)}/36 {r['id']} {r['successful_allocations']}/84 {r['status']}", flush=True)
    records.sort(key=lambda r: r["id"])
    _report(records)
    previous._verify_hashes(inputs)
    previous._verify_source()
    _save(OUTPUT / "COMPLETED.json", {"passed": True, "cases": 36, "allocations": 3024, "GP_fits": 0,
          "input_sha256": inputs,
          "artifact_sha256": {str(p): _hash(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file()}})
    _verify()


def _verify() -> None:
    previous._verify_source()
    seal = _read(OUTPUT / "COMPLETED.json")
    if seal.get("passed") is not True:
        raise ValueError("Passed uncapped completion required")
    previous._verify_hashes(seal["input_sha256"])
    previous._verify_hashes(seal["artifact_sha256"])
    paths = sorted((OUTPUT / "cases").glob("*.json"))
    if len(paths) != 36:
        raise ValueError("36 uncapped cases required")
    for path in paths:
        record = _read(path)
        if record["status"] != "complete" or record["successful_allocations"] != 84:
            raise ValueError("Incomplete uncapped case")
        previous._verify_hashes(record["input_sha256"])
        if record["policy"]["maximum_weight"] is not None:
            raise ValueError("Uncapped policy metadata required")
        if _hash(path.with_suffix(".npz")) != record["artifact_sha256"]:
            raise ValueError("Case arrays changed")
        arrays = previous._arrays(path.with_suffix(".npz"))
        source = previous._arrays(SOURCE / "cases" / path.with_suffix(".npz").name)
        for key in ["mean", "covariance", "actual", "dates", "assets", "seeds", "risk_free", "rank_ic", "joint_nll"]:
            np.testing.assert_array_equal(arrays[key], source[key])
        for w in arrays["weights"].reshape(-1, 15):
            audit_weights(w, Policy(7, 1))
    previous._verify_hashes(_read(OUTPUT / "THREE_WAY_BLEND_AUDIT.json")["input_sha256"])
    capped = _read(OUTPUT / "capped_diagnostic/t1_r5_i36_compact.json")
    if capped["status"] != "complete":
        raise ValueError("Capped diagnostic repair incomplete")
    print("Verified36/36 uncapped NE7 cases,3024 allocations; unchanged GP forecasts; capped edge fixed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    options = parser.parse_args()
    if not 1 <= options.workers <= 14:
        parser.error("One through fourteen workers required")
    _execute(options.workers) if options.execute else _verify()
