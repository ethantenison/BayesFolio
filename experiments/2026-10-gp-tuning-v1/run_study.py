"""One-command fit, evaluate and independently report three GP-tuning tracks.

Isolated research composition root. One shared fourteen-process fit pool closes
before portfolio-evaluation pools begin. All blend fractions are static across
months and seeds; no live trading, data refresh or baseline promotion occurs.
Returns are decimals and investment values dollars. Existing evidence is sealed.
Usage: uv run --locked --no-sync python <this_file> --launch|--execute|--verify|--status
--launch detaches this authorized finite experiment and writes durable log/status.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from campaign import (
    ALPHA,
    FRAMEWORK,
    HERE,
    OUTPUT,
    REPO,
    digest,
    fit_campaign,
    prepare,
    read_arrays,
    save,
    verify,
    verify_hashes,
)

sys.path.extend([str(HERE / "track1_scenarios"), str(HERE / "track2_asymmetric"),
                 str(HERE / "track3_complements"), str(FRAMEWORK)])


def _status(phase: str, **values: object) -> None:
    save(OUTPUT / "workflow_status.json", {"phase": phase, "updated_unix": time.time(), **values}, immutable=False)
    print(json.dumps({"phase": phase, **values}), flush=True)


def _baseline_case(allocator: str, identity: str) -> dict:
    return {"id": identity, "track": "control", "stage": "baseline", "allocator": allocator,
            "broad_id": "broad_r2_i15_exp", "compact_id": "compact_r2_i15_exp", "alpha": ALPHA,
            "cov_compact_fraction": .5, "third_id": None, "third_fraction": 0., "third_cov_fraction": 0.}


def _combine(summaries: dict) -> None:
    import numpy as np
    import pandas as pd
    from metrics import portfolio_performance
    from recipe_math import pareto_ids, select_candidates

    records = [json.loads(path.read_text()) for path in sorted((OUTPUT / "cases").glob("*.json"))]
    records = [row for row in records if row["case"]["track"] != "audit"]
    complete = [row for row in records if row["status"] == "complete"]
    selection = [{"id": row["id"], "period": "tuning_first16", **{
        key: row["periods"]["tuning_first16"][key] for key in ["sharpe", "total_return", "rank_ic"]}}
        for row in complete]
    chosen = select_candidates(selection, limit=3)
    frontier = pareto_ids(selection)
    by_id = {row["id"]: row for row in records}
    context = read_arrays(OUTPUT / "context.npz")
    benchmarks = []
    for name, key in [("Schwab aggressive proxy", "benchmark_schwab_total"),
                      ("Simple momentum top7", "benchmark_momentum_total")]:
        row = {"label": name, "periods": {}}
        for period, section in [("tuning_first16", slice(0, 16)), ("validation_last12", slice(16, 28)),
                                ("all28", slice(0, 28))]:
            stats = portfolio_performance(context[key][section], context["risk_free"][section])
            row["periods"][period] = {"sharpe": stats["conventional_sharpe"],
                                       "total_return": stats["compounded_total_return"],
                                       "max_drawdown": stats["max_drawdown_total_nav"]}
        benchmarks.append(row)
    table = [{"id": row["id"], "track": row["case"]["track"], "stage": row["case"]["stage"],
              "allocator": row["case"]["allocator"], "period": period, **{
                  key: values[key] for key in ["sharpe", "total_return", "rank_ic", "max_drawdown", "ending_value"]}}
             for row in complete for period, values in row["periods"].items()]
    save(OUTPUT / "ALL_COMPLETE_RESULTS.csv", pd.DataFrame(table).to_csv(index=False).encode())
    findings = []
    reference = by_id["control_hard_rank"]
    for identity in chosen:
        row = by_id[identity]
        train, later = row["periods"]["tuning_first16"], row["periods"]["validation_last12"]
        before, after = reference["periods"]["tuning_first16"], reference["periods"]["validation_last12"]
        objectives = ["sharpe", "total_return", "rank_ic"]
        gains = [key for key in objectives if later[key] > after[key] + 1e-10]
        losses = [key for key in objectives if later[key] < after[key] - 1e-10]
        findings.append({"candidate": identity, "static_recipe": row["case"],
                         "later12_improves_objectives": gains, "later12_worsens_objectives": losses,
                         "later12_delta_vs_frozen_hard_rank": {key: later[key] - after[key] for key in objectives},
                         "first16_delta_vs_frozen_hard_rank": {key: train[key] - before[key] for key in objectives},
                         "later12_pareto_dominates_reference": bool(gains and not losses),
                         "interpretation": "Favorable later-period trade-off evidence" if gains and not losses
                         else "Mixed or weaker later-period evidence; no unconditional improvement claim"})
    descriptive = {key: max(complete, key=lambda row: row["periods"]["all28"][key])["id"]
                   for key in ["sharpe", "total_return", "rank_ic"]}
    summary = {"status": "complete", "tracks": summaries, "complete_cases": len(complete),
               "failed_cases": len(records) - len(complete), "first16_selected_case_ids": chosen,
               "first16_global_pareto_ids": frontier, "benchmarks": benchmarks,
               "paired_findings": findings, "all28_descriptive_leaders_not_selection_inputs": descriptive,
               "automatic_promotion": False, "monthly_blend_adaptation": False,
               "limitations": ["All 28 months have previously been examined; no pristine holdout is claimed.",
                               "Later12 months never select a model pair or blending recipe in this campaign.",
                               "The finite blend menu is not a global continuous optimum.",
                               "GP seeds share one market history; explicit costs and taxes are excluded.",
                               "Scenario and hard-rank allocator economic constraints differ."]}
    save(OUTPUT / "GLOBAL_SUMMARY.json", summary)
    lines = ["# Three-track GP tuning: integrated findings", "",
             "All blending recipes are fixed across all months and seeds. Component/blend selection uses first16 only.",
             "Later12 is a temporal comparison on previously viewed history, not an untouched holdout.", "",
             f"Completed portfolio cases: {len(complete)}. Failed cases retained but not ranked: "
             f"{len(records) - len(complete)}.", "", "## Separate track reports", "",
             "- [Track1: scenarios](track1/RESULTS.md)",
             "- [Track2: asymmetric component settings and static blend tuning](track2/RESULTS.md)",
             "- [Track3: complementary third models](track3/RESULTS.md)", "",
             "## First16-selected global review candidates", "",
             "These are a trade-off set, not one scalarized winner or automatic production promotion.", "",
             "| Candidate | Period | Sharpe | Total return | Rank IC | Drawdown |", "|---|---|---:|---:|---:|---:|"]
    for identity in ["control_hard_rank", "control_scenarios", *chosen]:
        if identity not in by_id:
            continue
        record = by_id[identity]
        for period in ["tuning_first16", "validation_last12", "all28"]:
            metric = record["periods"][period]
            lines.append(f"| {identity} | {period} | {metric['sharpe']:.3f} | "
                         f"{metric['total_return']:.2%} | {metric['rank_ic']:.3f} | {metric['max_drawdown']:.2%} |")
    lines.extend(["", "## Integrated takeaways", ""])
    for finding in findings:
        delta = finding["later12_delta_vs_frozen_hard_rank"]
        lines.append(f"- {finding['candidate']}: later12 versus frozen hard rank: "
                     f"Sharpe {delta['sharpe']:+.3f}, return {delta['total_return']:+.2%}, "
                     f"rank IC {delta['rank_ic']:+.3f}. {finding['interpretation']}.")
    lines.extend(["", "Full-window leaders below are descriptive only, never inputs to a selected recipe:", ""])
    for objective, identity in descriptive.items():
        lines.append(f"- {objective}: {identity}.")
    lines.extend(["", "## Benchmark portfolio outcomes", "",
                  "| Benchmark | Period | Sharpe | Total return | Drawdown |", "|---|---|---:|---:|---:|"])
    for benchmark in benchmarks:
        for period, metric in benchmark["periods"].items():
            lines.append(f"| {benchmark['label']} | {period} | {metric['sharpe']:.3f} | "
                         f"{metric['total_return']:.2%} | {metric['max_drawdown']:.2%} |")
    lines.extend(["", "## Interpretation and boundaries", "",
                  "Interpret rank/budget effects using Track1 paired contrasts, not different legacy architectures.",
                  "Use Track2 fixed-blend versus retuned comparisons to separate component gains from blend gains.",
                  "Use Track3 incremental comparisons to assess whether error diversity actually improves portfolios.",
                  "Frozen-constraint failures are not rescued with replacement weights or relaxed limits.",
                  "NLL and covariance diagnostics are recorded but are not promotion objectives.",
                  "No baseline or production configuration has been changed.", ""])
    save(OUTPUT / "RESULTS.md", "\n".join(lines).encode())
    _plot(context, by_id, chosen)
    for identity in chosen:
        for period in ["tuning_first16", "validation_last12", "all28"]:
            if not np.isfinite(by_id[identity]["periods"][period]["sharpe"]):
                raise ValueError("Incomplete selected case")


def _plot(context: dict, records: dict, selected: list[str]) -> None:
    import io

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    origins = context["dates"].astype("datetime64[D]")
    endings = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    axis_dates = np.r_[origins[:1], endings]
    figure, axis = plt.subplots(figsize=(14, 8.5), layout="constrained")
    axis.axvline(axis_dates[16], color="gray", linestyle="--", alpha=.7)
    identities = ["control_hard_rank", *[name for name in selected if name != "control_hard_rank"]]
    for index, identity in enumerate(identities):
        record = records[identity]
        paths = read_arrays(OUTPUT / "cases" / f"{identity}.npz")["wealth"]
        label = "Frozen blended hard rank" if identity == "control_hard_rank" else (
            f"Selected {index}: {record['case']['track']}")
        axis.plot(axis_dates, paths.mean(axis=0), linewidth=2.2,
                  label=f"{label} | ${paths[:, -1].mean():,.0f}")
    for label, key in [("Schwab aggressive proxy", "benchmark_schwab_total"),
                       ("Simple momentum top7", "benchmark_momentum_total")]:
        path = 10000 * np.r_[1., np.cumprod(1 + context[key])]
        axis.plot(axis_dates, path, linewidth=2, linestyle="--", label=f"{label} | ${path[-1]:,.0f}")
    axis.set_title("GP tuning: fixed recipes selected on first16 months\n"
                   "Growth of $10,000 | June 2024–September 2026 | Monthly rebalancing")
    axis.set_xlabel("Month-end investment value; dashed divider separates first16 from later12")
    axis.set_ylabel("Investment value ($)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ticks = np.arange(1, 29, 3)
    axis.set_xticks(axis_dates[np.r_[ticks, 28]] if ticks[-1] != 28 else axis_dates[ticks])
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.legend(loc="upper left", fontsize=9)
    axis.grid(alpha=.2)
    figure.supxlabel("All blends fixed across months/seeds; means of three GP seed paths. Gross of costs/taxes.\n"
                    "Previously examined history; later12 is not a pristine holdout. No automatic baseline promotion.",
                    fontsize=9)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    save(OUTPUT / "GLOBAL_COMPARISON.png", buffer.getvalue())


def _execute(workers: int) -> None:
    from case_evaluator import evaluate_case

    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    started = time.monotonic()
    contract = prepare()
    _status("fitting", blend_policy="static across all months", workers=workers)
    fit_campaign(contract, workers)
    verify()
    _status("portfolio_evaluation", track=1)
    for allocator in ["hard_rank", "scenarios"]:
        result = evaluate_case(_baseline_case(allocator, f"control_{allocator}"), str(OUTPUT))
        if result["status"] != "complete":
            raise ValueError("Frozen control failed identity/portfolio scoring")
    track1 = importlib.import_module("run_track").run_track(str(OUTPUT), workers)
    _status("portfolio_evaluation", track=2)
    track2 = importlib.import_module("run_track2").run_track(str(OUTPUT), workers)
    candidates = track2.get("selected_best_cases", [])
    _status("portfolio_evaluation", track=3)
    track3 = importlib.import_module("run_track3").run_track(
        str(OUTPUT), workers, base_case=candidates[0] if candidates else None)
    _status("combining_independent_track_reports")
    _combine({"track1": track1, "track2": track2, "track3": track3})
    verify_hashes(contract["input_sha256"])
    excluded = {"COMPLETED.json", "campaign.log", "status.json", "workflow_status.json", "launch.json"}
    artifacts = {str(path): digest(path) for path in sorted(OUTPUT.rglob("*"))
                 if path.is_file() and path.name not in excluded and path.suffix != ".lock"}
    save(OUTPUT / "COMPLETED.json", {"passed": True, "input_sha256": contract["input_sha256"],
                                     "artifact_sha256": artifacts, "seconds": time.monotonic() - started,
                                     "implementation_sha256": {str(path): digest(path)
                                                               for path in HERE.rglob("*.py")}})
    _status("complete", complete_new_fits=2352, parity_audits=4,
            global_report=str(OUTPUT / "RESULTS.md"), elapsed_seconds=time.monotonic() - started)


def _verify() -> None:
    verify()
    seal = json.loads((OUTPUT / "COMPLETED.json").read_text())
    verify_hashes(seal["input_sha256"])
    verify_hashes(seal["artifact_sha256"])
    verify_hashes(seal["implementation_sha256"])
    print("Verified all three separate track reports, full case paths and final protected evidence", flush=True)


def _launch(workers: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / "COMPLETED.json").exists():
            raise ValueError("Campaign is already complete; use --verify")
    command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--workers", str(workers)]
    with (OUTPUT / "campaign.log").open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUTPUT / "launch.json", {"pid": process.pid, "command": command,
                                  "launched_unix": time.time(), "log": str(OUTPUT / "campaign.log")}, immutable=False)
    print(f"Launched finite campaign PID {process.pid}; log {OUTPUT / 'campaign.log'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ["launch", "execute", "verify", "status"]:
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    options = parser.parse_args()
    if not 1 <= options.workers <= 14:
        parser.error("One through fourteen shared workers required")
    if options.launch:
        _launch(options.workers)
    elif options.status:
        for filename in ["workflow_status.json", "status.json"]:
            path = OUTPUT / filename
            print(path.read_text() if path.exists() else f"No {filename} yet")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with (OUTPUT / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if options.verify:
                _verify()
            else:
                try:
                    _execute(options.workers)
                except Exception:
                    _status("failed_requires_repair_and_resume", traceback=traceback.format_exc())
                    raise
