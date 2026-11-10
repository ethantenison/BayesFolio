"""Read-only review of frozen per-model objective leaders, plus scenario diagnostics.

No new weight proposals, GP fits or later-period model selection. All reviewed
recipes were already frozen in the primary study; this expands only descriptive
scenario checks to each model's first16 objective extremes. New outputs preserve
the primary seal. Dollar NAV, decimal returns and fraction weights retain units.

Usage:
    uv run --locked --no-sync python <this_file> --execute
    uv run --locked --no-sync python <this_file> --verify
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PRIMARY = STUDY / "outputs"
OUTPUT = HERE / "outputs"
sys.path.insert(0, str(STUDY))
import run_joint as study  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: dict | bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only new review evidence is writable")
    content = value if isinstance(value, bytes) else (
        json.dumps(study.evaluator._jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    study.evaluator._immutable_bytes(path, content)


def _scenario(case: dict) -> dict:
    with patch.object(study, "OUTPUT", OUTPUT):
        return study._safe_full(case, "scenarios")


def _original_solver_parity(cases: list[dict]) -> dict:
    context = study._arrays(study.SOURCE / "context.npz")
    records = []
    for case in cases:
        original = study._arrays(PRIMARY / "cases" / f"{case['id']}_hard_rank.npz")
        recomputed = np.empty((3, 28, 15))
        for s in range(3):
            for t in range(28):
                solved = study.evaluator._dependency("hard").hard_rank(original["mean"][s, t],
                    original["covariance"][s, t], float(context["risk_budget"][t]))
                recomputed[s, t] = solved["weights"]
        delta = float(np.max(abs(recomputed - original["weights"])))
        alternate = {**original, "weights": recomputed}
        periods, _ = study.evaluator._scores(context, original["mean"], original["covariance"], alternate)
        old = study._read(PRIMARY / "cases" / f"{case['id']}_hard_rank.json")
        sharpe_delta = abs(periods["all28"]["sharpe"] - old["periods"]["all28"]["sharpe"])
        nav_delta = abs(periods["all28"]["ending_value"] - old["periods"]["all28"]["ending_value"])
        if delta > 1e-5 or sharpe_delta > 1e-5 or nav_delta > .03:
            raise ValueError("New candidate improvement does not survive the original solver")
        records.append({"case": case, "maximum_weight_difference": delta,
                        "full_period_sharpe_difference": sharpe_delta, "ending_dollars_difference": nav_delta})
    return {"passed": True, "original_conic_allocations_checked": 84 * len(cases), "records": records}


def _execute() -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    study._verify()
    frozen = study._read(PRIMARY / "TUNING_FROZEN.json")
    choices = {}
    per_model = {}
    for model, candidates in frozen["per_model_selected"].items():
        per_model[model] = {}
        for objective in study.OBJECTIVES:
            selected = max(candidates, key=lambda r: r[objective])
            choices[selected["id"]] = selected["case"]
            per_model[model][objective] = selected["id"]
    baseline = frozen["control_recipes"][0]
    choices[baseline["id"]] = baseline
    _save(OUTPUT / "TUNING_FROZEN.json", (PRIMARY / "TUNING_FROZEN.json").read_bytes())
    inputs = {str(PRIMARY / "COMPLETED.json"): _hash(PRIMARY / "COMPLETED.json"),
              str(Path(__file__)): _hash(Path(__file__))}
    _save(OUTPUT / "REVIEW_CONTRACT.json", {"input_sha256": inputs, "per_model_first16_leaders": per_model,
        "recipes_already_frozen_before_later12": True, "new_weight_proposals": 0, "GP_fits": 0,
        "expanded_scenario_checks_are_posthoc_descriptive": True})
    with ProcessPoolExecutor(max_workers=14, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_scenario, case) for case in choices.values()]
        scenarios = [f.result() for f in as_completed(futures)]
    if any(r["status"] != "complete" for r in scenarios):
        raise ValueError("A descriptive scenario check failed; retain evidence and inspect")
    lead_ids = list(dict.fromkeys(per_model["broad_roll48_r2_i15"].values()))
    _save(OUTPUT / "ORIGINAL_SOLVER_PARITY.json", _original_solver_parity([choices[i] for i in lead_ids]))
    _save(OUTPUT / "SUMMARY.json", {"per_model_first16_leaders": per_model, "scenario_cases": scenarios,
        "broad48_lead_ids": lead_ids, "all_weights_unchanged": True,
        "interpretation": "Per-model candidates were preselected; identifying the most consistent model is retrospective."})
    _plot(per_model["broad_roll48_r2_i15"], choices)
    study._verify()
    _save(OUTPUT / "COMPLETED.json", {"passed": True, "input_sha256": inputs,
        "scenario_cases": len(scenarios), "GP_fits": 0,
        "artifact_sha256": {str(p): _hash(p) for p in OUTPUT.rglob("*") if p.is_file()}})
    _verify()


def _plot(leaders: dict, choices: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.patheffects as effects
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    context = study._arrays(study.SOURCE / "context.npz")
    origins = context["dates"].astype("datetime64[D]")
    dates = np.r_[origins[:1], (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1]
    figure, axis = plt.subplots(figsize=(14, 8), layout="constrained")
    colors = {"sharpe": "#009E73", "total_return": "#E69F00", "rank_ic": "#CC79A7"}
    for objective, identity in leaders.items():
        values = study._arrays(PRIMARY / "cases" / f"{identity}_hard_rank.npz")["wealth"].mean(0)
        axis.plot(dates, values, linewidth=2, linestyle="--", color=colors[objective],
                  label=f"48m third GP: first16 {objective} leader | ${values[-1]:,.0f}")
    for label, key, color in [("Schwab aggressive", "benchmark_schwab_total", "#888888"),
                              ("Momentum top7", "benchmark_momentum_total", "#AAAAAA")]:
        values = 10000 * np.r_[1., np.cumprod(1 + context[key])]
        axis.plot(dates, values, color=color, linestyle=":", linewidth=2, label=f"{label} | ${values[-1]:,.0f}")
    base = study._arrays(study.SOURCE / "cases/control_hard_rank.npz")["wealth"].mean(0)
    line, = axis.plot(dates, base, color="#1565C0", linewidth=2.8, zorder=20,
                      label=f"Original hard-rank baseline | ${base[-1]:,.0f}")
    line.set_path_effects([effects.Stroke(linewidth=4.2, foreground="white"), effects.Normal()])
    handles, labels = axis.get_legend_handles_labels()
    axis.legend([handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]], loc="upper left", fontsize=9)
    axis.axvline(dates[16], linestyle="--", color="gray", alpha=.7)
    axis.set_title("Jointly retuned three-way blends with the 48-month broad GP\n"
                   "Per-model first16 objective leaders; later12 shown retrospectively")
    axis.set_xlabel("Month-end investment value | June2024–September2026")
    axis.set_ylabel("Investment value ($)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.grid(alpha=.2)
    figure.supxlabel("Same rank2/maxiter15 GP components; no refits; static weights across dates/seeds. "
                    "Three-seed mean NAV; gross of costs/taxes.\n"
                    "The most consistent model was identified retrospectively among preselected candidates. "
                    "No untouched holdout or automatic promotion.", fontsize=9)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    _save(OUTPUT / "BROAD48_COMPARISON.png", buffer.getvalue())


def _verify() -> None:
    study._verify()
    seal = study._read(OUTPUT / "COMPLETED.json")
    if seal.get("passed") is not True:
        raise ValueError("Passed review completion required")
    study.run_uncapped.previous._verify_hashes(seal["input_sha256"])
    study.run_uncapped.previous._verify_hashes(seal["artifact_sha256"])
    print("Verified per-model frozen candidates, descriptive scenario checks and original-solver parity", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    options = parser.parse_args()
    _execute() if options.execute else _verify()
