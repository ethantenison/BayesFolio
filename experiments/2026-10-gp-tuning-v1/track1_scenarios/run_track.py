"""Run and report the preregistered scenario-setting track without GP refits.

Research-only composition/I/O: one bounded case pool evaluates immutable component
panels after the campaign fit queue has finished. Recipes are fixed across months
and seeds. Returns are decimal, NAV dollars, IC unitless. Model shortlisting reads
only the first 16 months; later 12 and all 28 remain separate descriptive reports.

Usage:
    The parent imports ``run_track(output_root, workers=14)`` after all fits.
    This module does not start a fit, download data or alter previous artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from case_evaluator import evaluate_case  # noqa: E402

_RANKS = [1, 2, 5]
_ITERATIONS = [5, 15, 36, 75]
_SOURCES = ["broad", "compact", "blend"]
_PERIODS = ["tuning_first16", "validation_last12", "all28"]
_OBJECTIVES = ["sharpe", "total_return", "rank_ic"]
_REFERENCE_HARD = "t1_reference_hard_rank"
_REFERENCE_SCENARIO = "t1_r2_i15_blend"


def _recipes() -> ModuleType:
    name = "_bayesfolio_gp_tuning_track1_recipe_math"
    if name in sys.modules:
        return sys.modules[name]
    path = _HERE.parent / "track2_asymmetric/recipe_math.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Track 2's shared first-16 selection helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict | bytes) -> None:
    # Reuse the evaluator's exclusive-link writer: existing evidence cannot be
    # replaced, even by an accidentally concurrent invocation of this track.
    from case_evaluator import _immutable_bytes

    content = value if isinstance(value, bytes) else (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _immutable_bytes(path, content)


def build_cases() -> list[dict]:
    """Enumerate all 36 fixed scenario recipes and the hard-rank reference.

    Returns:
        37 JSON-compatible cases, with all component IDs tied to the campaign
        grid. All family/covariance fractions are constant across time/seeds.
    """
    cases = []
    for rank in _RANKS:
        for maxiter in _ITERATIONS:
            for source in _SOURCES:
                alpha = [0, 0, 0, 0] if source == "broad" else (
                    [1, 1, 1, 1] if source == "compact" else [0.5, 0, 0.375, 0.25])
                covariance = 0 if source == "broad" else (1 if source == "compact" else 0.5)
                cases.append({
                    "id": f"t1_r{rank}_i{maxiter}_{source}", "track": 1, "stage": "grid",
                    "allocator": "scenarios", "broad_id": f"broad_r{rank}_i{maxiter}_exp",
                    "compact_id": f"compact_r{rank}_i{maxiter}_exp", "alpha": alpha,
                    "cov_compact_fraction": covariance, "third_id": None,
                    "third_fraction": 0.0, "third_cov_fraction": 0.0,
                })
    reference = next(case for case in cases if case["id"] == _REFERENCE_SCENARIO)
    cases.append({**reference, "id": _REFERENCE_HARD, "stage": "reference", "allocator": "hard_rank"})
    return cases


def _manifest(root: Path, cases: list[dict]) -> dict:
    source_paths = [root / "FITS_COMPLETED.json", root / "context.npz", Path(__file__).resolve(),
                    _HERE / "case_evaluator.py", _HERE / "PLAN.md", _HERE / "plan.json",
                    _HERE.parent / "track2_asymmetric/recipe_math.py"]
    components = sorted({case[key] for case in cases for key in ["broad_id", "compact_id"]})
    source_paths.extend(root / "components" / f"{component}.npz" for component in components)
    return {"schema": "bayesfolio.gp_tuning.track1_case_manifest.v1", "cases": cases,
            "candidate_scenario_cases": 36, "additional_hard_rank_control": 1,
            "rank_grid": _RANKS, "maxiter_grid": _ITERATIONS,
            "selection_period": "tuning_first16", "maximum_selected_cases": 3,
            "selection_objectives_maximize": _OBJECTIVES,
            "input_sha256": {str(path): _hash(path) for path in source_paths},
            "static_recipes_all_months_and_seeds": True,
            "historical_75_matches_budget_not_July_architecture": True,
            "case_pool_worker_limit": 14, "fits_started_by_this_track": 0}


def _candidate_rows(records: list[dict], period: str) -> list[dict]:
    return [{"id": row["id"], "period": period,
             **{key: row["periods"][period][key] for key in _OBJECTIVES}}
            for row in records if row["status"] == "complete" and row["successful_allocations"] == 84
            and row["case"]["allocator"] == "scenarios"]


def _contrasts(records: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in records}
    pairs = []
    for source in _SOURCES:
        for maxiter in _ITERATIONS:
            for rank in [1, 5]:
                pairs.append((source, "rank_at_fixed_budget", f"rank{rank}_minus_rank2_at_iter{maxiter}",
                              f"t1_r{rank}_i{maxiter}_{source}", f"t1_r2_i{maxiter}_{source}"))
        for rank in _RANKS:
            for maxiter in [5, 36, 75]:
                pairs.append((source, "budget_at_fixed_rank", f"iter{maxiter}_minus_iter15_at_rank{rank}",
                              f"t1_r{rank}_i{maxiter}_{source}", f"t1_r{rank}_i15_{source}"))
    contrasts = []
    for source, kind, label, left_id, right_id in pairs:
        left, right = by_id[left_id], by_id[right_id]
        if any(row["status"] != "complete" or row["successful_allocations"] != 84 for row in [left, right]):
            contrasts.append({"source": source, "kind": kind, "label": label,
                              "left_id": left_id, "right_id": right_id, "status": "incomplete_pair"})
            continue
        for period in _PERIODS:
            lscore, rscore = left["periods"][period], right["periods"][period]
            contrast = {"source": source, "kind": kind, "label": label, "period": period,
                        "left_id": left_id, "right_id": right_id, "status": "complete",
                        "deltas": {key: lscore[key] - rscore[key] for key in _OBJECTIVES}}
            contrast["per_seed_deltas"] = [{"seed": lhs["seed"], **{
                key: lhs[key] - rhs[key] for key in _OBJECTIVES}}
                for lhs, rhs in zip(lscore["per_seed"], rscore["per_seed"], strict=True)]
            contrasts.append(contrast)
    return contrasts


def _benchmarks(root: Path) -> dict:
    from case_evaluator import _dependency

    metrics = _dependency("metrics")
    with np.load(root / "context.npz", allow_pickle=False) as context:
        result = {}
        for label, key in [("schwab_aggressive", "benchmark_schwab_total"),
                           ("momentum_top7", "benchmark_momentum_total")]:
            periods = {}
            for period, section in [("tuning_first16", slice(0, 16)), ("validation_last12", slice(16, 28)),
                                    ("all28", slice(0, 28))]:
                score = metrics.portfolio_performance(context[key][section], context["risk_free"][section])
                periods[period] = {"sharpe": score["conventional_sharpe"],
                                   "total_return": score["compounded_total_return"],
                                   "max_drawdown": score["max_drawdown_total_nav"],
                                   "ending_value": 10000 * (1 + score["compounded_total_return"]),
                                   "rank_ic": None}
            result[label] = periods
    return result


def _table(records: list[dict], selected: list[str], full_frontier: list[str]) -> bytes:
    output = io.StringIO(newline="")
    fields = ["id", "source", "rank", "maxiter", "status", "successful_allocations",
              "selected_from_first16", "all28_pareto"]
    fields.extend(f"{period}_{key}" for period in _PERIODS for key in [*_OBJECTIVES, "max_drawdown"])
    fields.extend(["all28_effective_assets", "all28_equity_weight"])
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        case = record["case"]
        source = "hard_rank_reference" if case["allocator"] == "hard_rank" else case["id"].rsplit("_", 1)[-1]
        component = case["broad_id"].split("_")
        row = {"id": record["id"], "source": source, "rank": int(component[1][1:]),
               "maxiter": int(component[2][1:]), "status": record["status"],
               "successful_allocations": record["successful_allocations"],
               "selected_from_first16": record["id"] in selected,
               "all28_pareto": record["id"] in full_frontier}
        for period in _PERIODS:
            for key in [*_OBJECTIVES, "max_drawdown"]:
                row[f"{period}_{key}"] = record["periods"].get(period, {}).get(key)
        for key in ["effective_assets", "equity_weight"]:
            row[f"all28_{key}"] = record["periods"].get("all28", {}).get(key)
        writer.writerow(row)
    return output.getvalue().encode()


def _format_metric(value: float | None, *, percent: bool = False) -> str:
    return "undefined" if value is None else (f"{value:.2%}" if percent else f"{value:.3f}")


def _report(summary: dict, records: list[dict]) -> bytes:
    by_id = {row["id"]: row for row in records}
    lines = [
        "# Track 1: scenario GP settings", "",
        "## Scope and completeness", "",
        f"{summary['complete_scenario_cases']}/36 scenario cases complete; "
        f"{summary['failed_scenario_cases']} failed. One separate frozen hard-rank control is included.",
        "Ranks 1/2/5 × maxiter 5/15/36/75; broad-only, compact-only and the fixed family blend.",
        "No month-specific or seed-specific blending, no refits in this stage and no automatic promotion.", "",
        "July used rank 5/maxiter 75, but also different features/tasks/kernel. "
        "This matches its fitting budget, not its architecture.",
        "Scenario NE≥10/cap25% differs from hard rank NE≥7/no cap/variance ceiling; "
        "their comparison is whole-strategy, not a pure allocator experiment.", "",
        "## First-16 shortlist; later results do not choose it", "",
        "The same three-objective deterministic Pareto selector used by Track 2 selects up to three cases "
        "using only first-16 Sharpe, compounded return and cross-sectional ranking IC. "
        "This reused history is not a pristine holdout.", "",
        "| Case | Period | Sharpe | Return | Rank IC | Drawdown |", "|---|---|---:|---:|---:|---:|",
    ]
    displayed = list(dict.fromkeys([*summary["selected_case_ids"], _REFERENCE_SCENARIO, _REFERENCE_HARD]))
    for identifier in displayed:
        record = by_id[identifier]
        if record["status"] != "complete":
            lines.append(f"| {identifier} | FAILED; not ranked | — | — | — | — |")
            continue
        for period in _PERIODS:
            score = record["periods"][period]
            lines.append(f"| {identifier} | {period} | {_format_metric(score['sharpe'])} "
                         f"| {_format_metric(score['total_return'], percent=True)} "
                         f"| {_format_metric(score['rank_ic'])} "
                         f"| {_format_metric(score['max_drawdown'], percent=True)} |")
    for benchmark, periods in summary["benchmarks"].items():
        for period in _PERIODS:
            score = periods[period]
            lines.append(f"| {benchmark} | {period} | {_format_metric(score['sharpe'])} "
                         f"| {_format_metric(score['total_return'], percent=True)} | not supplied "
                         f"| {_format_metric(score['max_drawdown'], percent=True)} |")
    lines.extend(["", "## Controlled paired effects", "",
                  "`summary.json` contains all paired rank and budget contrasts, including per-seed differences. "
                  "The following counts summarize positive/negative Sharpe deltas; they are not significance tests.",
                  "", "| Representation | Contrast family | Period | Positive / negative / flat / incomplete |",
                  "|---|---|---|---:|"])
    contrasts = summary["paired_contrasts"]
    for source in _SOURCES:
        for kind in ["rank_at_fixed_budget", "budget_at_fixed_rank"]:
            for period in _PERIODS:
                relevant = [row for row in contrasts if row["source"] == source and row["kind"] == kind]
                values = [row["deltas"]["sharpe"] for row in relevant
                          if row["status"] == "complete" and row["period"] == period]
                incomplete = sum(row["status"] != "complete" for row in relevant)
                count = (sum(value > 1e-12 for value in values), sum(value < -1e-12 for value in values),
                         sum(abs(value) <= 1e-12 for value in values), incomplete)
                lines.append(f"| {source} | {kind} | {period} | {' / '.join(str(value) for value in count)} |")
    lines.extend(["", "## Uncertainty, exposure and failures", "",
                  "Case JSON/NPZ files retain per-seed paths, observed covariance, joint/marginal NLL, "
                  "Mahalanobis error, fixed equal-weight calibration diagnostics, effective asset counts, "
                  "equity exposure and numerical solver routes. No calibration parameters are fitted. "
                  "NLL does not select or promote a model. Rankings use means, not portfolio weights.", "",
                  "Risk-budget infeasibility and numerical failures remain explicit failed cases. "
                  "No missing month is dropped or replaced by previous weights; "
                  "partial cases are excluded from Pareto lists."])
    if summary["failures"]:
        for row in summary["failures"]:
            lines.append(f"- {row['id']}: {row['successful_allocations']}/84 allocations; "
                         f"{json.dumps(row['failure_kind_counts'], sort_keys=True)}")
    else:
        lines.append("No incomplete cases.")
    lines.extend(["", "## Evidence and verification", "",
                  "`case_manifest.json` freezes all cases/code/input hashes before allocation; "
                  "`summary.json` records each sealed case JSON/NPZ hash and the selection protocol. "
                  "`table.csv` shows every complete/failed case and every reporting period. "
                  "Seeds share one market history; scenario draws are not new independent market evidence. "
                  "Returns are gross of explicit costs and taxes. Negative-mean signed-Sharpe multistart "
                  "agreement is not a proof of global optimality.", ""])
    return "\n".join(lines).encode()


def run_track(output_root: str, workers: int = 14) -> dict:
    """Evaluate all Track 1 cases and publish an immutable independent report.

    Args:
        output_root: Campaign outputs directory with sealed completed fits,
            context.npz and the 24 complete grid component panels.
        workers: Case-level process count, constrained to 1 through 14. The
            parent must call tracks sequentially after its fit queue stops.

    Returns:
        Summary including selected_case_ids and selected_best_case (the first
        first-16 Pareto shortlist recipe, or None). All later-period scores are
        descriptive only; no production configuration is promoted.

    Raises:
        ValueError: Fits are unfinished, worker count invalid or immutable
            evidence changed. Numerical case failures stay in the report.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 14:
        raise ValueError("The shared allocation queue supports one through 14 workers")
    root = Path(output_root).resolve()
    if not json.loads((root / "FITS_COMPLETED.json").read_text()).get("passed"):
        raise ValueError("All fitting jobs must finish before the allocation stage")
    cases = build_cases()
    manifest = _manifest(root, cases)
    folder = root / "track1"
    _write(folder / "case_manifest.json", manifest)
    if workers == 1:
        records = [evaluate_case(case, str(root)) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            tasks = [pool.submit(evaluate_case, case, str(root)) for case in cases]
            records = [task.result() for task in as_completed(tasks)]
    records.sort(key=lambda row: row["id"])
    selector = _recipes()
    tuning_rows = _candidate_rows(records, "tuning_first16")
    selected = selector.select_candidates(tuning_rows, limit=3)
    frontiers = {period: selector.pareto_ids(_candidate_rows(records, period)) for period in _PERIODS}
    failures = [{"id": row["id"], "successful_allocations": row["successful_allocations"],
                 "failure_kind_counts": row["failure_kind_counts"], "failures": row["failures"]}
                for row in records if row["status"] != "complete"]
    by_id = {row["id"]: row for row in records}
    complete = len(tuning_rows)
    summary = {"schema": "bayesfolio.gp_tuning.track1_results.v1", "track": 1,
               "selection_period": "tuning_first16", "selected_case_ids": selected,
               "selected_best_case": by_id[selected[0]]["case"] if selected else None,
               "selected_cases": [by_id[identifier]["case"] for identifier in selected],
               "complete_scenario_cases": complete, "failed_scenario_cases": 36 - complete,
               "hard_rank_reference_status": by_id[_REFERENCE_HARD]["status"],
               "pareto_case_ids_by_period": frontiers, "paired_contrasts": _contrasts(records),
               "failures": failures, "benchmarks": _benchmarks(root),
               "case_manifest_sha256": _hash(folder / "case_manifest.json"),
               "case_sha256": {str(root / "cases" / f"{row['id']}.{extension}"):
                               _hash(root / "cases" / f"{row['id']}.{extension}")
                               for row in records for extension in ["json", "npz"]},
               "automatic_promotion": False, "pristine_holdout_claimed": False,
               "pure_allocator_comparison": False, "old_July_architecture_reproduced": False}
    for name, expected in manifest["input_sha256"].items():
        if _hash(Path(name)) != expected:
            raise ValueError(f"Track 1 input changed while evaluating cases: {name}")
    _write(folder / "table.csv", _table(records, selected, frontiers["all28"]))
    _write(folder / "RESULTS.md", _report(summary, records))
    _write(folder / "summary.json", summary)
    return summary
