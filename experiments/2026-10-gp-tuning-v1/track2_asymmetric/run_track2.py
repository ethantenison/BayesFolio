"""Run and report bounded asymmetric-GP comparisons using completed fits only.

This experiment orchestration module owns static-recipe allocation batches and
reports, not GP fitting or production settings. Returns are decimals, IC and
Sharpe are unitless, covariance has squared-decimal-return units, and investment
values are dollars. Selection uses first-16-origin metrics only; all settings are
fixed across dates. No allocation starts merely by importing this module.

Usage: the parent campaign calls ``run_track(output_root, workers=14)`` after all
shared component NPZs are complete. Synthetic tests inject a fake evaluator.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import multiprocessing
import os
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import recipe_math as recipes

HERE = Path(__file__).resolve().parent
EVALUATOR_PATH = HERE.parent / "track1_scenarios/case_evaluator.py"
PERIODS = ("tuning_first16", "validation_last12", "all28")
BASELINE_COMPONENTS = ("broad_r2_i15_exp", "compact_r2_i15_exp")
EXPECTED_ALLOCATIONS = 84


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(root: Path, path: Path, value: Any) -> None:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Track artifacts must remain within the supplied campaign output root.")
    payload = value if isinstance(value, bytes) else (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Conflicting immutable Track 2 evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _evaluate_worker(case: dict[str, Any], output_root: str) -> dict[str, Any]:
    evaluator_root = str(EVALUATOR_PATH.parent)
    if evaluator_root not in sys.path:
        sys.path.insert(0, evaluator_root)
    from case_evaluator import evaluate_case

    return evaluate_case(case, output_root)


def _batch(cases: Sequence[dict[str, Any]], output_root: str, pool: ProcessPoolExecutor) -> list[dict[str, Any]]:
    futures = {pool.submit(_evaluate_worker, case, output_root): case for case in cases}
    results = []
    for future in as_completed(futures):
        case = futures[future]
        try:
            record = future.result()
        except Exception:
            # Preserve infrastructure failures as ineligible observations; never
            # manufacture replacement portfolios or partial-window performance.
            record = {
                "id": case["id"], "case": case, "status": "failed", "successful_allocations": 0,
                "expected_allocations": EXPECTED_ALLOCATIONS, "error": traceback.format_exc(), "periods": {},
            }
        if record.get("id") != case["id"] or record.get("case") != case:
            raise ValueError("Evaluator returned a different case identity or specification.")
        results.append(record)
    return sorted(results, key=lambda record: record["id"])


def _key(case: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        case["broad_id"], case["compact_id"], tuple(case["alpha"]), case["cov_compact_fraction"],
        case.get("third_id"), case.get("third_fraction", 0.0), case.get("third_cov_fraction", 0.0),
        case["allocator"],
    )


def _eligible(record: Mapping[str, Any]) -> bool:
    if (
        record.get("status") != "complete"
        or record.get("successful_allocations") != EXPECTED_ALLOCATIONS
        or record.get("expected_allocations") != EXPECTED_ALLOCATIONS
    ):
        return False
    for period in PERIODS:
        metrics = record.get("periods", {}).get(period, {})
        if any(not isinstance(metrics.get(key), (float, int)) or not np.isfinite(metrics[key])
               for key in recipes.OBJECTIVES):
            return False
    return True


def _selection_scores(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": record["id"], "period": recipes.SELECTION_PERIOD,
         **{key: record["periods"][recipes.SELECTION_PERIOD][key] for key in recipes.OBJECTIVES}}
        for record in records if _eligible(record)
    ]


def _selection_manifest(
    records: Sequence[dict[str, Any]], selected: Sequence[str], *, controls: Sequence[str] = ()
) -> dict[str, Any]:
    return {
        "selection_period": recipes.SELECTION_PERIOD,
        "validation_used_for_selection": False,
        "all28_used_for_selection": False,
        "weights_change_by_month": False,
        "required_successful_allocations": EXPECTED_ALLOCATIONS,
        "selection_method": "normalized_Pareto_ideal_proximity_then_farthest_first; lexical_ties",
        "controls": list(controls), "selected_ids": list(selected),
        "eligible_tuning_metrics": sorted(_selection_scores(records), key=lambda row: row["id"]),
        "ineligible_ids": sorted(record["id"] for record in records if not _eligible(record)),
    }


def _run_track(
    output_root: Path,
    evaluate_batch: Callable[[Sequence[dict[str, Any]]], list[dict[str, Any]]],
) -> dict[str, Any]:
    target = output_root / "track2"
    fits_path = output_root / "FITS_COMPLETED.json"
    if not json.loads(fits_path.read_text()).get("passed"):
        raise ValueError("The shared GP fit queue must complete before Track 2 can start.")
    available = [path.stem for path in sorted((output_root / "components").glob("*.npz"))]
    proposed_screen = [{**case, "allocator": "hard_rank"} for case in recipes.screen_cases(available)]
    context_path = output_root / "context.npz"
    with np.load(context_path, allow_pickle=False) as context:
        if context["seeds"].tolist() != [27, 127, 227] or context["assets"].tolist() != list(recipes.ASSETS):
            raise ValueError("Shared seed or ETF axes changed.")
        if len(context["dates"]) != 28:
            raise ValueError("Exactly 28 shared forecast origins required.")
        dates = context["dates"].astype(str).tolist()
    input_paths = [fits_path, context_path, *(output_root / "components" / f"{name}.npz" for name in available)]
    implementation_paths = [HERE / "recipe_math.py", HERE / "run_track2.py", HERE / "plan.json"]
    implementation_paths.append(EVALUATOR_PATH)
    contract = {
        "schema": "bayesfolio.gp_tuning.track2_execution.v1",
        "input_sha256": {str(path.resolve()): _digest(path) for path in input_paths},
        "implementation_sha256": {str(path.resolve()): _digest(path) for path in implementation_paths},
        "selection_dates": dates[:16], "validation_dates": dates[16:],
        "selection_uses_validation": False, "weights_fixed_across_dates_and_seeds": True,
        "maximum_unique_cases": 468, "minimum_successful_allocations_per_eligible_case": EXPECTED_ALLOCATIONS,
    }
    _save(output_root, target / "execution_contract.json", contract)
    all_records: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    proposal_mapping: list[dict[str, str]] = []

    def evaluate(proposals: Sequence[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
        pending = []
        pending_keys = set()
        for case in proposals:
            key = _key(case)
            if key not in by_key and key not in pending_keys:
                pending.append(case)
                pending_keys.add(key)
        records = evaluate_batch(pending) if pending else []
        if {record["id"] for record in records} != {case["id"] for case in pending}:
            raise ValueError("Allocation batch omitted or duplicated case records.")
        for record in records:
            if record["id"] in all_records:
                raise ValueError("Distinct numeric recipes cannot share one case ID.")
            all_records[record["id"]] = record
            by_key[_key(record["case"])] = record
        resolved = [by_key[_key(case)] for case in proposals]
        mappings = [{"stage": stage, "proposed_id": case["id"], "evaluated_id": record["id"]}
                    for case, record in zip(proposals, resolved, strict=True)]
        proposal_mapping.extend(mappings)
        _save(output_root, target / "stages" / f"{stage}_cases.json", {"cases": list(proposals), "mapping": mappings})
        _save(output_root, target / "stages" / f"{stage}_outcomes.json", {
            "complete_ids": sorted(record["id"] for record in resolved if _eligible(record)),
            "ineligible": [record for record in resolved if not _eligible(record)],
        })
        return resolved

    screen = evaluate(proposed_screen, "01_screen")
    baseline = next(record for record in screen if (
        record["case"]["broad_id"], record["case"]["compact_id"]
    ) == BASELINE_COMPONENTS)
    if not _eligible(baseline):
        raise ValueError("Original baseline is incomplete; fix the allocator/inputs before interpreting Track 2.")
    selected_pairs = recipes.select_candidates(
        _selection_scores(screen), limit=3, always_include=(baseline["id"],)
    )
    _save(output_root, target / "stages/01_screen_selection.json", _selection_manifest(
        screen, selected_pairs, controls=(baseline["id"],)
    ))
    pair_summaries = []
    for pair_index, pair_id in enumerate(selected_pairs, start=1):
        pair = all_records[pair_id]
        prefix = f"02_pair{pair_index}"
        proposed_means = [{**case, "allocator": "hard_rank"}
                          for case in recipes.mean_retuning_cases(pair["case"])]
        means = evaluate(proposed_means, f"{prefix}_mean")
        mean_finalists = recipes.select_candidates(_selection_scores(means), limit=3)
        _save(output_root, target / "stages" / f"{prefix}_mean_selection.json", _selection_manifest(
            means, mean_finalists
        ))
        proposed_covariances = [{**case, "allocator": "hard_rank"} for case in recipes.covariance_retuning_cases(
            [all_records[identifier]["case"] for identifier in mean_finalists]
        )]
        covariances = evaluate(proposed_covariances, f"{prefix}_covariance")
        best_mean = recipes.select_candidates(_selection_scores(means), limit=1)[0]
        best_final = recipes.select_candidates(_selection_scores(means + covariances), limit=1)[0]
        _save(output_root, target / "stages" / f"{prefix}_final_selection.json", _selection_manifest(
            means + covariances, [best_final]
        ))
        pair_summaries.append({
            "fixed_blend_case_id": pair_id, "mean_finalist_ids": mean_finalists,
            "selected_mean_case_id": best_mean, "selected_retuned_case_id": best_final,
            "broad_id": pair["case"]["broad_id"], "compact_id": pair["case"]["compact_id"],
        })
    records = list(all_records.values())
    if len(records) > 468:
        raise ValueError("Track 2 exceeded its preregistered static-recipe budget.")
    selected_best = recipes.select_candidates(_selection_scores(records), limit=3)
    _save(output_root, target / "stages/03_final_selection.json", _selection_manifest(records, selected_best))
    _save(output_root, target / "proposal_mapping.json", {"mappings": proposal_mapping})
    summary = _report(output_root, records, baseline["id"], selected_pairs, pair_summaries, selected_best)
    for paths in (contract["input_sha256"], contract["implementation_sha256"]):
        for name, expected in paths.items():
            if _digest(Path(name)) != expected:
                raise ValueError(f"Protected Track 2 input changed during evaluation: {name}")
    artifacts = sorted(path for path in target.rglob("*") if path.is_file() and path.name != "COMPLETED.json")
    source_records = [output_root / "cases" / f"{record['id']}.json" for record in records]
    _save(output_root, target / "COMPLETED.json", {
        "passed": True, "complete_case_count": summary["complete_case_count"],
        "ineligible_case_count": summary["ineligible_case_count"],
        "all_cases_successful": summary["ineligible_case_count"] == 0,
        "artifact_sha256": {str(path.resolve()): _digest(path) for path in artifacts},
        "case_record_sha256": {str(path.resolve()): _digest(path) for path in source_records if path.exists()},
    })
    return summary


def _period_scores(records: Sequence[dict[str, Any]], period: str) -> list[dict[str, Any]]:
    return [{"id": record["id"], **{key: record["periods"][period][key] for key in recipes.OBJECTIVES}}
            for record in records if _eligible(record)]


def _delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {period: {key: float(candidate["periods"][period][key] - reference["periods"][period][key])
                     for key in recipes.OBJECTIVES} for period in PERIODS}


def _benchmark_metrics(output_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    with np.load(output_root / "context.npz", allow_pickle=False) as context:
        risk_free = context["risk_free"].copy()
        benchmarks = {"Schwab aggressive proxy": context["benchmark_schwab_total"].copy(),
                      "Simple momentum top seven": context["benchmark_momentum_total"].copy()}
    result = {}
    for name, returns in benchmarks.items():
        if returns.shape != (28,) or risk_free.shape != (28,) or not np.isfinite(returns).all():
            raise ValueError("Benchmarks require 28 finite matching monthly returns.")
        periods = {}
        for period, subset in zip(PERIODS, (slice(0, 16), slice(16, 28), slice(0, 28)), strict=True):
            total = returns[subset]
            excess = total - risk_free[subset]
            volatility = float(np.std(excess, ddof=1))
            if volatility <= 0 or not np.isfinite(excess).all():
                raise ValueError("A conventional benchmark Sharpe requires finite returns and positive volatility.")
            wealth = np.r_[10000.0, 10000 * np.cumprod(1 + total)]
            periods[period] = {
                "sharpe": float(np.sqrt(12) * np.mean(excess) / volatility),
                "total_return": float(wealth[-1] / 10000 - 1), "ending_value": float(wealth[-1]),
                "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            }
        result[name] = periods
    return result


def _report(
    output_root: Path,
    records: Sequence[dict[str, Any]],
    baseline_id: str,
    selected_pairs: Sequence[str],
    pair_summaries: Sequence[dict[str, Any]],
    selected_best: Sequence[str],
) -> dict[str, Any]:
    by_id = {record["id"]: record for record in records}
    complete = [record for record in records if _eligible(record)]
    pair_results = []
    for pair in pair_summaries:
        fixed = by_id[pair["fixed_blend_case_id"]]
        mean = by_id[pair["selected_mean_case_id"]]
        final = by_id[pair["selected_retuned_case_id"]]
        pair_results.append({
            **pair, "fixed_blend_periods": fixed["periods"], "selected_mean_periods": mean["periods"],
            "selected_retuned_periods": final["periods"],
            "mean_tuning_minus_fixed_blend": _delta(mean, fixed),
            "covariance_tuning_minus_selected_mean": _delta(final, mean),
            "fixed_blend_minus_original_baseline": _delta(fixed, by_id[baseline_id]),
            "retuned_minus_original_baseline": _delta(final, by_id[baseline_id]),
            "selected_mean_case": mean["case"], "selected_retuned_case": final["case"],
        })
    descriptive_leaders = {
        period: {key: max(complete, key=lambda record: (record["periods"][period][key], record["id"]))["id"]
                 for key in recipes.OBJECTIVES} for period in PERIODS
    }
    summary = {
        "schema": "bayesfolio.gp_tuning.track2_results.v1", "track": "track2_asymmetric",
        "selection_period": recipes.SELECTION_PERIOD, "validation_used_for_selection": False,
        "all28_used_for_selection": False, "pristine_holdout": False, "weights_change_by_month": False,
        "unique_case_count": len(records), "complete_case_count": len(complete),
        "ineligible_case_count": len(records) - len(complete),
        "ineligible_ids": sorted(record["id"] for record in records if not _eligible(record)),
        "baseline_case_id": baseline_id, "baseline_periods": by_id[baseline_id]["periods"],
        "selected_component_pair_case_ids": list(selected_pairs), "pair_results": pair_results,
        "selected_best_case_ids": list(selected_best),
        "selected_best_cases": [by_id[identifier]["case"] for identifier in selected_best],
        "selected_best_metrics": {identifier: by_id[identifier]["periods"] for identifier in selected_best},
        "pareto_ids_by_period": {period: recipes.pareto_ids(_period_scores(records, period)) for period in PERIODS},
        "descriptive_leaders_by_period": descriptive_leaders,
        "descriptive_leaders_are_not_validation_selected": True,
        "benchmarks": _benchmark_metrics(output_root),
        "additional_gp_fits": 0,
    }
    target = output_root / "track2"
    _save(output_root, target / "summary.json", summary)
    fields = ["id", "status", "stage", "broad_id", "compact_id", "alpha", "cov_compact_fraction", "period",
              "successful_allocations", "sharpe", "total_return", "rank_ic", "max_drawdown", "ending_value",
              "selected_on_first16"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for record in sorted(records, key=lambda row: row["id"]):
        case = record["case"]
        for period in PERIODS:
            metrics = record.get("periods", {}).get(period, {})
            writer.writerow({
                "id": record["id"], "status": record["status"], "stage": case["stage"],
                "broad_id": case["broad_id"], "compact_id": case["compact_id"], "alpha": json.dumps(case["alpha"]),
                "cov_compact_fraction": case["cov_compact_fraction"], "period": period,
                "successful_allocations": record.get("successful_allocations", 0),
                **{key: metrics.get(key, "") for key in fields[9:14]},
                "selected_on_first16": record["id"] in selected_best,
            })
    _save(output_root, target / "tables.csv", buffer.getvalue().encode())
    _save(output_root, target / "RESULTS.md", _markdown(summary, by_id).encode())
    return summary


def _markdown(summary: dict[str, Any], by_id: Mapping[str, dict[str, Any]]) -> str:
    lines = [
        "# Track 2: asymmetric GP settings and static blending", "",
        f"Evaluated {summary['unique_case_count']} distinct recipes: {summary['complete_case_count']} complete, "
        f"{summary['ineligible_case_count']} ineligible. Each eligible recipe completed all 84 seed/month allocations.",
        "", "All GP fitting was shared; this track launched no GP fits. Hard rank retains NE >=7, no ETF cap, "
        "and the unchanged training-only historical-EW variance ceiling. No CVaR, fallback, or relaxation.",
        "", "## Selection and interpretation", "",
        "The first 16 origins alone selected component pairs and all fixed blending recipes. The last 12 origins "
        "are retrospective temporal validation; all 28 are exploratory because this history was previously inspected. "
        "No weights change by month. Validation/full-window leaders below are descriptive, not promotion decisions.",
        "", "## First-16-selected finalists", "",
        "| Case | Period | Sharpe | Total return | Rank IC | Max drawdown | Ending value |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    shown = list(dict.fromkeys([summary["baseline_case_id"], *summary["selected_best_case_ids"]]))
    details = []
    for index, identifier in enumerate(shown):
        label = "Original baseline" if index == 0 else f"First-16 finalist {index}"
        for period in PERIODS:
            metrics = by_id[identifier]["periods"][period]
            lines.append(f"| {label} | {period} | {metrics['sharpe']:.3f} | {metrics['total_return']:.2%} | "
                         f"{metrics['rank_ic']:.3f} | {metrics['max_drawdown']:.2%} | "
                         f"${metrics['ending_value']:,.0f} |")
        case = by_id[identifier]["case"]
        details.extend(["", f"{label}: `{identifier}`; broad `{case['broad_id']}`, compact `{case['compact_id']}`; "
                        f"compact family mean fractions `{case['alpha']}`; compact covariance "
                        f"fraction `{case['cov_compact_fraction']}`.", ""])
    for name, periods in summary["benchmarks"].items():
        for period in PERIODS:
            metrics = periods[period]
            lines.append(f"| {name} | {period} | {metrics['sharpe']:.3f} | {metrics['total_return']:.2%} | "
                         f"n/a | {metrics['max_drawdown']:.2%} | ${metrics['ending_value']:,.0f} |")
    lines.extend(details)
    lines.extend(["## Settings versus mean/covariance retuning", ""])
    for index, pair in enumerate(summary["pair_results"], start=1):
        lines.extend([f"### Admitted pair {index}: {pair['broad_id']} + {pair['compact_id']}", "",
                      "All three cases below were chosen using the same first-16-only rule; the later period was "
                      "not used to choose the winner.", "",
                      "| Comparison | Period | Delta Sharpe | Delta return | Delta rank IC |",
                      "|---|---|---:|---:|---:|"])
        for key, label in [("fixed_blend_minus_original_baseline", "Settings only minus original baseline"),
                           ("mean_tuning_minus_fixed_blend", "Mean tuning minus unchanged blend"),
                           ("covariance_tuning_minus_selected_mean", "Covariance tuning minus selected mean"),
                           ("retuned_minus_original_baseline", "Retuned minus original baseline")]:
            for period in PERIODS:
                delta = pair[key][period]
                lines.append(f"| {label} | {period} | {delta['sharpe']:+.3f} | {delta['total_return']:+.2%} | "
                             f"{delta['rank_ic']:+.3f} |")
        lines.extend(["", f"Fixed-blend case: `{pair['fixed_blend_case_id']}`. Mean-only selection: "
                      f"`{pair['selected_mean_case_id']}`. Retuned selection: "
                      f"`{pair['selected_retuned_case_id']}`.", ""])
    lines.extend(["## Descriptive full-window leaders (not validation-selected)", ""])
    for key, identifier in summary["descriptive_leaders_by_period"]["all28"].items():
        lines.append(f"- {key}: `{identifier}`.")
    lines.extend(["", "## Limitations and audit trail", "",
                  "The bounded component and blend menus are not exhaustive continuous optimization. Shortlisting "
                  "on one unchanged recipe can miss a component pair that needs substantially different weights. "
                  "Seeds are initialization replicates, not independent market histories. No pristine holdout, "
                  "transaction-cost adjustment, or future-performance guarantee is implied.", "",
                  "See `summary.json` for period-specific Pareto sets and first-16-selected recipes, `tables.csv` "
                  "for every period/case, `stages/*selection.json` for immutable selection-only evidence, and "
                  "`proposal_mapping.json` for exact duplicate reuse. Per-seed results and numerical diagnostics "
                  "remain in shared `cases/<id>.json` and `.npz` artifacts. Benchmarks and full-strategy comparisons "
                  "use the same shared monthly outcomes. Each subperiod ending value resets to $10,000. Benchmark "
                  "rank IC is n/a because these reference portfolios are not a comparable set of 15 GP forecasts.", ""])
    if summary["ineligible_ids"]:
        lines.extend(["Incomplete cases were excluded from selection and Pareto sets, not partially averaged:", "",
                      *[f"- `{identifier}`" for identifier in summary["ineligible_ids"]], ""])
    return "\n".join(lines)


def run_track(output_root: str, workers: int = 14) -> dict[str, Any]:
    """Evaluate Track 2 static recipes after the shared fit queue completes.

    Args:
        output_root: Campaign outputs containing ``context.npz`` and completed
            ``components/*.npz``. This function writes only its ``track2``
            reports and the shared evaluator's immutable ``cases`` artifacts.
        workers: One bounded allocation pool, from one to fourteen workers;
            this must not run concurrently with other campaign compute pools.

    Returns:
        Independent track summary, including ``selected_best_case_ids`` and
        ``selected_best_cases`` chosen on first-16-origin metrics only.

    Raises:
        ValueError: Missing evidence, incomplete baseline, invalid worker count,
            or any conflict with existing immutable artifacts.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 14:
        raise ValueError("Track 2 permits one allocation pool of at most fourteen workers.")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    root = Path(output_root).resolve()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=64) as pool:
        return _run_track(root, lambda cases: _batch(cases, str(root), pool))
