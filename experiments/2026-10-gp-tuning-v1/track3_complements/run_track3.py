"""Execute and independently report the bounded complementary-GP track.

This experiment-local I/O adapter consumes complete frozen component panels and
delegates unchanged portfolio allocation/scoring to the shared case evaluator.
It never launches GP fits. Mean/covariance blends are static across every month
and seed; return units are decimals, covariance units squared decimals, NAV USD.

Usage:
    Import ``run_track3.run_track(output_root, workers=14, base_case=None)`` from the parent
    campaign after all component panels are sealed. No work runs on import.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_CAMPAIGN = _HERE.parent
_PERIODS = {"tuning_first16": slice(0, 16), "validation_last12": slice(16, 28), "all28": slice(0, 28)}
_FAMILIES = {
    "all_etfs": slice(0, 15), "us_equity": slice(0, 4), "international_equity": slice(4, 8),
    "us_fixed_income": slice(8, 12), "international_fixed_income": slice(12, 15),
}
_OBJECTIVES = ("sharpe", "total_return", "rank_ic")
_FIELDS = (
    "broad_id", "compact_id", "alpha", "cov_compact_fraction", "third_id", "third_fraction", "third_cov_fraction",
)
_DEFAULT_BASE = {
    "broad_id": "broad_r2_i15_exp", "compact_id": "compact_r2_i15_exp", "alpha": [0.5, 0, 0.375, 0.25],
    "cov_compact_fraction": 0.5, "third_id": None, "third_fraction": 0.0, "third_cov_fraction": 0.0,
}


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__ or "").resolve() != path.resolve():
            raise ValueError(f"Module name collision: {name}")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load experiment helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _evaluator() -> ModuleType:
    path = _CAMPAIGN / "track1_scenarios/case_evaluator.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    # A canonical, importable module name is necessary for spawn's function
    # pickling even when the parent imported this orchestrator through a spec.
    return _module(path, "case_evaluator")


def _selector() -> ModuleType:
    return _module(_CAMPAIGN / "track2_asymmetric/recipe_math.py", "_bayesfolio_tuning_recipe_math")


def _math() -> ModuleType:
    return _module(_HERE / "complementarity_math.py", "_bayesfolio_tuning_complementarity_math")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _save(path: Path, value: bytes | dict) -> None:
    data = value if isinstance(value, bytes) else _encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to replace immutable Track 3 evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Concurrent conflicting Track 3 evidence: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _base(base_case: dict | None) -> dict:
    source = _DEFAULT_BASE if base_case is None else base_case
    if any(key not in source for key in _FIELDS):
        raise ValueError("A complete static two-model base recipe is required")
    result = {key: source[key] for key in _FIELDS}
    if result["third_id"] is not None or result["third_fraction"] != 0 or result["third_cov_fraction"] != 0:
        raise ValueError("The base must be a two-model recipe, not an existing third-model ensemble")
    if not isinstance(result["alpha"], list) or len(result["alpha"]) != 4:
        raise ValueError("Four fixed family fractions are required")
    if any(not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1
           for value in [*result["alpha"], result["cov_compact_fraction"]]):
        raise ValueError("Base blending fractions must be finite static scalars in [0, 1]")
    return json.loads(json.dumps(result, allow_nan=False))


def build_cases(component_ids: list[str], *, base_case: dict | None = None,
                stage: str = "primary_approved") -> tuple[list[dict], list[dict]]:
    """Construct the bounded static grid, deduplicating identical covariance recipes.

    Args:
        component_ids: Exactly the six preregistered third-model identifiers.
        base_case: Optional complete static two-model case from Track 2.
        stage: Primary reference or explicitly labeled secondary integration.

    Returns:
        One control plus 48 unique cases, and six alias records identifying the
        gamma=0.5 matched-covariance/half-covariance duplicate grid cells.

    Raises:
        ValueError: Components repeat, a base is not two-model/static, or the
            stage label is unsupported.
    """
    if len(component_ids) != 6 or len(set(component_ids)) != 6:
        raise ValueError("Exactly six distinct complementary components are required")
    if stage not in {"primary_approved", "secondary_track2"}:
        raise ValueError("Explicit primary or secondary Track 3 stage required")
    reference = _base(base_case)
    token = hashlib.sha256(json.dumps(reference, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
    prefix = "t3p" if stage == "primary_approved" else "t3s"
    control = {"id": f"{prefix}_control__{token}", "track": "track3_complements", "stage": stage,
               "allocator": "hard_rank", **reference}
    cases, aliases = [control], []
    for component in sorted(component_ids):
        for fraction in (0.125, 0.25, 0.5):
            seen: dict[float, str] = {}
            for mode, covariance in (("unchanged", 0.0), ("match_mean", fraction), ("half", 0.5)):
                identity = (f"{prefix}__{component}__m{round(10000 * fraction):05d}"
                            f"__v{round(10000 * covariance):05d}__{token}")
                if covariance in seen:
                    aliases.append({"third_id": component, "third_fraction": fraction,
                                    "third_cov_fraction": covariance, "grid_mode": mode,
                                    "evaluated_id": seen[covariance]})
                    continue
                seen[covariance] = identity
                cases.append({**control, "id": identity, "third_id": component, "third_fraction": fraction,
                              "third_cov_fraction": covariance})
    return cases, aliases


def select_representatives(records: list[dict], control_id: str) -> dict:
    """Select at most three complement recipes using only first-16-month scores.

    Args:
        records: All evaluated cases in a single stage, including the control.
        control_id: Mandatory complete no-third-model reference identifier.

    Returns:
        Stage status, failed IDs, full first-16 Pareto IDs, and at most three
        selected recipe IDs. Any failed case prevents a winner declaration;
        incomplete cases never receive partial-performance scores.

    Raises:
        ValueError: Duplicate case IDs, missing control, or malformed scores.
    """
    identifiers = [record["id"] for record in records]
    if len(set(identifiers)) != len(identifiers) or control_id not in identifiers:
        raise ValueError("Distinct records and an included control are required")
    failed = sorted(record["id"] for record in records if record.get("status") != "complete")
    if failed:
        return {"status": "incomplete", "failed_ids": failed, "first16_pareto_ids": [], "selected_ids": []}
    rows = [{"id": record["id"], "period": "tuning_first16",
             **{key: record["periods"]["tuning_first16"][key] for key in _OBJECTIVES}} for record in records]
    selector = _selector()
    selected = selector.select_candidates(rows, limit=4, always_include=[control_id])
    return {"status": "complete", "failed_ids": [], "first16_pareto_ids": selector.pareto_ids(rows),
            "selected_ids": [identifier for identifier in selected if identifier != control_id]}


def _initialize_allocation_worker() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _worker_failure(case: dict, root: Path, error: Exception) -> dict:
    # Exceptions outside the evaluator's normal failed-case path remain
    # explicit evidence, never an empty or partial success.
    record = {"id": case["id"], "case": case, "status": "failed", "periods": {},
              "failures": [{"stage": "worker_exception", "error": f"{type(error).__name__}: {error}"}]}
    _save(root / "track3/worker_failures" / f"{case['id']}.json", record)
    return record


def _evaluate_cases(cases: list[dict], root: Path, workers: int) -> list[dict]:
    records: dict[str, dict] = {}
    evaluator = _evaluator()
    if workers == 1:
        for case in cases:
            try:
                records[case["id"]] = evaluator.evaluate_case(case, str(root))
            except Exception as error:
                records[case["id"]] = _worker_failure(case, root, error)
    else:
        _initialize_allocation_worker()
        by_id = {case["id"]: case for case in cases}
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(evaluator.evaluate_case, case, str(root)): case["id"] for case in cases}
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as error:
                    record = _worker_failure(by_id[futures[future]], root, error)
                if record["id"] != futures[future]:
                    raise ValueError("Allocation worker returned another case's identity")
                records[record["id"]] = record
    return [records[case["id"]] for case in cases]


def _diversity(root: Path, context: dict, control_id: str, component_ids: list[str]) -> dict:
    control = _read(root / "cases" / f"{control_id}.npz")
    reference = control["mean"]
    report: dict[str, Any] = {}
    for component_id in component_ids:
        component = _read(root / "components" / f"{component_id}.npz")
        for axis in ("dates", "assets", "seeds"):
            np.testing.assert_array_equal(component[axis], context[axis])
        report[component_id] = {}
        for period, months in _PERIODS.items():
            report[component_id][period] = {}
            for family, assets in _FAMILIES.items():
                values = _math().residual_diversity(context["actual"][months, assets], reference[:, months, assets],
                                                   component["mean"][:, months, assets])
                means = {}
                valid_counts = {}
                for key in values[0]:
                    finite = [record[key] for record in values if record[key] is not None]
                    means[key] = float(np.mean(finite)) if finite else None
                    valid_counts[key] = len(finite)
                report[component_id][period][family] = {
                    "three_seed_mean": means, "valid_seed_counts": valid_counts,
                    "per_seed": [{"seed": int(seed), **record}
                                 for seed, record in zip(context["seeds"], values, strict=True)],
                }
    return report


def _benchmarks(context: dict) -> dict:
    metrics_path = _CAMPAIGN.parent / "2026-10-portfolio-evaluation-v1/metrics.py"
    metrics = _module(metrics_path, "_bayesfolio_tuning_track3_metrics")
    result = {}
    for name, key in (("schwab_aggressive", "benchmark_schwab_total"),
                      ("momentum_top7", "benchmark_momentum_total")):
        result[name] = {}
        for period, months in _PERIODS.items():
            performance = metrics.portfolio_performance(context[key][months], context["risk_free"][months])
            result[name][period] = {
                "sharpe": performance["conventional_sharpe"], "total_return": performance["compounded_total_return"],
                "max_drawdown": performance["max_drawdown_total_nav"],
                "ending_value": 10000 * (1 + performance["compounded_total_return"]),
                "rank_ic": None,
            }
    return result


def _table_rows(records: list[dict], control_id: str, selection: dict) -> list[dict]:
    control = next(record for record in records if record["id"] == control_id)
    rows = []
    for record in records:
        case = record["case"]
        for period in _PERIODS:
            row = {"id": record["id"], "stage": case["stage"], "status": record["status"],
                   "period": period, "control": record["id"] == control_id,
                   "selected_on_first16": record["id"] in selection["selected_ids"],
                   "third_id": case["third_id"], "third_fraction": case["third_fraction"],
                   "third_cov_fraction": case["third_cov_fraction"]}
            if record["status"] == "complete":
                values = record["periods"][period]
                for key in (*_OBJECTIVES, "max_drawdown", "ending_value", "effective_assets", "equity_weight",
                            "joint_nll", "marginal_nll", "sharpe_seed_sd", "return_seed_sd"):
                    row[key] = values.get(key)
                if control["status"] == "complete":
                    baseline = control["periods"][period]
                    for key in _OBJECTIVES:
                        row[f"delta_{key}"] = values[key] - baseline[key]
                        base_by_seed = {item["seed"]: item[key] for item in baseline["per_seed"]}
                        row[f"positive_seed_count_{key}"] = sum(
                            item[key] > base_by_seed[item["seed"]] for item in values["per_seed"])
                    flags = _math().incremental_pareto(
                        np.array([[baseline[key] for key in _OBJECTIVES]]),
                        np.array([[values[key] for key in _OBJECTIVES]]),
                    )
                    row["distinct_frontier_expansion_vs_control"] = flags["distinct_frontier_expansion"][0]
                    row["dominates_control"] = flags["dominates_any_baseline"][0]
            rows.append(row)
    return rows


def _csv_bytes(rows: list[dict]) -> bytes:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _markdown(summary: dict, rows: list[dict]) -> bytes:
    lines = ["# Track 3: complementary model results", "",
             f"Status: **{summary['status']}**. No production baseline was promoted.", "",
             "All blending fractions are fixed across all months, seeds, and ETFs. "
             "The original two-model reference remains family-dependent but static.", "",
             "Representatives were selected using only the first 16 origins. The later 12 "
             "are a temporal check, not an untouched holdout; all-28 results are exploratory.", ""]
    for stage, details in summary["stages"].items():
        lines.extend([f"## {stage}", "", f"Status: {details['status']}; "
                      f"{details['complete_cases']}/{details['case_count']} complete cases. "
                      f"{details['nominal_grid_cells']} nominal grid cells reduce to "
                      f"{details['unique_complement_cases']} distinct recipes plus the two-model control.", ""])
        if details["status"] != "complete":
            lines.extend(["The experiment is incomplete. No leader or representative is declared. "
                          "Failures are preserved in summary.json and per-case evidence.", ""])
            continue
        selected = set(details["selection"]["selected_ids"] + [details["control_id"]])
        lines.extend(["| Configuration | Period | Sharpe | Return | Rank IC | Drawdown | Ending $10k |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for row in rows:
            if row["stage"] == stage and row["id"] in selected:
                label = "Two-model control" if row["control"] else (
                    f"{row['third_id']}; mean {row['third_fraction']:.3f}, cov {row['third_cov_fraction']:.3f}")
                lines.append(f"| {label} | {row['period']} | {row['sharpe']:.3f} | "
                             f"{row['total_return']:.2%} | {row['rank_ic']:.3f} | "
                             f"{row['max_drawdown']:.2%} | ${row['ending_value']:,.0f} |")
        if not details["selection"]["selected_ids"]:
            lines.extend(["", "No distinct complement survived the first-16 Pareto selection against the control."])
        lines.extend(["", "Later-period and full-window values describe the frozen first-16 selections; "
                      "they did not choose the blend recipe. Detailed seed, exposure, NLL and "
                      "incremental-Pareto diagnostics remain in tables.csv.", ""])
    lines.extend(["## Benchmarks", "", "Benchmarks share the same dates and total-return convention. "
                  "Ranking IC is not assigned to the static Schwab portfolio.", "",
                  "| Benchmark | Period | Sharpe | Return | Drawdown | Ending $10k |",
                  "|---|---|---:|---:|---:|---:|"])
    for name, periods in summary["benchmarks"].items():
        for period, values in periods.items():
            sharpe = "undefined" if values["sharpe"] is None else f"{values['sharpe']:.3f}"
            lines.append(f"| {name} | {period} | {sharpe} | {values['total_return']:.2%} | "
                         f"{values['max_drawdown']:.2%} | ${values['ending_value']:,.0f} |")
    lines.extend(["", "## Interpretation guardrails", "",
                  "Low residual correlation alone is not an improvement: an inaccurate model can be "
                  "different but harmful. Inspect correction/error alignment, incremental portfolio "
                  "Pareto value, and seed/temporal consistency together. Family-specific diversity "
                  "is recorded separately for first16, later12, and all28 in diversity.json.", "",
                  "Arithmetic covariance pooling is a declared recipe, not an estimated cross-model "
                  "error covariance. No covariance repair, CVaR, constraint relaxation, fallback "
                  "weights, fitted calibration, trading costs or taxes are introduced. Three seeds "
                  "share one market history. Period NAVs each restart at $10,000; they are not "
                  "slices of a differently initialized capital path.", ""])
    return "\n".join(lines).encode()


def run_track(output_root: str, workers: int = 14, base_case: dict | None = None) -> dict:
    """Evaluate fixed third-model recipes and seal an independent Track 3 report.

    Args:
        output_root: Parent campaign outputs directory containing complete
            context.npz, components/{id}.npz, and a shared cases directory.
        workers: Single allocation-worker pool ceiling, between 1 and 14. The
            caller must not concurrently launch competing allocation/fit pools.
        base_case: Optional first-16-selected Track 2 two-model recipe. When
            present, add a separately labeled secondary integration stage; the
            primary approved-reference stage is always retained.

    Returns:
        Immutable summary with complete/incomplete status, preserved failures,
        static recipes, first-16 representatives, later-period descriptions,
        benchmarks and artifact fingerprints. Any failed case prevents a
        leader declaration for its stage. No GP fit or automatic promotion runs.

    Raises:
        ValueError: Worker count, panel axes, settings, or immutable evidence
            changed; missing input panels also prevent execution.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 14:
        raise ValueError("workers must be an integer between 1 and 14")
    root = Path(output_root).resolve()
    context = _read(root / "context.npz")
    if context["actual"].shape != (28, 15) or context["seeds"].tolist() != [27, 127, 227]:
        raise ValueError("Complete common 28-origin/15-ETF/three-seed context required")
    registry = json.loads((_HERE / "components.json").read_text())
    component_ids = [item["modelname"] for item in registry["components"]]
    stages = [("primary_approved", None)]
    if base_case is not None:
        stages.append(("secondary_track2", base_case))
    all_rows, stage_results, diversity = [], {}, {}
    inputs = [_HERE / "components.json", Path(__file__).resolve(), _HERE / "complementarity_math.py",
              _CAMPAIGN / "track1_scenarios/case_evaluator.py", _CAMPAIGN / "track2_asymmetric/recipe_math.py",
              root / "context.npz", *(root / "components" / f"{identifier}.npz" for identifier in component_ids)]
    for stage, reference in stages:
        cases, aliases = build_cases(component_ids, base_case=reference, stage=stage)
        for key in ("broad_id", "compact_id"):
            inputs.append(root / "components" / f"{cases[0][key]}.npz")
        input_hashes = {str(path): _digest(path) for path in sorted(set(inputs))}
        _save(root / "track3" / stage / "contract.json", {
            "schema": "bayesfolio.gp_tuning.track3.stage.v1", "cases": cases, "aliases": aliases,
            "input_sha256": input_hashes, "selection_period": "tuning_first16",
            "all_months_share_static_blend": True, "allocation_worker_ceiling": workers,
        })
        records = _evaluate_cases(cases, root, workers)
        control_id = cases[0]["id"]
        selection = select_representatives(records, control_id)
        failure_records = [{"id": record["id"], "failures": record.get("failures", [])}
                           for record in records if record["status"] != "complete"]
        stage_results[stage] = {
            "status": selection["status"], "control_id": control_id, "base_case": cases[0],
            "case_count": len(cases), "complete_cases": sum(record["status"] == "complete" for record in records),
            "nominal_grid_cells": 54, "unique_complement_cases": len(cases) - 1,
            "aliases": aliases, "selection": selection, "failures": failure_records,
        }
        try:
            diversity[stage] = {"status": "complete",
                                "components": _diversity(root, context, control_id, component_ids)}
        except (OSError, KeyError, ValueError, AssertionError) as error:
            diversity[stage] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            stage_results[stage]["status"] = "incomplete"
            stage_results[stage]["selection"]["selected_ids"] = []
        all_rows.extend(_table_rows(records, control_id, selection))
        _save(root / "track3" / stage / "evaluated_records.json", {"records": records})
    destination = root / "track3"
    _save(destination / "tables.csv", _csv_bytes(all_rows))
    _save(destination / "diversity.json", diversity)
    summary = {
        "schema": "bayesfolio.gp_tuning.track3.summary.v1",
        "status": ("complete" if all(value["status"] == "complete" for value in stage_results.values())
                   else "incomplete"),
        "stages": stage_results, "benchmarks": _benchmarks(context), "automatic_promotion": False,
        "selection_uses": "first16 only; static recipes then described on later12 and all28",
        "untouched_holdout": False, "CVaR": False, "fallback": False, "constraint_relaxation": False,
        "input_sha256": {str(path): _digest(path) for path in sorted(set(inputs))},
        "artifact_sha256": {str(path): _digest(path) for path in
                            [destination / "tables.csv", destination / "diversity.json"]},
    }
    _save(destination / "summary.json", summary)
    _save(destination / "RESULTS.md", _markdown(summary, all_rows))
    return summary
