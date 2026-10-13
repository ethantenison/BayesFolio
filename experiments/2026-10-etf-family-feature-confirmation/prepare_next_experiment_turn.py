"""Build the bounded evidence packet for Qwen's post-confirmation planning turn.

The packet is created only after every requested deterministic confirmation run
passes validation. It gives Qwen the measured Pareto frontier, frozen
hypervolume definition, exact evaluated configurations, and a small allowlist
of local research sources. It never launches a GP fit.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/prepare_next_experiment_turn.py \
      --seeds 27,127,227
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLANNING_DIR = EXPERIMENT_DIR / "planning"
DEFAULT_CONFIG = EXPERIMENT_DIR / "us_confirmation_config.json"
REQUIRED_SEEDS = [27, 127, 227]
RESEARCH_ALLOWLIST = [
    "experiments/2026-10-etf-family-feature-confirmation/README.md",
    "experiments/2026-10-etf-family-feature-confirmation/PLAN.md",
    "experiments/2026-10-etf-family-feature-confirmation/us_confirmation_config.json",
    "experiments/2026-07-us-equity-etf-features/experiment_plan.md",
    "experiments/2026-07-us-equity-etf-features/decision_log.md",
    "experiments/2026-07-us-equity-etf-features/redundancy_pruning_decisions.md",
    "experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py",
    "experiments/2026-07-international-equity-etf-features/README.md",
    "experiments/2026-07-international-equity-etf-features/runs/xgboost_shap_optuna_rankspace_400_20260709T0036Z/REPORT.md",
]


def parse_args() -> argparse.Namespace:
    """Parse the required completed seed panel and optional config path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", default="27,127,227")
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    """Parse and validate the fixed three-seed confirmation panel."""
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"The planning turn requires the complete seed panel {REQUIRED_SEEDS}; received {seeds}.")
    return seeds


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a required results file."""
    if not path.exists():
        raise FileNotFoundError(f"Required step-3 result is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def validate_step3_results(
    *,
    seeds: list[int],
    run_objectives: pd.DataFrame,
    aggregates: pd.DataFrame,
    validation: dict[str, Any],
    hypervolume: dict[str, Any],
) -> None:
    """Require a complete, common, validated three-seed result panel."""
    failures: list[str] = []
    if validation.get("seeds") != seeds:
        failures.append("validation seed panel")
    if validation.get("all_passed") is not True or validation.get("failures"):
        failures.append("run validation")
    if len(run_objectives) != 12 or set(run_objectives["seed"].astype(int)) != set(seeds):
        failures.append("12 run-level objective rows")
    if len(aggregates) != 4 or not aggregates["seed_count"].eq(3).all():
        failures.append("four three-seed configuration aggregates")
    if not aggregates["validation_passed"].astype(bool).all():
        failures.append("aggregate validation")
    if hypervolume.get("schema") != "bayesfolio.etf_family_confirmation_hypervolume.v1":
        failures.append("hypervolume schema")
    if not isinstance(hypervolume.get("value"), int | float) or hypervolume["value"] <= 0:
        failures.append("positive observed hypervolume")
    columns = [*run_objectives.columns, *aggregates.columns]
    if any("turnover" in column.lower() for column in columns):
        failures.append("turnover exclusion")
    if failures:
        raise ValueError(f"Step 3 is not ready for Qwen planning: {failures}")


def build_prompt(evidence_path: Path, evidence_sha256: str) -> str:
    """Return the constrained xhigh prompt for one post-results planning turn."""
    return f"""# Qwen step 4: plan the next deterministic experiment batch

Use xhigh reasoning for this single planning turn. Step 3 is complete and validated. Your goal is to propose the next
small deterministic batch most likely to improve the three-objective Pareto frontier and its fixed-reference
hypervolume. Treat this as sequential multi-objective experimental design inspired by Bayesian optimization, not as a
license to invent measured effects.

## First actions

1. Read `{evidence_path}` and verify its SHA-256 is `{evidence_sha256}`.
2. Immediately create `{PLANNING_DIR / 'NEXT_EXPERIMENT_PLAN.md'}` with a
   checkpoint containing the current frontier, current hypervolume, and the exploitation/exploration questions you
   will investigate. Do this before broader research so a useful artifact survives response exhaustion.
3. Perform bounded local research using only the evidence packet's `research_allowlist`. Use targeted `rg`, small line
   ranges, and small CSV selections; do not recurse through run directories or dump large artifacts. Allowlist entries
   are relative to the packet's `research_root` (`{REPO_ROOT}`).

## Required design

- Propose 4-8 new configurations: at least one exploitation candidate, one frontier-bridging candidate, and one
  genuinely exploratory candidate.
- Exploit promising neighborhoods of the measured frontier, bridge complementary strengths of existing nondominated
  configurations, and explore coherent cached feature/kernel/task-covariance ideas not tested in the four-config
  confirmation.
- Keep `SPY,MGK,VTV,IWM`, `t_index`, seeds `27,127,227`, the same 24 scored dates, rank 2, maxiter 15, 256 scenarios,
  target, optimizer settings, and cached-data-only rule fixed. Never include IJR, turnover, a time ablation, external
  data refresh, or the production-budget reference.
- Change one coherent feature or covariance hypothesis per candidate when possible. Specify exact parent
  configuration, feature additions/removals, model changes, and runner arguments. Mark unsupported ideas as deferred
  data/code gaps rather than executable candidates.
- IC, Gaussian NLL, and conventional Sharpe are the only objectives. Return and RMSE remain diagnostic-only.
- Do not claim numerical expected hypervolume improvement from four observations. Rank candidates by evidence,
  novelty, frontier-gap coverage, and falsifiability; actual hypervolume improvement is measured only after execution.
- Include deterministic execution order, controls, early stopping rules, run counts, and validation requirements.

## Required outputs

Write both files and do not launch GP fits or modify existing code/configuration:

1. `{PLANNING_DIR / 'NEXT_EXPERIMENT_PLAN.md'}`
2. `{PLANNING_DIR / 'next_experiment_candidates.json'}`

The JSON must conform to `{PLANNING_DIR / 'next_experiment_candidates.schema.json'}`.
Finish by running:

```bash
{REPO_ROOT / '.venv/bin/python'} \
  {EXPERIMENT_DIR / 'validate_next_experiment_plan.py'}
```

If validation fails, repair only the two planning outputs and rerun it. Stop after validation passes.
"""


def main() -> None:
    """Validate step 3 and write Qwen's bounded evidence packet and prompt."""
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    config = read_json(args.config)
    run_path = RESULTS_DIR / "us_confirmation_run_objectives.csv"
    aggregate_path = RESULTS_DIR / "us_confirmation_configuration_aggregates.csv"
    pareto_path = RESULTS_DIR / "us_confirmation_pareto.csv"
    validation_path = RESULTS_DIR / "us_confirmation_validation.json"
    hypervolume_path = RESULTS_DIR / "us_confirmation_hypervolume.json"
    run_objectives = pd.read_csv(run_path)
    aggregates = pd.read_csv(aggregate_path)
    pareto = pd.read_csv(pareto_path)
    validation = read_json(validation_path)
    hypervolume = read_json(hypervolume_path)
    validate_step3_results(
        seeds=seeds,
        run_objectives=run_objectives,
        aggregates=aggregates,
        validation=validation,
        hypervolume=hypervolume,
    )

    missing_sources = [relative for relative in RESEARCH_ALLOWLIST if not (REPO_ROOT / relative).exists()]
    if missing_sources:
        raise FileNotFoundError(f"Planning research sources are missing: {missing_sources}")
    input_paths = [run_path, aggregate_path, pareto_path, validation_path, hypervolume_path, args.config]
    evidence = {
        "schema": "bayesfolio.etf_family_next_experiment_evidence.v1",
        "stage": 4,
        "stage_3_status": "complete_and_validated",
        "seeds": seeds,
        "objectives": {
            "maximize": ["mean_ic", "sharpe"],
            "minimize": ["mean_gaussian_nll"],
            "diagnostic_only": ["cumulative_return", "rmse"],
            "turnover_excluded": True,
        },
        "hypervolume": hypervolume,
        "validation": validation,
        "evaluated_configurations": config["configurations"],
        "run_objectives": run_objectives.to_dict(orient="records"),
        "configuration_aggregates": aggregates.to_dict(orient="records"),
        "pareto_front": pareto.to_dict(orient="records"),
        "fixed_decisions": config["fixed_decisions"],
        "common_args": config["common_args"],
        "research_root": str(REPO_ROOT),
        "research_allowlist": RESEARCH_ALLOWLIST,
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in input_paths},
    }
    PLANNING_DIR.mkdir(parents=True, exist_ok=True)
    evidence_path = PLANNING_DIR / "step4_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_sha256 = sha256_file(evidence_path)
    prompt_path = PLANNING_DIR / "QWEN_STEP4_PROMPT.md"
    prompt_path.write_text(build_prompt(evidence_path, evidence_sha256), encoding="utf-8")
    print(json.dumps({"evidence": str(evidence_path), "sha256": evidence_sha256, "prompt": str(prompt_path)}, indent=2))


if __name__ == "__main__":
    main()
