"""Run planner-driven multitask GP from a CSV file.

This script reads a modeling CSV, maps user-provided input/output columns into
BayesFolio's planner-driven GP workflow, and prints fit status with compiled
mean/covariance configs.

Usage:
    poetry run python scripts/run_planned_gp_from_csv.py \
        --csv-path data.csv \
        --input-columns macro_1,macro_2,etf_1 \
        --target-column y_excess_lead \
        --task-column asset_id \
        --instruction-text "Use a matern kernel with ard for all input variables"

Optional live planner mode:
    poetry run python scripts/run_planned_gp_from_csv.py \
        --csv-path data.csv \
        --input-columns macro_1,macro_2,etf_1 \
        --target-column y_excess_lead \
        --task-column asset_id \
        --instruction-text "Use matern 1.5 on macro" \
        --require-live-planner
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from bayesfolio.engine.forecast import PlannedGPWorkflowOptions, run_planned_multitask_gp_from_dataframe


def _parse_comma_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def main() -> None:
    """Execute planner-driven multitask GP from CLI arguments."""

    parser = argparse.ArgumentParser(description="Run planner-driven multitask GP from a CSV file.")
    parser.add_argument("--csv-path", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--input-columns",
        required=True,
        help="Comma-separated non-task feature columns in model order.",
    )
    parser.add_argument("--target-column", required=True, help="Target return column (decimal units).")
    parser.add_argument("--task-column", required=True, help="Task/asset identifier column.")
    parser.add_argument(
        "--instruction-text",
        required=True,
        help="Free-form GP prior and kernel instructions passed to the planner workflow.",
    )
    parser.add_argument(
        "--require-live-planner",
        action="store_true",
        help="Fail fast if live planner endpoint is unavailable.",
    )
    parser.add_argument(
        "--planner-timeout-seconds",
        type=float,
        default=20.0,
        help="Planner HTTP timeout in seconds.",
    )

    args = parser.parse_args()
    input_columns = _parse_comma_list(args.input_columns)
    df = pd.read_csv(args.csv_path)

    options = PlannedGPWorkflowOptions(
        planner_timeout_seconds=args.planner_timeout_seconds,
        require_live_planner=args.require_live_planner,
    )

    artifacts = run_planned_multitask_gp_from_dataframe(
        df=df,
        input_columns=input_columns,
        output_columns=[args.target_column, args.task_column],
        instruction_text=args.instruction_text,
        options=options,
    )

    print(f"final_status={artifacts.result.final_status}")
    print(f"planner_client_status={artifacts.result.planner_client_status}")
    print(f"attempt_count={artifacts.result.fit_validation.attempt_count}")
    print("mean_config_json=")
    print(json.dumps(artifacts.result.mean_config, indent=2, sort_keys=True))
    print("covar_config_json=")
    print(json.dumps(artifacts.result.covar_config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
