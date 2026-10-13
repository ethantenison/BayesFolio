"""Run the approved 100-row macro split kernel batch.

The batch fixes the U.S. equity ETF feature set and portfolio settings while
varying only the new macro_market and macro_credit_inflation block kernels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WRAPPER = SCRIPT_DIR / "run_first_round_walkforward.py"
RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_CANDIDATE_CSV = (
    Path("/Users/et/.openclaw/agents/bayes-ml-engineer/workspace/outbox")
    / "macro_split_kernel_100_proposed_configs_v2.csv"
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    role: str
    feature_set: str
    gp_experiment: str
    time_modulation_mode: str
    maxiter: int
    covar_grouping_proposal: str
    include_three_way_interaction: bool
    include_time_etf_market_three_way: bool
    include_time_etf_credit_three_way: bool
    include_macro_market_credit_interaction: bool
    include_macro_market_credit_etf_three_way: bool
    include_time_macro_market_credit_three_way: bool
    macro_market_kernel_id: str
    macro_market_ard: bool
    macro_credit_inflation_kernel_id: str
    macro_credit_inflation_ard: bool
    implementation_status: str


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y"}


def read_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            candidates.append(
                Candidate(
                    candidate_id=row["candidate_id"],
                    role=row["role"],
                    feature_set=row["feature_set"],
                    gp_experiment=row["gp_experiment"],
                    time_modulation_mode=row["time_modulation_mode"],
                    maxiter=int(row["maxiter"]),
                    covar_grouping_proposal=row["covar_grouping_proposal"],
                    include_three_way_interaction=parse_bool(row["include_three_way_interaction"]),
                    include_time_etf_market_three_way=parse_bool(
                        row.get("include_time_etf_market_three_way", "false")
                    ),
                    include_time_etf_credit_three_way=parse_bool(
                        row.get("include_time_etf_credit_three_way", "false")
                    ),
                    include_macro_market_credit_interaction=parse_bool(
                        row["include_macro_market_credit_interaction"]
                    ),
                    include_macro_market_credit_etf_three_way=parse_bool(
                        row.get("include_macro_market_credit_etf_three_way", "false")
                    ),
                    include_time_macro_market_credit_three_way=parse_bool(
                        row.get("include_time_macro_market_credit_three_way", "false")
                    ),
                    macro_market_kernel_id=row["macro_market_kernel_id"],
                    macro_market_ard=parse_bool(row["macro_market_ard"]),
                    macro_credit_inflation_kernel_id=row["macro_credit_inflation_kernel_id"],
                    macro_credit_inflation_ard=parse_bool(row["macro_credit_inflation_ard"]),
                    implementation_status=row["implementation_status"],
                )
            )
    return candidates


def candidate_command(
    candidate: Candidate,
    *,
    output_root: Path,
    run_id: str,
    feature_path: Path | None,
    tickers: str | None,
    max_windows: int,
    posterior_scenarios: int,
) -> list[str]:
    command = [
        sys.executable,
        str(WRAPPER),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--max-windows",
        str(max_windows),
        "--maxiter",
        str(candidate.maxiter),
        "--posterior-scenarios",
        str(posterior_scenarios),
        "--resume-scenarios",
        "--gp-experiment",
        candidate.gp_experiment,
        "--time-modulation-mode",
        candidate.time_modulation_mode,
        "--upperlng",
        "0.35",
        "--nea",
        "3",
        "--gp-riskfolio-rm",
        "CVaR",
        "--gp-riskfolio-obj",
        "MaxRet",
        "--feature-set",
        candidate.feature_set,
        "--covar-grouping-proposal",
        candidate.covar_grouping_proposal,
        "--macro-market-kernel-id",
        candidate.macro_market_kernel_id,
        "--macro-market-ard",
        "true" if candidate.macro_market_ard else "false",
        "--macro-credit-inflation-kernel-id",
        candidate.macro_credit_inflation_kernel_id,
        "--macro-credit-inflation-ard",
        "true" if candidate.macro_credit_inflation_ard else "false",
    ]
    if feature_path is not None:
        command.extend(["--feature-path", str(feature_path)])
    if tickers is not None:
        command.extend(["--tickers", tickers])
    if candidate.include_three_way_interaction:
        command.append("--include-three-way-interaction")
    if candidate.include_time_etf_market_three_way:
        command.append("--include-time-etf-market-three-way")
    if candidate.include_time_etf_credit_three_way:
        command.append("--include-time-etf-credit-three-way")
    if candidate.include_macro_market_credit_interaction:
        command.append("--include-macro-market-credit-interaction")
    if candidate.include_macro_market_credit_etf_three_way:
        command.append("--include-macro-market-credit-etf-three-way")
    if candidate.include_time_macro_market_credit_three_way:
        command.append("--include-time-macro-market-credit-three-way")
    return command


def read_metrics(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "portfolio_summary.csv"
    forecast_path = run_dir / "forecast_metrics_overall.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    gp = summary.loc[summary["strategy"].eq("gp_scenarios_riskfolio")].iloc[0]
    ew = summary.loc[summary["strategy"].eq("equal_weight")].iloc[0]
    row = {
        "cumulative_return": float(gp["cumulative_return"]),
        "return_vs_equal_weight": float(gp["cumulative_return"] - ew["cumulative_return"]),
        "sharpe": float(gp["sharpe"]) if not pd.isna(gp["sharpe"]) else math.nan,
        "max_drawdown": float(gp["max_drawdown"]),
        "avg_turnover": float(gp["avg_turnover"]),
        "mean_ic": float(gp["mean_ic"]) if not pd.isna(gp["mean_ic"]) else math.nan,
        "median_ic": float(gp["median_ic"]) if not pd.isna(gp["median_ic"]) else math.nan,
        "equal_weight_return": float(ew["cumulative_return"]),
    }
    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path).iloc[0]
        row.update(
            {
                "rmse": float(forecast["rmse"]),
                "mean_gaussian_nll": float(forecast["mean_gaussian_nll"]),
                "mean_pred_std": float(forecast["mean_pred_std"]),
                "coverage_80": float(forecast["coverage_80"]),
            }
        )
    return row


def run_candidate(
    candidate: Candidate,
    *,
    output_root: Path,
    feature_path: Path | None,
    tickers: str | None,
    max_windows: int,
    posterior_scenarios: int,
) -> dict[str, Any]:
    run_id = f"full_{candidate.candidate_id}_{candidate.role}"
    run_dir = output_root / run_id
    command = candidate_command(
        candidate,
        output_root=output_root,
        run_id=run_id,
        feature_path=feature_path,
        tickers=tickers,
        max_windows=max_windows,
        posterior_scenarios=posterior_scenarios,
    )
    if (run_dir / "portfolio_summary.csv").exists():
        status = "cached"
        stdout_tail = ""
        stderr_tail = ""
    else:
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        status = "complete" if result.returncode == 0 else "failed"
        stdout_tail = result.stdout[-4000:]
        stderr_tail = result.stderr[-4000:]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "agent_subprocess_stdout_tail.txt").write_text(stdout_tail)
        (run_dir / "agent_subprocess_stderr_tail.txt").write_text(stderr_tail)
    row = {
        **candidate.__dict__,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
        "command": " ".join(command),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    if status in {"complete", "cached"}:
        row.update(read_metrics(run_dir))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="macro_split_kernel_100_full_20260706T1600Z")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--posterior-scenarios", type=int, default=256)
    parser.add_argument("--expected-candidates", type=int, default=100)
    args = parser.parse_args()

    candidates = read_candidates(args.candidate_csv)
    if args.expected_candidates and len(candidates) != args.expected_candidates:
        raise ValueError(
            f"Expected {args.expected_candidates} candidates, found {len(candidates)} in {args.candidate_csv}"
        )

    output_dir = RUNS_DIR / args.run_id
    full_root = output_dir / "full_runs"
    full_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.candidate_csv, output_dir / "search_space.csv")

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_candidate,
                candidate,
                output_root=full_root,
                feature_path=args.feature_path,
                tickers=args.tickers,
                max_windows=args.max_windows,
                posterior_scenarios=args.posterior_scenarios,
            )
            for candidate in candidates
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            print(f"full: {len(rows)}/{len(candidates)} complete", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "all_results.csv", index=False)
    completed = results[results["status"].isin(["complete", "cached"])].copy()
    if not completed.empty:
        completed.sort_values(["cumulative_return", "mean_ic"], ascending=[False, False]).to_csv(
            output_dir / "leaderboard.csv",
            index=False,
        )

    manifest = {
        "run_id": args.run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "complete",
        "candidate_count": len(candidates),
        "completed_count": int(len(completed)),
        "failed_count": int((~results["status"].isin(["complete", "cached"])).sum()),
        "max_windows": args.max_windows,
        "posterior_scenarios": args.posterior_scenarios,
        "candidate_csv": str(args.candidate_csv),
        "feature_path": str(args.feature_path) if args.feature_path is not None else None,
        "tickers": args.tickers,
        "search_space": "search_space.csv",
        "all_results": "all_results.csv",
        "leaderboard": "leaderboard.csv",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
