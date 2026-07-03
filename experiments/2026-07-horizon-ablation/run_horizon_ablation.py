"""Run a tracked monthly-vs-3-week horizon ablation for the July portfolio span.

This experiment keeps the July full3 multitask GP configuration fixed and
changes only the feature/label/rebalance horizon:

* monthly control: BME
* 3-week candidate: 3W-FRI

Feature artifacts are built first, then the two GP walk-forward jobs run in
parallel. MLflow is the tracker/artifact store, with local manifests alongside
the run outputs for auditable lineage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand  # noqa: E402
from bayesfolio.core.settings import Horizon, Interval  # noqa: E402
from bayesfolio.engine.features import build_features_dataset, make_default_feature_providers  # noqa: E402
from bayesfolio.io import ParquetArtifactStore  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = REPO_ROOT / "experiments" / "2026-06-portfolio-optimization" / "run_monthly_optimization_walkforward.py"
ARTIFACT_ROOT = Path("/Users/et/.bayesfolio/artifacts/features/horizon_ablation_20260703")
MLFLOW_DIR = EXPERIMENT_DIR / "mlruns"
RUNS_DIR = EXPERIMENT_DIR / "runs"
START_DATE = date(2021, 3, 1)
LOOKBACK_DATE = date(2019, 3, 1)
END_DATE = date(2026, 7, 2)
EVAL_MIN_SCORED_DATE = "2025-06-01"
STARTING_VALUE = 10_000.0

ETF_TICKERS = [
    "SPY",
    "MGK",
    "VTV",
    "IJR",
    "IWM",
    "VNQ",
    "VNQI",
    "VEA",
    "VWO",
    "VSS",
    "BND",
    "IEF",
    "BNDX",
    "LQD",
    "HYG",
    "EWX",
    "VWOB",
    "HYEM",
]

SELECTED_ETF_COLS = [
    "baspread",
    "ret_kurt",
    "chmom",
    "mom12m",
    "mom36m",
    "cs_mom_rank",
    "max_dd_6m",
    "ma_signal",
    "ret_autocorr",
    "vol_z",
]

SELECTED_MACRO_COLS = [
    "hy_spread",
    "hy_spread_chg_1p",
    "hy_spread_z_12p",
    "vix_slope",
    "vix_ts_z_12p",
    "vix",
    "spy_flow_z_12p",
    "spy_ret",
    "erp",
    "cpi_chg_12p",
    "cpi_chg_1p",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
]


@dataclass(frozen=True)
class HorizonRunConfig:
    label: str
    horizon: str
    periods_per_year: float
    rebalance_frequency_label: str
    run_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=75)
    parser.add_argument("--posterior-scenarios", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--skip-feature-build", action="store_true")
    return parser.parse_args()


def json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def append_trace(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **payload}
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def git_dirty_summary() -> str:
    return git_output("status", "--short")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


def build_feature_artifact(config: HorizonRunConfig, run_dir: Path, *, seed: int, skip: bool) -> Path:
    feature_path = ARTIFACT_ROOT / f"portfolio_etf_macro_features_{config.run_id}.parquet"
    if skip and feature_path.exists():
        return feature_path

    command = BuildFeaturesDatasetCommand.model_validate(
        {
            "schema": "bayesfolio.features_dataset.command",
            "tickers": ETF_TICKERS,
            "drop_assets": [],
            "lookback_date": LOOKBACK_DATE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "interval": Interval.DAILY,
            "horizon": config.horizon,
            "etf_cols": SELECTED_ETF_COLS,
            "macro_cols": SELECTED_MACRO_COLS,
            "drop_etf_cols": [],
            "drop_macro_cols": [],
            "clip_quantile": 0.99,
            "seed": seed,
            "artifact_name": feature_path.name,
            "include_unlabeled_tail": True,
        }
    )
    providers = make_default_feature_providers(cache_root="artifacts/cache")
    artifact_store = ParquetArtifactStore(base_dir=str(ARTIFACT_ROOT))
    result = build_features_dataset(command=command, providers=providers, artifact_store=artifact_store)
    resolved = resolve_uri(str(result.artifact.uri))
    write_json(
        run_dir / "feature_artifact_manifest.json",
        {
            "schema": "bayesfolio.horizon_ablation.feature_artifact.v1",
            "label": config.label,
            "horizon": config.horizon,
            "path": resolved,
            "sha256": sha256_file(resolved),
            "row_count": result.artifact.row_count,
            "column_count": result.artifact.column_count,
            "diagnostics": result.diagnostics,
        },
    )
    return resolved


def runner_command(
    config: HorizonRunConfig,
    feature_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(RUNNER_PATH),
        "--feature-path",
        str(feature_path),
        "--output-dir",
        str(output_dir),
        "--run-id",
        config.run_id,
        "--max-windows",
        "100",
        "--maxiter",
        str(args.maxiter),
        "--seed",
        str(args.seed),
        "--posterior-scenarios",
        str(args.posterior_scenarios),
        "--include-live-window",
        "--min-scored-date",
        EVAL_MIN_SCORED_DATE,
        "--periods-per-year",
        str(config.periods_per_year),
        "--rebalance-frequency-label",
        config.rebalance_frequency_label,
        "--min-inferred-noise-level",
        "0.0025",
        "--upperlng",
        "0.25",
        "--nea",
        "10",
        "--gp-experiment",
        "positive_no_prior",
        "--input-transform-mode",
        "botorch_normalize",
        "--time-modulation-mode",
        "lengthscale_only",
        "--kernel-proposal",
        "none",
        "--kernel-half-life-months",
        "36.0",
        "--kernel-changepoint-date",
        "2021-03-31",
        "--kernel-changepoint-width-months",
        "6.0",
        "--kernel-composition-proposal",
        "e_plus_m_plus_t_plus_et_plus_mt_plus_emt",
        "--mean-kind",
        "multitask_constant",
        "--scenario-mean-scale",
        "1.0",
        "--turnover-blend",
        "0.50",
        "--lengthscale-floor",
        "0.02",
        "--outputscale-floor",
        "0.01",
        "--outputscale-prior-median",
        "0.05",
        "--outputscale-prior-sigma",
        "0.75",
        "--task-noise-floor-raw-std",
        "0.005",
    ]


def update_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "manifest.json"
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text())
    current.update(payload)
    write_json(path, current)


def log_summary_metrics(summary_path: Path, *, prefix: str) -> dict[str, float]:
    summary = pd.read_csv(summary_path)
    metrics: dict[str, float] = {}
    for _, row in summary.iterrows():
        strategy = str(row["strategy"])
        for column, value in row.items():
            if column == "strategy" or not isinstance(value, (int, float, np.floating)) or pd.isna(value):
                continue
            key = f"{prefix}.{strategy}.{column}"
            metrics[key] = float(value)
            mlflow.log_metric(key, float(value))
    return metrics


def run_walkforward(
    config: HorizonRunConfig,
    feature_path: Path,
    run_dir: Path,
    mlflow_run_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trace_path = run_dir / "agent_trace.jsonl"
    output_dir = run_dir / "portfolio_walkforward"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = runner_command(config, feature_path, output_dir, args)
    write_json(run_dir / "resolved_config.json", {**asdict(config), "command": command})
    append_trace(trace_path, "runner_start", command=command)
    update_manifest(
        run_dir,
        {
            "status": "running",
            "command": command,
            "artifact_paths": {"feature_path": feature_path, "walkforward_output_dir": output_dir},
        },
    )

    stdout_path = logs_dir / "runner.stdout.log"
    stderr_path = logs_dir / "runner.stderr.log"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    append_trace(trace_path, "runner_finish", returncode=process.returncode)

    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_artifact(run_dir / "resolved_config.json", artifact_path="lineage")
        mlflow.log_artifact(trace_path, artifact_path="lineage")
        mlflow.log_artifact(stdout_path, artifact_path="logs")
        mlflow.log_artifact(stderr_path, artifact_path="logs")
        if output_dir.exists():
            mlflow.log_artifacts(str(output_dir), artifact_path="portfolio_walkforward")
        status = "complete" if process.returncode == 0 else "failed"
        mlflow.set_tag("status", status)
        metrics = {}
        if process.returncode == 0:
            metrics = log_summary_metrics(output_dir / "portfolio_summary.csv", prefix=config.label)
        update_manifest(
            run_dir,
            {
                "status": status,
                "returncode": process.returncode,
                "metrics": metrics,
                "metric_file_path": output_dir / "portfolio_summary.csv",
                "plot_paths": [output_dir / "equity_curve.png", output_dir / "drawdown_curve.png"],
                "artifact_paths": {
                    "feature_path": feature_path,
                    "walkforward_output_dir": output_dir,
                    "stdout": stdout_path,
                    "stderr": stderr_path,
                },
            },
        )
        mlflow.log_artifact(run_dir / "manifest.json", artifact_path="lineage")

    return {"label": config.label, "status": status, "returncode": process.returncode, "run_dir": run_dir}


def equity_curve(run_dir: Path) -> pd.Series:
    returns = pd.read_csv(run_dir / "portfolio_walkforward" / "portfolio_returns.csv")
    frame = returns[
        (returns["strategy"] == "gp_scenarios_riskfolio") & (~returns["is_live_window"].astype(bool))
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    return (1.0 + frame["return"].astype(float)).cumprod() * STARTING_VALUE


def add_start_anchor(curve: pd.Series) -> pd.Series:
    anchor_date = pd.Timestamp(EVAL_MIN_SCORED_DATE)
    anchored = pd.concat([pd.Series([STARTING_VALUE], index=[anchor_date]), curve])
    return anchored[~anchored.index.duplicated(keep="last")].sort_index()


def markdown_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def build_comparison_artifacts(configs: list[HorizonRunConfig], run_dirs: dict[str, Path], parent_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    curves: dict[str, pd.Series] = {}
    for config in configs:
        summary = pd.read_csv(run_dirs[config.label] / "portfolio_walkforward" / "portfolio_summary.csv")
        gp = summary[summary["strategy"] == "gp_scenarios_riskfolio"].iloc[0].to_dict()
        rows.append({"label": config.label, "horizon": config.horizon, **gp})
        curves[config.label] = equity_curve(run_dirs[config.label])

    comparison = pd.DataFrame(rows)
    comparison.to_csv(parent_dir / "comparison_metrics.csv", index=False)
    anchored_curves = {label: add_start_anchor(curve) for label, curve in curves.items()}
    curve_df = pd.DataFrame(anchored_curves).sort_index()
    curve_df.to_csv(parent_dir / "comparison_equity_curves.csv", index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, curve in anchored_curves.items():
        curve.plot(ax=ax, marker="o", linewidth=2.0, markersize=4, label=label)
    ax.set_title("Full3 GP Horizon Ablation: Portfolio Value from $10,000")
    ax.set_ylabel("Portfolio value")
    ax.set_xlabel("Realized rebalance date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(parent_dir / "comparison_equity_curve.png", dpi=160)
    plt.close(fig)

    metric_cols = ["cumulative_return", "cagr", "annualized_vol", "sharpe", "max_drawdown", "mean_ic"]
    report = [
        "# Horizon Ablation Report",
        "",
        "Decision question: does 3W-FRI improve the fixed July full3 multitask GP portfolio workflow versus BME?",
        "",
        "Both runs use the same universe, features, GP configuration, seed, Riskfolio CVaR/Sharpe settings, "
        "and calendar input span. The realized evaluation window is selected by `min_scored_date=2025-06-01`, "
        "so each horizon uses all available native rebalance periods over that calendar span.",
        "",
        "## GP Strategy Metrics",
        "",
        markdown_table(comparison[["label", "horizon", *metric_cols]]),
        "",
        "## Artifacts",
        "",
        f"- Metrics: `{parent_dir / 'comparison_metrics.csv'}`",
        f"- Equity curve: `{parent_dir / 'comparison_equity_curve.png'}`",
        f"- Per-run directories: `{RUNS_DIR}`",
        "",
        "## Caveats",
        "",
        (
            "- The monthly and 3-week candidates have different native rebalance counts; "
            "this is the point of the horizon ablation, but it means window-by-window "
            "returns are not one-to-one paired."
        ),
        (
            "- July/live construction rows are logged but excluded from realized performance "
            "metrics because their labels are unavailable."
        ),
    ]
    (parent_dir / "decision_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parent_id = f"{timestamp}_monthly_vs_three_week_full3_july_span"
    configs = [
        HorizonRunConfig(
            label="monthly",
            horizon=Horizon.MONTHLY.value,
            periods_per_year=12.0,
            rebalance_frequency_label="monthly",
            run_id=f"{timestamp}_monthly_full3_july_span_control",
        ),
        HorizonRunConfig(
            label="three_week",
            horizon=Horizon.THREE_WEEK.value,
            periods_per_year=365.25 / 21.0,
            rebalance_frequency_label="three_week",
            run_id=f"{timestamp}_three_week_full3_july_span_ablation",
        ),
    ]

    parent_dir = RUNS_DIR / parent_id
    parent_dir.mkdir(parents=True, exist_ok=False)
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR.resolve()}")
    mlflow.set_experiment("bayesfolio_horizon_ablation_july_span")

    parent_manifest = {
        "schema": "bayesfolio.horizon_ablation.parent_manifest.v1",
        "run_id": parent_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "git_sha": git_output("rev-parse", "HEAD"),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "decision_question": (
            "Compare monthly BME versus 3W-FRI horizon for the fixed July full3 multitask GP workflow."
        ),
        "baseline": "monthly BME",
        "candidate": "3W-FRI",
        "target_metric": "gp_scenarios_riskfolio Sharpe and cumulative return over the July-portfolio calendar span",
        "data_window": {
            "lookback_date": LOOKBACK_DATE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "evaluation_min_scored_date": EVAL_MIN_SCORED_DATE,
        },
        "configs": [asdict(config) for config in configs],
        "tracker": {"type": "mlflow", "tracking_uri": f"file://{MLFLOW_DIR.resolve()}"},
    }
    write_json(parent_dir / "manifest.json", parent_manifest)
    write_json(parent_dir / "experiment_plan.json", parent_manifest)

    run_dirs: dict[str, Path] = {}
    feature_paths: dict[str, Path] = {}
    mlflow_run_ids: dict[str, str] = {}
    results: list[dict[str, Any]] = []

    with mlflow.start_run(run_name=parent_id) as parent_run:
        mlflow.set_tag("status", "running")
        mlflow.set_tag("run_role", "parent")
        mlflow.log_params(
            {
                "baseline": "monthly_BME",
                "candidate": "three_week_3W-FRI",
                "eval_min_scored_date": EVAL_MIN_SCORED_DATE,
                "maxiter": args.maxiter,
                "posterior_scenarios": args.posterior_scenarios,
                "seed": args.seed,
            }
        )
        parent_manifest["tracker"]["parent_mlflow_run_id"] = parent_run.info.run_id
        write_json(parent_dir / "manifest.json", parent_manifest)
        mlflow.log_artifact(parent_dir / "manifest.json", artifact_path="lineage")
        mlflow.log_artifact(parent_dir / "experiment_plan.json", artifact_path="lineage")

        for config in configs:
            run_dir = RUNS_DIR / config.run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            run_dirs[config.label] = run_dir
            manifest = {
                "schema": "bayesfolio.horizon_ablation.child_manifest.v1",
                "run_id": config.run_id,
                "parent_run_id": parent_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "status": "building_features",
                "git_sha": parent_manifest["git_sha"],
                "git_dirty_summary_at_start": parent_manifest["git_dirty_summary_at_start"],
                "baseline_or_incumbent_run_id": "monthly" if config.label != "monthly" else None,
                "candidate_label": config.label,
                "resolved_config_path": run_dir / "resolved_config.json",
                "agent_trace_path": run_dir / "agent_trace.jsonl",
                "split_window_definition": {
                    "lookback_date": LOOKBACK_DATE,
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "evaluation_min_scored_date": EVAL_MIN_SCORED_DATE,
                },
                "next_run_rationale": "Controlled horizon ablation requested by user; no autonomous search.",
            }
            write_json(run_dir / "manifest.json", manifest)
            append_trace(run_dir / "agent_trace.jsonl", "feature_build_start", config=asdict(config))
            with mlflow.start_run(run_name=config.run_id, nested=True) as child_run:
                mlflow_run_ids[config.label] = child_run.info.run_id
                mlflow.set_tag("run_role", "child")
                mlflow.set_tag("status", "building_features")
                mlflow.set_tag("horizon_label", config.label)
                mlflow.log_params(asdict(config))
                mlflow.log_params(
                    {
                        "maxiter": args.maxiter,
                        "posterior_scenarios": args.posterior_scenarios,
                        "seed": args.seed,
                    }
                )
                feature_path = build_feature_artifact(config, run_dir, seed=args.seed, skip=args.skip_feature_build)
                feature_paths[config.label] = feature_path
                mlflow.log_artifact(run_dir / "feature_artifact_manifest.json", artifact_path="lineage")
                mlflow.log_artifact(feature_path, artifact_path="features")
                update_manifest(
                    run_dir,
                    {
                        "status": "features_built",
                        "mlflow_run_id": child_run.info.run_id,
                        "feature_path": feature_path,
                        "feature_sha256": sha256_file(feature_path),
                    },
                )
                mlflow.log_artifact(run_dir / "manifest.json", artifact_path="lineage")
            append_trace(run_dir / "agent_trace.jsonl", "feature_build_finish", feature_path=feature_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    run_walkforward,
                    config,
                    feature_paths[config.label],
                    run_dirs[config.label],
                    mlflow_run_ids[config.label],
                    args,
                )
                for config in configs
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        failed = [result for result in results if result["status"] != "complete"]
        if failed:
            parent_manifest["status"] = "failed"
            parent_manifest["child_results"] = results
            write_json(parent_dir / "manifest.json", parent_manifest)
            mlflow.set_tag("status", "failed")
            mlflow.log_artifact(parent_dir / "manifest.json", artifact_path="lineage")
            raise RuntimeError(f"One or more child runs failed: {failed}")

        build_comparison_artifacts(configs, run_dirs, parent_dir)
        comparison = pd.read_csv(parent_dir / "comparison_metrics.csv")
        for _, row in comparison.iterrows():
            for metric in ["cumulative_return", "cagr", "annualized_vol", "sharpe", "max_drawdown", "mean_ic"]:
                value = row.get(metric)
                if isinstance(value, (int, float, np.floating)) and not pd.isna(value):
                    mlflow.log_metric(f"{row['label']}.gp_scenarios_riskfolio.{metric}", float(value))
        mlflow.log_artifacts(str(parent_dir), artifact_path="comparison")
        parent_manifest["status"] = "complete"
        parent_manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        parent_manifest["child_results"] = results
        parent_manifest["artifact_paths"] = {
            "comparison_metrics": parent_dir / "comparison_metrics.csv",
            "comparison_equity_curve": parent_dir / "comparison_equity_curve.png",
            "decision_report": parent_dir / "decision_report.md",
        }
        write_json(parent_dir / "manifest.json", parent_manifest)
        mlflow.set_tag("status", "complete")
        mlflow.log_artifact(parent_dir / "manifest.json", artifact_path="lineage")

    print(
        json.dumps(
            {"parent_run_id": parent_id, "parent_dir": parent_dir, "results": results}, indent=2, default=json_default
        )
    )


if __name__ == "__main__":
    main()
