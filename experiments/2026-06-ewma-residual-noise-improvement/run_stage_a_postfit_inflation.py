"""Stage A diagnostic: post-fit EWMA scenario inflation.

This isolates whether the EWMA hl1/scale0.50 benefit comes from changing the GP
fit or from adding residual-risk dispersion to the portfolio scenarios after a
plain GP fit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_DIR = Path(__file__).resolve().parent
PORTFOLIO_EXPERIMENT = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "run_monthly_optimization_walkforward.py"
)
PORT_SPEC = importlib.util.spec_from_file_location("portfolio_walkforward", PORTFOLIO_EXPERIMENT)
if PORT_SPEC is None or PORT_SPEC.loader is None:
    raise RuntimeError(f"Unable to load portfolio experiment from {PORTFOLIO_EXPERIMENT}")
portfolio_exp = importlib.util.module_from_spec(PORT_SPEC)
sys.modules[PORT_SPEC.name] = portfolio_exp
PORT_SPEC.loader.exec_module(portfolio_exp)

task_exp = portfolio_exp.task_exp

OUTPUT_ROOT = EXPERIMENT_DIR / "outputs" / "stage_a"
PLAIN_RUN = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "outputs"
    / "runs"
    / "20260621_signed_lkj_eta_2_plain_24w"
)
FULL_REFIT_EWMA_RUN = (
    REPO_ROOT
    / "experiments"
    / "2026-06-ewma-residual-noise-improvement"
    / "outputs"
    / "runs"
    / "20260621_ewma_hl1_scale_050_24w"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-run", type=Path, default=PLAIN_RUN)
    parser.add_argument("--full-refit-ewma-run", type=Path, default=FULL_REFIT_EWMA_RUN)
    parser.add_argument("--feature-path", type=Path, default=portfolio_exp.DEFAULT_FEATURE_PATH)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--upperlng", type=float, default=0.20)
    parser.add_argument("--nea", type=int, default=10)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def git_dirty_summary() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    run_id = args.run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_postfit_ewma_hl1_scale050"
    return OUTPUT_ROOT / run_id


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text())


def load_eval_noise(full_refit_run: Path) -> pd.DataFrame:
    noise = pd.read_csv(full_refit_run / "noise_model_diagnostics.csv", parse_dates=["date", "window_date"])
    eval_noise = noise[noise["noise_role"].eq("eval")].copy()
    eval_noise["date"] = pd.to_datetime(eval_noise["window_date"]).dt.normalize()
    return eval_noise[["date", "asset_id", "raw_noise_std", "raw_noise_variance"]]


def inflate_predictions(plain_predictions: pd.DataFrame, eval_noise: pd.DataFrame) -> pd.DataFrame:
    predictions = plain_predictions.merge(eval_noise, on=["date", "asset_id"], how="left")
    if predictions["raw_noise_std"].isna().any():
        missing = predictions[predictions["raw_noise_std"].isna()][["date", "asset_id"]].head(10)
        raise ValueError(f"Missing EWMA eval noise for prediction rows:\n{missing}")
    predictions["plain_y_std"] = predictions["y_std"].astype(float)
    predictions["ewma_noise_y_std"] = predictions["raw_noise_std"].astype(float)
    predictions["y_std"] = np.sqrt(predictions["plain_y_std"] ** 2 + predictions["ewma_noise_y_std"] ** 2)
    predictions["score"] = predictions["y_pred"].astype(float) / np.clip(
        predictions["y_std"].astype(float),
        1e-12,
        None,
    )
    return predictions[
        [
            "date",
            "asset_id",
            "y_true",
            "y_pred",
            "y_std",
            "plain_y_std",
            "ewma_noise_y_std",
            "score",
        ]
    ]


def write_manifest(args: argparse.Namespace, output_dir: Path, dates: list[pd.Timestamp]) -> None:
    manifest = {
        "schema": "bayesfolio.ewma_stage_a_postfit_inflation.manifest.v1",
        "command": " ".join(sys.argv),
        "git_sha": git_sha(),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "feature_path": str(resolve_existing(args.feature_path)),
        "feature_sha256": sha256_file(resolve_existing(args.feature_path)),
        "output_dir": str(output_dir),
        "plain_run": str(resolve_existing(args.plain_run)),
        "full_refit_ewma_run": str(resolve_existing(args.full_refit_ewma_run)),
        "question": (
            "Does EWMA hl1/scale0.50 help mainly through post-fit portfolio scenario risk inflation, "
            "or through changing the fitted GP?"
        ),
        "variant": {
            "label": "postfit_ewma_hl1_scale_050",
            "scenario_semantics": (
                "plain GP observation-posterior scenarios plus independent Gaussian residual EWMA noise "
                "drawn after fit"
            ),
            "prediction_std_semantics": "sqrt(plain_y_std^2 + ewma_noise_y_std^2)",
            "seed": args.seed,
            "upperlng": args.upperlng,
            "nea": args.nea,
        },
        "windows": {
            "n": len(dates),
            "first": dates[0].date().isoformat(),
            "last": dates[-1].date().isoformat(),
        },
        "input_manifests": {
            "plain": load_manifest(resolve_existing(args.plain_run)),
            "full_refit_ewma": load_manifest(resolve_existing(args.full_refit_ewma_run)),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build_postfit_run(args: argparse.Namespace, output_dir: Path) -> None:
    plain_run = resolve_existing(args.plain_run)
    full_refit_run = resolve_existing(args.full_refit_ewma_run)
    feature_path = resolve_existing(args.feature_path)
    plain_predictions = pd.read_csv(plain_run / "gp_window_predictions.csv", parse_dates=["date"])
    eval_noise = load_eval_noise(full_refit_run)
    predictions = inflate_predictions(plain_predictions, eval_noise)
    dates = sorted(pd.to_datetime(predictions["date"]).drop_duplicates())

    output_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(args, output_dir, dates)

    df = task_exp.load_features(feature_path)
    first_scenario_path = plain_run / f"gp_scenarios_{dates[0].date().isoformat()}.csv"
    final_universe = pd.read_csv(first_scenario_path, nrows=0).columns.tolist()
    rng = np.random.default_rng(args.seed)

    return_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    weight_frames: dict[str, list[pd.Series]] = {
        "gp_scenarios_riskfolio": [],
        "historical_y_ewma2_riskfolio": [],
        "equal_weight": [],
    }

    for window_date in dates:
        date_label = window_date.date().isoformat()
        print(f"post-fit inflation {date_label}", flush=True)
        plain_scenarios = pd.read_csv(plain_run / f"gp_scenarios_{date_label}.csv").reindex(columns=final_universe)
        noise_std = (
            predictions[predictions["date"].eq(window_date)]
            .set_index("asset_id")["ewma_noise_y_std"]
            .reindex(final_universe)
            .astype(float)
        )
        noise = rng.normal(loc=0.0, scale=noise_std.to_numpy(float), size=plain_scenarios.shape)
        scenarios = plain_scenarios.astype(float) + noise
        scenarios.to_csv(output_dir / f"gp_scenarios_{date_label}.csv", index=False)

        train_df = df[(df["date"] < window_date) & df[task_exp.TARGET_COL].notna()].copy()
        eval_df = df[df["date"] == window_date].copy()
        eval_returns = (
            eval_df.set_index(eval_df["asset_id"].astype(str))[task_exp.TARGET_COL]
            .reindex(final_universe)
            .astype(float)
        )
        gp_weights = portfolio_exp.optimize_riskfolio(
            scenarios,
            method_mu="hist",
            method_cov="hist",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        hist_panel = train_df.pivot(index="date", columns="asset_id", values=task_exp.TARGET_COL).reindex(
            columns=final_universe
        )
        hist_weights = portfolio_exp.optimize_riskfolio(
            hist_panel,
            method_mu="ewma2",
            method_cov="ewma2",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        ew_weights = portfolio_exp.equal_weight(final_universe)
        window_predictions = predictions[predictions["date"].eq(window_date)]
        gp_ic = portfolio_exp.information_coefficient(window_predictions, final_universe)
        ic_rows.append({"date": date_label, "strategy": "gp_scenarios_riskfolio", "ic": gp_ic})

        weights_by_strategy = {
            "gp_scenarios_riskfolio": gp_weights.reindex(final_universe).fillna(0.0),
            "historical_y_ewma2_riskfolio": hist_weights.reindex(final_universe).fillna(0.0),
            "equal_weight": ew_weights.reindex(final_universe).fillna(0.0),
        }
        for strategy, weights in weights_by_strategy.items():
            weight_frames[strategy].append(pd.Series(weights, name=window_date))
            return_rows.append(
                {
                    "date": date_label,
                    "strategy": strategy,
                    "return": portfolio_exp.realized_return(weights, eval_returns),
                    "gp_ic": gp_ic if strategy == "gp_scenarios_riskfolio" else math.nan,
                }
            )

    returns_df = pd.DataFrame(return_rows)
    ic_df = pd.DataFrame(ic_rows)
    weights_output: list[pd.DataFrame] = []
    strategy_returns: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, Any]] = []

    for strategy, rows in weight_frames.items():
        weights = pd.DataFrame(rows)
        weights.index = pd.to_datetime(weights.index)
        weights.index.name = "date"
        weights = weights.reindex(columns=final_universe).fillna(0.0)
        weights_long = weights.reset_index().melt(id_vars="date", var_name="asset_id", value_name="weight")
        weights_long["strategy"] = strategy
        weights_output.append(weights_long)
        strategy_ret = (
            returns_df[returns_df["strategy"].eq(strategy)]
            .assign(date=lambda frame: pd.to_datetime(frame["date"]))
            .set_index("date")["return"]
            .astype(float)
        )
        strategy_returns[strategy] = strategy_ret
        summary = {
            "strategy": strategy,
            **portfolio_exp.performance_stats(
                strategy_ret,
                weights,
                starting_value=portfolio_exp.STARTING_VALUE,
            ),
        }
        summary["mean_ic"] = float(ic_df["ic"].mean()) if strategy == "gp_scenarios_riskfolio" else math.nan
        summary["median_ic"] = float(ic_df["ic"].median()) if strategy == "gp_scenarios_riskfolio" else math.nan
        summary_rows.append(summary)

    pd.concat(weights_output, ignore_index=True).to_csv(output_dir / "portfolio_weights.csv", index=False)
    returns_df.to_csv(output_dir / "portfolio_returns.csv", index=False)
    predictions.to_csv(output_dir / "gp_window_predictions.csv", index=False)
    ic_df.to_csv(output_dir / "gp_window_ic.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "portfolio_summary.csv", index=False)
    eval_noise.assign(noise_role="eval").to_csv(output_dir / "noise_model_diagnostics.csv", index=False)
    portfolio_exp.plot_equity_and_drawdown(strategy_returns, output_dir)


def load_gp_summary(label: str, run_dir: Path) -> pd.Series:
    frame = pd.read_csv(run_dir / "portfolio_summary.csv")
    row = frame[frame["strategy"].eq("gp_scenarios_riskfolio")].iloc[0].copy()
    row["run_label"] = label
    return row


def write_stage_a_report(args: argparse.Namespace, output_dir: Path) -> None:
    plain_run = resolve_existing(args.plain_run)
    full_refit_run = resolve_existing(args.full_refit_ewma_run)
    runs = {
        "plain_gp": plain_run,
        "postfit_ewma_hl1_scale_050": output_dir,
        "full_refit_ewma_hl1_scale_050": full_refit_run,
    }
    summary = pd.DataFrame([load_gp_summary(label, run_dir) for label, run_dir in runs.items()])
    summary.to_csv(output_dir / "stage_a_portfolio_summary.csv", index=False)

    returns = []
    weights = []
    for label, run_dir in runs.items():
        ret = pd.read_csv(run_dir / "portfolio_returns.csv", parse_dates=["date"])
        ret = ret[ret["strategy"].eq("gp_scenarios_riskfolio")].copy()
        ret["run_label"] = label
        returns.append(ret)
        w = pd.read_csv(run_dir / "portfolio_weights.csv", parse_dates=["date"])
        w = w[w["strategy"].eq("gp_scenarios_riskfolio")].copy()
        w["run_label"] = label
        weights.append(w)
    returns_df = pd.concat(returns, ignore_index=True)
    weights_df = pd.concat(weights, ignore_index=True)
    returns_df.to_csv(output_dir / "stage_a_gp_returns.csv", index=False)
    weights_df.to_csv(output_dir / "stage_a_gp_weights.csv", index=False)

    base = returns_df[returns_df["run_label"].eq("plain_gp")][["date", "return"]].rename(
        columns={"return": "plain_return"}
    )
    paired_rows = []
    for label, group in returns_df[~returns_df["run_label"].eq("plain_gp")].groupby("run_label", observed=True):
        paired = group[["date", "return"]].merge(base, on="date", how="inner")
        delta = paired["return"].to_numpy(float) - paired["plain_return"].to_numpy(float)
        paired_rows.append(
            {
                "run_label": label,
                "n_windows": len(delta),
                "mean_monthly_delta": float(delta.mean()),
                "median_monthly_delta": float(np.median(delta)),
                "positive_delta_share": float((delta > 0).mean()),
                "sum_monthly_delta": float(delta.sum()),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "stage_a_paired_return_deltas.csv", index=False)

    visual_dir = output_dir / "visual_checks"
    visual_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, group in returns_df.groupby("run_label", observed=True):
        curve = (1.0 + group.sort_values("date")["return"].astype(float)).cumprod() * portfolio_exp.STARTING_VALUE
        ax.plot(group.sort_values("date")["date"], curve, marker="o", label=label)
    ax.set_title("Stage A GP Scenario Portfolio Equity")
    ax.set_ylabel("Portfolio value")
    ax.set_xlabel("Rebalance date")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(visual_dir / "stage_a_equity_curve.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = summary.sort_values("sharpe", ascending=False)
    ax.bar(ordered["run_label"], ordered["sharpe"], color=["#4c78a8", "#59a14f", "#f28e2b"])
    ax.set_title("Stage A Sharpe")
    ax.set_ylabel("Sharpe")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(visual_dir / "stage_a_sharpe.png", dpi=170)
    plt.close(fig)

    report = [
        "# Stage A EWMA Source-of-Benefit Diagnostic",
        "",
        f"Run directory: `{output_dir}`",
        "",
        "## Question",
        "",
        (
            "Does EWMA hl1/scale0.50 mainly help as post-fit scenario risk inflation, "
            "or because the fixed-noise refit changes the GP mean/latent uncertainty?"
        ),
        "",
        "## Compared Runs",
        "",
        "- `plain_gp`: existing 24-window signed multitask GP baseline.",
        "- `postfit_ewma_hl1_scale_050`: plain GP scenarios plus independent EWMA residual noise after fit.",
        "- `full_refit_ewma_hl1_scale_050`: existing fixed-noise EWMA refit.",
        "",
        "## GP Scenario Portfolio Summary",
        "",
        portfolio_exp.markdown_table(
            summary[
                [
                    "run_label",
                    "cumulative_return",
                    "cagr",
                    "annualized_vol",
                    "sharpe",
                    "max_drawdown",
                    "avg_turnover",
                    "mean_ic",
                ]
            ].sort_values("sharpe", ascending=False)
        ),
        "",
        "## Paired Return Deltas Versus Plain GP",
        "",
        portfolio_exp.markdown_table(paired),
        "",
        "## Visual Checks",
        "",
        "- `visual_checks/stage_a_equity_curve.png`: path comparison.",
        "- `visual_checks/stage_a_sharpe.png`: headline risk-adjusted comparison.",
        "- `equity_curve.png` and `drawdown_curve.png`: post-fit run internal strategy checks.",
        "",
        "## Caveats",
        "",
        "- The post-fit variant adds EWMA residual noise on top of the saved plain GP observation-posterior scenarios.",
        "- This is a scenario-only diagnostic; it does not produce new GP latent posterior samples.",
        "- No transaction costs, taxes, slippage, or liquidity filters are applied.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def run(args: argparse.Namespace) -> None:
    output_dir = resolve_output_dir(args)
    build_postfit_run(args, output_dir)
    write_stage_a_report(args, output_dir)
    summary = pd.read_csv(output_dir / "stage_a_portfolio_summary.csv")
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
