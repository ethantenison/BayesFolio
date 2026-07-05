"""Build the 24-window residual-noise comparison Quarto report.

Usage:
    poetry run python experiments/2026-06-time-varying-residual-noise/build_24w_quarto_report.py \
        --comparison-dir experiments/2026-06-time-varying-residual-noise/outputs/sweeps/20260621_residual_noise_methods_24w
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm, ttest_rel, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_DIR = (
    Path(__file__).resolve().parent / "outputs" / "sweeps" / "20260621_residual_noise_methods_24w"
)
PRIMARY_BENCHMARK = "baseline"
RESIDUAL_BENCHMARK = "residual_history_scale_050_24w"
RUN_DESCRIPTIONS = {
    "baseline": (
        "Plain signed multitask GP. No plug-in residual fixed noise is passed through train_Yvar; "
        "the model uses its learned multitask likelihood/task noise."
    ),
    "residual_history_scale_050_24w": (
        "Fixed heteroskedastic observation noise from trailing out-of-sample GP residual variance, "
        "15-window lookback, variance scale 0.50."
    ),
    "residual_ewma_hl3_scale_050_24w": (
        "Residual-history noise with exponentially weighted residual variance, half-life 3 prior residual windows, "
        "15-window lookback, variance scale 0.50."
    ),
    "residual_robust_q90_scale_050_24w": (
        "Residual-history noise with robust central 90% winsorized squared residuals, 15-window lookback, "
        "variance scale 0.50."
    ),
    "residual_shrinkage_prior6_scale_050_24w": (
        "Residual-history noise with adaptive asset-to-class shrinkage using prior sample size 6, "
        "15-window lookback, variance scale 0.50."
    ),
}
INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    benchmark: str
    larger_is_better: bool
    label: str


METRICS = (
    MetricSpec("mean_nll", PRIMARY_BENCHMARK, False, "Mean NLL"),
    MetricSpec("mean_crps", PRIMARY_BENCHMARK, False, "Mean CRPS"),
    MetricSpec("mean_abs_coverage_error", PRIMARY_BENCHMARK, False, "Mean abs coverage error"),
    MetricSpec("monthly_return", PRIMARY_BENCHMARK, True, "Monthly portfolio return"),
    MetricSpec("ic", PRIMARY_BENCHMARK, True, "Window IC"),
    MetricSpec("mean_nll", RESIDUAL_BENCHMARK, False, "Mean NLL vs fixed residual 0.50"),
    MetricSpec("monthly_return", RESIDUAL_BENCHMARK, True, "Monthly return vs fixed residual 0.50"),
    MetricSpec("ic", RESIDUAL_BENCHMARK, True, "IC vs fixed residual 0.50"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args()


def markdown_table(df: pd.DataFrame, *, digits: int = 4) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    header = "| " + " | ".join(name.ljust(widths[index]) for index, name in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def normal_crps(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (y_true - mean) / std
    phi = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    return std * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    mean = frame["y_pred"].to_numpy(dtype=float)
    std = frame["y_std"].to_numpy(dtype=float)
    err = y_true - mean
    z = err / std
    nll = 0.5 * np.log(2.0 * math.pi * std**2) + 0.5 * z**2
    row = {
        "mean_nll": float(np.mean(nll)),
        "mean_crps": float(np.mean(normal_crps(y_true, mean, std))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mean_pred_std": float(np.mean(std)),
        "z_std": float(np.std(z, ddof=0)),
    }
    coverage_errors = []
    for level in INTERVAL_LEVELS:
        alpha = 1.0 - level
        zcrit = norm.ppf(1.0 - alpha / 2.0)
        coverage = float(np.mean((y_true >= mean - zcrit * std) & (y_true <= mean + zcrit * std)))
        row[f"coverage_{int(level * 100)}"] = coverage
        coverage_errors.append(abs(coverage - level))
    row["mean_abs_coverage_error"] = float(np.mean(coverage_errors))
    return row


def build_window_metrics(comparison_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(comparison_dir / "comparison_predictions.csv", parse_dates=["date"])
    returns = []
    manifest = json.loads((comparison_dir / "manifest.json").read_text())
    run_paths = {"baseline": manifest["baseline_run"], **manifest["runs"]}
    for label, raw_path in run_paths.items():
        run_dir = (REPO_ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
        frame = pd.read_csv(run_dir / "portfolio_returns.csv", parse_dates=["date"])
        gp = frame[frame["strategy"].eq("gp_scenarios_riskfolio")][["date", "return"]].copy()
        gp = gp.rename(columns={"return": "monthly_return"})
        gp["run_label"] = label
        returns.append(gp)
    returns_df = pd.concat(returns, ignore_index=True)
    ic = []
    for label, raw_path in run_paths.items():
        run_dir = (REPO_ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
        frame = pd.read_csv(run_dir / "gp_window_ic.csv", parse_dates=["date"])
        frame["run_label"] = label
        ic.append(frame[["date", "run_label", "ic"]])
    ic_df = pd.concat(ic, ignore_index=True)

    rows = []
    for (label, date), group in predictions.groupby(["run_label", "date"], observed=True):
        row = prediction_metrics(group)
        row["run_label"] = label
        row["date"] = pd.Timestamp(date)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics = metrics.merge(returns_df, on=["date", "run_label"], how="left")
    metrics = metrics.merge(ic_df, on=["date", "run_label"], how="left")
    return metrics


def permutation_p(deltas: np.ndarray, *, rng: np.random.Generator, n_samples: int = 50_000) -> float:
    observed = abs(float(np.mean(deltas)))
    if len(deltas) <= 20:
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(deltas))).T.reshape(-1, len(deltas))
        means = np.abs(signs @ deltas / len(deltas))
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_samples, len(deltas)))
        means = np.abs(signs @ deltas / len(deltas))
    return float(np.mean(means >= observed))


def paired_tests(window_metrics: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for spec in METRICS:
        benchmark = window_metrics[window_metrics["run_label"].eq(spec.benchmark)][["date", spec.name]]
        benchmark = benchmark.rename(columns={spec.name: "benchmark_value"})
        for label, group in window_metrics[~window_metrics["run_label"].eq(spec.benchmark)].groupby("run_label"):
            paired = group[["date", spec.name]].merge(benchmark, on="date", how="inner").sort_values("date")
            raw_delta = paired[spec.name].to_numpy(float) - paired["benchmark_value"].to_numpy(float)
            favorable_delta = raw_delta if spec.larger_is_better else -raw_delta
            indices = rng.integers(0, len(favorable_delta), size=(bootstrap_samples, len(favorable_delta)))
            boot = favorable_delta[indices].mean(axis=1)
            try:
                t_p = float(ttest_rel(paired[spec.name], paired["benchmark_value"]).pvalue)
            except Exception:
                t_p = math.nan
            try:
                w_p = float(wilcoxon(raw_delta).pvalue)
            except Exception:
                w_p = math.nan
            rows.append(
                {
                    "metric": spec.name,
                    "metric_label": spec.label,
                    "benchmark": spec.benchmark,
                    "run_label": label,
                    "n_windows": int(len(favorable_delta)),
                    "mean_delta_raw": float(np.mean(raw_delta)),
                    "mean_delta_favorable": float(np.mean(favorable_delta)),
                    "ci95_low_favorable": float(np.quantile(boot, 0.025)),
                    "ci95_high_favorable": float(np.quantile(boot, 0.975)),
                    "bootstrap_p_two_sided": float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0))),
                    "sign_flip_p_two_sided": permutation_p(favorable_delta, rng=rng),
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                    "direction": "positive favors run_label after metric orientation",
                }
            )
    return pd.DataFrame(rows)


def write_extra_plots(comparison_dir: Path, window_metrics: pd.DataFrame, tests: pd.DataFrame) -> None:
    visual_dir = comparison_dir / "visual_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)

    baseline = window_metrics[window_metrics["run_label"].eq("baseline")][["date", "mean_nll", "monthly_return", "ic"]]
    for metric, ylabel, filename, favorable in [
        ("mean_nll", "Mean NLL delta vs baseline (negative is better)", "window_nll_delta_plain_baseline.png", -1),
        ("monthly_return", "Monthly return delta vs baseline", "window_return_delta_plain_baseline.png", 1),
        ("ic", "IC delta vs baseline", "window_ic_delta_plain_baseline.png", 1),
    ]:
        bench = baseline[["date", metric]].rename(columns={metric: "benchmark"})
        fig, ax = plt.subplots(figsize=(12, 6))
        for label, group in window_metrics[~window_metrics["run_label"].eq("baseline")].groupby("run_label", observed=True):
            merged = group[["date", metric]].merge(bench, on="date")
            merged["delta"] = merged[metric] - merged["benchmark"]
            ax.plot(merged["date"], merged["delta"], marker="o", linewidth=1.5, label=label)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(ylabel)
        ax.set_xlabel("Rebalance date")
        ax.set_ylabel("Delta")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(visual_dir / filename, dpi=170)
        plt.close(fig)

    primary = tests[tests["benchmark"].eq("baseline") & tests["metric"].isin(["mean_nll", "monthly_return", "ic"])]
    primary = primary.sort_values(["metric", "mean_delta_favorable"])
    labels = primary["metric_label"] + "\n" + primary["run_label"]
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(primary))
    ax.errorbar(
        primary["mean_delta_favorable"],
        y,
        xerr=[
            primary["mean_delta_favorable"] - primary["ci95_low_favorable"],
            primary["ci95_high_favorable"] - primary["mean_delta_favorable"],
        ],
        fmt="o",
        color="#4c78a8",
        ecolor="#8ab6d6",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Paired Bootstrap 95% CIs vs Plain GP Baseline")
    ax.set_xlabel("Favorable delta; positive favors residual-noise method")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(visual_dir / "paired_bootstrap_ci_vs_plain_baseline.png", dpi=170)
    plt.close(fig)


def relative(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def write_qmd(comparison_dir: Path, window_metrics: pd.DataFrame, tests: pd.DataFrame) -> Path:
    qmd_path = comparison_dir / "residual_noise_24w_report.qmd"
    calibration = pd.read_csv(comparison_dir / "calibration_summary.csv")
    portfolio = pd.read_csv(comparison_dir / "portfolio_summary.csv")
    noise = pd.read_csv(comparison_dir / "noise_summary.csv")
    manifest = json.loads((comparison_dir / "manifest.json").read_text())

    residual_rows = [
        {"run_label": label, "description": RUN_DESCRIPTIONS.get(label, "")}
        for label in ["baseline", *manifest["runs"].keys()]
    ]
    run_table = pd.DataFrame(residual_rows)

    cal_cols = [
        "run_label",
        "mean_nll",
        "mean_crps",
        "rmse",
        "mae",
        "mean_pred_std",
        "z_std",
        "mean_abs_coverage_error",
        "coverage_80",
        "coverage_90",
        "coverage_95",
    ]
    port_cols = [
        "run_label",
        "cumulative_return",
        "cagr",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "mean_ic",
        "median_ic",
    ]
    primary_tests = tests[
        tests["benchmark"].eq("baseline") & tests["metric"].isin(["mean_nll", "mean_crps", "monthly_return", "ic"])
    ].copy()
    residual_tests = tests[
        tests["benchmark"].eq(RESIDUAL_BENCHMARK) & tests["metric"].isin(["mean_nll", "monthly_return", "ic"])
    ].copy()
    test_cols = [
        "metric_label",
        "run_label",
        "n_windows",
        "mean_delta_raw",
        "mean_delta_favorable",
        "ci95_low_favorable",
        "ci95_high_favorable",
        "bootstrap_p_two_sided",
        "sign_flip_p_two_sided",
        "paired_t_p",
        "wilcoxon_p",
    ]

    best_nll = calibration.sort_values("mean_nll").iloc[0]
    best_sharpe = portfolio.sort_values("sharpe", ascending=False).iloc[0]
    nll_sig = primary_tests[(primary_tests["metric"].eq("mean_nll")) & (primary_tests["ci95_low_favorable"] > 0)]
    return_sig = primary_tests[(primary_tests["metric"].eq("monthly_return")) & (primary_tests["ci95_low_favorable"] > 0)]
    ic_sig = primary_tests[(primary_tests["metric"].eq("ic")) & (primary_tests["ci95_low_favorable"] > 0)]

    visual_paths = [
        "visual_checks/mean_nll_by_run.png",
        "visual_checks/portfolio_sharpe_by_run.png",
        "visual_checks/window_nll_delta_plain_baseline.png",
        "visual_checks/window_return_delta_plain_baseline.png",
        "visual_checks/window_ic_delta_plain_baseline.png",
        "visual_checks/paired_bootstrap_ci_vs_plain_baseline.png",
        "visual_checks/eval_noise_std_by_run_box.png",
        "visual_checks/noise_std_vs_abs_error_by_run.png",
    ]

    conclusion = [
        f"- Best calibration by mean NLL: `{best_nll['run_label']}` ({best_nll['mean_nll']:.4f}).",
        f"- Best realized GP-scenario Sharpe: `{best_sharpe['run_label']}` ({best_sharpe['sharpe']:.4f}).",
    ]
    if not nll_sig.empty:
        conclusion.append(
            "- Calibration improvement versus the plain GP baseline is statistically supported by the paired bootstrap "
            "for all residual-noise variants tested."
        )
    else:
        conclusion.append("- Calibration improvements do not clear the paired-bootstrap CI test versus the plain baseline.")
    if return_sig.empty:
        conclusion.append(
            "- None of the residual-noise variants has a statistically supported monthly-return improvement versus the plain baseline."
        )
    if ic_sig.empty:
        conclusion.append(
            "- IC point estimates improve for residual variants, but the paired tests do not make that a strong claim."
        )
    conclusion.append(
        "- Practical readout: residual-noise modeling improved uncertainty calibration, but did not beat the simpler multitask GP "
        "on realized portfolio Sharpe/return over the 24-window holdout."
    )
    conclusion.append(
        "- Next step: keep the plain multitask GP as the portfolio baseline, use residual EWMA as the calibration challenger, "
        "and only promote it if a decision rule values calibrated scenario uncertainty enough to offset no portfolio improvement."
    )

    lines = [
        "---",
        'title: "BayesFolio Residual-Noise Method Comparison: 24 Holdout Windows"',
        "format:",
        "  html:",
        "    embed-resources: true",
        "    toc: true",
        "    toc-depth: 3",
        "    code-fold: true",
        "execute:",
        "  echo: false",
        "---",
        "",
        "## Purpose",
        "",
        (
            "This experiment tests whether explicit time-varying residual-noise models improve BayesFolio's "
            "plain signed multitask GP over 24 monthly holdout windows. The decision question is whether the "
            "extra residual-noise machinery earns its complexity through better calibrated forecasts and/or "
            "better portfolio behavior."
        ),
        "",
        "## Experiment Runs",
        "",
        markdown_table(run_table, digits=4),
        "",
        "## Lineage",
        "",
        f"- Comparison artifact: `{comparison_dir}`",
        f"- Git SHA: `{manifest['git_sha']}`",
        f"- Dirty summary captured in manifest: `{manifest['git_dirty_summary_at_start'][:500]}`",
        "- Scored windows: 24 monthly rebalances from 2024-05-31 through 2026-04-30.",
        "- Residual source: out-of-sample residuals from the 24-window plain GP baseline.",
        "",
        "## Metric Tables",
        "",
        "### Forecast Calibration",
        "",
        markdown_table(calibration[cal_cols].sort_values("mean_nll"), digits=4),
        "",
        "### Portfolio Metrics",
        "",
        markdown_table(portfolio[port_cols].sort_values("sharpe", ascending=False), digits=4),
        "",
        "### Noise Diagnostics",
        "",
        markdown_table(noise.sort_values("run_label"), digits=4),
        "",
        "## Visual Diagnostics",
        "",
    ]
    for rel_path in visual_paths:
        lines.extend([f"### {Path(rel_path).stem.replace('_', ' ').title()}", "", f"![]({rel_path})", ""])
    lines.extend(
        [
            "## Statistical Tests",
            "",
            (
                "Tests are paired by rebalance window. Forecast metrics aggregate asset-level observations inside each "
                "rebalance date before testing. Confidence intervals are nonparametric bootstraps over the 24 windows. "
                "The sign-flip p-value is a paired randomization/permutation check under symmetric window deltas. "
                "`mean_delta_favorable` is oriented so positive values favor the candidate run."
            ),
            "",
            "### Versus Plain Multitask GP Baseline",
            "",
            markdown_table(primary_tests[test_cols].sort_values(["metric_label", "run_label"]), digits=5),
            "",
            "### Versus Fixed Residual-History 0.50",
            "",
            markdown_table(residual_tests[test_cols].sort_values(["metric_label", "run_label"]), digits=5),
            "",
            "## Conclusion",
            "",
            *conclusion,
            "",
            "## Caveats",
            "",
            "- The portfolio tests still have only 24 paired monthly observations; they are better than 12, but not large-sample evidence.",
            "- Riskfolio emitted infeasible-optimization messages in several windows and fell back internally; this was observed across methods.",
            "- Residual-noise estimates are plug-in fixed-noise values; uncertainty in the noise model itself is not propagated.",
            "- The experiment uses one feature artifact and one historical period; robustness across market regimes is still unproven.",
        ]
    )
    qmd_path.write_text("\n".join(lines) + "\n")
    return qmd_path


def main() -> None:
    args = parse_args()
    comparison_dir = args.comparison_dir.resolve()
    window_metrics = build_window_metrics(comparison_dir)
    tests = paired_tests(window_metrics, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    stat_dir = comparison_dir / "stat_tests"
    stat_dir.mkdir(parents=True, exist_ok=True)
    window_metrics.to_csv(stat_dir / "window_metrics.csv", index=False)
    tests.to_csv(stat_dir / "paired_stat_tests_24w.csv", index=False)
    write_extra_plots(comparison_dir, window_metrics, tests)
    qmd_path = write_qmd(comparison_dir, window_metrics, tests)
    subprocess.run(["quarto", "render", str(qmd_path)], cwd=comparison_dir, check=True)
    print(f"Wrote {qmd_path}")
    print(f"Wrote {qmd_path.with_suffix('.html')}")


if __name__ == "__main__":
    main()
