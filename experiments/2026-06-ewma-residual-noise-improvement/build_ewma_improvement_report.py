"""Build the EWMA residual-noise improvement Quarto report."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm, ttest_rel, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPARISON_DIR = EXPERIMENT_DIR / "outputs" / "sweeps" / "20260621_ewma_improvement_final"
CURRENT_EWMA = "ewma_hl3_scale_050"
PLAIN_BASELINE = "baseline"
FINAL_CHALLENGER = "ewma_hl1_scale_050"
CALIBRATION_WINNER = "ewma_hl2_scale_025"
INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def markdown_table(df: pd.DataFrame, *, digits: int = 4) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    headers = [str(column) for column in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    head = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def normal_crps(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (y_true - mean) / std
    phi = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    return std * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(float)
    mean = frame["y_pred"].to_numpy(float)
    std = frame["y_std"].to_numpy(float)
    err = y_true - mean
    z = err / std
    nll = 0.5 * np.log(2.0 * math.pi * std**2) + 0.5 * z**2
    row = {
        "mean_nll": float(np.mean(nll)),
        "mean_crps": float(np.mean(normal_crps(y_true, mean, std))),
        "rmse": float(np.sqrt(np.mean(err**2))),
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


def run_parts(label: str) -> tuple[float | None, float | None]:
    if not label.startswith("ewma_hl"):
        return None, None
    rest = label.removeprefix("ewma_hl")
    half_life_raw, scale_raw = rest.split("_scale_")
    return float(half_life_raw), float(scale_raw) / 100.0


def resolve_run_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def build_window_metrics(comparison_dir: Path) -> pd.DataFrame:
    manifest = json.loads((comparison_dir / "manifest.json").read_text())
    run_paths = {PLAIN_BASELINE: manifest["baseline_run"], **manifest["runs"]}
    predictions = pd.read_csv(comparison_dir / "comparison_predictions.csv", parse_dates=["date"])
    rows = []
    for (label, date), group in predictions.groupby(["run_label", "date"], observed=True):
        row = prediction_metrics(group)
        row["run_label"] = label
        row["date"] = pd.Timestamp(date)
        rows.append(row)
    metrics = pd.DataFrame(rows)

    returns = []
    ics = []
    for label, raw_path in run_paths.items():
        run_dir = resolve_run_dir(raw_path)
        ret = pd.read_csv(run_dir / "portfolio_returns.csv", parse_dates=["date"])
        ret = ret[ret["strategy"].eq("gp_scenarios_riskfolio")][["date", "return"]].rename(
            columns={"return": "monthly_return"}
        )
        ret["run_label"] = label
        returns.append(ret)
        ic = pd.read_csv(run_dir / "gp_window_ic.csv", parse_dates=["date"])
        ic["run_label"] = label
        ics.append(ic[["date", "run_label", "ic"]])
    metrics = metrics.merge(pd.concat(returns), on=["date", "run_label"], how="left")
    metrics = metrics.merge(pd.concat(ics), on=["date", "run_label"], how="left")
    return metrics


def sign_flip_p(deltas: np.ndarray, *, rng: np.random.Generator, samples: int = 50_000) -> float:
    observed = abs(float(deltas.mean()))
    if len(deltas) <= 18:
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(deltas))).T.reshape(-1, len(deltas))
    else:
        signs = rng.choice([-1.0, 1.0], size=(samples, len(deltas)))
    means = np.abs(signs @ deltas / len(deltas))
    return float(np.mean(means >= observed))


def paired_tests(window_metrics: pd.DataFrame, *, seed: int, bootstrap_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    specs = [
        ("mean_nll", False, PLAIN_BASELINE),
        ("mean_crps", False, PLAIN_BASELINE),
        ("mean_abs_coverage_error", False, PLAIN_BASELINE),
        ("monthly_return", True, PLAIN_BASELINE),
        ("ic", True, PLAIN_BASELINE),
        ("mean_nll", False, CURRENT_EWMA),
        ("monthly_return", True, CURRENT_EWMA),
        ("ic", True, CURRENT_EWMA),
    ]
    for metric, larger_is_better, benchmark_label in specs:
        benchmark = window_metrics[window_metrics["run_label"].eq(benchmark_label)][["date", metric]].rename(
            columns={metric: "benchmark"}
        )
        for label, group in window_metrics[~window_metrics["run_label"].eq(benchmark_label)].groupby("run_label"):
            paired = group[["date", metric]].merge(benchmark, on="date", how="inner")
            raw_delta = paired[metric].to_numpy(float) - paired["benchmark"].to_numpy(float)
            favorable_delta = raw_delta if larger_is_better else -raw_delta
            boot = favorable_delta[
                rng.integers(0, len(favorable_delta), size=(bootstrap_samples, len(favorable_delta)))
            ].mean(axis=1)
            try:
                paired_t_p = float(ttest_rel(paired[metric], paired["benchmark"]).pvalue)
            except Exception:
                paired_t_p = math.nan
            try:
                wilcoxon_p = float(wilcoxon(raw_delta).pvalue)
            except Exception:
                wilcoxon_p = math.nan
            rows.append(
                {
                    "metric": metric,
                    "benchmark": benchmark_label,
                    "run_label": label,
                    "n_windows": int(len(favorable_delta)),
                    "mean_delta_raw": float(raw_delta.mean()),
                    "mean_delta_favorable": float(favorable_delta.mean()),
                    "ci95_low_favorable": float(np.quantile(boot, 0.025)),
                    "ci95_high_favorable": float(np.quantile(boot, 0.975)),
                    "bootstrap_p_two_sided": float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0))),
                    "sign_flip_p_two_sided": sign_flip_p(favorable_delta, rng=rng),
                    "paired_t_p": paired_t_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )
    return pd.DataFrame(rows)


def write_plots(comparison_dir: Path, calibration: pd.DataFrame, portfolio: pd.DataFrame, tests: pd.DataFrame) -> None:
    visual_dir = comparison_dir / "visual_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)
    ewma = calibration[calibration["run_label"].str.startswith("ewma_")].copy()
    ewma[["half_life", "variance_scale"]] = ewma["run_label"].apply(lambda x: pd.Series(run_parts(x)))
    port = portfolio[portfolio["run_label"].str.startswith("ewma_")].copy()
    port[["half_life", "variance_scale"]] = port["run_label"].apply(lambda x: pd.Series(run_parts(x)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, data, ylabel, filename in [
        ("mean_nll", ewma[ewma["variance_scale"].eq(0.50)], "Mean NLL", "half_life_mean_nll.png"),
        ("sharpe", port[port["variance_scale"].eq(0.50)], "Sharpe", "half_life_sharpe.png"),
    ]:
        ax.clear()
        ordered = data.sort_values("half_life")
        ax.plot(ordered["half_life"], ordered[metric], marker="o")
        ax.set_xlabel("EWMA half-life")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by half-life at variance scale 0.50")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(visual_dir / filename, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for half_life, group in ewma[ewma["half_life"].isin([1.0, 2.0])].groupby("half_life"):
        ordered = group.sort_values("variance_scale")
        axes[0].plot(ordered["variance_scale"], ordered["mean_nll"], marker="o", label=f"hl {half_life:g}")
    axes[0].set_title("Mean NLL by variance scale")
    axes[0].set_xlabel("Variance scale")
    axes[0].set_ylabel("Mean NLL")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for half_life, group in port[port["half_life"].isin([1.0, 2.0])].groupby("half_life"):
        ordered = group.sort_values("variance_scale")
        axes[1].plot(ordered["variance_scale"], ordered["sharpe"], marker="o", label=f"hl {half_life:g}")
    axes[1].set_title("Sharpe by variance scale")
    axes[1].set_xlabel("Variance scale")
    axes[1].set_ylabel("Sharpe")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(visual_dir / "scale_sweep_nll_sharpe.png", dpi=170)
    plt.close(fig)

    primary = tests[
        tests["benchmark"].eq(PLAIN_BASELINE)
        & tests["metric"].isin(["mean_nll", "monthly_return", "ic"])
        & tests["run_label"].isin([FINAL_CHALLENGER, CALIBRATION_WINNER, CURRENT_EWMA])
    ].copy()
    primary["label"] = primary["metric"] + "\n" + primary["run_label"]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(primary))
    ax.errorbar(
        primary["mean_delta_favorable"],
        y,
        xerr=[
            primary["mean_delta_favorable"] - primary["ci95_low_favorable"],
            primary["ci95_high_favorable"] - primary["mean_delta_favorable"],
        ],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(primary["label"], fontsize=8)
    ax.set_xlabel("Favorable paired delta; positive favors candidate")
    ax.set_title("Paired bootstrap CIs vs plain GP baseline")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(visual_dir / "paired_ci_key_candidates.png", dpi=170)
    plt.close(fig)


def write_qmd(comparison_dir: Path, tests: pd.DataFrame) -> Path:
    qmd = comparison_dir / "ewma_improvement_report.qmd"
    calibration = pd.read_csv(comparison_dir / "calibration_summary.csv")
    portfolio = pd.read_csv(comparison_dir / "portfolio_summary.csv")
    noise = pd.read_csv(comparison_dir / "noise_summary.csv")
    manifest = json.loads((comparison_dir / "manifest.json").read_text())

    run_notes = pd.DataFrame(
        [
            {
                "variant": "Half-life sweep",
                "values": "1, 2, 3, 6, 9, 12 at scale 0.50",
                "why": "Tests whether monthly ETF residual noise should react quickly to recent forecast errors or smooth over longer residual history.",
            },
            {
                "variant": "Variance scale sweep",
                "values": "0.25, 0.50, 0.75, 1.00 for hl1; 0.25, 0.50, 0.75 for hl2",
                "why": "Separates the smoothing speed from the total amount of fixed observation noise injected into the GP.",
            },
            {
                "variant": "Fixed residual benchmark",
                "values": "residual history scale 0.50",
                "why": "Checks whether EWMA adds value over a simpler rolling residual variance estimator.",
            },
        ]
    )
    cal_cols = [
        "run_label",
        "mean_nll",
        "mean_crps",
        "rmse",
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
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "mean_ic",
        "median_ic",
    ]
    test_cols = [
        "metric",
        "benchmark",
        "run_label",
        "mean_delta_raw",
        "mean_delta_favorable",
        "ci95_low_favorable",
        "ci95_high_favorable",
        "bootstrap_p_two_sided",
        "sign_flip_p_two_sided",
    ]
    key_tests = tests[
        tests["run_label"].isin([FINAL_CHALLENGER, CALIBRATION_WINNER, CURRENT_EWMA])
        & tests["metric"].isin(["mean_nll", "monthly_return", "ic"])
    ]
    best_nll = calibration.sort_values("mean_nll").iloc[0]
    best_sharpe = portfolio.sort_values("sharpe", ascending=False).iloc[0]
    current = calibration[calibration["run_label"].eq(CURRENT_EWMA)].iloc[0]
    challenger = calibration[calibration["run_label"].eq(FINAL_CHALLENGER)].iloc[0]
    lines = [
        "---",
        'title: "BayesFolio EWMA Residual-Noise Improvement Experiment"',
        "format:",
        "  html:",
        "    embed-resources: true",
        "    toc: true",
        "    toc-depth: 3",
        "execute:",
        "  echo: false",
        "---",
        "",
        "## Purpose",
        "",
        (
            "This experiment varies the residual EWMA noise method to see whether the previous best calibration "
            "candidate can be improved without giving up portfolio behavior versus the plain multitask GP."
        ),
        "",
        "## What Varied And Why",
        "",
        markdown_table(run_notes),
        "",
        "## Lineage",
        "",
        f"- Comparison artifact: `{comparison_dir}`",
        f"- Git SHA: `{manifest['git_sha']}`",
        "- Holdout: 24 monthly windows from 2024-05-31 through 2026-04-30.",
        "- Residual source: out-of-sample residuals from the 24-window plain multitask GP baseline.",
        "",
        "## Forecast Calibration Metrics",
        "",
        markdown_table(calibration[cal_cols].sort_values("mean_nll"), digits=4),
        "",
        "## Portfolio Metrics",
        "",
        markdown_table(portfolio[port_cols].sort_values("sharpe", ascending=False), digits=4),
        "",
        "## Noise Diagnostics",
        "",
        markdown_table(noise.sort_values("run_label"), digits=4),
        "",
        "## Visual Diagnostics",
        "",
        "### Half-Life Mean NLL",
        "",
        "![](visual_checks/half_life_mean_nll.png)",
        "",
        "### Half-Life Sharpe",
        "",
        "![](visual_checks/half_life_sharpe.png)",
        "",
        "### Scale Sweep",
        "",
        "![](visual_checks/scale_sweep_nll_sharpe.png)",
        "",
        "### Key Paired CIs",
        "",
        "![](visual_checks/paired_ci_key_candidates.png)",
        "",
        "### Existing Comparison Plots",
        "",
        "![](visual_checks/mean_nll_by_run.png)",
        "",
        "![](visual_checks/portfolio_sharpe_by_run.png)",
        "",
        "![](visual_checks/window_mean_nll_delta_vs_baseline.png)",
        "",
        "## Statistical Tests",
        "",
        (
            "Tests are paired by rebalance window. For NLL/CRPS/coverage error, negative raw deltas are favorable, "
            "so `mean_delta_favorable` flips the sign. For return and IC, positive raw deltas are favorable."
        ),
        "",
        markdown_table(key_tests[test_cols].sort_values(["benchmark", "metric", "run_label"]), digits=5),
        "",
        "## Conclusion",
        "",
        f"- Best calibration by mean NLL: `{best_nll['run_label']}` at `{best_nll['mean_nll']:.4f}`.",
        f"- Best realized Sharpe: `{best_sharpe['run_label']}` at `{best_sharpe['sharpe']:.4f}`.",
        (
            f"- The practical EWMA challenger is `{FINAL_CHALLENGER}`: it improves mean NLL from "
            f"`{current['mean_nll']:.4f}` to `{challenger['mean_nll']:.4f}` versus current hl3/scale0.50, "
            "and it has the best EWMA Sharpe."
        ),
        (
            f"- `{CALIBRATION_WINNER}` has the best mean NLL, but it gives up some Sharpe versus "
            f"`{FINAL_CHALLENGER}`, so it is a calibration-only challenger rather than the balanced recommendation."
        ),
        "- The plain multitask GP remains very competitive on cumulative return, but no longer has the best Sharpe in this EWMA sweep.",
        "- Recommendation: promote `ewma_hl1_scale_050` as the next EWMA candidate to test against plain GP in future portfolio workflows; do not expand to a more complex noise model yet.",
        "",
        "## Caveats",
        "",
        "- These are still 24 paired monthly windows; useful, but not large-sample evidence.",
        "- Multiple variants were screened, so the best candidate needs a future confirmation run before being treated as settled.",
        "- Riskfolio emitted known infeasibility/fallback warnings across runs; one scale-1.00 run also emitted a solver accuracy warning.",
        "- The EWMA noise estimates remain plug-in fixed-noise values, not a fully joint noise posterior.",
    ]
    qmd.write_text("\n".join(lines) + "\n")
    return qmd


def main() -> None:
    args = parse_args()
    comparison_dir = args.comparison_dir.resolve()
    window_metrics = build_window_metrics(comparison_dir)
    tests = paired_tests(window_metrics, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    stat_dir = comparison_dir / "stat_tests"
    stat_dir.mkdir(parents=True, exist_ok=True)
    window_metrics.to_csv(stat_dir / "window_metrics.csv", index=False)
    tests.to_csv(stat_dir / "paired_stat_tests.csv", index=False)
    calibration = pd.read_csv(comparison_dir / "calibration_summary.csv")
    portfolio = pd.read_csv(comparison_dir / "portfolio_summary.csv")
    write_plots(comparison_dir, calibration, portfolio, tests)
    qmd = write_qmd(comparison_dir, tests)
    subprocess.run(["quarto", "render", str(qmd)], cwd=comparison_dir, check=True)
    print(qmd)
    print(qmd.with_suffix(".html"))


if __name__ == "__main__":
    main()
