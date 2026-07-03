"""Build the GP vs EWMA portfolio backtest report."""

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
from scipy.stats import ttest_rel, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "reports" / "20260621_gp_vs_ewma_24w"
PLAIN_RUN = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "outputs"
    / "runs"
    / "20260621_signed_lkj_eta_2_plain_24w"
)
EWMA_RUN = (
    REPO_ROOT
    / "experiments"
    / "2026-06-ewma-residual-noise-improvement"
    / "outputs"
    / "runs"
    / "20260621_ewma_hl1_scale_050_24w"
)
STARTING_VALUE = 10_000.0
PERIODS_PER_YEAR = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plain-run", type=Path, default=PLAIN_RUN)
    parser.add_argument("--ewma-run", type=Path, default=EWMA_RUN)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def git_dirty_summary() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()


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
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body])


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text())


def load_strategy_returns(run_dir: Path, *, run_label: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "portfolio_returns.csv", parse_dates=["date"])
    rows = []
    mapping = {
        "gp_scenarios_riskfolio": run_label,
        "historical_y_ewma2_riskfolio": "historical_riskfolio_ewma2",
        "equal_weight": "equal_weight",
    }
    for raw_strategy, strategy_label in mapping.items():
        if strategy_label in {"historical_riskfolio_ewma2", "equal_weight"} and run_label != "plain_gp_scenarios":
            continue
        part = frame[frame["strategy"].eq(raw_strategy)][["date", "return"]].copy()
        part["strategy"] = strategy_label
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def load_weights(run_dir: Path, *, run_label: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "portfolio_weights.csv", parse_dates=["date"])
    mapping = {
        "gp_scenarios_riskfolio": run_label,
        "historical_y_ewma2_riskfolio": "historical_riskfolio_ewma2",
        "equal_weight": "equal_weight",
    }
    rows = []
    for raw_strategy, strategy_label in mapping.items():
        if strategy_label in {"historical_riskfolio_ewma2", "equal_weight"} and run_label != "plain_gp_scenarios":
            continue
        part = frame[frame["strategy"].eq(raw_strategy)].copy()
        part["strategy"] = strategy_label
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def drawdown(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def performance_stats(group: pd.DataFrame, weights: pd.DataFrame) -> dict[str, float]:
    returns = group.sort_values("date")["return"].astype(float)
    equity = (1.0 + returns).cumprod()
    cagr = float(equity.iloc[-1] ** (PERIODS_PER_YEAR / len(returns)) - 1.0)
    vol = float(returns.std(ddof=0) * math.sqrt(PERIODS_PER_YEAR))
    sharpe = float(cagr / vol) if vol > 0 else math.nan
    strat_weights = weights[weights["strategy"].eq(group["strategy"].iloc[0])].sort_values(["date", "asset_id"])
    wide = strat_weights.pivot(index="date", columns="asset_id", values="weight").fillna(0.0)
    turnover = wide.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = wide.iloc[0].abs().sum()
    return {
        "n_rebalances": float(len(returns)),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(returns),
        "terminal_value": float(STARTING_VALUE * equity.iloc[-1]),
        "mean_monthly_return": float(returns.mean()),
        "hit_rate": float((returns > 0).mean()),
        "avg_turnover": float(turnover.mean()),
        "max_weight": float(wide.max(axis=1).max()),
    }


def build_tables(plain_run: Path, ewma_run: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = pd.concat(
        [
            load_strategy_returns(plain_run, run_label="plain_gp_scenarios"),
            load_strategy_returns(ewma_run, run_label="ewma_hl1_scale_050_gp_scenarios"),
        ],
        ignore_index=True,
    )
    weights = pd.concat(
        [
            load_weights(plain_run, run_label="plain_gp_scenarios"),
            load_weights(ewma_run, run_label="ewma_hl1_scale_050_gp_scenarios"),
        ],
        ignore_index=True,
    )
    plain_summary = pd.read_csv(plain_run / "portfolio_summary.csv")
    ewma_summary = pd.read_csv(ewma_run / "portfolio_summary.csv")
    summary_rows = []
    source_rows = [
        (ewma_summary, "gp_scenarios_riskfolio", "ewma_hl1_scale_050_gp_scenarios"),
        (plain_summary, "gp_scenarios_riskfolio", "plain_gp_scenarios"),
        (plain_summary, "historical_y_ewma2_riskfolio", "historical_riskfolio_ewma2"),
        (plain_summary, "equal_weight", "equal_weight"),
    ]
    for frame, source_strategy, strategy_label in source_rows:
        row = frame[frame["strategy"].eq(source_strategy)].iloc[0].to_dict()
        row["strategy"] = strategy_label
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)
    returns.to_csv(output_dir / "strategy_returns.csv", index=False)
    weights.to_csv(output_dir / "strategy_weights.csv", index=False)
    summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    return returns, weights, summary


def paired_tests(returns: pd.DataFrame, *, seed: int, bootstrap_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    benchmark = returns[returns["strategy"].eq("plain_gp_scenarios")][["date", "return"]].rename(
        columns={"return": "benchmark_return"}
    )
    rows = []
    for strategy, group in returns[~returns["strategy"].eq("plain_gp_scenarios")].groupby("strategy", observed=True):
        paired = group[["date", "return"]].merge(benchmark, on="date", how="inner").sort_values("date")
        delta = paired["return"].to_numpy(float) - paired["benchmark_return"].to_numpy(float)
        boot = delta[rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))].mean(axis=1)
        signs = rng.choice([-1.0, 1.0], size=(50_000, len(delta)))
        sign_means = np.abs(signs @ delta / len(delta))
        observed = abs(float(delta.mean()))
        try:
            paired_t_p = float(ttest_rel(paired["return"], paired["benchmark_return"]).pvalue)
        except Exception:
            paired_t_p = math.nan
        try:
            wilcoxon_p = float(wilcoxon(delta).pvalue)
        except Exception:
            wilcoxon_p = math.nan
        rows.append(
            {
                "strategy": strategy,
                "benchmark": "plain_gp_scenarios",
                "n_windows": len(delta),
                "mean_monthly_return_delta": float(delta.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "bootstrap_p_two_sided": float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0))),
                "sign_flip_p_two_sided": float(np.mean(sign_means >= observed)),
                "paired_t_p": paired_t_p,
                "wilcoxon_p": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows)


def write_plots(returns: pd.DataFrame, weights: pd.DataFrame, output_dir: Path) -> None:
    visual_dir = output_dir / "visual_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)
    ordered = [
        "plain_gp_scenarios",
        "ewma_hl1_scale_050_gp_scenarios",
        "historical_riskfolio_ewma2",
        "equal_weight",
    ]
    colors = {
        "plain_gp_scenarios": "#4c78a8",
        "ewma_hl1_scale_050_gp_scenarios": "#59a14f",
        "historical_riskfolio_ewma2": "#f28e2b",
        "equal_weight": "#9c755f",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy in ordered:
        group = returns[returns["strategy"].eq(strategy)].sort_values("date")
        equity = STARTING_VALUE * (1.0 + group["return"]).cumprod()
        ax.plot(group["date"], equity, label=strategy, color=colors[strategy], linewidth=2)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(visual_dir / "equity_curve.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy in ordered:
        group = returns[returns["strategy"].eq(strategy)].sort_values("date")
        ax.plot(group["date"], drawdown(group["return"]), label=strategy, color=colors[strategy], linewidth=2)
    ax.set_title("Drawdown Curve")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(visual_dir / "drawdown_curve.png", dpi=170)
    plt.close(fig)

    summary = pd.read_csv(output_dir / "portfolio_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    summary.sort_values("sharpe", ascending=False).plot.bar(x="strategy", y="sharpe", ax=axes[0], legend=False)
    axes[0].set_title("Sharpe")
    axes[0].tick_params(axis="x", rotation=35)
    summary.sort_values("cumulative_return", ascending=False).plot.bar(
        x="strategy", y="cumulative_return", ax=axes[1], legend=False, color="#59a14f"
    )
    axes[1].set_title("Cumulative Return")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(visual_dir / "summary_bars.png", dpi=170)
    plt.close(fig)

    gp_weights = weights[weights["strategy"].isin(["plain_gp_scenarios", "ewma_hl1_scale_050_gp_scenarios"])]
    avg_weights = gp_weights.groupby(["strategy", "asset_id"], observed=True)["weight"].mean().reset_index()
    pivot = avg_weights.pivot(index="asset_id", columns="strategy", values="weight").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.28)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Average GP Scenario Weights")
    fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(visual_dir / "average_gp_weights.png", dpi=170)
    plt.close(fig)


def write_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    manifest = {
        "schema": "bayesfolio.gp_ewma_portfolio_backtest.report.v1",
        "git_sha": git_sha(),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "plain_run": str(args.plain_run),
        "ewma_run": str(args.ewma_run),
        "output_dir": str(output_dir),
        "holdout_windows": "24 monthly windows, 2024-05-31 through 2026-04-30",
        "riskfolio_gp": {
            "model": "Classic",
            "rm": "CVaR",
            "obj": "Sharpe",
            "method_mu": "hist",
            "method_cov": "hist",
            "hist": True,
            "upperlng": 0.20,
            "nea": 10,
            "alpha": 0.5,
        },
        "riskfolio_historical": {
            "model": "Classic",
            "rm": "CVaR",
            "obj": "Sharpe",
            "method_mu": "ewma2",
            "method_cov": "ewma2",
            "hist": True,
            "upperlng": 0.20,
            "nea": 10,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_qmd(output_dir: Path, summary: pd.DataFrame, tests: pd.DataFrame, plain_run: Path, ewma_run: Path) -> Path:
    qmd = output_dir / "gp_vs_ewma_portfolio_backtest.qmd"
    summary_cols = [
        "strategy",
        "cumulative_return",
        "cagr",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "mean_monthly_return",
        "hit_rate",
        "avg_turnover",
        "max_weight",
    ]
    test_cols = [
        "strategy",
        "benchmark",
        "mean_monthly_return_delta",
        "ci95_low",
        "ci95_high",
        "bootstrap_p_two_sided",
        "sign_flip_p_two_sided",
        "paired_t_p",
        "wilcoxon_p",
    ]
    best_sharpe = summary.sort_values("sharpe", ascending=False).iloc[0]
    best_return = summary.sort_values("cumulative_return", ascending=False).iloc[0]
    lines = [
        "---",
        'title: "BayesFolio GP vs EWMA Portfolio Backtest"',
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
            "This report compares the plain signed multitask GP portfolio against the best EWMA residual-noise "
            "GP portfolio over the same 24 monthly holdout windows. Both GP policies use posterior scenarios "
            "fed into Riskfolio. The report also includes the historical Riskfolio baseline and equal weight."
        ),
        "",
        "## Strategy Definitions",
        "",
        "- `plain_gp_scenarios`: plain signed multitask GP scenarios + Riskfolio.",
        "- `ewma_hl1_scale_050_gp_scenarios`: EWMA residual-noise GP with half-life 1 and variance scale 0.50 + Riskfolio.",
        "- `historical_riskfolio_ewma2`: historical excess-return Riskfolio baseline.",
        "- `equal_weight`: monthly equal-weight baseline.",
        "",
        "## Riskfolio Configuration",
        "",
        "- GP scenario optimization: `Classic`, `CVaR`, `Sharpe`, `method_mu=hist`, `method_cov=hist`, `upperlng=0.20`, `nea=10`, `alpha=0.5`.",
        "- Historical optimization: `Classic`, `CVaR`, `Sharpe`, `method_mu=ewma2`, `method_cov=ewma2`, `upperlng=0.20`, `nea=10`.",
        "",
        "## Lineage",
        "",
        f"- Plain GP run: `{plain_run}`",
        f"- EWMA GP run: `{ewma_run}`",
        f"- Report output: `{output_dir}`",
        "- Holdout: 24 monthly rebalances from 2024-05-31 through 2026-04-30.",
        "- No transaction costs, taxes, slippage, liquidity filters, or execution delay are applied.",
        "",
        "## Metrics",
        "",
        markdown_table(summary[summary_cols], digits=4),
        "",
        "## Visual Diagnostics",
        "",
        "### Equity Curve",
        "",
        "![](visual_checks/equity_curve.png)",
        "",
        "### Drawdown Curve",
        "",
        "![](visual_checks/drawdown_curve.png)",
        "",
        "### Summary Bars",
        "",
        "![](visual_checks/summary_bars.png)",
        "",
        "### Average GP Scenario Weights",
        "",
        "![](visual_checks/average_gp_weights.png)",
        "",
        "## Paired Monthly Return Tests",
        "",
        (
            "Tests are paired by rebalance month versus the plain GP scenario strategy. Positive deltas favor "
            "the strategy named in the row."
        ),
        "",
        markdown_table(tests[test_cols].sort_values("mean_monthly_return_delta", ascending=False), digits=5),
        "",
        "## Conclusion",
        "",
        f"- Best Sharpe: `{best_sharpe['strategy']}` at `{best_sharpe['sharpe']:.4f}`.",
        f"- Best cumulative return: `{best_return['strategy']}` at `{best_return['cumulative_return']:.4f}`.",
        (
            "- The EWMA GP scenario strategy has the best risk-adjusted performance, but the plain GP scenario "
            "strategy still has the highest cumulative return."
        ),
        (
            "- The historical Riskfolio and equal-weight baselines are competitive but trail both GP scenario "
            "policies on Sharpe in this 24-window comparison."
        ),
        (
            "- Paired monthly return tests should be treated as uncertainty checks, not proof; 24 windows is still "
            "a small sample and the return deltas are modest."
        ),
        "",
        "## Caveats",
        "",
        "- No transaction costs or slippage are applied, so turnover differences are not penalized.",
        "- Riskfolio solver fallback/infeasibility warnings were observed in earlier source runs and remain a reporting caveat.",
        "- The EWMA residual-noise estimates are plug-in fixed-noise values.",
        "- The best EWMA setting was selected from a parameter sweep; confirm it on a future holdout before treating it as settled.",
    ]
    qmd.write_text("\n".join(lines) + "\n")
    return qmd


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    load_manifest(args.plain_run)
    load_manifest(args.ewma_run)
    returns, weights, summary = build_tables(args.plain_run, args.ewma_run, output_dir)
    tests = paired_tests(returns, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    tests.to_csv(output_dir / "paired_return_tests.csv", index=False)
    write_plots(returns, weights, output_dir)
    write_manifest(args, output_dir)
    qmd = write_qmd(output_dir, summary, tests, args.plain_run, args.ewma_run)
    subprocess.run(["quarto", "render", str(qmd)], cwd=output_dir, check=True)
    print(qmd)
    print(qmd.with_suffix(".html"))


if __name__ == "__main__":
    main()
