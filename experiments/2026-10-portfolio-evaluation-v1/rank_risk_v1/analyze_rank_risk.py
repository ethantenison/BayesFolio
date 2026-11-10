"""Render fixed-grid rank-risk tradeoffs and compile paired source comparisons.

Usage: .venv/bin/python <this_file> after run_rank_risk.py --execute.
Research-only derived outputs; no allocations, GP fits or calibration. Portfolio
returns and volatility are decimals; Sharpe and rank scores are dimensionless.
"""

from __future__ import annotations

import io
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-rank-risk-mpl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import run_rank_risk as study  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

LABELS = {
    "half": "50/50 GP covariance",
    "broad": "Broad GP covariance",
    "historical": "Historical covariance",
    "diagonal_half": "GP variances only",
}
COLORS = {"half": "#245bb2", "broad": "#268b73", "historical": "#b16a28", "diagonal_half": "#a34486"}


def _save_figure(figure: plt.Figure, filename: str) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    path = study.OUTPUT / filename
    payload = buffer.getvalue()
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Conflicting immutable figure")
    if not path.exists():
        path.write_bytes(payload)
    plt.close(figure)


def _paired(rows: list[dict]) -> list[dict]:
    paired = []
    context = study.base._context()
    for row in rows:
        case = row["case"]
        reference = next(
            r
            for r in rows
            if r["case"]["prediction"] == case["prediction"]
            and r["case"]["covariance"] == "half"
            and r["case"]["risk_aversion"] == 0
        )
        historical = next(
            r
            for r in rows
            if r["case"]["prediction"] == case["prediction"]
            and r["case"]["covariance"] == "historical"
            and r["case"]["risk_aversion"] == case["risk_aversion"]
        )
        with (
            np.load(study.OUTPUT / "cases" / f"{row['id']}.npz", allow_pickle=False) as panel,
            np.load(study.OUTPUT / "cases" / f"{historical['id']}.npz", allow_pickle=False) as history,
        ):
            difference = np.max(np.abs(panel["weights"] - history["weights"]), axis=2)
            family_weights = {
                name: float(panel["weights"][..., [context["assets"].index(a) for a in assets]].sum(axis=2).mean())
                for name, assets in context["families"].items()
            }
        paired.append(
            {
                "id": row["id"],
                "Sharpe_difference_vs_prior_half_control": row["conventional_sharpe"]
                - reference["conventional_sharpe"],
                "return_difference_vs_prior_half_control": row["compounded_total_return"]
                - reference["compounded_total_return"],
                "volatility_difference_vs_prior_half_control": row["annualized_excess_vol"]
                - reference["annualized_excess_vol"],
                "Sharpe_difference_vs_historical_same_lambda": row["conventional_sharpe"]
                - historical["conventional_sharpe"],
                "changed_months_vs_historical_same_lambda": int((difference > 1e-5).sum()),
                "maximum_weight_difference_vs_historical_same_lambda": float(difference.max()),
                "mean_family_weights": family_weights,
            }
        )
    return paired


def _plot(rows: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for offset, prediction in enumerate(study.RECIPES):
        for method in study.METHODS:
            local = sorted(
                (r for r in rows if r["case"]["prediction"] == prediction and r["case"]["covariance"] == method),
                key=lambda r: r["case"]["risk_aversion"],
            )
            lambdas = [row["case"]["risk_aversion"] for row in local]
            for column, metric in enumerate(["conventional_sharpe", "annualized_excess_vol"]):
                axes[offset, column].plot(
                    lambdas,
                    [row[metric] for row in local],
                    marker="o",
                    markersize=4,
                    color=COLORS[method],
                    label=LABELS[method],
                )
        title = "Prior Sharpe prediction recipe" if prediction == "sharpe" else "Prior ranking and Sharpe recipe"
        axes[offset, 0].set_title(f"{title}\nConventional Sharpe")
        axes[offset, 1].set_title(f"{title}\nRealized annualized volatility")
        axes[offset, 1].yaxis.set_major_formatter(PercentFormatter(1))
        for axis in axes[offset]:
            axis.set_xscale("symlog", linthresh=0.05)
            axis.set_xticks(study.RISK_AVERSIONS, [f"{value:g}" for value in study.RISK_AVERSIONS])
            axis.set_xlabel("Continuous risk penalty lambda; 0 = prior objective")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Rank and risk allocation: fixed forecasts, different risk tradeoffs\n"
        "Means of three seeds on 24 reused months; gross returns; no GP refits",
        fontsize=13,
    )
    _save_figure(figure, "rank_risk_tradeoffs.png")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    for axis, prediction in zip(axes, study.RECIPES, strict=True):
        for method in study.METHODS:
            local = [r for r in rows if r["case"]["prediction"] == prediction and r["case"]["covariance"] == method]
            axis.plot(
                [r["diagnostics"]["rank_advantage_retained_vs_own_zero"] for r in local],
                [r["conventional_sharpe"] for r in local],
                marker="o",
                markersize=4,
                color=COLORS[method],
                label=LABELS[method],
            )
            for row in local:
                if row["case"]["risk_aversion"] in {0, 0.25, 1, 2}:
                    axis.annotate(
                        f"{row['case']['risk_aversion']:g}",
                        (row["diagnostics"]["rank_advantage_retained_vs_own_zero"], row["conventional_sharpe"]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                        color=COLORS[method],
                    )
        axis.set_title("Prior Sharpe recipe" if prediction == "sharpe" else "Prior ranking and Sharpe recipe")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_xlabel("Allocation rank advantage retained vs own zero-penalty portfolio")
        axis.set_ylabel("Conventional Sharpe")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("How much ranking preference is sacrificed for risk? Point labels are lambda.", fontsize=13)
    _save_figure(figure, "rank_retention.png")


def _main() -> None:
    evidence = study._read(study.OUTPUT / "EVALUATED.json")
    study.base._verify(evidence["input_sha256"])
    study.base._verify(evidence["artifact_sha256"])
    result = study._read(study.OUTPUT / "comparison.json")
    rows = result["summaries"]
    study._write(study.OUTPUT / "paired_comparisons.json", {"summaries": _paired(rows)})
    _plot(rows)
    print("Compiled paired covariance effects, family exposures and rank-risk visualizations")


if __name__ == "__main__":
    _main()
