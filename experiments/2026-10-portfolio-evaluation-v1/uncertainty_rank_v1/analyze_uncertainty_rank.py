"""Compile paired score-method effects and render cached portfolio comparisons.

Usage: .venv/bin/python <this_file> after run_uncertainty_rank.py --execute.
Only new derived experiment outputs are writable. Returns/NAV changes are decimal;
no allocation, GP fitting, recalibration or source changes occur here.
"""

from __future__ import annotations

import io
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-expected-rank-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-expected-rank-cache")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import run_uncertainty_rank as study  # noqa: E402

LABELS = {
    "hard_rank": "Hard ranks\ncontrol",
    "expected_full": "Expected ranks\nfull covariance",
    "expected_diagonal": "Expected ranks\ndiagonal uncertainty",
    "reranked_expected_full": "Hard re-rank\nexpected scores",
}
COLORS = {
    "hard_rank": "#245bb2",
    "expected_full": "#218772",
    "expected_diagonal": "#b17825",
    "reranked_expected_full": "#a64a7f",
}


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
    context = study.base._context()
    results = []
    for row in rows:
        case = row["case"]
        control = next(
            r for r in rows if r["case"]["prediction"] == case["prediction"] and r["case"]["decision"] == "hard_rank"
        )
        with np.load(study.OUTPUT / "cases" / f"{row['id']}.npz", allow_pickle=False) as panel:
            family_weights = {
                name: float(panel["weights"][..., [context["assets"].index(a) for a in assets]].sum(axis=-1).mean())
                for name, assets in context["families"].items()
            }
        results.append(
            {
                "id": row["id"],
                "role": case["role"],
                "Sharpe_delta": row["conventional_sharpe"] - control["conventional_sharpe"],
                "return_delta": row["compounded_total_return"] - control["compounded_total_return"],
                "volatility_delta": row["annualized_excess_vol"] - control["annualized_excess_vol"],
                "decision_IC_delta": row["decision_IC"] - control["decision_IC"],
                "seed_Sharpe_deltas": (
                    np.array(row["seed_values_conventional_sharpe"]) - control["seed_values_conventional_sharpe"]
                ).tolist(),
                "family_weights": family_weights,
            }
        )
    return results


def _plot(rows: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for offset, prediction in enumerate(study.RECIPES):
        local = {r["case"]["decision"]: r for r in rows if r["case"]["prediction"] == prediction}
        title = "Prior Sharpe prediction recipe" if prediction == "sharpe" else "Prior ranking and Sharpe recipe"
        for column, metric in enumerate(["conventional_sharpe", "decision_IC"]):
            axis = axes[offset, column]
            values = [local[method][metric] for method in study.METHODS]
            bars = axis.bar(np.arange(4), values, color=[COLORS[method] for method in study.METHODS], width=0.65)
            bars[2].set_hatch("//")
            bars[3].set_hatch("//")
            axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4, fontsize=10)
            axis.set_xticks(np.arange(4), [LABELS[method] for method in study.METHODS], fontsize=8)
            axis.set_title(f"{title}\n{'Conventional Sharpe' if column == 0 else 'Allocation score ranking IC'}")
            axis.set_ylim(min(0, min(values) * 1.15), max(values) * 1.15)
            axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Uncertainty-aware ranking with the original allocator\n"
        "Fixed means and allocation covariance; hatched bars are diagnostics; three seeds on 24 reused months",
        fontsize=13,
    )
    _save_figure(figure, "expected_rank_comparison.png")

    context = study.base._context()
    dates = np.asarray(context["dates"], dtype="datetime64[M]")
    timeline = np.concatenate([dates[:1], dates + 1])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for axis, prediction in zip(axes, study.RECIPES, strict=True):
        for method in study.METHODS:
            with np.load(study.OUTPUT / "cases" / f"{prediction}__{method}.npz", allow_pickle=False) as panel:
                raw = np.sum(panel["weights"] * context["actual"], axis=-1)
                total = (1 + raw) * (1 + context["risk_free"]) - 1
                nav = np.concatenate([np.ones((3, 1)), np.cumprod(1 + total, axis=-1)], axis=-1)
            axis.plot(
                timeline,
                nav.mean(axis=0),
                color=COLORS[method],
                label=LABELS[method].replace("\n", " "),
                linestyle="--" if method in {"expected_diagonal", "reranked_expected_full"} else "-",
            )
        axis.set_title("Prior Sharpe recipe" if prediction == "sharpe" else "Prior ranking and Sharpe recipe")
        axis.set_xlabel("Return target month; initial capital shown before first target")
        axis.set_ylabel("Mean individual-seed NAV; starting capital = 1")
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Same forecasts and portfolio policy, different ranking signals; gross of costs", fontsize=13)
    _save_figure(figure, "expected_rank_wealth.png")


def _main() -> None:
    evidence = study._read(study.OUTPUT / "EVALUATED.json")
    study.base._verify(evidence["input_sha256"])
    study.base._verify(evidence["artifact_sha256"])
    rows = study._read(study.OUTPUT / "comparison.json")["summaries"]
    study._write(study.OUTPUT / "paired_comparisons.json", {"summaries": _paired(rows)})
    _plot(rows)
    print("Compiled paired score-method effects and rendered ranking and portfolio charts")


if __name__ == "__main__":
    _main()
