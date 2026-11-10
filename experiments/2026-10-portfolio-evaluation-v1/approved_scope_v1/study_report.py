"""Batch tables, raw-weight exports and unsmoothed US forecast plots.

Usage: called after all fits and portfolio cases by run_study.py. Research-only
reporting writes outputs/; it changes no fitted models or selection policy.
Returns are discounted-excess decimals, performance is gross, and all months are
reused exploratory data. No stability-window, holdout, or new feature experiments.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from study_common import BASELINE, ROOT, US_MASK, _context, _save


def _plots(contract: dict) -> None:
    os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime_cache/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    context = _context()
    families = context["families"]
    common_us = next(name for name, item in contract["groups"].items() if item["families"] == ["us_equity"])
    dates = pd.to_datetime(context["dates"])
    for mask, family_name, joint_name in [
        ("US_5_input", US_MASK, "joint_existing_us_mask"),
        ("joint_14_input", common_us, BASELINE),
    ]:
        with np.load(ROOT / "components" / f"{family_name}.npz", allow_pickle=False) as values:
            family_mean, family_assets = values["mean"].copy(), values["assets"].tolist()
        with np.load(ROOT / "components" / f"{joint_name}.npz", allow_pickle=False) as values:
            joint_mean, joint_assets = values["mean"].copy(), values["assets"].tolist()
        fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True)
        for column, asset in enumerate(families["us_equity"]):
            index = context["assets"].index(asset)
            actual = context["actual"][:, index]
            for row, seed in enumerate([27, 127, 227]):
                axis = axes[row, column]
                axis.plot(dates, actual, color="#2166ac", label="Actual", marker="o", markersize=2)
                axis.plot(dates, family_mean[row, :, family_assets.index(asset)], color="#238b45", label="US-only")
                axis.plot(dates, joint_mean[row, :, joint_assets.index(asset)], color="#cb181d", label="Full joint")
                axis.axhline(0, color="gray", linewidth=0.7, linestyle="--")
                axis.set_title(f"{asset} | seed {seed}")
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.grid(alpha=0.2)
                axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                if column == 0:
                    axis.set_ylabel("Next-month excess return")
                if row == 2:
                    axis.set_xlabel("Forecast origin")
            limits = [axis.get_ylim() for axis in axes[:, column]]
            low, high = min(v[0] for v in limits), max(v[1] for v in limits)
            for axis in axes[:, column]:
                axis.set_ylim(low, high)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97))
        fig.suptitle(f"Matched task-universe comparison | {mask.replace('_', ' ')} | rank 2, maxiter 15", y=1.0)
        fig.text(
            0.5,
            0.005,
            "Identical raw inputs, targets and forecast dates. No smoothing or averaging over seeds.",
            ha="center",
        )
        fig.tight_layout(rect=(0, 0.025, 1, 0.94))
        fig.savefig(ROOT / f"US_equity_{mask}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def _report(contract: dict) -> None:
    context = _context()
    policies = pd.read_csv(ROOT / "policy_metrics.csv")
    ranking = pd.read_csv(ROOT / "assembly_ranking.csv")
    wide_ic = ranking.groupby(["configuration", "scope"]).mean_ic.mean().unstack().add_prefix("IC_")
    policies = policies.merge(wide_ic, on="configuration", how="left", validate="many_to_one")
    policies.to_csv(ROOT / "all_policy_and_ranking_metrics.csv", index=False)
    weights, fixed, failures = [], [], []
    for path in sorted((ROOT / "portfolio_cases").glob("*.json")):
        artifact = json.loads(path.read_text())
        case = artifact["case"]
        for seed in artifact["seeds"]:
            fixed.extend(seed["fixed_reference_scores"])
            for month in seed["months"]:
                if month["status"] != "solved":
                    failures.append({**case, "seed": seed["seed"], **month})
                for index, asset in enumerate(context["assets"]):
                    weights.append(
                        {
                            **case,
                            "seed": seed["seed"],
                            "date": month["date"],
                            "asset": asset,
                            "weight": None if month["weights"] is None else month["weights"][index],
                            "status": month["status"],
                            "fallback_used": False,
                        }
                    )
    pd.DataFrame(weights).to_csv(ROOT / "monthly_weights.csv", index=False)
    fixed_frame = pd.DataFrame(fixed).drop_duplicates(["configuration", "risk_source", "seed", "portfolio"])
    fixed_frame.to_csv(ROOT / "fixed_reference_scores.csv", index=False)
    _save(ROOT / "failed_allocations.json", failures)
    eligible = policies[policies.complete_seeds.eq(3)].copy()
    kind = {model["name"]: model["kind"] for model in contract["models"]}
    policies["model_kind"] = policies.configuration.map(kind)
    comparable = policies[
        policies.objective.eq("rank")
        & policies.budget_factor.eq(1)
        & policies.risk_source.eq("historical")
        & (policies.minimum.isna() | policies.minimum.eq(10))
    ]
    comparable.to_csv(ROOT / "complete_assembly_comparison.csv", index=False)
    paired = eligible[eligible.objective.eq("rank") & eligible.risk_source.isin(["GP", "historical"])].copy()
    paired["minimum_label"] = paired.minimum.fillna(0).astype(int)
    pairs = (
        paired.pivot(
            index=["configuration", "minimum_label", "budget_factor"],
            columns="risk_source",
            values="conventional_sharpe",
        )
        .dropna()
        .reset_index()
    )
    pairs["GP_minus_historical_Sharpe"] = pairs.GP - pairs.historical
    pairs.to_csv(ROOT / "paired_covariance_comparisons.csv", index=False)
    diagnostics = pd.read_csv(ROOT / "ETF_diagnostics.csv")
    common_us = next(name for name, item in contract["groups"].items() if item["families"] == ["us_equity"])
    selected = diagnostics[diagnostics.configuration.isin([US_MASK, BASELINE, common_us, "joint_existing_us_mask"])]
    selected.to_csv(ROOT / "US_equity_controlled_diagnostics.csv", index=False)
    task_ranking = pd.read_csv(ROOT / "ranking_diagnostics.csv")
    task_ranking[task_ranking.configuration.isin(contract["groups"])].to_csv(
        ROOT / "matched_task_group_ranking.csv", index=False
    )
    _plots(contract)
    _save(
        ROOT / "scope_coverage.json",
        {
            "1": {
                "status": "completed",
                "artifact": "policy_metrics.csv",
                "effective_floors": [None, 3, 5, 7, 10, 12],
                "variance_budget_multipliers": [0.5, 1, 1.5],
            },
            "2": {
                "status": "completed",
                "artifact": "US_equity_controlled_diagnostics.csv",
                "caution": "controlled comparisons diagnose associations_not_universal_causal_proof",
            },
            "3": {"status": "completed", "artifact": "complete_assembly_comparison.csv", "selected_assemblies": 16},
            "4": {"status": "completed", "artifact": "matched_task_group_ranking.csv", "task_groups": 15},
            "5": {
                "status": "completed",
                "artifact": "fixed_reference_scores.csv",
                "calibration_uses": "earlier_errors_only",
            },
            "6": {
                "status": "completed",
                "artifact": "policy_metrics.csv",
                "objectives": ["rank", "smooth_rank", "shrunk_return"],
            },
            "8": {"status": "completed", "artifact": "assembly_ranking.csv"},
            "7": {"status": "excluded_not_run"},
            "9": {"status": "excluded_not_run"},
        },
    )
    lines = [
        "# Approved ETF follow up results",
        "",
        "## Scope",
        "",
        "Completed approved areas 1–6 and 8. Areas 7 and 9 were excluded: no new feature masks, kernel redesign, "
        "training-window alternatives, holdout redesign or additional stability studies.",
        "",
        "The study retained rank 2, maxiter 15, the same 84-month expanding initialization, 24 forecast origins, "
        "three seeds, IWM and t_index. No CVaR, ETF caps, failed-weight fallback, or automatic promotion.",
        "",
        "## Comparable complete portfolios",
        "",
        "The following rows all use historical risk, the original variance budget and the same rank objective. "
        "Family assemblies replace all four families together. Cross-family GP covariance is not fabricated.",
        "",
        "| Effective floor | Model | Sharpe | Pooled IC | Mean effective count |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for minimum in [None, 10]:
        group = comparable[
            comparable.complete_seeds.eq(3)
            & (comparable.minimum.isna() if minimum is None else comparable.minimum.eq(minimum))
        ]
        for row in group.nlargest(5, "conventional_sharpe").itertuples():
            lines.append(
                f"| {'None' if minimum is None else minimum} | {row.configuration} | "
                f"{row.conventional_sharpe:.3f} | {row.IC_pooled:.3f} | {row.mean_effective_count:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation and detailed outputs",
            "",
            "These are selected, gross, exploratory results on previously reused months. A high selected Sharpe "
            "is not proof of future superiority; no new untouched-period test was performed.",
            "",
            "- Diversification, budget and objective comparisons: `all_policy_and_ranking_metrics.csv`.",
            "- Like-for-like covariance comparisons: `paired_covariance_comparisons.csv`.",
            "- Complete family/region/joint portfolios: `complete_assembly_comparison.csv`.",
            "- U.S. equity time-series tracking, amplitude and RMSE: `US_equity_controlled_diagnostics.csv`.",
            "- Matched-family ranking by task group: `matched_task_group_ranking.csv`.",
            "- Within-family versus pooled ranking: `assembly_ranking.csv`.",
            "- Fixed-weight uncertainty diagnostics: `fixed_reference_scores.csv`.",
            "- Every monthly ETF weight and failed solve: `monthly_weights.csv` and `failed_allocations.json`.",
            "- Original per-seed paths and risk multipliers: `portfolio_cases/`.",
            "- Fitted kernel/noise/transform diagnostics: `runs/*/fitted_parameters/`.",
            "",
            f"There were {len(failures)} failed monthly allocation solves. Failed weights remain null; full-period "
            "performance is omitted unless all 24 months and all three seeds completed.",
            "",
            "Prequential scaling adjusts GP variance using only earlier equal-weight errors, with a prior "
            "strength of six and initial multiplier one. Fixed blending is a declared experimental arm, not "
            "a fallback. No covariance eigenvalues or solver weights were repaired.",
            "",
            "## Verification",
            "",
            "`run_study.py --verify-only` checks the immutable plan, original evidence, fit checkpoints, "
            "all policy artifacts and completed outputs. See the parent README for exact commands.",
            "",
        ]
    )
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
