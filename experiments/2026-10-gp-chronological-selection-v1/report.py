"""Seal matched portfolio comparisons after chronological selections are frozen.

Experiment-only reporting I/O; no GP fitting, new hypotheses or test-driven
selection. Reuse complete covariance/distribution diagnostics from the existing
framework. Returns are decimal, allocation weights fractions and NAV dollars.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from selection_math import FAMILY, blend, recipe


def _family_ic(actual: np.ndarray, mean: np.ndarray) -> dict:
    result = {}
    for family, label in enumerate(["us_equity", "international_equity", "us_fixed_income",
                                   "international_fixed_income"]):
        predicted = rankdata(mean[:, :, FAMILY == family], axis=-1)
        realized = rankdata(actual[:, FAMILY == family], axis=-1)
        predicted -= predicted.mean(-1, keepdims=True)
        realized -= realized.mean(-1, keepdims=True)
        denominator = np.sqrt(np.sum(predicted**2, -1) * np.sum(realized**2, -1))
        if (denominator <= 0).any():
            result[label] = {"rank_ic": None, "undefined_due_to_ties": True}
        else:
            ic = np.einsum("sta,ta->st", predicted, realized) / denominator
            result[label] = {"rank_ic": float(ic.mean()), "per_seed": ic.mean(1).tolist()}
    return result


def _describe(case: dict, names: list[str], configs: dict) -> list[dict]:
    result = []
    for family, row in enumerate(case["counts"]):
        experts = [{"expert": name, "fraction": value / 64, "rank": configs[name]["rank"],
                    "maxiter": configs[name]["maxiter"], "rolling_months": configs[name]["rolling_months"],
                    "inputs": configs[name]["inputs"]}
                   for name, value in zip(names, row, strict=True) if value]
        result.append({"family": family, "experts": experts})
    return result


def create_report(contract: dict, context: dict, means: np.ndarray, covariance: np.ndarray,
                  selections: list[dict], incumbent: dict) -> None:
    """Evaluate frozen recipes and save all paths, diagnostics, figures and seals.

    Args:
        contract: Nineteen registered feature/GP/window experts.
        context: Complete52-month forecast/outcome axes; only last28 are scored here.
        means: Complete expert/seed/month/ETF prediction ledger in decimal units.
        covariance: Fixed50/50 broad/compact observed covariance, squared decimals.
        selections: Four earlier-only frozen multi-month selection records.
        incumbent: Prior two-model static family recipe on the same expert axis.
    """
    import case_evaluator as evaluator
    from joint_math import RankAllocator
    from run_study import CAMPAIGN, OUTPUT, arrays, digest, read, save, snapshot, status, verify_hashes

    sys.path.insert(0, str(CAMPAIGN / "track1_scenarios"))
    frozen = read(OUTPUT / "SELECTIONS_FROZEN.json")
    verify_hashes(frozen["selection_sha256"])
    names = list(contract["configs"])
    common = {key: context[key][24:] for key in ["dates", "actual", "risk_free", "risk_budget"]}
    common.update(assets=context["assets"], seeds=context["seeds"])
    later_means, later_covariance = means[:, :, 24:], covariance[:, 24:]
    candidates = {"current_baseline": (blend(later_means, incumbent), {"recipe": incumbent})}
    for index, name in enumerate(names):
        candidates["expert_" + name] = (later_means[index], {"expert": name})
    for objective in ["sharpe", "ranking", "compromise"]:
        first = selections[0]["selected"][objective]["recipe"]
        candidates["locked28_" + objective] = (
            blend(later_means, first), {"recipe": first, "selection_months": 24,
                                      "static_all28": True,
                                      "selection_sha256": digest(OUTPUT / "selections/fold_0.json")})
        blocked = np.empty((3, 28, 15))
        rules = []
        for fold, selection in enumerate(selections):
            case = selection["selected"][objective]["recipe"]
            block = slice(fold * 7, (fold + 1) * 7)
            blocked[:, block] = blend(later_means[:, :, block], case)
            rules.append({"fold": fold, "origins": selection["test_origins"],
                          "selection_sha256": digest(OUTPUT / "selections" / f"fold_{fold}.json"),
                          "family_experts": _describe(case, names, contract["configs"])})
        candidates["blocked7_" + objective] = (blocked, {"static_within_each7_month_block": True, "rules": rules})
    # Explicitly descriptive prior challengers; their old outcome-informed
    # coefficients never enter the earlier-only selection search or shortlist.
    challenger_counts = {
        "prior48_sharpe_challenger": [[24, 38, 2], [63, 0, 1], [0, 4, 60], [1, 3, 60]],
        "prior48_return_challenger": [[19, 33, 12], [64, 0, 0], [0, 4, 60], [4, 1, 59]],
        "prior48_rank_challenger": [[39, 25, 0], [54, 10, 0], [0, 6, 58], [0, 0, 64]],
    }
    positions = [names.index(x) for x in ["broad_r2_i15_exp", "compact_r2_i15_exp", "broad_roll48_r2_i15"]]
    for name, counts in challenger_counts.items():
        expanded = np.zeros((4, len(names)), dtype=int)
        expanded[:, positions] = counts
        case = recipe(expanded)
        candidates[name] = (blend(later_means, case), {"recipe": case,
                             "prior_research_selected_descriptive_control": True,
                             "covariance_matched_to_fixed50_50": True})
    records = []
    allocator = RankAllocator()
    for number, (name, (mean, definition)) in enumerate(candidates.items(), 1):
        destination = OUTPUT / "cases" / f"{name}.json"
        if destination.exists():
            row = read(destination)
            if row["definition"] != definition or row["artifact_sha256"] != digest(destination.with_suffix(".npz")):
                raise ValueError("Changed completed portfolio case")
            records.append(row)
            continue
        weights, constrained = allocator.allocate(mean, later_covariance, common["risk_budget"])
        values = {**{key: common[key] for key in ["dates", "assets", "seeds", "actual", "risk_free"]},
                  "mean": mean, "covariance": later_covariance, "weights": weights,
                  "raw_weights": weights.copy(), "allocated": np.ones((3, 28), dtype=bool),
                  "effective_count": 1 / np.sum(weights**2, axis=-1),
                  "equity_weight": weights[:, :, :8].sum(-1),
                  "predicted_variance": np.einsum("sta,stab,stb->st", weights, later_covariance, weights)}
        periods, calibration = evaluator._scores(common, mean, later_covariance, values)
        metrics = evaluator._dependency("metrics")
        portfolio_actual = np.einsum("sta,ta->st", weights, common["actual"])
        portfolio_mean = np.sum(weights * mean, -1)
        portfolio_std = np.sqrt(values["predicted_variance"])
        calibration["allocated_portfolio"] = [metrics.calibration_summary(metrics.gaussian_scores(
            portfolio_actual[s], portfolio_mean[s], portfolio_std[s])) for s in range(3)]
        periods["all28"]["rmse"] = float(np.sqrt(np.mean((mean - common["actual"])**2)))
        snapshot(destination.with_suffix(".npz"), values)
        row = {"id": name, "status": "complete", "definition": definition, "periods": periods,
               "family_ranking": _family_ic(common["actual"], mean), "calibration": calibration,
               "risk_constrained_pairs": constrained, "complete_allocations": 84,
               "artifact_sha256": digest(destination.with_suffix(".npz")),
               "selections_frozen_sha256": digest(OUTPUT / "SELECTIONS_FROZEN.json")}
        save(destination, evaluator._jsonable(row))
        records.append(row)
        status("portfolio_evaluation", complete_cases=number, total_cases=len(candidates))
    prior = arrays(CAMPAIGN / "outputs/context.npz")
    control = arrays(OUTPUT / "cases/current_baseline.npz")
    np.testing.assert_allclose(control["weights"], prior["baseline_hard_weights"], rtol=0, atol=3e-7)
    np.testing.assert_allclose(control["mean"], prior["baseline_mean"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(control["covariance"], prior["baseline_covariance"], rtol=0, atol=1e-15)
    table = [{"configuration": row["id"], **{key: row["periods"]["all28"][key] for key in
              ["sharpe", "total_return", "rank_ic", "max_drawdown", "rmse", "joint_nll_per_asset"]}}
             for row in records]
    _plot(records, prior, OUTPUT)
    baseline = next(row for row in records if row["id"] == "current_baseline")["periods"]["all28"]
    takeaways = []
    for name in ["locked28_sharpe", "blocked7_sharpe", "expert_rich_r2_i15_exp"]:
        row = next(item for item in records if item["id"] == name)["periods"]["all28"]
        takeaways.append(f"- `{name}` versus current baseline: Sharpe {row['sharpe'] - baseline['sharpe']:+.4f}; "
                         f"compounded return {100 * (row['total_return'] - baseline['total_return']):+.2f} pp; "
                         f"rank IC {row['rank_ic'] - baseline['rank_ic']:+.4f}.")
    features = []
    full = next(row for row in records if row["id"] == "expert_rich_r2_i15_exp")["periods"]["all28"]
    for row in records:
        if row["id"].startswith("expert_rich_without_"):
            score = row["periods"]["all28"]
            features.append({"removed_block": row["id"], **{
                key + "_change": score[key] - full[key] for key in ["sharpe", "total_return", "rank_ic"]}})
    selections_readout = [{"fold": selection["fold"], "last_selection_origin": selection["ledger"]["last_origin"],
                           "test_origins": selection["test_origins"], "policies": {
                               objective: _describe(row["recipe"], names, contract["configs"])
                               for objective, row in selection["selected"].items()}}
                          for selection in selections]
    save(OUTPUT / "SELECTED_INPUTS_AND_WEIGHTS.json", selections_readout)
    save(OUTPUT / "COMPARISON.json", {"cases": records, "matched_rich_ablations": features})
    lines = ["# Chronological GP selection results", "", "## Summary and scope", "",
             "19 experts; 2,208 new fits plus two independent parity checks. Inputs, GP settings and",
             "blend weights were selected only from earlier out-of-sample forecast ledgers.",
             "Fixed50/50 covariance, unchanged hard rank, NE>=7, no cap, no CVaR or fallback.", "",
             "## Findings and impact", "", *takeaways, "",
             "These are retrospective comparisons. Earlier feature searches and today's catalog design",
             "used inspected history; no result is an untouched confirmation or automatic promotion.",
             "The existing broad/compact controls retain their inherited feature-selection bias.", "",
             "## Matched full28 comparisons", "", pd.DataFrame(table).to_markdown(index=False), "",
             "Prior48 challengers are descriptive controls with prior outcome-informed coefficients.",
             "`locked28` uses one recipe chosen before the first test origin; `blocked7` changes recipes",
             "only at four preregistered seven-month boundaries, never month by month.", "",
             "## Conditional feature ablations", "", pd.DataFrame(features).to_markdown(index=False), "",
             "Changes are removal minus full eligible-rich r2/i15 expanding at the same allocator/covariance.",
             "A feature block's result is conditional, not proof of intrinsic or universal importance.", "",
             "## Interpretation and next decision", "",
             "Compare locked and blocked policies separately: any advantage of blocked selection includes",
             "periodic reselection, not merely a superior fixed blend. Examine seed and temporal-half",
             "evidence in cases/ and selections/ before deciding whether richer representations help.",
             "A richer expert's standalone weakness does not establish that its ensemble contribution is zero.",
             "Do not promote from this report; freeze challengers for future unseen monthly outcomes.", "",
             "## Verification", "", "Baseline moments reproduced exactly and weights match within3e-7.",
             "Every case has84 allocations and all three seeds. Selected inputs/weights are recorded in",
             "SELECTED_INPUTS_AND_WEIGHTS.json; full covariance and uncertainty diagnostics are in cases/.",
             "Reproduce/verify with the commands in ../README.md. No production or prior evidence changed.", ""]
    save(OUTPUT / "RESULTS.md", "\n".join(lines).encode())
    verify_hashes(contract["input_sha256"])
    artifacts = [*sorted((OUTPUT / "components").glob("*.npz")),
                 *sorted((OUTPUT / "cases").glob("*")), *sorted((OUTPUT / "selections").glob("*.json")),
                 OUTPUT / "FEATURE_AUDIT.json", OUTPUT / "CONTRACT.json", OUTPUT / "FITS_COMPLETED.json",
                 OUTPUT / "SELECTIONS_FROZEN.json", OUTPUT / "SELECTED_INPUTS_AND_WEIGHTS.json",
                 OUTPUT / "COMPARISON.json", OUTPUT / "COMPARISON.png", OUTPUT / "RESULTS.md"]
    save(OUTPUT / "COMPLETED.json", {"passed": True, "complete_cases": len(records), "new_fits": 2208,
                                     "audit_fits": 2, "frozen_cutoffs": 4, "pristine_holdout": False,
                                     "artifact_sha256": {str(path): digest(path) for path in artifacts}})
    status("complete", complete_cases=len(records), completed_new_fits=2208, separate_audit_fits=2,
           report=str(OUTPUT / "RESULTS.md"), plot=str(OUTPUT / "COMPARISON.png"))


def _plot(records: list[dict], prior: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    from run_study import arrays, save

    figure, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
    dates = pd.to_datetime(prior["dates"]) + pd.offsets.BMonthEnd(1)
    x = pd.DatetimeIndex([pd.to_datetime(prior["dates"][0]), *dates])
    panels = [
        [("current_baseline", "Current hard-rank baseline", "#2456a6"),
         ("expert_rich_r2_i15_exp", "Richer GP alone, matched allocator", "#a75927"),
         ("locked28_sharpe", "Earlier-selected, fixed28-month Sharpe recipe", "#1c8c67"),
         ("blocked7_sharpe", "Earlier-selected, frozen7-month Sharpe recipes", "#a83e6f")],
        [("current_baseline", "Current hard-rank baseline", "#2456a6"),
         ("locked28_ranking", "Fixed28-month ranking recipe", "#319362"),
         ("locked28_compromise", "Fixed28-month compromise recipe", "#aa7536"),
         ("prior48_return_challenger", "Prior48 return challenger (descriptive)", "#934d9e")],
    ]
    for axis, series in zip(axes, panels, strict=True):
        for name, label, color in series:
            values = arrays(output / "cases" / f"{name}.npz")["wealth"]
            axis.plot(x, values.mean(0), color=color, label=f"{label} | ${values[:, -1].mean():,.0f}")
            axis.fill_between(x, values.min(0), values.max(0), color=color, alpha=.08)
        for key, label, color in [("benchmark_schwab_total", "Schwab aggressive", "#9b7b10"),
                                  ("benchmark_momentum_total", "Simple momentum top7", "#66736d")]:
            nav = 10000 * np.r_[1., np.cumprod(1 + prior[key])]
            axis.plot(x, nav, color=color, linestyle="--", label=f"{label} | ${nav[-1]:,.0f}")
        axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        axis.set_ylabel("Investment value")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="upper left")
    axes[0].set_title("Matched GP representations and chronological selection | Growth of $10,000")
    axes[1].set_title("Fixed-recipe tradeoffs and prior48 challenger")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    figure.text(.5, .015, "Monthly rebalancing; gross of explicit costs/taxes. "
                "Shading: numerical seed range, not confidence.\n"
                "Retrospective research, not a pristine holdout. Same fixed50/50 covariance and NE>=7 allocator.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .055, 1, 1))
    import io

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    save(output / "COMPARISON.png", buffer.getvalue())
