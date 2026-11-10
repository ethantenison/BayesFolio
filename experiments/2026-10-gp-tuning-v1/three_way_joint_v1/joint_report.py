"""Evidence-grounded reporting of first16-selected static three-way ensembles.

Research reporting only: no model selection based on later/full-window metrics,
GP fits or portfolio allocation. Uses decimal returns and dollar saved NAV paths.
Original controls, earlier nested blends, independent covariance counterfactuals,
and the two standing benchmarks are kept distinct.
"""

from __future__ import annotations

import io

import numpy as np


def report(full: list[dict], tuning: dict, selected: dict, global_selected: list[dict], controls: list[dict]) -> None:
    """Write period-separated findings, exact blend recipes, and comparison plot.

    Args:
        full: Complete or explicitly failed frozen final hard-rank/scenario cases.
        tuning: First16-only candidate evaluations, including ineligible records.
        selected: Up to five first16 representatives per third-model candidate.
        global_selected: Five global first16 representatives including objective extremes.
        controls: Original, refined and compromise two-model boundary recipes.
    """
    from joint_math import FIXED_COV, recipe
    from run_joint import OUTPUT, SOURCE, _arrays, _read, _save

    complete = [r for r in full if r["status"] == "complete"]
    by_id = {r["id"]: r for r in full}
    baseline = _read(SOURCE / "cases/control_hard_rank.json")
    control = by_id[f"{controls[0]['id']}_hard_rank"]
    if control["status"] != "complete":
        raise ValueError("The baseline identity control must be complete")
    original_arrays = _arrays(SOURCE / "cases/control_hard_rank.npz")
    new_arrays = _arrays(OUTPUT / "cases" / f"{control['id']}.npz")
    for key in ["mean", "covariance"]:
        np.testing.assert_allclose(new_arrays[key], original_arrays[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(new_arrays["weights"], original_arrays["weights"], rtol=0, atol=3e-7)
    for period in baseline["periods"]:
        for key in ["sharpe", "total_return", "rank_ic"]:
            if abs(control["periods"][period][key] - baseline["periods"][period][key]) > 3e-6:
                raise ValueError("Fast solver changed the baseline materially")
    original_nested = [_read(p) for p in (SOURCE / "cases").glob("t3p__*.json")]
    previous_best = {model: max([r for r in original_nested if r["case"]["third_id"] == model],
                               key=lambda r: r["periods"]["tuning_first16"]["sharpe"])
                     for model in selected}
    period_names = ["tuning_first16", "validation_last12", "all28"]
    findings, covariance_findings = [], []
    for choice in global_selected:
        row = by_id[f"{choice['id']}_hard_rank"]
        if row["status"] != "complete":
            findings.append({"id": row["id"], "status": "failed", "error": row["error"]})
            continue
        deltas = {p: {k: row["periods"][p][k] - baseline["periods"][p][k]
                      for k in ["sharpe", "total_return", "rank_ic"]} for p in period_names}
        findings.append({"id": row["id"], "status": "complete", "recipe": row["recipe"],
                         "deltas_vs_original": deltas,
                         "later12_dominates_baseline": all(v >= -1e-10 for v in deltas["validation_last12"].values())
                         and any(v > 1e-10 for v in deltas["validation_last12"].values())})
        c = row["recipe"]
        fixed = recipe(c["third_id"], np.array(c["mean_counts"]), FIXED_COV)
        reference = by_id[f"{fixed['id']}_hard_rank"]
        if reference["status"] == "complete":
            covariance_findings.append({"id": row["id"], "fixed_covariance_case": reference["id"],
                "same_prediction_weights": True, "covariance_counts": c["covariance_counts"],
                "delta_vs_same_means_fixed_covariance": {p: {k: row["periods"][p][k] - reference["periods"][p][k]
                    for k in ["sharpe", "total_return", "rank_ic"]} for p in period_names}})
    hard = [r for r in complete if r["allocator"] == "hard_rank"]
    leaders = {k: max(hard, key=lambda r: r["periods"]["all28"][k])["id"]
               for k in ["sharpe", "total_return", "rank_ic"]}
    summary = {"status": "complete", "GP_fits": 0, "tuning_recipes": len(tuning),
               "failed_tuning_recipes": sum(r.get("status") == "failed" for r in tuning.values()),
               "full_cases": len(full), "complete_full_cases": len(complete),
               "failed_full_cases": [r for r in full if r["status"] != "complete"],
               "global_first16_choices": global_selected, "per_model_first16_choices": selected,
               "findings": findings, "covariance_findings": covariance_findings,
               "previous_nested_first16_sharpe_best": {k: v["id"] for k, v in previous_best.items()},
               "all28_descriptive_leaders_not_selection_inputs": leaders,
               "scenario_checks_are_not_scenario_retuning": True, "automatic_promotion": False,
               "pristine_holdout": False, "weights_vary_by_month_or_seed": False}
    _save(OUTPUT / "SUMMARY.json", summary)
    _save(OUTPUT / "FINAL_RECIPES.json", {
        "family_order": ["US equity", "International equity", "US FI", "International FI"],
        "component_order": ["broad", "compact", "third"], "denominator": 64,
        "recipes": [r["case"] for r in global_selected]})
    lines = ["# Joint three-way blend retuning", "",
             f"Evaluated {len(tuning):,} distinct first16 recipes. "
             f"{summary['failed_tuning_recipes']} were ineligible; no partial objectives were used. "
             f"Final full paths: {len(complete)}/{len(full)} complete. No GP refits or production promotion.", "",
             "Each family independently chooses broad/compact/third fractions summing to one, static "
             "across months/seeds. International-equity compact and third weights are free too. "
             "Both current components stay rank2/maxiter15; all six third candidates were searched. "
             "Mean tuning starts with fixed50/50 broad/compact covariance; three-way covariance weights "
             "are then separately tuned, followed by a joint mean polish. "
             "This is a bounded deterministic menu/refinement search, not a globally certified continuous optimum.", "",
             "First16 alone choose all proposals/finalists. Later12 and all28 are descriptive on "
             "previously viewed history, not untouched validation. Objectives are conventional Sharpe, "
             "compounded return and cross-sectional ranking IC. NLL is diagnostic only. "
             "Hard rank retains NE>=7, no cap and its original variance ceiling. "
             "Secondary scenario checks use NE>=7/no cap and no variance ceiling; "
             "they check the frozen hard-rank finalists, not independently scenario-tuned recipes.", "",
             "## First16-selected global alternatives", "",
             "| Candidate | Period | Sharpe | Return | IC | Ending $10k |",
             "|---|---|---:|---:|---:|---:|"]
    display = [("Original hard-rank baseline", baseline)]
    display.extend((f"Choice {i + 1}: {choice['case']['third_id']}", by_id[f"{choice['id']}_hard_rank"])
                   for i, choice in enumerate(global_selected))
    for label, row in display:
        if row["status"] != "complete":
            lines.append(f"| {label} | FAILED, never ranked | — | — | — | — |")
            continue
        for p in period_names:
            v = row["periods"][p]
            lines.append(f"| {label} | {p} | {v['sharpe']:.4f} | {v['total_return']:.2%} | "
                         f"{v['rank_ic']:.4f} | ${v['ending_value']:,.0f} |")
    lines.extend(["", "## Exact joint prediction and covariance fractions", "",
                  "| Choice | Family | Broad | Compact | Third |", "|---|---|---:|---:|---:|"])
    for i, choice in enumerate(global_selected):
        for family, weights in zip(["US equity", "International equity", "US FI", "International FI"],
                                    choice["case"]["mean_counts"], strict=True):
            lines.append(f"| {i + 1} | {family} | {weights[0] / 64:.4%} | "
                         f"{weights[1] / 64:.4%} | {weights[2] / 64:.4%} |")
        lines.append(f"| {i + 1} | Covariance (global) | "
                     + " | ".join(f"{x / 64:.4%}" for x in choice["case"]["covariance_counts"]) + " |")
    lines.extend(["", "## What survived the later-period check?", ""])
    for i, finding in enumerate(findings):
        if finding["status"] != "complete":
            lines.append(f"- Choice{i + 1}: incomplete; no performance inference.")
            continue
        d = finding["deltas_vs_original"]["validation_last12"]
        lines.append(f"- Choice{i + 1}: later12 delta vs original: Sharpe{d['sharpe']:+.4f}, "
                     f"return{d['total_return']:+.2%}, IC{d['rank_ic']:+.4f}. "
                     f"Dominates all three baseline objectives: {finding['later12_dominates_baseline']}.")
    lines.extend(["", "## Each third model: prior restricted blend versus joint retuning", "",
                  "Rows below choose each method's Sharpe leader on first16 only, not full28.", "",
                  "| Third candidate | Method | Period | Sharpe | Return | IC |", "|---|---|---|---:|---:|---:|"])
    for model, choices in selected.items():
        chosen = max(choices, key=lambda r: r["sharpe"])
        joint = by_id[f"{chosen['id']}_hard_rank"]
        for method, row in [("Previous fixed base ratio", previous_best[model]), ("Joint all-three retuning", joint)]:
            if row["status"] != "complete":
                continue
            for p in period_names:
                v = row["periods"][p]
                lines.append(f"| {model} | {method} | {p} | {v['sharpe']:.4f} | "
                             f"{v['total_return']:.2%} | {v['rank_ic']:.4f} |")
    lines.extend(["", "## Boundaries", "",
                  "Candidate simplex weights may be zero: the search does not force a useless third component. "
                  "Different mixtures may produce identical ranks/portfolios. Objective ties at1e-10 "
                  "prefer a simpler/lower-third-weight recipe; exact optimal blend weights may not be identified.",
                  "PSD covariance arithmetic pooling is a declared risk recipe, not an estimated "
                  "cross-model error covariance or a fully calibrated ensemble distribution.",
                  "The analytical allocator is the KKT solution of the same rank objective whenever "
                  "the variance constraint is satisfied; otherwise the same conic problem is solved. "
                  "The exact baseline's allocations and metrics were parity-checked. "
                  "No fallback portfolios, CVaR, weight clipping, risk relaxation or fitted uncertainty calibration.",
                  "Seeds share one market history. Returns are gross of explicit costs/taxes. "
                  "More recipe testing increases selection risk; this does not establish future outperformance.", ""])
    _save(OUTPUT / "RESULTS.md", "\n".join(lines).encode())
    csv = ["id,allocator,period,sharpe,total_return,rank_ic,max_drawdown,ending_value"]
    for row in complete:
        for p, v in row["periods"].items():
            csv.append(",".join(map(str, [row["id"], row["allocator"], p, *[v[k] for k in
                ["sharpe", "total_return", "rank_ic", "max_drawdown", "ending_value"]]])))
    _save(OUTPUT / "ALL_FINAL_RESULTS.csv", ("\n".join(csv) + "\n").encode())
    _plot(display)


def _plot(display: list[tuple[str, dict]]) -> None:
    import matplotlib
    from run_joint import OUTPUT, SOURCE, _arrays, _save

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.patheffects as effects
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    context = _arrays(SOURCE / "context.npz")
    dates = context["dates"].astype("datetime64[D]")
    ends = (dates.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    axis_dates = np.r_[dates[:1], ends]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1]}, layout="constrained")
    baseline = _arrays(SOURCE / "cases/control_hard_rank.npz")["wealth"].mean(0)
    palette = ["#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    for i, (_, row) in enumerate(display[1:]):
        if row["status"] != "complete":
            continue
        values = _arrays(OUTPUT / "cases" / f"{row['id']}.npz")["wealth"].mean(0)
        top.plot(axis_dates, values, color=palette[i], linewidth=2, linestyle="--",
                 label=f"Joint choice{i + 1} | ${values[-1]:,.0f}")
        bottom.plot(axis_dates, values - baseline, color=palette[i], label=f"Choice{i + 1}")
    for label, key, color in [("Schwab aggressive", "benchmark_schwab_total", "#777777"),
                              ("Momentum top7", "benchmark_momentum_total", "#999999")]:
        values = 10000 * np.r_[1., np.cumprod(1 + context[key])]
        top.plot(axis_dates, values, color=color, linestyle=":", linewidth=2,
                 label=f"{label} | ${values[-1]:,.0f}")
    line, = top.plot(axis_dates, baseline, linewidth=2.8, color="#1565C0", zorder=20,
                    label=f"Original hard-rank baseline | ${baseline[-1]:,.0f}")
    line.set_path_effects([effects.Stroke(linewidth=4.4, foreground="white"), effects.Normal()])
    bottom.axhline(0, color="#1565C0", linewidth=2)
    for axis in [top, bottom]:
        axis.axvline(axis_dates[16], color="gray", linestyle="--", alpha=.7)
        axis.grid(alpha=.2)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    top.set_title("Joint three-way fixed blends: first16-selected alternatives\n"
                  "Growth of $10,000 | June2024–September2026 | Monthly rebalancing")
    top.set_ylabel("Investment value ($)")
    handles, labels = top.get_legend_handles_labels()
    top.legend([handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]], loc="upper left", fontsize=8.5)
    bottom.set_ylabel("Difference vs baseline ($)")
    bottom.legend(loc="best", ncol=5, fontsize=8)
    bottom.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    bottom.set_xlabel("Month-end; divider marks the later12 retrospective check")
    figure.supxlabel("All recipes static across dates/seeds. Means of three seed NAVs. Gross of costs/taxes.\n"
                    "All history previously viewed; no untouched holdout or automatic promotion.", fontsize=9)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    _save(OUTPUT / "COMPARISON.png", buffer.getvalue())
