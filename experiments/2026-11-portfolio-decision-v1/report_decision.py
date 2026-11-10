"""Render the frozen November decision without fitting, scoring or selecting.

Experiment-only report I/O reads root-supplied decisions, validated fractional
weights and saved single-portfolio paths. Returns are decimal, portfolio weights
fractions, covariance squared decimals and wealth dollars. No historical source,
candidate definition or live allocation is changed by rendering.

Usage:
    from report_decision import render
    render(config, validation, comparison, november_weights_dataframe)
The root-owned reproducibility command assembles and validates these inputs.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import FAMILIES, HERE, arrays, save

PERIOD_LABELS = {"earlier24": "Jun 2022–May 2024 (24 months)",
                 "later28": "Jun 2024–Sep 2026 (28 months)",
                 "later29": "Jun 2024–Oct 2026 (29 months)",
                 "all52": "Jun 2022–Sep 2026 (52 months)",
                 "all53": "Jun 2022–Oct 2026 (53 months)"}
FAMILY_LABELS = dict(zip(FAMILIES, ["U.S. equity", "International equity", "U.S. fixed income",
                                  "International fixed income"], strict=True))


def _percent(value: object, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * float(value):.{digits}f}%"


def _number(value: object, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else HERE / path


def _period_table(records: list[dict], periods: list[str]) -> str:
    rows = []
    for record in records:
        for period in periods:
            values = record[period]
            rows.append({"Configuration": record["label"], "Period": PERIOD_LABELS.get(period, period),
                         "Sharpe": _number(values.get("sharpe")),
                         "Return": _percent(values.get("total_return")),
                         "Rank IC": _number(values.get("rank_ic")),
                         "Max drawdown": _percent(values.get("max_drawdown")),
                         "Average equity": _percent(values.get("equity_weight"))})
    return pd.DataFrame(rows).to_markdown(index=False)


def _gate_table(record: dict) -> str:
    promotion = record.get("promotion", {})
    gates = promotion.get("gates", {})
    if not gates:
        return "See the frozen promotion evidence in COMPARISON.json and HISTORICAL_DECISION.json."
    return pd.DataFrame([{"Preregistered gate": key, "Passed": bool(value)}
                         for key, value in gates.items()]).to_markdown(index=False)


def _alternatives_table(records: list[dict], selected_id: str, incumbent_id: str) -> str:
    failed_labels = {
        "all52_sharpe_improvement_0p05": "Sharpe gain ≥0.05",
        "all52_IC_improvement_0p01": "Rank IC gain ≥0.01",
        "every_leave_one_month_out_sharpe_positive": "every leave-one-month-out Sharpe gain >0",
        "no_family_IC_loss_over_0p05": "no family IC loss >0.05",
    }
    selected = next(record for record in records if record["id"] == selected_id)
    rows = []
    for record in records:
        values = record["all52"]
        promotion = record["promotion"]
        if record["id"] == incumbent_id:
            reason = "Incumbent reference"
        elif record["id"] == selected_id:
            reason = "Selected; all frozen gates passed"
        elif not promotion["eligible"]:
            failed = [failed_labels.get(key, key) for key, passed in promotion["gates"].items() if not passed]
            reason = "Failed: " + "; ".join(failed)
        else:
            dominated = all(selected["all52"][key] > values[key] for key in ["sharpe", "total_return", "rank_ic"])
            reason = ("All gates passed; selected is higher on all three objectives" if dominated
                      else "All gates passed; not selected by frozen decision")
        rows.append({"Configuration": record["label"], "Sharpe": _number(values["sharpe"], 4),
                     "Cumulative return": _percent(values["total_return"], 3),
                     "Rank IC": _number(values["rank_ic"], 4), "Frozen decision/status": reason})
    return pd.DataFrame(rows).to_markdown(index=False)


def _selection_note(records: list[dict], selected_id: str, incumbent_id: str) -> str:
    if selected_id == incumbent_id:
        return ("No challenger justified replacing the incumbent under the frozen promotion rule. "
                "The table below distinguishes the incumbent reference from rejected challengers.")
    selected = next(record for record in records if record["id"] == selected_id)
    eligible = [record for record in records if record["promotion"]["eligible"]]
    others = [record for record in eligible if record["id"] != selected_id]
    dominated = all(all(selected["all52"][key] > record["all52"][key]
                        for key in ["sharpe", "total_return", "rank_ic"]) for record in others)
    if dominated:
        return (f"{len(eligible)} finalists passed every frozen gate. The selected recipe is higher than each of "
                f"the other {len(others)} eligible recipes on all three 52-month objectives: Sharpe, cumulative "
                "return and rank IC. It is the sole eligible Pareto candidate; the closest-to-incumbent "
                "tie-break was not needed. Ineligible candidates were excluded by the recorded gates, "
                "even if an individual objective was higher.")
    return ("The supplied frozen decision selected this recipe from eligible candidates. "
            "The alternatives table records failed gates separately from eligible alternatives.")


def _comparison_table(records: list[dict], benchmarks: dict, period: str, selected_id: str) -> str:
    rows = []
    for record in records:
        values = record[period]
        rows.append({"Configuration": record["label"], "Selected": record["id"] == selected_id,
                     "Frozen gates passed": bool(record.get("promotion", {}).get("eligible", False)),
                     "Sharpe": _number(values.get("sharpe")),
                     "Cumulative return": _percent(values.get("total_return")),
                     "Rank IC": _number(values.get("rank_ic")),
                     "Max drawdown": _percent(values.get("max_drawdown")),
                     "October return": _percent(record.get("october_return"))})
    for key, label in [("schwab_aggressive", "Schwab aggressive proxy"),
                       ("momentum_top7", "Simple momentum top7")]:
        values = benchmarks[key][period]
        rows.append({"Configuration": label, "Selected": False, "Frozen gates passed": "Not applicable",
                     "Sharpe": _number(values.get("sharpe")),
                     "Cumulative return": _percent(values.get("total_return")),
                     "Rank IC": "—", "Max drawdown": _percent(values.get("max_drawdown")),
                     "October return": _percent(benchmarks[key].get("october_return"))})
    return pd.DataFrame(rows).to_markdown(index=False)


def _raw_checks(validation: dict) -> dict:
    portfolio = validation.get("portfolio", {})
    if "raw_checks" in portfolio:
        return portfolio["raw_checks"]
    if "raw_checks" in validation:
        return validation["raw_checks"]
    return portfolio.get("raw", validation.get("raw", {}))


def _plot(config: dict, comparison: dict, records: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    dates = np.asarray(comparison["dates"]).astype(str)
    if dates.shape != (53,):
        raise ValueError("Report requires exactly 53 complete historical origins")
    market_dates = pd.to_datetime(dates)
    endpoints = market_dates + pd.offsets.BMonthEnd(1)
    x_full = pd.DatetimeIndex([market_dates[0], *endpoints])
    selected_id, incumbent_id = comparison["selected_id"], comparison["incumbent_id"]
    series = []
    for identifier, label, color in [
        (incumbent_id, "Incumbent hard-rank", "#2456a6"),
        (selected_id, "Selected November configuration", "#17865b"),
    ]:
        record = records[identifier]
        artifact = arrays(_path(record["artifact_path"]))
        np.testing.assert_array_equal(artifact["dates"].astype(str), dates)
        wealth = np.asarray(artifact["wealth"], dtype=float)
        if wealth.shape != (54,) or not np.isfinite(wealth).all() or (wealth <= 0).any():
            raise ValueError("Complete single-portfolio54-point wealth path required")
        series.append((label, color, wealth, identifier == selected_id and label.startswith("Selected")))
    for key, label, color in [("schwab_aggressive", "Schwab aggressive proxy", "#a37b18"),
                              ("momentum_top7", "Simple momentum: top7 equal weight", "#6b6e70")]:
        artifact = arrays(_path(comparison["benchmark"][key]["artifact_path"]))
        np.testing.assert_array_equal(artifact["dates"].astype(str), dates)
        wealth = np.asarray(artifact["wealth"], dtype=float)
        if wealth.shape != (54,) or not np.isfinite(wealth).all() or (wealth <= 0).any():
            raise ValueError("Complete benchmark wealth path required")
        series.append((label, color, wealth, False))
    figure, axes = plt.subplots(2, 1, figsize=(13.8, 10.4))
    titles = ["Full replay: June 2022–October 2026",
              "Recent replay: June 2024–October 2026; reset to $10,000"]
    for axis, start, title in zip(axes, [0, 24], titles, strict=True):
        x = x_full[start:]
        for label, color, wealth, selected in series:
            path = wealth[start:] / wealth[start] * 10000
            benchmark = label.startswith("Schwab") or label.startswith("Simple")
            identical = selected and selected_id == incumbent_id
            style = "--" if benchmark or identical else "-"
            width = 1.9 if benchmark or identical else 2.7
            if identical:
                label += " (same as incumbent)"
            axis.plot(x, path, color=color, linewidth=width, linestyle=style,
                      label=f"{label} | ${path[-1]:,.0f}", zorder=3 if selected else 2)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Investment value ($)")
        axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6 if start == 0 else 3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axis.grid(alpha=.2)
        axis.legend(fontsize=9, loc="upper left", framealpha=.94)
        axis.set_xlim(x[0], x[-1])
    figure.suptitle("November 2026 portfolio decision | Growth of $10,000\n"
                    "One pooled GP forecast and one allocator solve per month", fontsize=14)
    axes[1].set_xlabel("Realized month-end investment value")
    figure.text(.5, .018,
                "Retrospective research; gross of explicit costs/taxes. "
                "One portfolio per method—not averaged seed portfolios or confidence bands.\n"
                "All GP methods: fixed 50/50 covariance, hard rank, effective assets ≥7, "
                "no ETF cap, no CVaR. October outcome did not select the configuration.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .065, 1, .935))
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170)
    plt.close(figure)
    save(HERE / "COMPARISON.png", buffer.getvalue())


def render(config: dict, validation: dict, comparison: dict, weights: pd.DataFrame) -> None:
    """Render immutable decision/report artifacts from root-validated evidence.

    Args:
        config: Selected recipe, exact settings/cutoffs, fit counters and execution
            note. The renderer never alters or chooses the supplied recipe.
        validation: Root-owned acceptance checks; passed must be explicit.
        comparison: Matched candidate records, dates, and the two entries in
            benchmark, with saved53-origin/54-wealth-point artifacts. Returns
            are decimal units.
        weights: Fifteen ETF rows, asset_id and raw decimal weight columns,
            including zeros in the approved asset order.

    Raises:
        ValueError: Missing selected/incumbent cases, axes, complete paths,
            or the all-asset weight table required for truthful reporting.
    """
    records_list = comparison["records"]
    records = {row["id"]: row for row in records_list}
    if len(records) != len(records_list):
        raise ValueError("Comparison case IDs must be unique")
    selected_id, incumbent_id = comparison["selected_id"], comparison["incumbent_id"]
    if selected_id not in records or incumbent_id not in records:
        raise ValueError("Selected and incumbent evidence must both be present")
    if weights["asset_id"].astype(str).tolist() != config["assets"] or len(weights) != 15:
        raise ValueError("Decision table must retain all fifteen ETFs in the approved order")
    raw_weights = pd.to_numeric(weights["weight"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(raw_weights).all():
        raise ValueError("Report cannot hide nonfinite raw weights")
    selected, incumbent = records[selected_id], records[incumbent_id]
    selected_recipe = config["selected_recipe"]
    if selected_recipe["id"] != selected_id:
        raise ValueError("Selected recipe and scored comparison disagree")
    fractions = np.asarray(selected_recipe["fractions"], dtype=float)
    if (fractions.shape != (4, 3) or not np.isfinite(fractions).all()
            or (fractions < 0).any() or not np.allclose(fractions.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Four fixed three-component prediction simplexes required")
    family_table = pd.DataFrame([
        {"Family": FAMILY_LABELS[family], "Broad expanding": _percent(row[0], 4),
         "Compact expanding": _percent(row[1], 4), "Broad rolling 48": _percent(row[2], 4)}
        for family, row in zip(FAMILIES, fractions, strict=True)])
    weight_table = pd.DataFrame({"ETF": config["assets"],
                                "Weight (decimal)": [f"{value:.12f}" for value in raw_weights],
                                "Weight (%)": [f"{100 * value:.4f}%" for value in raw_weights]})
    exposure = [float(raw_weights[start:end].sum()) for start, end in [(0, 4), (4, 8), (8, 12), (12, 15)]]
    exposure_table = pd.DataFrame({"Family": [FAMILY_LABELS[family] for family in FAMILIES],
                                   "November target allocation": [_percent(value, 4) for value in exposure]})
    fixed_income_assets = [asset for asset, value in zip(config["assets"][8:], raw_weights[8:], strict=True)
                           if value > 0]
    exposure_note = (f"The November target is {_percent(sum(exposure[:2]))} equities and "
                     f"{_percent(sum(exposure[2:]))} fixed income. "
                     "Minimum effective assets 7 constrains concentration; it does not require an equity/fixed-income "
                     "balance or a holding in every family. "
                     "No family allocation floor or extra constraint was imposed. "
                     "Nonzero fixed-income target holdings: " + (", ".join(fixed_income_assets) or "none") + ".")
    checks = _raw_checks(validation)
    check_table = pd.DataFrame([
        {"Check": label, "Recorded value": _number(checks.get(key), digits)}
        for key, label, digits in [
            ("weight_sum", "Sum of raw weights", 12),
            ("minimum_weight", "Minimum raw weight", 12),
            ("maximum_weight", "Maximum raw weight", 8),
            ("effective_assets", "Effective asset count", 8),
            ("predicted_variance", "Predicted variance (decimal squared)", 10),
            ("variance_ceiling", "Causal variance ceiling (decimal squared)", 10),
            ("variance_to_ceiling", "Predicted variance / ceiling", 8),
        ]])
    retained = bool(config["incumbent_retained"])
    decision = ("Retain the incumbent: the predefined evidence does not justify replacement."
                if retained else "Use the selected challenger: it passed the frozen conservative promotion rule.")
    status = ("Validated target-weight recommendation" if validation.get("passed") is True
              else "NOT VALIDATED—provisional")
    deltas = {key: selected["all52"][key] - incumbent["all52"][key]
              for key in ["sharpe", "total_return", "rank_ic"]}
    reproducer = config.get("reproduction_command", "uv run --locked --no-sync python "
                            "experiments/2026-11-portfolio-decision-v1/run_decision.py --verify")
    execution = config["execution_note"]
    vintage = ("The vendor's MGK adjusted-price overlap showed an approximately 0.08247% vintage discrepancy. "
               "A documented historical dividend revision was recognized but not back-applied to frozen history. "
               "This is a data-vintage limitation, not an additional strategy return or an unrecorded repair.")
    promotion_note = (
        "The selection decision was frozen using the 52 earlier completed return months "
        "before October outcomes were scored. "
        "October is an appended descriptive check, not a reason to change the chosen configuration. "
        "Operational gates are conservative decision hurdles, not statistical significance tests "
        "or a proof of future superiority.")
    policy = (
        "Apply fixed family prediction fractions, average means equally across seeds 27/127/227, "
        "average the approved 50/50 broad/compact observed covariance pool equally across those seeds, "
        "and allocate once. Every historical finalist is replayed with exactly this single-portfolio policy. "
        "The old research curves averaging three separately optimized seed portfolios "
        "are not used as its performance record.")
    fixed = (
        "GP rank 2/maxiter 15; t_index retained; raw hy_spread and its change excluded. "
        "Hard-rank allocation, long-only/full investment, effective assets ≥7, no ETF cap, and the unchanged "
        "causal historical-EW variance ceiling. Fixed 50/50 broad/compact covariance. "
        "No CVaR, uncertainty recalibration, continuous risk penalty, new forest blender, "
        "monthly discretionary blend selection, fallback weights or constraint relaxation.")
    alternatives = _alternatives_table(records_list, selected_id, incumbent_id)
    selection_note = _selection_note(records_list, selected_id, incumbent_id)
    prior48 = next(record for record in records_list if record["label"] == "prior48_sharpe")
    alternative_note = (
        f"For example, prior48_sharpe has slightly higher 52-month Sharpe "
        f"({_number(prior48['all52']['sharpe'], 4)} versus {_number(selected['all52']['sharpe'], 4)}) and "
        f"cumulative return ({_percent(prior48['all52']['total_return'], 3)} versus "
        f"{_percent(selected['all52']['total_return'], 3)}), but fails `no_family_IC_loss_over_0p05`. "
        "prior48_return and prior48_ranking fail that same family noninferiority gate. "
        "The frozen rule deliberately rejects those tradeoffs; it does not simply pick the highest Sharpe.")
    benchmark = comparison["benchmark"]["schwab_aggressive"]["all53"]
    benchmark_note = (
        f"The selected recipe is not a universal benchmark winner: over 53 months its cumulative return is "
        f"{_percent(selected['all53']['total_return'])}, versus {_percent(benchmark['total_return'])} for "
        f"Schwab aggressive. Its conventional Sharpe is {_number(selected['all53']['sharpe'])}, versus "
        f"{_number(benchmark['sharpe'])} for Schwab. Higher risk-adjusted performance did not mean higher "
        "ending wealth in this panel.")
    family_tradeoff = (
        f"The 52-month family tradeoff is also visible: U.S. fixed-income within-family IC falls from "
        f"{_number(incumbent['all52']['family_ic'][2])} to {_number(selected['all52']['family_ic'][2])}, "
        "within the allowed 0.05 loss, while international fixed-income IC increases from "
        f"{_number(incumbent['all52']['family_ic'][3])} to {_number(selected['all52']['family_ic'][3])}. "
        "The pooled ranking gain is not an improvement in every family.")
    decision_lines = [
        "# November 2026 ETF portfolio decision", "", f"**{status}.** {decision}", "",
        f"Selected configuration: **{selected_recipe['label']}** (`{selected_id}`).", "",
        f"Logical as-of date: `{config['as_of_date']}`. Market-data/forecast origin: `{config['market_origin']}`. "
        f"Forecast month: `{config['forecast_month']}`.", "",
        "October 31 is a Saturday; the latest completed regular-session bar is October 30. "
        "No fictitious Saturday market bar or unavailable November training target is used.", "",
        "## Decision and evidence", "",
        f"Against the identically pooled incumbent over the 52-month historical decision panel: "
        f"Sharpe difference {deltas['sharpe']:+.3f}; cumulative-return difference "
        f"{100 * deltas['total_return']:+.2f} percentage points; ranking-IC difference {deltas['rank_ic']:+.3f}.",
        "", promotion_note, "", _gate_table(selected), "",
        selection_note, "", alternative_note, "",
        "The 15-finalist frozen decision table and exact rejection reasons are in COMPARISON.md; "
        "COMPARISON.json preserves the unrounded metrics and gate flags.",
        "", _period_table([selected, incumbent] if selected_id != incumbent_id else [selected],
                             ["earlier24", "later28", "later29", "all52", "all53"]), "",
        benchmark_note, "", family_tradeoff, "",
        "## Exact live policy", "", policy, "", fixed, "",
        "### Fixed prediction blend", "", family_table.to_markdown(index=False), "",
        "These are prediction coefficients, not ETF portfolio weights. They do not change by month or seed. "
        "The covariance pool remains 50/50 even when prediction coefficients differ.", "",
        "## November target weights", "", weight_table.to_markdown(index=False), "",
        "All fifteen assets are included, including zero weights. Percentages above are display-rounded; "
        "NOVEMBER_WEIGHTS.csv retains raw decimal optimizer weights and is the machine-readable source.", "",
        "### Family exposure", "", exposure_table.to_markdown(index=False), "", exposure_note, "",
        "## Validation", "", check_table.to_markdown(index=False), "",
        "See VALIDATION.json for strict solver acceptance, raw-weight feasibility, independent conic parity, "
        "data/target availability, source fingerprints and deterministic reproduction checks. "
        "No infeasible month, fit failure or portfolio was silently substituted.", "",
        "Fit-budget evidence:", "", "```json", json.dumps(config["fit_counts"], indent=2, sort_keys=True), "```", "",
        "## Uncertainty and limitations", "",
        "The earlier/later reversal is real in the legacy saved results: different historical periods favored "
        "different configurations. Much of the pooled ranking reversal was between ETF families; "
        "month-specific exposure changes mattered more than average equity weight alone. "
        "The supporting posthoc diagnosis is outputs/reversal_analysis.md.", "",
        "All examined historical periods and candidate ideas have been inspected during research. "
        "These are retrospective comparisons, not a pristine holdout or a claim of a globally best future portfolio. "
        "NLL and covariance diagnostics do not independently establish ranking or investment value. "
        "Seed variation is numerical sensitivity, not independent market histories or a confidence interval.", "",
        vintage, "",
        "Backtests are gross of explicit trading costs/taxes and use the established return convention. "
        "The plotted fixed-weight Schwab proxy and top7 momentum benchmark are the only standing benchmarks.", "",
        "## Execution timing and scope", "", execution, "",
        "The output is target weights for human review, not orders. No trades or brokerage changes were made; "
        "production defaults and prior experiment evidence remain unchanged.", "",
        "## Reproduce and verify", "", "```sh", reproducer, "```", "",
        "CONFIG.json records exact model inputs/kernels/windows, pooling, covariance, allocator and cutoff. "
        "VALIDATION.json and the reproduction manifest carry the acceptance results and fingerprints.", ""]
    family_rows = []
    for record in records_list:
        values = record["all53"]
        family_ic = np.asarray(values["family_ic"], dtype=float)
        if family_ic.shape != (4,) or not np.isfinite(family_ic).all():
            raise ValueError("Every candidate must supply four finite family_ic diagnostics")
        for family, ic in zip(FAMILIES, family_ic, strict=True):
            family_rows.append({"Configuration": record["label"], "Family": FAMILY_LABELS[family],
                                "Within-family IC": _number(ic)})
    comparison_lines = [
        "# Matched November configuration comparison", "",
        "## Summary", "", "Every GP row is one pooled forecast and one allocation per month—"
        "not a mean of seed portfolios.",
        "The full panel contains 53 origins predicting June 2022--October 2026. "
        "October did not select the recommendation.",
        "", policy, "", fixed, "",
        "## Frozen decision panel: 52 months through September 2026", "",
        selection_note, "", alternative_note, "", alternatives, "",
        "All 15 finalists are shown. Gate failures and eligible-but-dominated alternatives are different "
        "reasons for exclusion. The incumbent is a reference, not a failed challenger. "
        "October outcomes are excluded from this selection table; exact gate IDs and unrounded values "
        "are in COMPARISON.json.", "",
        "## Full 53-month outcomes: June 2022–October 2026", "",
        _comparison_table(records_list, comparison["benchmark"], "all53", selected_id), "",
        benchmark_note, "",
        "## Recent 29-month outcomes: June 2024–October 2026", "",
        _comparison_table(records_list, comparison["benchmark"], "later29", selected_id), "",
        "Gate status refers to the frozen52-month decision panel, not a fresh test using October's return. "
        "Benchmarks are references, not eligible GP candidates. Returns are gross and cumulative.", "",
        "## Earlier/later matched comparison", "", _period_table(records_list, ["earlier24", "later28"]), "",
        "## Family rankings over 53 months", "", pd.DataFrame(family_rows).to_markdown(index=False), "",
        "These are Spearman rankings within each ETF family. They are not pooled rank IC, family Sharpe, "
        "or an attribution of compounded return.", "", family_tradeoff, "",
        "## Visualization", "", "![Matched single-portfolio comparison](COMPARISON.png)", "",
        "Both panels use the same saved return paths. The lower panel resets all methods to$10,000 "
        "at the start of the later 29-month period. There are no seed-confidence bands.", "",
        "## Limitations", "", promotion_note, "", vintage, "", execution, "",
        "Existing histories and candidate ideas were already inspected. See DECISION.md for the chosen "
        "configuration, all 15 target weights, uncertainty, and the exact reproduction command.", ""]
    _plot(config, comparison, records)
    save(HERE / "DECISION.md", ("\n".join(decision_lines)).encode())
    save(HERE / "COMPARISON.md", ("\n".join(comparison_lines)).encode())
