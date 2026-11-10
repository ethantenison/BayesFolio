"""Render supplied November v2 decisions without scoring or choosing policies.

This isolated experiment I/O renderer reads root-owned summaries, raw fractional
weights and saved pooled portfolio paths. Returns are decimals, covariance is
squared decimals, and displayed NAV is dollars. It does not fit, allocate,
compute a new Pareto frontier, inspect extra outcomes, or modify prior evidence.

Usage:
    from v2_report import render
    render(config, validation, comparison, november_weights_dataframe)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from v2_common import ASSETS, FAMILIES, HERE, arrays, save

FAMILY_LABELS = {"us_equity": "U.S. equity", "international_equity": "International equity",
                 "us_fixed_income": "U.S. fixed income", "international_fixed_income": "International fixed income"}
PERIOD_LABELS = {"development": "Development: June 2022–October 2025 (41 months)",
                 "latest12": "Retrospective confirmation: November 2025–October 2026 (12 months)",
                 "full": "Full stitched history: June 2022–October 2026 (53 months)"}


def _percent(value: object, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * float(value):.{digits}f}%"


def _number(value: object, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _json(value: object) -> str:
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported report evidence: {type(item).__name__}")

    return json.dumps(value, indent=2, sort_keys=True, default=convert, allow_nan=False)


def _summary(value: dict) -> dict:
    return value["summary"] if "summary" in value else value


def _metrics(record: dict, period: str) -> dict:
    if period == "latest12":
        return _summary(record["full"]["latest12"])
    return _summary(record[period])


def _metric_row(label: str, values: dict) -> dict:
    return {"Configuration": label, "CAGR": _percent(values.get("cagr")),
            "Cumulative return": _percent(values.get("total_return")),
            "Max drawdown": _percent(values.get("max_drawdown")),
            "Sharpe": _number(values.get("sharpe")), "14-ETF rank IC": _number(values.get("rank_ic")),
            "Mean effective assets": _number(values.get("effective_assets")),
            "Worst month": _percent(values.get("worst_month"))}


def _metrics_table(records: list[dict], benchmarks: dict, period: str) -> str:
    rows = [_metric_row(record["policy"]["label"], _metrics(record, period)) for record in records]
    if period != "development":
        for benchmark in benchmarks.values():
            values = benchmark["latest12"] if period == "latest12" else benchmark["summary"]
            rows.append(_metric_row(benchmark["label"], _summary(values)))
    return pd.DataFrame(rows).to_markdown(index=False)


def _gate_table(record: dict) -> str:
    gates = record["selection"].get("gates", {})
    if not gates:
        return "No additional gate map was supplied; see the frozen root decision evidence."
    return pd.DataFrame([{"Frozen criterion": key, "Passed": bool(value)}
                         for key, value in gates.items()]).to_markdown(index=False)


def _alternatives_table(records: list[dict], selected_id: str, controls: list[str]) -> str:
    rows = []
    for record in records:
        selection = record["selection"]
        failed = [key for key, value in selection.get("gates", {}).items() if not value]
        status = "Selected" if record["id"] == selected_id else (
            "Eligible alternative" if selection["eligible"] else "Ineligible: " + (", ".join(failed) or "frozen rule"))
        values = _metrics(record, "development")
        rows.append({"Configuration": record["policy"]["label"], "Control": record["id"] in controls,
                     "Development CAGR": _percent(values["cagr"]),
                     "Development drawdown": _percent(values["max_drawdown"]),
                     "Frozen distance": _number(selection.get("distance"), 5), "Decision/status": status})
    return pd.DataFrame(rows).to_markdown(index=False)


def _frontier_note(comparison: dict, records: dict[str, dict]) -> str:
    frontier = comparison.get("eligible_frontier_ids", [])
    if not frontier:
        return "The eligible primary-objective frontier is supplied by the frozen root decision."
    tied_zero = all(abs(float(records[identifier]["selection"]["distance"])) <= 1e-8 for identifier in frontier)
    if tied_zero:
        return (f"All {len(frontier)} eligible Pareto candidates have fixed-anchor distance 0: each meets both "
                "one-sided aspirations. The final choice therefore used the frozen tie chain—development annual-block "
                "stability, conventional Sharpe, eligible14 ranking IC, then recipe simplicity/ID. "
                "The selected policy was not a unique CAGR/drawdown dominator.")
    return (f"There are {len(frontier)} eligible primary-objective Pareto candidates. The recorded frozen "
            "distance and tie rule determine the choice; the renderer does not recalculate that frontier.")


def _cost_comparison_table(records: list[dict], benchmarks: dict, identifiers: list[str]) -> str:
    rows = []
    for record in records:
        if record["id"] in identifiers:
            cases = _metrics(record, "full")["cost_cases"]
            for bps in ["0", "10"]:
                values = cases[bps]
                rows.append({"Configuration": record["policy"]["label"], "Cost (bps per traded dollar)": bps,
                             "CAGR": _percent(values["cagr"]), "Cumulative return": _percent(values["total_return"]),
                             "Max drawdown": _percent(values["max_drawdown"])})
    for benchmark in benchmarks.values():
        for bps in ["0", "10"]:
            values = benchmark["cost_sensitivity"][bps]
            rows.append({"Configuration": benchmark["label"], "Cost (bps per traded dollar)": bps,
                         "CAGR": _percent(values["cagr"]), "Cumulative return": _percent(values["total_return"]),
                         "Max drawdown": _percent(values["max_drawdown"])})
    return pd.DataFrame(rows).to_markdown(index=False)


def _distribution_note(comparison: dict, selected_id: str) -> str:
    diagnostics = comparison.get("distribution_diagnostics")
    source = HERE / "DISTRIBUTION_DIAGNOSTICS.json"
    if diagnostics is None and source.exists():
        diagnostics = json.loads(source.read_text())
    if diagnostics is None:
        return "See the linked distribution diagnostics for the actual forecast-error/uncertainty checks."
    full = diagnostics["finalists_eligible14"][selected_id]["blocks"]["full53"]
    allocated, reference = full["allocated_portfolio_projection"], full["fixed_eligible_EW_reference"]
    return (
        f"Across the full53 months, the selected portfolio has mean squared standardized forecast error "
        f"z²={_number(allocated['mean_squared_standardized_error'])} and nominal95% interval coverage "
        f"{_percent(allocated['nominal95_coverage'])}; fixed eligible equal-weight z²="
        f"{_number(reference['mean_squared_standardized_error'])}, versus Gaussian reference values of 1 and 95%. "
        "This forecast-error/uncertainty mismatch can reflect mean and variance errors; it does not establish "
        "calibrated crash protection, and remains diagnostic without fitting, recalibration or selection changes.")


def _policy_tables(policy: dict) -> tuple[str, str]:
    prediction = np.asarray(policy["prediction_fractions"], dtype=float)
    experts = list(policy["prediction_experts"])
    if prediction.shape != (4, len(experts)) or not np.isfinite(prediction).all():
        raise ValueError("Four aligned fixed family prediction-fraction rows are required")
    family_rows = [{"Family": FAMILY_LABELS[family],
                    **{str(expert): _percent(value, 4) for expert, value in zip(experts, row, strict=True)}}
                   for family, row in zip(FAMILY_LABELS, prediction, strict=True)]
    covariance = np.asarray(policy["covariance_fractions"], dtype=float)
    covariance_experts = list(policy["covariance_experts"])
    if covariance.shape == (len(covariance_experts) + 1,):
        covariance_experts.append("causal historical eligible14")
    if covariance.shape != (len(covariance_experts),) or not np.isfinite(covariance).all():
        raise ValueError("One aligned fixed covariance-fraction vector is required")
    covariance_rows = [{"Covariance expert": expert, "Fixed coefficient": _percent(value, 4)}
                       for expert, value in zip(covariance_experts, covariance, strict=True)]
    return pd.DataFrame(family_rows).to_markdown(index=False), pd.DataFrame(covariance_rows).to_markdown(index=False)


def _stress_table(stress: dict) -> str:
    rows = []
    labels = {
        "equity20_FI5": "Assumed equities −20%; all fixed income −5%",
        "equity30_HY15_otherFI5": "Assumed equities −30%; HYG/HYEM −15%; other fixed income −5%",
    }
    for name, shock in stress.get("hypothetical_shocks", {}).items():
        rows.append({"Test": labels.get(name, name), "Period": "Hypothetical simultaneous asset shock",
                     "Portfolio return / drawdown": _percent(shock["portfolio_return"]),
                     "Scope": "Conditional arithmetic loss; no assigned probability"})
    march = stress.get("march2020", {})
    if march.get("available"):
        rows.append({"Test": "November weights applied to March 2020", "Period": "March 2020",
                     "Portfolio return / drawdown": _percent(march["return"]),
                     "Scope": "Static historical sensitivity; no GP forecast"})
    if "static_path_max_drawdown" in stress:
        period = f"{stress.get('historical_first_return_month')}–{stress.get('historical_last_return_month')}"
        rows.append({"Test": "Static monthly-reset path maximum drawdown",
                     "Period": period,
                     "Portfolio return / drawdown": _percent(stress["static_path_max_drawdown"]),
                     "Scope": "Same November targets reset monthly throughout history"})
    for key in ["1", "3", "12"]:
        if key in stress.get("worst_windows", {}):
            window = stress["worst_windows"][key]
            rows.append({"Test": f"Worst static {key}-month cumulative return",
                         "Period": f"{window['start_month']}–{window['end_month']}",
                         "Portfolio return / drawdown": _percent(window["return"]),
                         "Scope": "Fixed November target historical sensitivity"})
    return pd.DataFrame(rows).to_markdown(index=False) if rows else "Static stress evidence was not supplied."


def _paths(path: str, dates: np.ndarray) -> dict:
    source = Path(path)
    artifact = arrays(source if source.is_absolute() else HERE / source)
    np.testing.assert_array_equal(artifact["dates"].astype(str), dates)
    for key in ["wealth", "drawdown"]:
        if np.asarray(artifact[key]).shape != (54,) or not np.isfinite(artifact[key]).all():
            raise ValueError(f"Complete saved 54-point {key} path required")
    if (artifact["wealth"] <= 0).any():
        raise ValueError("Positive complete wealth paths required")
    return artifact


def _plots(comparison: dict, records: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    dates = np.asarray(comparison["dates53"]).astype(str)
    if dates.shape != (53,):
        raise ValueError("Exactly 53 historical forecast origins required")
    origins = pd.to_datetime(dates)
    endpoints = origins + pd.offsets.BMonthEnd(1)
    x = pd.DatetimeIndex([origins[0], *endpoints])
    selected = comparison["selected_id"]
    controls = comparison["control_ids"]
    identifiers = list(dict.fromkeys([*controls, selected]))
    series = []
    for index, identifier in enumerate(identifiers):
        record = records[identifier]
        is_selected = identifier == selected
        color = "#17865b" if is_selected else ["#2456a6", "#7d5aab", "#008c99"][index % 3]
        label = ("Selected: " if is_selected else "Control: ") + record["policy"]["label"]
        series.append((label, color, _paths(record["artifact_path"], dates), False, is_selected))
    for key, color in [("schwab_aggressive", "#a37b18"), ("momentum_top7", "#64686c")]:
        benchmark = comparison["benchmarks"][key]
        series.append((benchmark["label"], color, _paths(benchmark["artifact_path"], dates), True, False))

    figure, axes = plt.subplots(2, 1, figsize=(14, 10.4))
    for axis, start, title in zip(axes, [0, 41], ["Full history: June 2022–October 2026",
                                                "Retrospective confirmation: November 2025–October 2026"], strict=True):
        for label, color, artifact, benchmark, chosen in series:
            wealth = artifact["wealth"][start:] / artifact["wealth"][start] * 10000
            axis.plot(x[start:], wealth, label=f"{label} | ${wealth[-1]:,.0f}", color=color,
                      linewidth=2.8 if chosen else 1.8, linestyle="--" if benchmark else "-",
                      zorder=4 if chosen else 2)
        axis.set_title(title + " | reset to $10,000", fontsize=11)
        axis.set_ylabel("Investment value ($)")
        axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6 if start == 0 else 2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="upper left")
        axis.set_xlim(x[start], x[-1])
    axes[1].set_xlabel("Realized month-end value")
    figure.suptitle("November v2 | One pooled forecast and one allocation per rebalance", fontsize=14)
    figure.text(.5, .014,
                "Gross of explicit costs/taxes. MGK is a GP training task with zero allocation. "
                "Only Schwab aggressive and simple momentum are standing benchmarks.\n"
                "The latest12 panel was locked during development, but prior research inspected this history; "
                "it is retrospective confirmation, not a pristine holdout. No seed-confidence bands.",
                ha="center", fontsize=8.5)
    figure.tight_layout(rect=(0, .065, 1, .95))
    _save_figure(figure, "COMPARISON.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5.4))
    for label, color, artifact, benchmark, chosen in series:
        axis.plot(x, artifact["drawdown"], label=label, color=color, linewidth=2.6 if chosen else 1.6,
                  linestyle="--" if benchmark else "-", zorder=4 if chosen else 2)
    axis.set_title("Stitched drawdown | initial investment and all 53 return months retained")
    axis.set_ylabel("Drawdown from running high")
    axis.set_xlabel("Realized month-end")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.legend(fontsize=8, loc="lower left")
    axis.grid(alpha=.2)
    figure.tight_layout()
    _save_figure(figure, "DRAWDOWN.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for axis, period, title in zip(axes, ["development", "full"],
                                 ["Development41", "Full53 descriptive outcomes"], strict=True):
        for index, record in enumerate(records.values(), start=1):
            values = _metrics(record, period)
            chosen, control = record["id"] == selected, record["id"] in controls
            eligible = record["selection"]["eligible"]
            color = "#17865b" if chosen else ("#2456a6" if control else ("#777e85" if eligible else "#b9665c"))
            marker = "*" if chosen else ("s" if control else ("o" if eligible else "x"))
            axis.scatter(-values["max_drawdown"], values["cagr"], color=color, marker=marker,
                         s=180 if chosen else 45, zorder=4 if chosen else 2)
            axis.annotate(str(index), (-values["max_drawdown"], values["cagr"]), xytext=(5, 5),
                          textcoords="offset points", fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("Maximum drawdown loss magnitude (smaller is better)")
        axis.set_ylabel("CAGR (larger is better)")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.2)
    figure.suptitle("Primary objective plane | recorded finalists and controls", fontsize=13)
    figure.text(.5, .015,
                "Numbering follows the finalist table. Green star: selected; blue square: control; "
                "gray circle: eligible alternative; red cross: ineligible.\n"
                "Selection flags come from the frozen root decision. This renderer does not recalculate a frontier. "
                "Confirmation/full-history results did not reopen the search.", ha="center", fontsize=8.5)
    figure.tight_layout(rect=(0, .09, 1, .94))
    _save_figure(figure, "PARETO.png")
    plt.close(figure)


def _save_figure(figure, name: str) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170)
    save(HERE / name, buffer.getvalue())


def render(config: dict, validation: dict, comparison: dict, weights_frame: pd.DataFrame) -> None:
    """Write reports/figures from supplied decisions and immutable saved paths.

    Args:
        config: Root-selected policy, exact GP specs/settings, timing, search/fit
            counts, execution note, frozen rule/reason and reproduction command.
        validation: Root-owned acceptance results; passed must be explicit to
            label final November weights validated.
        comparison: Finalist records with development/full/latest12 summaries,
            frozen selection flags, two benchmarks, dates53 and saved NPZ paths.
        weights_frame: All fifteen aligned asset_id and decimal weight rows;
            MGK must remain exactly zero. No rounding changes the source weights.

    Raises:
        ValueError: Incomplete/misaligned evidence or conflicting selected IDs.
    """
    records_list = comparison["records"]
    records = {record["id"]: record for record in records_list}
    selected_id = comparison["selected_id"]
    if len(records) != len(records_list) or selected_id not in records:
        raise ValueError("Unique finalists and explicit selected evidence required")
    if set(comparison["benchmarks"]) != {"schwab_aggressive", "momentum_top7"}:
        raise ValueError("Exactly the two approved standing benchmarks required")
    if weights_frame["asset_id"].astype(str).tolist() != ASSETS:
        raise ValueError("All fifteen target weights must retain approved asset order")
    weights = pd.to_numeric(weights_frame["weight"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(weights).all() or weights[ASSETS.index("MGK")] != 0:
        raise ValueError("Finite target fractions and exact MGK zero weight required")
    selected = records[selected_id]
    policy = selected["policy"]
    supplied_policy = config["selected_policy"]
    if supplied_policy.get("id", selected_id) != selected_id:
        raise ValueError("Configuration and supplied selected policy disagree")
    prediction_table, covariance_table = _policy_tables(policy)
    weights_table = pd.DataFrame({"ETF": ASSETS, "Weight (decimal)": [f"{value:.12f}" for value in weights],
                                 "Weight (%)": [_percent(value, 4) for value in weights]})
    family_exposure = {name: float(weights[[ASSETS.index(asset) for asset in members]].sum())
                       for name, members in FAMILIES.items()}
    exposure_table = pd.DataFrame([{"Family": FAMILY_LABELS[name], "November target weight": _percent(value, 4)}
                                   for name, value in family_exposure.items()])
    stress = comparison.get("stress", {})
    stress_table = _stress_table(stress)
    maximum_asset = ASSETS[int(np.argmax(weights))]
    fixed_income_exposure = family_exposure["us_fixed_income"] + family_exposure["international_fixed_income"]
    concentration_note = (
        f"**Current target concentration:** {maximum_asset} is {_percent(float(weights.max()), 3)}; "
        f"equities are {_percent(family_exposure['us_equity'] + family_exposure['international_equity'], 3)} "
        f"and fixed income {_percent(fixed_income_exposure, 3)}. "
        f"HYEM alone is {_percent(float(weights[ASSETS.index('HYEM')]), 3)}. "
        "The selected policy's effective-count minimum is a concentration measure, not an equity/fixed-income balance.")
    shocks = stress.get("hypothetical_shocks", {})
    if "equity20_FI5" in shocks and "equity30_HY15_otherFI5" in shocks:
        concentration_note += (
            f" Under the two specified common-shock assumptions, the current targets return "
            f"{_percent(shocks['equity20_FI5']['portfolio_return'])} and "
            f"{_percent(shocks['equity30_HY15_otherFI5']['portfolio_return'])}, respectively. "
            "**Lower historical drawdown does not establish protection from a common future equity shock.**")
    stress_note = (
        "The hypothetical rows apply the stated asset losses directly to the frozen November target weights. "
        "The historical rows reset those same targets monthly against adjusted total-return proxy history; "
        "they use no historical GP forecasts and are not a backtest of the GP allocation strategy. "
        "These conditional/static losses have no assigned probability, calibrated future-crash interpretation "
        "or assumed recovery path. They do not establish a personally acceptable loss tolerance. "
        "The frozen selection rule and constraints were maintained after inspecting the November weights "
        "and these diagnostics; no new selection or constraint was introduced.")
    stress_assumptions = "\n".join(f"- {assumption}" for assumption in stress.get("assumptions", []))
    distribution_note = (
        "[Distribution diagnostics](DISTRIBUTION_DIAGNOSTICS.md) and "
        "[machine-readable evidence](DISTRIBUTION_DIAGNOSTICS.json) examine predictive distributions only. "
        "They do not trigger fitting, uncertainty recalibration, a changed selection rule or another policy search.")
    periods = ["development", "latest12", "full"]
    period_rows = [{"Period": PERIOD_LABELS[period], **{key: value for key, value in
                    _metric_row(policy["label"], _metrics(selected, period)).items() if key != "Configuration"}}
                   for period in periods]
    controls = comparison["control_ids"]
    frontier_note = _frontier_note(comparison, records)
    reason = config.get("decision_reason", "The root-owned frozen decision selected this complete policy; "
                        "the exact gate flags and distance records below show its recorded acceptance.")
    rule = config.get("selection_rule", "See the immutable preregistration and root selection artifacts.")
    status = "Validated November target-weight recommendation" if validation.get("passed") is True else (
        "PROVISIONAL—validation gates have not all passed")
    costs = _metrics(selected, "full").get("cost_cases", {})
    cost_table = (pd.DataFrame([{"Cost (bps per traded dollar)": key,
                                "Cumulative return": _percent(value.get("total_return")),
                                "CAGR": _percent(value.get("cagr")),
                                "Max drawdown": _percent(value.get("max_drawdown")),
                                "Sharpe": _number(value.get("sharpe"))} for key, value in costs.items()])
                  .to_markdown(index=False) if costs else "See the recorded transaction-cost diagnostic artifacts.")
    cost_comparison = _cost_comparison_table(records_list, comparison["benchmarks"], [selected_id, *controls])
    selected_cost = _metrics(selected, "full")["cost_cases"]
    schwab_cost = comparison["benchmarks"]["schwab_aggressive"]["cost_sensitivity"]
    cost_tradeoff = (
        f"The gross cumulative-return comparison is {_percent(selected_cost['0']['total_return'])} selected versus "
        f"{_percent(schwab_cost['0']['total_return'])} Schwab aggressive. At 10 bps per gross traded dollar it is "
        f"{_percent(selected_cost['10']['total_return'])} versus {_percent(schwab_cost['10']['total_return'])}; "
        "the selected policy's return edge reverses in that cost case. Its recorded maximum drawdown at 10 bps "
        f"is {_percent(selected_cost['10']['max_drawdown'])}, versus "
        f"{_percent(schwab_cost['10']['max_drawdown'])} for Schwab. Costs remain the predeclared sensitivity "
        "check rather than a new selection objective; the frozen policy is unchanged.")
    raw_checks = validation.get("portfolio", {}).get("raw_checks", {})
    actual_ne = raw_checks.get("effective_assets", float(1 / np.sum(weights**2)))
    raw_note = (
        f"Actual November effective assets are {_number(actual_ne, 5)}, above the policy floor "
        f"{policy['minimum_effective_assets']}. The minimum raw optimizer weight is {float(weights.min()):.3g}, "
        "within the predeclared −1e−8 numerical tolerance; it is preserved without clipping or renormalization.")
    if "variance_to_ceiling" in raw_checks:
        raw_note += (f" The variance-ceiling ratio is {_number(raw_checks['variance_to_ceiling'], 10)}; "
                     "the risk ceiling is active within the recorded numerical acceptance tolerance.")
    distribution_actual = _distribution_note(comparison, selected_id)
    timing = config.get("execution_note", config.get("executionnote", "See CONFIG.json for execution timing."))
    reproduce = config.get("reproduction_command", config.get("reprocmd", "See REPRODUCTION.md."))
    settings = {key: config[key] for key in ["GPspecs", "GP_specs", "asof", "cutoff", "fit_counts", "fitcounts",
                                            "search_counts", "searchcounts", "policy_settings", "policysettings"]
                if key in config}
    family_ic = _metrics(selected, "full").get("family_ic", {})
    family_ic_table = pd.DataFrame([{"Family": FAMILY_LABELS[key], "Full53 within-family IC": _number(value)}
                                    for key, value in family_ic.items()]).to_markdown(index=False)
    base_policy = (
        "Predictions and covariance are pooled equally across seeds 27, 127 and 227 before one allocation per month. "
        "The static family prediction blend and covariance recipe below apply across dates and seeds. "
        "MGK remains one of the 15 GP training tasks, receives exactly zero portfolio weight, and is excluded "
        "from the 14-ETF pooled/family ranking diagnostics. The allocator uses the eligible principal covariance.")
    objectives = (
        "Compounded return/CAGR and drawdown are the primary objectives. Conventional Sharpe and fourteen-ETF "
        "ranking IC are secondary evidence, with NLL/RMSE diagnostic. IC is not a veto that automatically "
        "discards a useful return/drawdown tradeoff. No turnover objective, CVaR, uncertainty recalibration, "
        "monthly discretionary blending, negative-return floor or previous-weight fallback is used.")
    chronology = (
        "Development includes all first 41 return months. The first five, June–October 2022, remain a partial "
        "diagnostic block and remain in every stitched wealth/drawdown path; they are not discarded. The three "
        "complete development blocks are November 2022–October 2023, November 2023–October 2024 and "
        "November 2024–October 2025. The latest12 block, November 2025–October 2026, was locked while this "
        "round's policies were developed and the shortlist/rule frozen. Older research already inspected these "
        "dates: final confirmation is retrospective, not a pristine independent holdout. No retuning follows it.")
    disclosure = (
        "The reference experts retain their established economic features, transformations, target definition, "
        "publication lags and training-window anchors: compact expanding, broad expanding and broad rolling48. "
        "Rolling48 means 48 labeled training months, not a 12-month evaluation block. The exact evaluated "
        "GP ranks, kernels/priors and fitting settings—including any explicitly registered limited variant—" 
        "are recorded in CONFIG.json/GP specs. No unrecorded kernel, feature, window or iteration change "
        "is implied by a different allocator or blend. t_index remains in the model; raw hy_spread and "
        "its change are not reintroduced.")
    risk_note = (
        f"The selected hard-rank policy uses minimum effective assets {policy['minimum_effective_assets']}, "
        f"risk multiplier {policy['risk_multiplier']}, and equity cap "
        f"{_percent(policy.get('equity_cap')) if policy.get('equity_cap') is not None else 'none'}. "
        "Effective count measures concentration; it does not itself guarantee an equity/fixed-income balance. "
        "There is no ETF position cap. The causal risk-ceiling recipe and any static exposure constraint "
        "are fixed parts of the selected policy, with feasibility recorded in VALIDATION.json.")
    limitations = (
        "These are retrospective research paths, gross of explicit costs/taxes unless a cost row says otherwise. "
        "Development selection, temporal dependence, prior inspection and limited sample size constrain the "
        "strength of inference. Paired block-bootstrap ranges are sensitivity diagnostics, not calibrated future "
        "win probabilities or crash protection. Seed variation is numerical sensitivity, not independent "
        "market evidence. Better NLL or a more attractive November allocation does not prove future superiority.")
    design_notes = config.get("design_notes")
    design_note_text = design_notes if isinstance(design_notes, str) else "```json\n" + _json(design_notes) + "\n```"
    design_note_lines = (["## Registered design limitations", "",
                          design_note_text, ""]
                         if design_notes is not None else [])
    decision_lines = [
        "# November 2026 multiobjective portfolio decision", "", f"**{status}.**", "",
        f"Selected complete policy: **{policy['label']}** (`{selected_id}`).", "", str(reason), "",
        concentration_note, "",
        "## Decision evidence", "", objectives, "", "Frozen selection rule:", "", str(rule), "",
        frontier_note, "", _gate_table(selected), "", pd.DataFrame(period_rows).to_markdown(index=False), "",
        cost_tradeoff, "",
        "The complete finalist alternatives table is in COMPARISON.md; root-owned flags and distances "
        "are reported without recomputing a winner.", "",
        "## Exact model and allocation policy", "", base_policy, "", risk_note, "", disclosure, "",
        "### Fixed prediction coefficients", "", prediction_table, "",
        "These are prediction coefficients, not portfolio weights.", "",
        "### Fixed covariance coefficients", "", covariance_table, "",
        "## November ETF target weights", "", weights_table.to_markdown(index=False, disable_numparse=True), "",
        "All 15 rows, including zeros and MGK=0, are retained. Percentages are display-rounded only; "
        "NOVEMBER_WEIGHTS.csv preserves unmodified decimal optimizer output.", "",
        "### Family exposures", "", exposure_table.to_markdown(index=False), "", raw_note, "",
        "## Conditional and static stress of the November target", "", stress_table, "", stress_note, "",
        distribution_actual, "",
        stress_assumptions, "", distribution_note, "",
        "## Timing, validation and reproduction", "",
        "Logical as-of October 31, 2026 is a Saturday. The latest completed regular-session market bar is "
        "October 30, predicting November; no unavailable November target is used for training.", "",
        str(timing), "",
        "The gap between Friday's information and potential Monday November 2 execution matters: a decision "
        "made after the close cannot assume a retrospective Friday-close fill. Target weights are not orders.", "",
        "VALIDATION.json records freshness/availability, raw-weight numerical tolerances, effective count, "
        "risk/exposure feasibility, strict solver acceptance, independent checks and reproducibility. "
        "Accepted numerical residuals remain raw; no clipping or renormalization is applied.", "",
        "```sh", str(reproduce), "```", "",
        "## Robustness, costs and limitations", "", chronology, "", cost_table, "",
        "Cost rows charge the first cash-to-assets purchase and subsequent drift-aware purchases plus sales, "
        "per gross dollar traded, with exact self-financing funding. They are diagnostic, not objectives.", "",
        family_ic_table, "", limitations, "", *design_note_lines,
        "No trades, brokerage changes, commits or pushes were made. Prior datasets/experiments and production "
        "defaults are protected. See REPRODUCTION.md, the experiment ledger and STATUS.json for exact evidence.", "",
        "## Recorded settings and budgets", "", "```json", _json(settings), "```", "",
    ]
    alternatives = _alternatives_table(records_list, selected_id, controls)
    numbered = pd.DataFrame([{"Figure number": index, "Policy ID": record["id"],
                              "Configuration": record["policy"]["label"]}
                             for index, record in enumerate(records_list, start=1)]).to_markdown(index=False)
    comparison_lines = [
        "# November v2 matched policy comparison", "", objectives, "", base_policy, "", chronology, "",
        "## November target concentration and conditional stress", "", concentration_note, "", raw_note, "",
        stress_table, "", stress_note, "", distribution_actual, "", stress_assumptions, "", distribution_note, "",
        "## Frozen development decision and alternatives", "", str(rule), "", str(reason), "", frontier_note, "",
        alternatives, "",
        "Eligible alternatives were not automatically selected. Exact recorded distances/gates, rather than "
        "a newly calculated objective ordering, explain the frozen decision.", "",
        "## Development outcomes", "", _metrics_table(records_list, comparison["benchmarks"], "development"), "",
        "## Latest twelve-month retrospective confirmation", "",
        _metrics_table(records_list, comparison["benchmarks"], "latest12"), "",
        "## Full stitched 53-month outcomes", "", _metrics_table(records_list, comparison["benchmarks"], "full"), "",
        "Only Schwab aggressive and eligible-14 standalone simple momentum are standing benchmarks. "
        "Control model policies are research comparisons, not additional standing benchmarks.", "",
        "## Primary objective plane", "", "![Primary objective plane](PARETO.png)", "", numbered, "",
        "## Cumulative wealth and stitched drawdown", "", "![Cumulative wealth](COMPARISON.png)", "",
        "![Stitched drawdown](DRAWDOWN.png)", "",
        "Each curve is one pooled forecast followed by one allocation per date. It is not an average of "
        "separately optimized seed portfolios. The latest12 wealth panel resets to $10,000; stitched "
        "drawdown retains the original investment peak and all 53 months.", "",
        "## Selected policy robustness evidence", "", "```json", _json(selected.get("robustness", {})), "```", "",
        "## Matched cost sensitivity: selected, controls and benchmarks", "", cost_comparison, "", cost_tradeoff, "",
        "## Selected policy cost sensitivity", "", cost_table, "",
        "0/5/10/20 bps per gross traded dollar are diagnostic fixed-policy replays, not a turnover objective. "
        "Initial purchases are charged; subsequent trades use drifted pretrade holdings. No failed month "
        "or negative return is silently dropped or floored.", "",
        "## Limitations", "", limitations, "", *design_note_lines, str(timing), "",
        "See DECISION.md for the selected policy and all fifteen November target weights. Source/data/code "
        "hashes, raw moments/optimizer evidence and one-command verification are in the reproduction artifacts.", "",
    ]
    _plots(comparison, records)
    save(HERE / "DECISION.md", ("\n".join(decision_lines)).encode())
    save(HERE / "COMPARISON.md", ("\n".join(comparison_lines)).encode())
