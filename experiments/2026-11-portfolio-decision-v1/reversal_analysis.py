"""Audit the previously evaluated chronological GP performance reversal.

Experiment-only I/O orchestration reads sealed forecasts, outcomes and legacy
seedwise allocations. It replays only the two previously tested earlier-period
recipes to recover their missing monthly weights, never fits a GP or proposes a
new portfolio candidate. Returns are decimal, risk squared decimal, weights
fractions. The separate November pool-first decision is not evaluated here.

Usage:
    uv run --locked --no-sync python <this_file> --execute
    uv run --locked --no-sync python <this_file> --verify
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE.parent / "2026-10-gp-chronological-selection-v1/outputs"
RANK_SOURCE = HERE.parent / "2026-10-gp-tuning-v1/three_way_joint_v1/joint_math.py"
FAMILY_LABELS = ["us_equity", "international_equity", "us_fixed_income", "international_fixed_income"]
FAMILY = np.repeat(np.arange(4), [4, 4, 4, 3])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen local helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rank_ic(actual: np.ndarray, mean: np.ndarray) -> np.ndarray:
    predicted = rankdata(mean, axis=-1)
    realized = rankdata(actual, axis=-1)
    predicted -= predicted.mean(-1, keepdims=True)
    realized -= realized.mean(-1, keepdims=True)
    denominator = np.sqrt(np.sum(predicted**2, -1) * np.sum(realized**2, -1))
    if (denominator <= 0).any():
        raise ValueError("Undefined ranking diagnostics; no month may silently disappear")
    return np.einsum("sta,ta->st", predicted, realized) / denominator


def _forecast_summary(actual: np.ndarray, mean: np.ndarray) -> dict:
    result = {"pooled_rank_ic": float(_rank_ic(actual, mean).mean())}
    for family, label in enumerate(FAMILY_LABELS):
        mask = FAMILY == family
        result[label] = float(_rank_ic(actual[:, mask], mean[:, :, mask]).mean())
    result["rmse"] = float(np.sqrt(np.mean((mean - actual)**2)))
    return result


def _rank_decomposition(actual: np.ndarray, mean: np.ndarray) -> dict:
    # This exactly decomposes covariance of GLOBAL ETF ranks, not the average
    # of four within-family Spearman correlations. These are accounting terms.
    predicted = rankdata(mean, axis=-1)
    realized = rankdata(actual, axis=-1)
    predicted -= predicted.mean(-1, keepdims=True)
    realized -= realized.mean(-1, keepdims=True)
    denominator = np.sqrt(np.sum(predicted**2, -1) * np.sum(realized**2, -1))
    within, between = np.zeros(mean.shape[:2]), np.zeros(mean.shape[:2])
    for family in range(4):
        mask = FAMILY == family
        x, y = predicted[:, :, mask], realized[:, mask]
        x_center, y_center = x.mean(-1), y.mean(-1)
        within += np.einsum("sta,ta->st", x - x_center[:, :, None], y - y_center[:, None])
        between += int(mask.sum()) * x_center * y_center
    pooled = _rank_ic(actual, mean)
    np.testing.assert_allclose((within + between) / denominator, pooled, rtol=0, atol=1e-14)
    return {"within_family_global_rank_component": float((within / denominator).mean()),
            "between_family_global_rank_component": float((between / denominator).mean()),
            "pooled_rank_ic": float(pooled.mean()),
            "interpretation": "Exact global-rank covariance decomposition; not causal importance or mean family IC."}


def _performance(actual: np.ndarray, rf: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> dict:
    total = (1 + np.einsum("sta,ta->st", weights, actual)) * (1 + rf) - 1
    excess = total - rf
    std = excess.std(1, ddof=1)
    if (std <= 0).any() or not np.isfinite(total).all() or (total <= -1).any():
        raise ValueError("Undefined complete-period performance")
    return {"months": len(actual), "sharpe": float((np.sqrt(12) * excess.mean(1) / std).mean()),
            "total_return": float((np.prod(1 + total, axis=1) - 1).mean()),
            "rank_ic": float(_rank_ic(actual, mean).mean()),
            "annualized_excess_volatility": float((np.sqrt(12) * std).mean()),
            "family_weights": {label: float(weights[:, :, FAMILY == i].sum(-1).mean())
                               for i, label in enumerate(FAMILY_LABELS)},
            "effective_assets": float((1 / np.sum(weights**2, -1)).mean())}


def _attribution(context: dict, base: dict, selected: dict) -> dict:
    actual, rf = context["actual"], context["risk_free"]
    gap = (base["weights"] - selected["weights"]) * actual[None] * (1 + rf)[None, :, None]
    total_base = (1 + np.einsum("sta,ta->st", base["weights"], actual)) * (1 + rf) - 1
    total_selected = (1 + np.einsum("sta,ta->st", selected["weights"], actual)) * (1 + rf) - 1
    np.testing.assert_allclose(gap.sum(-1), total_base - total_selected, rtol=0, atol=5e-16)
    monthly = gap.sum(-1).mean(0)
    family = {}
    for i, label in enumerate(FAMILY_LABELS):
        mask = FAMILY == i
        family[label] = {"baseline_mean_weight": float(base["weights"][:, :, mask].sum(-1).mean()),
                         "selected_mean_weight": float(selected["weights"][:, :, mask].sum(-1).mean()),
                         "mean_monthly_gap_basis_points": float(gap[:, :, mask].sum(-1).mean() * 1e4)}
    assets = [{"asset": str(asset), "baseline_mean_weight": float(base["weights"][:, :, i].mean()),
               "selected_mean_weight": float(selected["weights"][:, :, i].mean()),
               "mean_monthly_gap_basis_points": float(gap[:, :, i].mean() * 1e4)}
              for i, asset in enumerate(context["assets"])]
    month_rows = []
    for t, date in enumerate(context["dates"]):
        month_rows.append({"origin": str(date),
                           "realized_month": (pd.Timestamp(str(date)) + pd.offsets.BMonthEnd(1)).strftime("%Y-%m"),
                           "mean_gap_basis_points": float(monthly[t] * 1e4),
                           "baseline_mean_total_return": float(total_base[:, t].mean()),
                           "selected_mean_total_return": float(total_selected[:, t].mean()),
                           "baseline_family_weights": {label: float(base["weights"][:, t, FAMILY == i].sum(-1).mean())
                                                       for i, label in enumerate(FAMILY_LABELS)},
                           "selected_family_weights": {
                               label: float(selected["weights"][:, t, FAMILY == i].sum(-1).mean())
                               for i, label in enumerate(FAMILY_LABELS)}})
    leave_one_out = []
    for t, date in enumerate(context["dates"]):
        keep = np.arange(len(actual)) != t
        b = _performance(actual[keep], rf[keep], base["mean"][:, keep], base["weights"][:, keep])
        s = _performance(actual[keep], rf[keep], selected["mean"][:, keep], selected["weights"][:, keep])
        leave_one_out.append({"omitted_origin": str(date), **{
            "baseline_minus_selected_" + key: b[key] - s[key] for key in ["sharpe", "total_return", "rank_ic"]}})
    summed = float(monthly.sum())
    return {"direction": "baseline minus earlier-selected blend",
            "mean_monthly_gap_basis_points": float(monthly.mean() * 1e4), "families": family, "assets": assets,
            "monthly": month_rows, "positive_months": int((monthly > 0).sum()),
            "negative_months": int((monthly < 0).sum()),
            "largest_positive_month_share_of_summed_arithmetic_gap": float(monthly.max() / summed)
            if abs(summed) > 1e-14 else None,
            "leave_one_month_out": leave_one_out,
            "interpretation": "Arithmetic return-gap accounting only; not attribution of Sharpe or compounded return. "
                              "Leave-one-month-out is a diagnostic, never omission from the actual backtest."}


def _write(path: Path, value: dict | str) -> None:
    if not path.resolve().is_relative_to((HERE / "outputs").resolve()) or not path.name.startswith("reversal"):
        raise ValueError("Only this agent's new reversal outputs are writable")
    content = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Conflicting immutable reversal evidence")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def analyze(output: Path, frozen_plan: Path) -> dict:
    """Audit existing reversal paths and replay only missing legacy early weights.

    Args:
        output: This experiment's new outputs directory.
        frozen_plan: Already frozen November preregistration; required before replay.

    Returns:
        Reproducible diagnostics, hashes and limitations. No November selection
        decision, pooled live policy, new candidate or GP fit is made here.
    """
    if not frozen_plan.exists():
        raise ValueError("November candidate menu and decision gates must be frozen first")
    plan = _read(frozen_plan)
    if not plan:
        raise ValueError("Empty preregistration cannot authorize replay")
    contract_path = SOURCE / "CONTRACT.json"
    baseline_path = HERE.parent / "2026-10-portfolio-evaluation-v1/CURRENT_BASELINES.json"
    paths = [contract_path, baseline_path, SOURCE / "context52.npz", SOURCE / "selections/fold_0.json",
             SOURCE / "selections/fold_0_search.json", SOURCE / "cases/current_baseline.npz",
             SOURCE / "cases/locked28_sharpe.npz", RANK_SOURCE, frozen_plan, Path(__file__).resolve()]
    contract, baseline = _read(contract_path), _read(baseline_path)
    names = list(contract["configs"])
    paths.extend(SOURCE / "components" / f"{name}.npz" for name in names)
    hashes = {str(path.resolve()): _digest(path) for path in paths}
    context = _arrays(SOURCE / "context52.npz")
    panels = [_arrays(SOURCE / "components" / f"{name}.npz") for name in names]
    for panel in panels:
        for key in ["dates", "assets", "seeds"]:
            np.testing.assert_array_equal(panel[key], context[key])
    if context["actual"].shape != (52, 15) or context["seeds"].tolist() != [27, 127, 227]:
        raise ValueError("Exact legacy forecast ledger required")
    all_means = np.stack([panel["mean"] for panel in panels])
    incumbent = next(item for item in baseline["baselines"] if item["id"] == "sharpe_leader")
    alpha = np.array([incumbent["compact_mean_fraction_by_family"][label] for label in FAMILY_LABELS])[FAMILY]
    base_mean = (1 - alpha) * all_means[names.index("broad_r2_i15_exp")] + (
        alpha * all_means[names.index("compact_r2_i15_exp")])
    selection = _read(SOURCE / "selections/fold_0.json")["selected"]["sharpe"]
    counts = np.array(selection["recipe"]["counts"])
    locked_mean = np.einsum("ksta,ak->sta", all_means, counts[FAMILY] / 64)
    covariance = .5 * (panels[names.index("broad_r2_i15_exp")]["covariance"]
                       + panels[names.index("compact_r2_i15_exp")]["covariance"])
    old_base, old_locked = _arrays(SOURCE / "cases/current_baseline.npz"), _arrays(SOURCE / "cases/locked28_sharpe.npz")
    np.testing.assert_allclose(base_mean[:, 24:], old_base["mean"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(locked_mean[:, 24:], old_locked["mean"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(covariance[:, 24:], old_base["covariance"], rtol=0, atol=1e-15)
    allocator = _module(RANK_SOURCE, "_november_reversal_legacy_rank").RankAllocator()
    early_base, base_constrained = allocator.allocate(
        base_mean[:, :24], covariance[:, :24], context["risk_budget"][:24])
    early_locked, locked_constrained = allocator.allocate(
        locked_mean[:, :24], covariance[:, :24], context["risk_budget"][:24])
    base_weights = np.concatenate([early_base, old_base["weights"]], axis=1)
    locked_weights = np.concatenate([early_locked, old_locked["weights"]], axis=1)
    search = _read(SOURCE / "selections/fold_0_search.json")
    base_counts = np.zeros((4, len(names)), dtype=int)
    base_counts[:, names.index("compact_r2_i15_exp")] = np.rint(alpha[[0, 4, 8, 12]] * 64).astype(int)
    base_counts[:, names.index("broad_r2_i15_exp")] = 64 - base_counts[:, names.index("compact_r2_i15_exp")]
    saved_base = next(row for row in search if row["recipe"]["counts"] == base_counts.tolist())
    for mean, weights, expected in [(base_mean, base_weights, saved_base), (locked_mean, locked_weights, selection)]:
        score = _performance(context["actual"][:24], context["risk_free"][:24], mean[:, :24], weights[:, :24])
        for key in ["sharpe", "total_return", "rank_ic"]:
            np.testing.assert_allclose(score[key], expected[key], rtol=0, atol=1e-10)
    models = {"baseline": base_mean, "earlier_selected": locked_mean,
              **{name: all_means[names.index(name)] for name in
                 ["broad_r2_i15_exp", "broad_roll48_r2_i15", "rich_r2_i15_exp", "rich_roll48_r2_i15"]}}
    periods = {"earlier24": slice(0, 24), "later28": slice(24, 52)}
    diagnostics = {}
    for label, mean in models.items():
        diagnostics[label] = {}
        for period, section in periods.items():
            diagnostics[label][period] = _forecast_summary(context["actual"][section], mean[:, section])
            diagnostics[label][period]["rank_decomposition"] = _rank_decomposition(
                context["actual"][section], mean[:, section])
    portfolio, attribution = {}, {}
    for period, section in periods.items():
        actual, rf = context["actual"][section], context["risk_free"][section]
        portfolio[period] = {"baseline": _performance(actual, rf, base_mean[:, section], base_weights[:, section]),
                             "earlier_selected": _performance(actual, rf, locked_mean[:, section],
                                                               locked_weights[:, section])}
        local = {**context, "actual": actual, "risk_free": rf, "dates": context["dates"][section]}
        attribution[period] = _attribution(local,
            {"mean": base_mean[:, section], "weights": base_weights[:, section]},
            {"mean": locked_mean[:, section], "weights": locked_weights[:, section]})
    result = {"schema": "bayesfolio.november.reversal_diagnostic.v1", "passed": True,
              "policy": "legacy three separately allocated seeds; not November pooled-first policy",
              "inputs_sha256": hashes, "forecast_diagnostics": diagnostics,
              "portfolio_performance": portfolio, "arithmetic_return_gap_attribution": attribution,
              "early_replay_parity_with_saved_metrics": True,
              "early_risk_constrained_pairs": {"baseline": base_constrained, "selected": locked_constrained},
              "training_window_counts": {"expanding_earlier": [60, 83], "expanding_later": [84, 111],
                                          "rolling_both_periods": 48},
              "earlier_selected_family_recipe": [
                  {"family": FAMILY_LABELS[i], "experts": [{"id": names[j], "fraction": int(value) / 64}
                    for j, value in enumerate(row) if value]} for i, row in enumerate(counts)],
              "limitations": ["Posthoc accounting on inspected history; no causal identification or pristine holdout.",
                              "Seeds are numerical repetitions of one market history.",
                              "Earlier-selected recipe includes rank1/maxiter5 experts; diagnostic only, "
                              "not an eligible November finalist.",
                              "Fixed48 experts change rank performance across periods: expanding-history growth "
                              "cannot alone explain the reversal, but regime/model/sample effects remain confounded.",
                              "Global rank between-family contribution is not average within-family rank IC.",
                              "Family return contributions sum arithmetic return gaps, "
                              "not compounded return or Sharpe."]}
    for name, expected in hashes.items():
        if _digest(Path(name)) != expected:
            raise ValueError(f"Protected source changed during audit: {name}")
    destination = output / "reversal_analysis.json"
    _write(destination, result)
    _write(output / "reversal_analysis.md", _markdown(result))
    return result


def _markdown(result: dict) -> str:
    metrics = result["portfolio_performance"]
    rows = []
    for period, models in metrics.items():
        for name, values in models.items():
            rows.append({"period": period, "legacy_strategy": name, **{
                key: values[key] for key in ["sharpe", "total_return", "rank_ic"]}})
    ranks = result["forecast_diagnostics"]
    rank_rows = [{"strategy": name, "period": period, **values["rank_decomposition"]}
                 for name in ["baseline", "earlier_selected"] for period, values in ranks[name].items()]
    for row in rank_rows:
        row.pop("interpretation")
    later = result["arithmetic_return_gap_attribution"]["later28"]
    family_rows = [{"family": label, **values} for label, values in later["families"].items()]
    top = sorted(later["monthly"], key=lambda row: row["mean_gap_basis_points"], reverse=True)[:5]
    month_rows = [{key: row[key] for key in ["origin", "realized_month", "mean_gap_basis_points",
                                            "baseline_mean_total_return", "selected_mean_total_return"]} for row in top]
    gaps = later["leave_one_month_out"]
    min_sharpe = min(row["baseline_minus_selected_sharpe"] for row in gaps)
    max_sharpe = max(row["baseline_minus_selected_sharpe"] for row in gaps)
    return "\n".join([
        "# Cached performance-reversal diagnosis", "", "## Summary", "",
        "- The earlier-selected blend won in June2022--May2024 but lost in June2024--September2026.",
        "- Most of the baseline's pooled-rank reversal is between ETF families, "
        "not uniformly improved within-family selection.",
        "- Later portfolio differences reflect month-varying family exposure and ETF choices; "
        "average equity exposure alone hides them.",
        "", "This audits the already-evaluated legacy seedwise policy. "
        "It does not choose a November model or replace the required pool-first replay.",
        "", "## Matched legacy-period findings", "", pd.DataFrame(rows).to_markdown(index=False),
        "", "## Pooled rank decomposition", "", pd.DataFrame(rank_rows).to_markdown(index=False),
        "", "The within term uses within-family deviations of GLOBAL ETF ranks. "
        "It is not the mean of family-specific Spearman ICs.",
        "", "## Later arithmetic return-gap accounting", "", pd.DataFrame(family_rows).to_markdown(index=False),
        "", "Positive values favor the baseline. These sum the mean monthly arithmetic gap, "
        "not compound return or Sharpe.",
        "", "## Largest later favorable months", "", pd.DataFrame(month_rows).to_markdown(index=False),
        "", f"Baseline wins {later['positive_months']}/28 later months. The largest favorable month represents "
        f"{100 * later['largest_positive_month_share_of_summed_arithmetic_gap']:.1f}% of the summed arithmetic gap.",
        "Removing any single later month leaves the baseline-minus-selected Sharpe difference "
        f"between {min_sharpe:.4f} and {max_sharpe:.4f}.",
        "", "At the February2025 origin, the baseline held no U.S. equity "
        "while the earlier-selected blend held about66.7%; "
        "March U.S. equity returns were negative. At the October2024 origin "
        "the baseline instead held about65.9% U.S. equity, "
        "versus27.6% selected, before a strong November U.S. equity month. "
        "This is posthoc accounting, not causal proof of timing ability.",
        "", "## Training history and interpretation", "",
        "Expanding training grows60--83 dates in the earlier period and84--111 in the later period; "
        "rolling controls retain48 throughout.",
        "Broad rolling48 pooled IC also changes from approximately-.0776 to+.1096. "
        "Growth in the count of expanding observations is "
        "therefore not the sole possible explanation; market conditions, observation content, "
        "features and fitted kernels remain confounded.",
        "", "## Impact and proposed use", "",
        "Do not infer a universal best feature set from the later winner. "
        "Preserve cross-family ranking and exposure diagnostics in November selection.",
        "The old selected recipe's rank1/maxiter5 components are ineligible for November "
        "and are used only to explain the prior result.",
        "Root-owned pooled-first matched comparisons, not this legacy report, must decide November promotion.",
        "", "## Verification", "",
        "Reconstructed means match sealed later cases within1e-15. "
        "Replayed earlier metrics match saved tuning records within1e-10.",
        "Global rank decomposition and arithmetic return attribution were checked as numerical identities. "
        "Protected source hashes are unchanged.",
        "Run `uv run --locked --no-sync python "
        "experiments/2026-11-portfolio-decision-v1/reversal_analysis.py --verify`.", ""])


def main() -> None:
    """Execute the bounded legacy audit or verify existing hashed diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.execute:
        result = analyze(HERE / "outputs", HERE / "PREREGISTRATION.json")
        print(json.dumps({"passed": result["passed"], "gp_fits": 0,
                          "earlier_replay_parity": result["early_replay_parity_with_saved_metrics"]}))
    else:
        result = _read(HERE / "outputs/reversal_analysis.json")
        for name, expected in result["inputs_sha256"].items():
            if _digest(Path(name)) != expected:
                raise ValueError(f"Protected audit input changed: {name}")
        if not result["passed"]:
            raise ValueError("Incomplete reversal audit")
        print(json.dumps({"passed": True, "protected_inputs": len(result["inputs_sha256"])}))


if __name__ == "__main__":
    main()
