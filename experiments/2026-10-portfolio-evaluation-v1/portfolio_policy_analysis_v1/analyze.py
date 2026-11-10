"""Compare completed paired portfolio policies without changing fitted studies.

Usage: .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_analysis_v1/analyze.py
Use --verify-only to check saved input, script and output hashes. Research-only
rankings separate covariance sources and conditional family comparisons. Returns
are decimal, Sharpe is annualized arithmetic excess, and all 24 months are reused.
No GP fits, allocation solves, promotion, or production changes occur here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE / "outputs"
POLICY = HERE.parent / "portfolio_policy_comparison_v1/outputs"
FITS = HERE.parent / "full_rerun_v1/outputs"
BASELINE = "global_e6c41f2020fd"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _pareto(values: np.ndarray) -> np.ndarray:
    """Return nondominated rows with all objectives expressed as maximization."""
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Finite objective matrix required")
    return np.array([not np.any(np.all(values >= row, axis=1) & np.any(values > row, axis=1)) for row in values])


def _bootstrap(left: np.ndarray, right: np.ndarray, draws: int = 4000) -> dict:
    """Paired circular three-month blocks; seeds remain dependent replications."""
    if left.shape != (3, 24) or right.shape != left.shape:
        raise ValueError("Three aligned seeds and twenty-four months required")
    rng = np.random.default_rng(27)
    starts = rng.integers(24, size=(draws, 8))
    indices = ((starts[..., None] + np.arange(3)) % 24).reshape(draws, 24)

    def sharpes(panel: np.ndarray) -> np.ndarray:
        return np.sqrt(12) * panel.mean(axis=-1) / panel.std(axis=-1, ddof=1)

    differences = (sharpes(left[:, indices]) - sharpes(right[:, indices])).mean(axis=0)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "mean_seed_sharpe_difference": float(np.mean(sharpes(left) - sharpes(right))),
        "conditional_95pct_interval": [float(lower), float(upper)],
        "draws": draws,
        "block_months": 3,
        "selection_adjusted": False,
        "untouched_holdout": False,
    }


def _pair(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "configuration", "covariance_policy"]
    paired = frame.pivot(index=keys, columns="variant", values="conventional_sharpe").dropna().reset_index()
    paired["diversified_minus_risk_only"] = paired.risk_budget_plus_effective - paired.risk_budget_only
    return paired


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = json.loads((ROOT / "manifest.json").read_text())
        for path, digest in {**manifest["input_sha256"], **manifest["output_sha256"]}.items():
            if _digest(Path(path)) != digest:
                raise ValueError(f"Changed analysis evidence: {path}")
        print("Verified saved rankings, paired comparisons, masks and analysis evidence")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    inputs = {str(Path(__file__).resolve()): _digest(Path(__file__))}

    def read(path: Path, expected: dict | None = None) -> bytes:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected[str(path.resolve())]:
            raise ValueError(f"Source seal differs: {path}")
        inputs[str(path.resolve())] = digest
        return data

    seal = json.loads(read(POLICY / "COMPLETED.json"))
    contract = json.loads(read(POLICY / "policy_contract.json"))
    if not seal["passed"] or seal["signature"] != _digest(POLICY / "policy_contract.json"):
        raise ValueError("Incomplete policy study")
    csv = POLICY / "all_499_policy_metrics.csv"
    read(csv, seal["artifact_sha256"])
    forecast = pd.read_csv(csv)
    csv = POLICY / "all_portfolio_policies.csv"
    read(csv, seal["artifact_sha256"])
    portfolios = pd.read_csv(csv)
    if len(forecast) != 499 or len(portfolios) != 1068:
        raise ValueError("Wrong configuration/policy counts")
    names = [
        "configuration",
        "input_count_excluding_time",
        "mean_ic",
        "pooled_15ETF_ic",
        "equal_family_ic",
        "joint_nll_per_asset",
        "mean_marginal_nll",
        "rmse",
        "linear_rank_conventional_sharpe",
    ]
    frame = portfolios.merge(forecast[names], on="configuration", validate="many_to_one")
    seeds, failures, fixed, returns = [], [], [], {}
    for candidate in contract["candidates"]:
        name = candidate["configuration"]
        stamp = json.loads(read(POLICY / "checkpoints" / f"{name}.json", seal["artifact_sha256"]))
        evidence = json.loads(read(POLICY / "portfolios" / f"{name}.json", stamp["result"]["artifact_sha256"]))
        for seed_index, row in enumerate(evidence["seeds"]):
            for key, value in row["portfolios"].items():
                source, variant = key.split("__")
                info = {
                    "configuration": name,
                    "family": candidate["family"],
                    "seed": row["seed"],
                    "covariance_policy": source,
                    "variant": variant,
                    "status": value["status"],
                }
                for month in value["months"]:
                    if month["status"] != "solved":
                        failures.append({**info, **month})
                if value["status"] != "solved":
                    continue
                performance = value["performance"]
                calibration_source = "exact_observed_GP" if source == "GP" else "historical_control_not_GP"
                calibration = next(v for v in value["risk_calibration"] if v["covariance_source"] == calibration_source)
                seeds.append(
                    {
                        **info,
                        **{
                            metric: performance[metric]
                            for metric in [
                                "conventional_sharpe",
                                "first_12_sharpe",
                                "last_12_sharpe",
                                "annualized_excess_vol",
                                "compounded_total_return",
                                "max_drawdown_total_nav",
                                "mean_effective_count",
                                "largest_weight",
                            ]
                        },
                        **{
                            f"adaptive_{metric}": calibration[metric]
                            for metric in [
                                "error_rms_to_predicted_sd_rms",
                                "coverage_95",
                                "mean_nll",
                                "mean_crps",
                            ]
                        },
                    }
                )
                returns.setdefault((name, key), np.full((3, 24), np.nan))[seed_index] = performance["arithmetic_excess"]
        if candidate["family"] == "joint":
            upstream = json.loads(read(FITS / "evaluation_checkpoints" / f"{name}.json"))
            evaluation = json.loads(read(FITS / "evaluation" / f"{name}.json", upstream["result"]["artifact_sha256"]))
            for seed, row in zip([27, 127, 227], evaluation["seeds"], strict=True):
                for value in row["fixed_GP_portfolio_distributions"]:
                    fixed.append({"configuration": name, "seed": seed, **value})
    seed_frame = pd.DataFrame(seeds)
    keys = ["family", "configuration", "covariance_policy", "variant"]
    seed_diagnostics = (
        seed_frame.groupby(keys)
        .agg(
            minimum_seed_sharpe=("conventional_sharpe", "min"),
            adaptive_error_rms_to_predicted_sd_rms=("adaptive_error_rms_to_predicted_sd_rms", "mean"),
            adaptive_coverage_95=("adaptive_coverage_95", "mean"),
        )
        .reset_index()
    )
    frame = frame.merge(seed_diagnostics, on=keys, validate="one_to_one")
    frame["eligible"] = frame.complete_seeds.eq(3) & frame.solved_months.eq(72)
    frame["sharpe_rank"] = (
        frame.where(frame.eligible)
        .groupby(["family", "covariance_policy", "variant"])
        .conventional_sharpe.rank(ascending=False, method="min")
    )
    frame["pareto_ic_nll_sharpe"] = False
    for _, group in frame[frame.eligible].groupby(["family", "covariance_policy", "variant"]):
        ranking = group.pooled_15ETF_ic if group.iloc[0].family == "joint" else group.mean_ic
        objectives = np.column_stack([ranking, -group.joint_nll_per_asset, group.conventional_sharpe])
        frame.loc[group.index, "pareto_ic_nll_sharpe"] = _pareto(objectives)
    ranked = frame.sort_values(["family", "covariance_policy", "variant", "sharpe_rank", "configuration"])
    leaders = ranked[ranked.sharpe_rank.le(3)].copy()
    paired = _pair(frame[frame.eligible])
    statistics = []
    for (family, source), group in paired.groupby(["family", "covariance_policy"]):
        delta = group.diversified_minus_risk_only
        statistics.append(
            {
                "family": family,
                "covariance_policy": source,
                "paired_configurations": len(group),
                "diversified_higher_sharpe": int((delta > 0).sum()),
                "mean_sharpe_change": delta.mean(),
                "median_sharpe_change": delta.median(),
                "policy_sharpe_rank_spearman": group.risk_budget_only.corr(
                    group.risk_budget_plus_effective, method="spearman"
                ),
            }
        )
    bootstrap = {}
    for row in leaders[leaders.family.eq("joint") & leaders.sharpe_rank.eq(1)].itertuples():
        key = f"{row.covariance_policy}__{row.variant}"
        bootstrap[f"{row.configuration}__{key}__minus_baseline"] = _bootstrap(
            returns[(row.configuration, key)], returns[(BASELINE, key)]
        )
    selected = set(leaders.configuration) | {BASELINE, "global_bb1b8f9500f2", "global_b48732adb186"}
    masks = [candidate for candidate in contract["candidates"] if candidate["configuration"] in selected]
    for name, table in {
        "rankings": ranked,
        "top_three_by_group": leaders,
        "paired_policy_deltas": paired,
        "paired_policy_summary": pd.DataFrame(statistics),
        "seed_diagnostics": seed_frame,
        "fixed_reference_scores": pd.DataFrame(fixed),
        "pareto": ranked[ranked.pareto_ic_nll_sharpe],
    }.items():
        table.to_csv(ROOT / f"{name}.csv", index=False)
    _save(ROOT / "selected_masks.json", masks)
    _save(ROOT / "paired_exploratory_bootstrap.json", bootstrap)
    _save(ROOT / "failed_months.json", failures)
    summary = {
        "configurations": 499,
        "configuration_covariance_policy_rows": len(frame),
        "complete_rows": int(frame.eligible.sum()),
        "failed_monthly_solves": len(failures),
        "baseline": BASELINE,
        "new_fits": 0,
        "automatic_promotion": False,
        "joint_minimum_effective": 10,
        "family_conditional_minimum_effective": 3,
        "family_comparison": "one_family_replaced_other_three_families_frozen_ranking_anchors",
        "pareto": "within_family_and_covariance_and_policy_max_IC_min_joint_NLL_max_Sharpe",
        "holdout": "reused_24_months_exploratory_not_selection_adjusted",
    }
    _save(ROOT / "summary.json", summary)
    for path, digest in inputs.items():
        if _digest(Path(path)) != digest:
            raise ValueError("Input changed during analysis")
    outputs = [p for p in ROOT.iterdir() if p.name != "manifest.json" and p.is_file()]
    _save(ROOT / "manifest.json", {"input_sha256": inputs, "output_sha256": {str(p): _digest(p) for p in outputs}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
