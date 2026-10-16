"""Consolidate all fixed-income search rounds into final Pareto artifacts.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/summarize_fixed_income_search.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from summarize_us_confirmation import hypervolume_3d, is_dominated

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
BASE_ROUND_NAMES = ["block_screen", "targeted_ablation", "feature_refinement", "combination_refinement"]
FAMILIES = {
    "us_fixed_income": {
        "title": "U.S. fixed income",
        "universe": ["BND", "IEF", "LQD", "HYG"],
        "prefix": "us_fixed_income",
        "round_names": BASE_ROUND_NAMES,
        "selections": [
            ("primary", "USFI-COMPACT-NO-CS+RET-AUTOCORR"),
            ("challenger", "USFI-COMPACT+VOL-Z"),
            ("calibration_anchor", "USFI-TREND"),
        ],
    },
    "international_fixed_income": {
        "title": "International fixed income",
        "universe": ["BNDX", "VWOB", "HYEM"],
        "prefix": "international_fixed_income",
        "round_names": [
            *BASE_ROUND_NAMES,
            "sparsity_ablation",
            "sparsity_combination",
            "sparsity_final",
        ],
        "selections": [
            ("primary", "INTFI-PRIMARY-NO-LAG-NO-CPI-NO-GOLD"),
            ("high_ic_challenger", "INTFI-PRIMARY-NO-LAG-NO-ERP"),
            ("sparse_sharpe_challenger", "INTFI-PRIMARY-NO-LAG-NO-ERP-NO-CPI-MOM-NO-GOLD-RATIO"),
            ("calibration_anchor", "INTFI-TREND+MARKET-RISK"),
        ],
    },
}
OBJECTIVE_COLUMNS = ["mean_ic", "mean_gaussian_nll", "sharpe"]
RATE_CHALLENGE_CONFIG = EXPERIMENT_DIR / "us_fixed_income_rate_no_lag_screen_config.json"
RATE_CHALLENGE_RESULTS = RESULTS_DIR / "us_fixed_income_rate_no_lag_screen_configuration_aggregates.csv"


def pareto_mask(frame: pd.DataFrame) -> list[bool]:
    """Return the three-objective nondominance mask."""
    view = frame.loc[:, ["configuration", *OBJECTIVE_COLUMNS]]
    return [not is_dominated(row, view) for _, row in view.iterrows()]


def objective_hypervolume(frame: pd.DataFrame) -> float:
    """Calculate fixed-reference hypervolume in transformed maximization space."""
    points = [
        (float(row.mean_ic), -float(row.mean_gaussian_nll), float(row.sharpe))
        for row in frame.itertuples()
        if row.mean_ic > 0.0 and row.mean_gaussian_nll < 0.0 and row.sharpe > 0.0
    ]
    return hypervolume_3d(points, (0.0, 0.0, 0.0))


def load_feature_map(family: str) -> dict[str, list[str]]:
    """Resolve labels to exact frozen input lists across all family configs."""
    feature_map: dict[str, list[str]] = {}
    for path in sorted(EXPERIMENT_DIR.glob("*fixed_income*_config.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("family") != family:
            continue
        for item in [*payload.get("comparison_configurations", []), *payload["configurations"]]:
            label = str(item["label"])
            columns = [str(value) for value in item["expected_input_columns"]]
            previous = feature_map.setdefault(label, columns)
            if previous != columns:
                raise ValueError(f"Conflicting input definitions for {label}.")
    return feature_map


def load_family(family: str, prefix: str, round_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and deduplicate all configuration aggregates for a family."""
    frames: list[pd.DataFrame] = []
    cumulative: list[pd.DataFrame] = []
    progression: list[dict[str, Any]] = []
    for round_name in round_names:
        path = RESULTS_DIR / f"{prefix}_{round_name}_configuration_aggregates.csv"
        frame = pd.read_csv(path)
        frame.insert(1, "observed_in_round", round_name)
        frames.append(frame)
        cumulative.append(frame)
        current = pd.concat(cumulative, ignore_index=True)
        current = current.drop_duplicates(subset="configuration", keep="last")
        progression.append(
            {
                "round": round_name,
                "unique_configurations": len(current),
                "hypervolume": objective_hypervolume(current),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    metric_columns = [*OBJECTIVE_COLUMNS, "cumulative_return_diagnostic_only", "rmse_diagnostic_only"]
    for configuration, group in combined.groupby("configuration", sort=False):
        if len(group) > 1 and not (group[metric_columns].nunique(dropna=False) == 1).all():
            raise ValueError(f"Repeated aggregate changed across rounds: {configuration}.")
    round_membership = combined.groupby("configuration", sort=False)["observed_in_round"].agg(",".join)
    combined = combined.drop_duplicates(subset="configuration", keep="last").copy()
    combined["observed_in_round"] = combined["configuration"].map(round_membership)
    combined["pareto_nondominated"] = pareto_mask(combined)
    if not bool(combined["validation_passed"].all()):
        raise ValueError(f"{family} contains an invalid aggregate.")
    return combined, pd.DataFrame(progression)


def selected_row(frame: pd.DataFrame, label: str, role: str, feature_map: dict[str, list[str]]) -> dict[str, Any]:
    """Return a serialized recommendation with exact inputs and metrics."""
    row = frame.loc[frame["configuration"] == label].iloc[0]
    return {
        "role": role,
        "configuration": label,
        "input_columns": feature_map[label],
        "mean_ic": float(row["mean_ic"]),
        "mean_gaussian_nll": float(row["mean_gaussian_nll"]),
        "sharpe": float(row["sharpe"]),
        "cumulative_return_diagnostic_only": float(row["cumulative_return_diagnostic_only"]),
        "rmse_diagnostic_only": float(row["rmse_diagnostic_only"]),
    }


def load_rate_challenge() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load the separately comparable recent-history rate-aware lag challenge."""
    frame = pd.read_csv(RATE_CHALLENGE_RESULTS)
    if not bool(frame["validation_passed"].all()):
        raise ValueError("Rate-aware lag challenge contains an invalid aggregate.")
    payload = json.loads(RATE_CHALLENGE_CONFIG.read_text(encoding="utf-8"))
    feature_map = {
        str(item["label"]): [str(value) for value in item["expected_input_columns"]]
        for item in payload["configurations"]
    }
    frame = frame.copy()
    frame["pareto_nondominated"] = pareto_mask(frame)
    return frame, feature_map


def main() -> None:
    """Write final family-wide Pareto tables, recommendations, and study summary."""
    recommendations: dict[str, Any] = {
        "schema": "bayesfolio.fixed_income_feature_search_recommendations.v1",
        "objectives": {"maximize": ["mean_ic", "sharpe"], "minimize": ["mean_gaussian_nll"]},
        "diagnostic_only": ["cumulative_return", "rmse"],
        "turnover_excluded": True,
        "promotion_stage": "longer-history/full-budget confirmation",
        "families": {},
    }
    report = [
        "# Fixed-Income ETF Feature Search",
        "",
        "This report consolidates the U.S. and international fixed-income searches across block screening,",
        "targeted ablations, feature-level add/remove tests, combination refinement, and sparsity refinement.",
        "",
        "The Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher conventional annualized",
        "Sharpe. Cumulative return and RMSE are recorded only; turnover is excluded. All results use cached data,",
        "seeds `27,127,227`, 24 common monthly scored windows, rank 2, maxiter 15, 256 scenarios, and `t_index`.",
        "",
    ]

    for family, metadata in FAMILIES.items():
        frame, progression = load_family(family, str(metadata["prefix"]), list(metadata["round_names"]))
        feature_map = load_feature_map(family)
        pareto = frame.loc[frame["pareto_nondominated"]].copy()
        selected = [selected_row(frame, str(label), str(role), feature_map) for role, label in metadata["selections"]]
        recommendations["families"][family] = {
            "universe": metadata["universe"],
            "configurations_evaluated": len(frame),
            "valid_fits": len(frame) * 3,
            "final_hypervolume": objective_hypervolume(frame),
            "promotions": selected,
        }
        frame.to_csv(RESULTS_DIR / f"{metadata['prefix']}_all_configurations.csv", index=False)
        pareto.to_csv(RESULTS_DIR / f"{metadata['prefix']}_final_pareto.csv", index=False)
        progression.to_csv(RESULTS_DIR / f"{metadata['prefix']}_hypervolume_progression.csv", index=False)

        report.extend(
            [
                f"## {metadata['title']}",
                "",
                f"Universe: `{', '.join(metadata['universe'])}`. Evaluated `{len(frame)}` unique configurations "
                f"and `{len(frame) * 3}` valid fits.",
                "",
                "### Final Pareto front",
                "",
                pareto[
                    [
                        "configuration",
                        "mean_ic",
                        "mean_gaussian_nll",
                        "sharpe",
                        "cumulative_return_diagnostic_only",
                        "rmse_diagnostic_only",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
                "### Hypervolume progression",
                "",
                progression.to_markdown(index=False, floatfmt=".9f"),
                "",
                "### Promotion set",
                "",
            ]
        )
        for item in selected:
            report.append(
                f"- **{item['role']} — {item['configuration']}**: IC `{item['mean_ic']:.6f}`, "
                f"NLL `{item['mean_gaussian_nll']:.6f}`, Sharpe `{item['sharpe']:.6f}`; "
                f"inputs: `{', '.join(item['input_columns'])}`."
            )
        report.append("")

        if family == "us_fixed_income":
            rate_frame, rate_feature_map = load_rate_challenge()
            rate_pareto = rate_frame.loc[rate_frame["pareto_nondominated"]].copy()
            rate_selected = [
                selected_row(
                    rate_frame,
                    "USFI-RATE-REAL-RATE-NO-LAG",
                    "balanced_no_lag_challenger",
                    rate_feature_map,
                ),
                selected_row(
                    rate_frame,
                    "USFI-RATE-BALANCED-NO-LAG",
                    "no_lag_calibration_anchor",
                    rate_feature_map,
                ),
            ]
            recommendations["families"][family]["rate_aware_lag_challenge"] = {
                "training_panel": ["2021-03-31", "2026-04-30"],
                "scored_panel": ["2024-05-31", "2026-04-30"],
                "configurations_evaluated": len(rate_frame),
                "valid_fits": len(rate_frame) * 3,
                "pareto_no_lag_count": int(
                    rate_pareto["configuration"].str.endswith("-NO-LAG").sum()
                ),
                "pareto_count": len(rate_pareto),
                "promotions": rate_selected,
            }
            report.extend(
                [
                    "### Rate-aware lag challenge",
                    "",
                    "A subsequent cached-only challenge added 10 bond-specific rate-state variables and evaluated",
                    "22 explicitly paired with-lag/no-lag configurations (66 fits) on the same 24 scored dates.",
                    "The rate source limits training to 62 complete months from 2021-03 through 2026-04, so its",
                    "raw objectives remain separate from the 2013-history front above.",
                    "",
                    rate_pareto[
                        [
                            "configuration",
                            "mean_ic",
                            "mean_gaussian_nll",
                            "sharpe",
                            "cumulative_return_diagnostic_only",
                            "rmse_diagnostic_only",
                        ]
                    ].to_markdown(index=False, floatfmt=".6f"),
                    "",
                    "Five of the six rate-aware Pareto points omit `lag_y_excess_lead`.",
                    "",
                ]
            )
            for item in rate_selected:
                report.append(
                    f"- **{item['role']} — {item['configuration']}**: IC `{item['mean_ic']:.6f}`, "
                    f"NLL `{item['mean_gaussian_nll']:.6f}`, Sharpe `{item['sharpe']:.6f}`; "
                    f"inputs: `{', '.join(item['input_columns'])}`."
                )
            report.extend(
                [
                    "",
                    "The real-rate no-lag model beat its exact with-lag pair on IC and Sharpe in all three seeds",
                    "and on NLL in two of three seeds. The balanced no-lag model beat its exact with-lag pair on",
                    "NLL and Sharpe in all three seeds, with only a `0.005556` aggregate IC reduction. The prior",
                    "lag requirement was therefore conditional on the original feature/training regime, not a",
                    "stable U.S. fixed-income property. See `US_FIXED_INCOME_RATE_NO_LAG_RESULTS.md` for details.",
                    "",
                ]
            )

    report.extend(
        [
            "## Findings and recommendation",
            "",
            "- The long-history U.S. fixed-income search favored local time-series features. Generic credit/inflation",
            "  and broad macro bundles were noise at this budget. The paired rate-aware challenge later showed that",
            "  `lag_y_excess_lead` is not generally required: sparse no-lag local and real-rate models occupied five",
            "  of six Pareto positions.",
            "- International fixed income required macro context, but the broad model was over-specified. The final",
            "  primary removes `hy_spread_chg_1m`, `copper_ret`, `lag_y_excess_lead`, `cpi_mom`, and",
            "  `gold_crude_ratio` while retaining `oil_ret`; it dominates the former 21-input primary.",
            "- `INTFI-TREND+COMMODITY-RET` remains mathematically nondominated only because it has the best NLL;",
            "  its negative IC means it contributes no hypervolume above the fixed zero reference and is not promoted.",
            "- Promote each listed primary and challenger to longer-history/full-budget confirmation.",
            "  Keep the calibration anchor in that confirmation so selection remains genuinely multi-objective.",
            "  Do not replace production inputs from this screen alone.",
            "",
            "## Validation and limitations",
            "",
            "- Every reported configuration passed exact universe, date, seed, input-column, and model-control checks.",
            "- The search is conditional on the cached 2013-08 through 2026-04 data and the 24 scored months from",
            "  2024-05 through 2026-04. It is evidence-directed rather than exhaustive over every feature subset.",
            "- The rate-aware challenge uses the same scored months but a 2021-03 through 2026-04 complete training",
            "  panel. Its paired comparisons are valid, but its raw metrics are not pooled into the long-history",
            "  front.",
            "- The next confirmation should preserve a common date panel and use the production optimization budget",
            "  before any production promotion decision.",
        ]
    )

    (RESULTS_DIR / "fixed_income_final_recommendations.json").write_text(
        json.dumps(recommendations, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "FIXED_INCOME_FEATURE_SELECTION_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
