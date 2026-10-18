"""Audit matched training histories and report the U.S. fixed-income lag challenge.

This experiment-only entry point reads frozen configs, cached panels, and run
outputs. It verifies per-window training dates/rows before reporting IC, Gaussian
NLL, conventional annualized Sharpe, decimal cumulative returns, and decimal RMSE.
Long and recent training regimes receive separate Pareto fronts.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/summarize_us_fixed_income_lag_history.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from orchestrate_international_experiments import sha256_file, validate_config
from summarize_fixed_income_search import pareto_mask
from summarize_international_experiments import read_dates

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
SEEDS = [27, 127, 227]
HISTORY_COLUMNS = ["date", "train_rows", "train_date_min", "train_date_max", "unique_train_dates"]
OBJECTIVES = ["mean_ic", "mean_gaussian_nll", "sharpe"]


def expected_training_history(panel: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """Return the expected expanding-window date bounds and row counts.

    Args:
        panel: Complete cached asset/date panel with decimal excess-return labels.
        dates: Ordered monthly evaluation dates in ISO format.

    Returns:
        One row per evaluation date, using only earlier labeled observations.
    """
    rows = []
    for date in dates:
        train = panel.loc[(panel["date"] < pd.Timestamp(date)) & panel["y_excess_lead"].notna()]
        rows.append(
            {
                "date": date,
                "train_rows": len(train),
                "train_date_min": train["date"].min().date().isoformat(),
                "train_date_max": train["date"].max().date().isoformat(),
                "unique_train_dates": train["date"].nunique(),
            }
        )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def validate_training_history(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Reject a run whose actual training bounds or observation counts differ.

    Args:
        actual: Saved per-window training-history rows from one fit.
        expected: Expected history calculated directly from its frozen panel.

    Raises:
        ValueError: If a date, row count, or training date bound differs.
    """
    observed = actual.loc[:, HISTORY_COLUMNS].reset_index(drop=True)
    if not observed.equals(expected.reset_index(drop=True)):
        raise ValueError("Run training dates or row counts differ from the frozen common panel.")


def _audit_history(history: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    path = EXPERIMENT_DIR / f"us_fixed_income_lag_matched_{history}_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    paths = validate_config(config, path)
    panel = pd.read_parquet(paths["feature"])
    dates = read_dates(paths["dates"])
    expected = expected_training_history(panel, dates)
    if expected["train_date_min"].nunique() != 1 or expected.iloc[0]["train_date_min"] != config["history_start"]:
        raise ValueError(f"Unexpected {history} training-history start.")
    expected.to_csv(RESULTS_DIR / f"us_fixed_income_lag_matched_{history}_training_history.csv", index=False)
    audits: list[dict[str, object]] = []
    for item in [*config["configurations"], *config["comparison_configurations"]]:
        for seed in SEEDS:
            run_dir = paths["output_root"] / f"{item['id']}_s{seed}"
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest["feature_sha256"] != sha256_file(paths["feature"]):
                raise ValueError(f"{run_dir.name}: feature hash mismatch.")
            if manifest["training_history"] != {
                "mode": "all_available_before_window",
                "train_months": None,
                "window_rule": "date < window_date",
            }:
                raise ValueError(f"{run_dir.name}: unexpected training policy.")
            actual_path = run_dir / "window_training_history.csv"
            validate_training_history(pd.read_csv(actual_path), expected)
            audits.append(
                {
                    "run_id": run_dir.name,
                    "feature_sha256": manifest["feature_sha256"],
                    "training_history_sha256": sha256_file(actual_path),
                    "history_start": config["history_start"],
                    "n_windows": len(expected),
                    "passed": True,
                }
            )
    prefix = config["results_prefix"]
    aggregates = pd.read_csv(RESULTS_DIR / f"{prefix}_configuration_aggregates.csv")
    runs = pd.read_csv(RESULTS_DIR / f"{prefix}_run_objectives.csv")
    if not aggregates["validation_passed"].all() or not runs["validation_passed"].all():
        raise ValueError(f"{history}: invalid objective outputs.")
    designs = {item["label"]: item for item in config["configurations"]}
    for item in config["comparison_configurations"]:
        include_lag = "lag_y_excess_lead" in item["expected_input_columns"]
        suffix = "_with_lag" if include_lag else "_no_lag"
        designs[item["label"]] = {
            **item,
            "design": item["id"].removeprefix("usfi_rate_").removesuffix(suffix),
            "include_lag": include_lag,
        }
    for frame in (aggregates, runs):
        frame["history"] = history
        frame["design"] = frame["configuration"].map(lambda label: designs[label]["design"])
        frame["include_lag"] = frame["configuration"].map(lambda label: designs[label]["include_lag"])
    aggregates["pareto_nondominated"] = pareto_mask(aggregates)
    return aggregates, runs, audits


def main() -> None:
    """Validate every run's training rows and write paired lag/history evidence."""
    long_panel = pd.read_parquet(EXPERIMENT_DIR / "artifacts/us_fixed_income_cached_inputs_20261016.parquet")
    recent_panel = pd.read_parquet(EXPERIMENT_DIR / "artifacts/us_fixed_income_rate_cached_inputs_20261016.parquet")
    common = long_panel.loc[long_panel["date"].isin(recent_panel["date"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(common, recent_panel[common.columns], check_exact=True)
    if read_dates(EXPERIMENT_DIR / "artifacts/us_fixed_income_scored_dates_24.txt") != read_dates(
        EXPERIMENT_DIR / "artifacts/us_fixed_income_rate_scored_dates_24.txt"
    ):
        raise ValueError("The long and recent scored dates differ.")

    aggregate_frames, run_frames, audits = [], [], []
    for history in ("long", "recent"):
        aggregates, runs, history_audits = _audit_history(history)
        aggregate_frames.append(aggregates)
        run_frames.append(runs)
        audits.extend(history_audits)
    aggregate = pd.concat(aggregate_frames, ignore_index=True)
    runs = pd.concat(run_frames, ignore_index=True)
    pairs: list[dict[str, object]] = []
    for (history, design), frame in runs.groupby(["history", "design"], sort=False):
        without = frame.loc[~frame["include_lag"]].set_index("seed").sort_index()
        with_lag = frame.loc[frame["include_lag"]].set_index("seed").sort_index()
        if without.index.tolist() != SEEDS or with_lag.index.tolist() != SEEDS:
            raise ValueError(f"Incomplete seed pair: {history}/{design}.")
        deltas = without[OBJECTIVES] - with_lag[OBJECTIVES]
        row: dict[str, object] = {"history": history, "design": design}
        for objective in OBJECTIVES:
            row[f"no_lag_{objective}"] = float(without[objective].mean())
            row[f"with_lag_{objective}"] = float(with_lag[objective].mean())
            row[f"delta_{objective}"] = float(deltas[objective].mean())
            sign = -1 if objective == "mean_gaussian_nll" else 1
            row[f"no_lag_seed_wins_{objective}"] = int((deltas[objective] * sign > 1e-10).sum())
        pairs.append(row)
    paired = pd.DataFrame(pairs)
    aggregate.to_csv(RESULTS_DIR / "us_fixed_income_lag_matched_all_aggregates.csv", index=False)
    paired.to_csv(RESULTS_DIR / "us_fixed_income_lag_matched_pair_deltas.csv", index=False)

    candidate_recommendations: list[dict[str, object]] = []
    recommendation = {
        "default_training_policy": "expanding from 2013-08-30",
        "recommendation": "retain lag_y_excess_lead for the current long-history U.S. fixed-income incumbent",
        "lag_effect_depends_on_training_history": True,
        "production_changed": False,
        "candidates": candidate_recommendations,
    }
    for history, design, include_lag, role in [
        ("long", "balanced", True, "retain_current_incumbent"),
        ("recent", "balanced", False, "recent_policy_challenger"),
        ("recent", "no_mom12_autocorr", False, "recent_policy_calibration_challenger"),
    ]:
        row = aggregate.loc[
            (aggregate["history"] == history)
            & (aggregate["design"] == design)
            & (aggregate["include_lag"] == include_lag)
        ].iloc[0]
        config_path = EXPERIMENT_DIR / f"us_fixed_income_lag_matched_{history}_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        item = next(c for c in config["configurations"] if c["label"] == row["configuration"])
        candidate_recommendations.append(
            {
                "role": role,
                "configuration": row["configuration"],
                "training_start": config["history_start"],
                "input_columns": item["expected_input_columns"],
                **{objective: float(row[objective]) for objective in OBJECTIVES},
            }
        )
    (RESULTS_DIR / "us_fixed_income_lag_matched_recommendations.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )

    original = pd.read_csv(RESULTS_DIR / "us_fixed_income_final_pareto.csv")
    fresh_long = aggregate.loc[(aggregate["history"] == "long") & aggregate["include_lag"]]
    original_values = sorted(tuple(row) for row in original[OBJECTIVES].to_numpy())
    fresh_values = sorted(tuple(row) for row in fresh_long[OBJECTIVES].to_numpy())
    reproduced_original = bool(np.allclose(original_values, fresh_values, atol=1e-10, rtol=1e-10))
    validation = {
        "all_passed": True,
        "seeds": SEEDS,
        "scored_dates": ["2024-05-31", "2026-04-30"],
        "shared_cached_feature_values_exact": True,
        "long_incumbent_metrics_reproduced": reproduced_original,
        "runs_audited": len(audits),
        "runs": audits,
    }
    (RESULTS_DIR / "us_fixed_income_lag_matched_history_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# U.S. Fixed-Income Lag: Matched Training-History Comparison",
        "",
        "Universe: BND, IEF, LQD, HYG. Seeds: 27, 127, 227. Scored months: May 2024 through April 2026.",
        "Rank 2, maxiter 15, 256 scenarios; t_index retained. Objectives: maximize IC and conventional Sharpe,",
        "minimize Gaussian NLL. Cumulative decimal return and decimal RMSE are record-only; turnover excluded.",
        "",
        "All eight original Pareto designs were rerun with and without lag for both history starts (96 fresh fits).",
        "Eight previously evaluated rate-model fits per seed are included in the recent comparison only, after",
        "checking their frozen feature hash and every window's training rows and date bounds (24 reused fits).",
        "Long and recent input values are exactly identical on overlapping dates. All 120 fits were audited.",
        f"The eight original with-lag long-history results reproduced: {reproduced_original}.",
        "",
        "## Incumbent input design: direct four-way comparison",
        "",
        "Balanced inputs: t_index, chmom, mom12m, ma_signal, ret_autocorr, with optional lag_y_excess_lead.",
        "",
        aggregate.loc[aggregate["design"] == "balanced", ["history", "include_lag", *OBJECTIVES]].to_markdown(
            index=False, floatfmt=".6f"
        ),
        "",
    ]
    for history, start in (("long", "2013-08-30"), ("recent", "2021-03-31")):
        selected = aggregate.loc[(aggregate["history"] == history) & aggregate["pareto_nondominated"]]
        lines.extend(
            [
                f"## {history.title()} history: expanding from {start}",
                "",
                selected[
                    ["design", "include_lag", *OBJECTIVES, "cumulative_return_diagnostic_only", "rmse_diagnostic_only"]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
                "### Removing lag: no-lag minus with-lag",
                "",
                paired.loc[
                    paired["history"] == history,
                    [
                        "design",
                        "delta_mean_ic",
                        "delta_mean_gaussian_nll",
                        "delta_sharpe",
                        "no_lag_seed_wins_mean_ic",
                        "no_lag_seed_wins_mean_gaussian_nll",
                        "no_lag_seed_wins_sharpe",
                    ],
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    lines.extend(
        [
            "## Matched-history findings",
            "",
        ]
    )
    for history in ("long", "recent"):
        front = aggregate.loc[(aggregate["history"] == history) & aggregate["pareto_nondominated"]]
        comparisons = paired.loc[paired["history"] == history]
        dominated_with_lag = (
            (comparisons["delta_mean_ic"] >= -1e-10)
            & (comparisons["delta_mean_gaussian_nll"] <= 1e-10)
            & (comparisons["delta_sharpe"] >= -1e-10)
            & (comparisons[["delta_mean_ic", "delta_mean_gaussian_nll", "delta_sharpe"]].abs().max(axis=1) > 1e-10)
        )
        lines.append(
            f"- {history.title()}: {int((~front['include_lag']).sum())}/{len(front)} Pareto configurations omit lag; "
            f"removing lag improves aggregate Sharpe in {int((comparisons['delta_sharpe'] > 1e-10).sum())}/"
            f"{len(comparisons)} pairs and dominates its with-lag pair in {int(dominated_with_lag.sum())} designs."
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Retain lag_y_excess_lead for the current long-history U.S. fixed-income incumbent.",
            "The long-history balanced model remains the stronger IC/Sharpe choice; shorter-history no-lag models",
            "remain alternative training policies with calibration tradeoffs. Removing lag is not a justified",
            "family-wide simplification under the current long-history policy.",
            "",
            "The previous suggestion that richer rate inputs explain lag redundancy was too strong: local-only",
            "no-lag models also improve when trained from 2021. Training-history sensitivity is directly supported;",
            "the economic mechanism causing that sensitivity has not been established by these experiments.",
            "",
            "## Interpretation boundary",
            "",
            "Each within-history comparison uses identical observation dates and expanding training rules.",
            "Across histories, feature values and evaluation months are held fixed; training history varies.",
            "Rate variables are cached only from 2021; their performance on 2013-start history remains untested.",
            "The 24 evaluation months have already informed model selection; these results refine the experiment",
            "and require further confirmation before production promotion.",
            "",
            "## Verification",
            "",
            "```bash",
            "uv run --locked python "
            "experiments/2026-10-etf-family-feature-confirmation/summarize_us_fixed_income_lag_history.py",
            "uv run --locked pytest -q "
            "experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py",
            "```",
            "",
        ]
    )
    (RESULTS_DIR / "US_FIXED_INCOME_LAG_MATCHED_HISTORY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(paired.to_string(index=False))
    print(f"Validated {len(audits)} runs; original incumbent metrics reproduced: {reproduced_original}.")


if __name__ == "__main__":
    main()
