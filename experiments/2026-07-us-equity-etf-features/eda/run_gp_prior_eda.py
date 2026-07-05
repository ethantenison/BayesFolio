"""GP-focused EDA for the U.S. equity ETF feature experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
EDA_DIR = EXPERIMENT_DIR / "eda"
FIGURE_DIR = EDA_DIR / "figures"

ARTIFACTS = {
    "BME": ARTIFACT_DIR / "us_equity_full_feature_candidates_20260705T135534Z_bme.parquet",
    "3W-FRI": ARTIFACT_DIR / "us_equity_full_feature_candidates_20260705T135547Z_3w_fri.parquet",
}

TARGET = "y_excess_lead"
ID_COLS = {"t_index", "date", "asset_id", TARGET}


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    frames = {horizon: pd.read_parquet(path) for horizon, path in ARTIFACTS.items()}
    feature_cols = [
        column
        for column in frames["BME"].columns
        if column not in ID_COLS and pd.api.types.is_numeric_dtype(frames["BME"][column])
    ]

    task_summary = build_task_summary(frames)
    task_summary.to_csv(EDA_DIR / "gp_task_output_summary.csv", index=False)

    task_corrs = {horizon: target_pivot(frame).corr() for horizon, frame in frames.items()}
    eig_summary = build_task_eigen_summary(task_corrs)
    eig_summary.to_csv(EDA_DIR / "gp_task_correlation_eigenvalues.csv", index=False)

    redundancy = build_redundancy_summary(frames, feature_cols)
    redundancy.to_csv(EDA_DIR / "gp_feature_redundancy_pairs.csv", index=False)

    feature_tail = build_feature_tail_summary(frames, feature_cols)
    feature_tail.to_csv(EDA_DIR / "gp_feature_tail_summary.csv", index=False)

    distance_summary, distance_samples = build_distance_summary(frames, feature_cols)
    distance_summary.to_csv(EDA_DIR / "gp_standardized_distance_summary.csv", index=False)

    plot_task_correlations(task_corrs)
    plot_task_eigens(task_corrs)
    plot_target_scale(task_summary)
    plot_feature_redundancy(redundancy)
    plot_feature_tails(feature_tail)
    plot_distance_diagnostics(distance_samples, distance_summary)
    write_report(task_summary, eig_summary, redundancy, feature_tail, distance_summary)


def target_pivot(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot(index="date", columns="asset_id", values=TARGET).sort_index()


def build_task_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for horizon, frame in frames.items():
        for asset, group in frame.groupby("asset_id"):
            target = group[TARGET].dropna()
            rows.append(
                {
                    "horizon": horizon,
                    "asset_id": asset,
                    "n": int(target.count()),
                    "mean": float(target.mean()),
                    "std": float(target.std(ddof=1)),
                    "p05": float(target.quantile(0.05)),
                    "p50": float(target.quantile(0.50)),
                    "p95": float(target.quantile(0.95)),
                    "positive_share": float((target > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_task_eigen_summary(task_corrs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for horizon, corr in task_corrs.items():
        eigvals = np.linalg.eigvalsh(corr.to_numpy())
        eigvals = np.sort(eigvals)[::-1]
        shares = eigvals / eigvals.sum()
        for index, (value, share) in enumerate(zip(eigvals, shares, strict=True), start=1):
            rows.append(
                {
                    "horizon": horizon,
                    "component": index,
                    "eigenvalue": float(value),
                    "variance_share": float(share),
                    "cumulative_variance_share": float(shares[:index].sum()),
                }
            )
    return pd.DataFrame(rows)


def build_redundancy_summary(frames: dict[str, pd.DataFrame], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for horizon, frame in frames.items():
        corr = frame[feature_cols].corr(min_periods=24)
        for i, left in enumerate(feature_cols):
            for right in feature_cols[i + 1 :]:
                value = corr.loc[left, right]
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "horizon": horizon,
                        "feature_a": left,
                        "feature_b": right,
                        "corr": float(value),
                        "abs_corr": float(abs(value)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["horizon", "abs_corr"], ascending=[True, False])


def build_feature_tail_summary(frames: dict[str, pd.DataFrame], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for horizon, frame in frames.items():
        for feature in feature_cols:
            values = pd.to_numeric(frame[feature], errors="coerce").dropna()
            median = values.median()
            mad = (values - median).abs().median()
            robust_z = (values - median).abs() / (1.4826 * mad) if mad > 0 else pd.Series(np.zeros(len(values)))
            rows.append(
                {
                    "horizon": horizon,
                    "feature": feature,
                    "n": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "skew": float(values.skew()),
                    "kurtosis": float(values.kurt()),
                    "p01": float(values.quantile(0.01)),
                    "p99": float(values.quantile(0.99)),
                    "max_abs_robust_z": float(robust_z.max()),
                    "share_abs_robust_z_gt_5": float((robust_z > 5).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_distance_summary(
    frames: dict[str, pd.DataFrame], feature_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    samples = {}
    for horizon, frame in frames.items():
        x = frame[feature_cols].astype(float)
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        matrix = x.to_numpy()
        diffs = matrix[:, None, :] - matrix[None, :, :]
        distances = np.sqrt(np.mean(diffs**2, axis=2))
        upper = distances[np.triu_indices_from(distances, k=1)]
        samples[horizon] = upper
        rows.append(
            {
                "horizon": horizon,
                "n_rows": int(len(frame)),
                "n_features": int(len(feature_cols)),
                "p05_normalized_distance": float(np.quantile(upper, 0.05)),
                "p25_normalized_distance": float(np.quantile(upper, 0.25)),
                "median_normalized_distance": float(np.quantile(upper, 0.50)),
                "p75_normalized_distance": float(np.quantile(upper, 0.75)),
                "p95_normalized_distance": float(np.quantile(upper, 0.95)),
            }
        )
    return pd.DataFrame(rows), samples


def plot_task_correlations(task_corrs: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, (horizon, corr) in zip(axes, task_corrs.items(), strict=True):
        image = axis.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_title(f"{horizon}: target task correlation")
        axis.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(corr.index)), corr.index)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                axis.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.savefig(FIGURE_DIR / "gp_target_task_correlations.png", dpi=200)
    plt.close(fig)


def plot_task_eigens(task_corrs: dict[str, pd.DataFrame]) -> None:
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(1, 6)
    for horizon, corr in task_corrs.items():
        eigvals = np.sort(np.linalg.eigvalsh(corr.to_numpy()))[::-1]
        shares = eigvals / eigvals.sum()
        axis.plot(x, shares, marker="o", label=horizon)
    axis.set_title("Task-correlation eigenvalue shares")
    axis.set_xlabel("Correlation principal component")
    axis.set_ylabel("Share of task-correlation variance")
    axis.set_xticks(x)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(FIGURE_DIR / "gp_task_correlation_eigens.png", dpi=200)
    plt.close(fig)


def plot_target_scale(task_summary: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for horizon, group in task_summary.groupby("horizon", sort=False):
        axis.scatter(group["std"] * 100, group["mean"] * 100, s=80, label=horizon)
        for _, row in group.iterrows():
            axis.text(row["std"] * 100, row["mean"] * 100, row["asset_id"], fontsize=8, ha="left", va="bottom")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Output mean vs volatility by ETF task")
    axis.set_xlabel("Target std (%)")
    axis.set_ylabel("Target mean (%)")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(FIGURE_DIR / "gp_target_mean_vs_std.png", dpi=200)
    plt.close(fig)


def plot_feature_redundancy(redundancy: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, (horizon, group) in zip(axes, redundancy.groupby("horizon", sort=False), strict=True):
        top = group.head(15).iloc[::-1]
        labels = [f"{a} | {b}" for a, b in zip(top["feature_a"], top["feature_b"], strict=True)]
        axis.barh(labels, top["corr"], color=np.where(top["corr"] >= 0, "#69a761", "#d95f5f"))
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(f"{horizon}: most redundant feature pairs")
        axis.set_xlabel("Pearson correlation")
    fig.savefig(FIGURE_DIR / "gp_feature_redundancy_top_pairs.png", dpi=200)
    plt.close(fig)


def plot_feature_tails(feature_tail: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, (horizon, group) in zip(axes, feature_tail.groupby("horizon", sort=False), strict=True):
        top = group.sort_values("max_abs_robust_z", ascending=False).head(15).iloc[::-1]
        axis.barh(top["feature"], top["max_abs_robust_z"], color="#7aa6c2")
        axis.axvline(5, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"{horizon}: largest robust-z feature outliers")
        axis.set_xlabel("Max absolute robust z-score")
    fig.savefig(FIGURE_DIR / "gp_feature_tail_outliers.png", dpi=200)
    plt.close(fig)


def plot_distance_diagnostics(distance_samples: dict[str, np.ndarray], distance_summary: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for horizon, values in distance_samples.items():
        axis.hist(values, bins=40, alpha=0.45, density=True, label=horizon)
        median = distance_summary.loc[distance_summary["horizon"] == horizon, "median_normalized_distance"].iloc[0]
        axis.axvline(median, linestyle="--", linewidth=1.5)
    axis.set_title("Standardized feature-space pairwise distances")
    axis.set_xlabel("sqrt(mean squared standardized feature difference)")
    axis.set_ylabel("Density")
    axis.legend()
    fig.savefig(FIGURE_DIR / "gp_standardized_feature_distances.png", dpi=200)
    plt.close(fig)


def write_report(
    task_summary: pd.DataFrame,
    eig_summary: pd.DataFrame,
    redundancy: pd.DataFrame,
    feature_tail: pd.DataFrame,
    distance_summary: pd.DataFrame,
) -> None:
    lines = [
        "# GP Prior And Configuration EDA\n",
        "This pass is aimed at choices for GP and multitask GP configuration, not at proving forecast skill.\n",
        "\n## Output And Task Structure\n",
    ]
    for horizon, group in task_summary.groupby("horizon", sort=False):
        pooled_std = group["std"].mean()
        mean_abs = group["mean"].abs().mean()
        lines.append(
            f"- `{horizon}`: average per-task target std `{pooled_std:.4f}` "
            f"({pooled_std * 100:.2f}%), average absolute task mean `{mean_abs:.4f}` "
            f"({mean_abs * 100:.2f}%).\n"
        )
        eig = eig_summary[eig_summary["horizon"] == horizon]
        pc1 = eig.loc[eig["component"] == 1, "variance_share"].iloc[0]
        pc2 = eig.loc[eig["component"] == 2, "cumulative_variance_share"].iloc[0]
        lines.append(
            f"  Task-correlation PC1 explains `{pc1:.2%}`; first two PCs explain `{pc2:.2%}`.\n"
        )

    lines.extend(
        [
            "\nImplication: start MTGP coregionalization with rank 1-2, not a fully flexible task covariance. "
            "The tasks are related enough for pooling, but style/size differences mean rank 1 alone may be "
            "too restrictive.\n",
            "\n## Feature Space And Kernel Implications\n",
        ]
    )
    for _, row in distance_summary.iterrows():
        lines.append(
            f"- `{row['horizon']}` standardized feature-space median pairwise distance is "
            f"`{row['median_normalized_distance']:.2f}` with 5-95% range "
            f"`{row['p05_normalized_distance']:.2f}`-`{row['p95_normalized_distance']:.2f}`.\n"
        )
    lines.append(
        "\nImplication: standardize all predictors before GP fitting. Use conservative lengthscale priors "
        "or ARD regularization; short lengthscales across all 44 features would overfit these small panels.\n"
    )

    lines.append("\n## Redundancy Pressure\n")
    for horizon, group in redundancy.groupby("horizon", sort=False):
        top = group.head(5)
        lines.append(f"\n`{horizon}` highest absolute feature correlations:\n")
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['feature_a']}` vs `{row['feature_b']}`: corr `{row['corr']:.3f}`\n"
            )
    lines.append(
        "\nImplication: the first experiment batch should compare compact blocks against broader blocks. "
        "Momentum variants, volatility variants, and curve/macro variables contain overlapping information.\n"
    )

    tail_top = feature_tail.sort_values("max_abs_robust_z", ascending=False).groupby("horizon").head(5)
    lines.append("\n## Tail And Robustness Pressure\n")
    for horizon, group in tail_top.groupby("horizon", sort=False):
        lines.append(f"\n`{horizon}` largest feature outliers:\n")
        for _, row in group.iterrows():
            lines.append(
                f"- `{row['feature']}`: max robust z `{row['max_abs_robust_z']:.1f}`, "
                f"share >5 `{row['share_abs_robust_z_gt_5']:.2%}`\n"
            )
    lines.append(
        "\nImplication: keep clipping/robust scaling in the feature pipeline and consider Student-t or "
        "noise-floor sensitivity for targets when evaluating GP fits.\n"
    )

    lines.extend(
        [
            "\n## Starting Prior Suggestions To Test\n",
            "- Mean: per-task constant mean near zero, with weak task offsets around the observed 0.5%-1.1% "
            "one-period scale.\n",
            "- Observation noise: initialize around 3%-6% target std by asset/horizon; test a floor so the GP "
            "cannot explain all noise as signal.\n",
            "- Task covariance: LKJ or low-rank coregionalization with moderate shrinkage toward positive "
            "correlation; compare rank 1 vs rank 2.\n",
            "- Kernel: standardized inputs, ARD RBF or Matern, conservative lengthscale prior, and feature-block "
            "ablations before all-feature ARD.\n",
            "- Output scale: regularize strongly enough that posterior means can move, but avoid letting weak "
            "feature correlations create spurious sharp forecasts.\n",
            "\n## Files\n",
            "- `figures/gp_target_task_correlations.png`\n",
            "- `figures/gp_task_correlation_eigens.png`\n",
            "- `figures/gp_target_mean_vs_std.png`\n",
            "- `figures/gp_feature_redundancy_top_pairs.png`\n",
            "- `figures/gp_feature_tail_outliers.png`\n",
            "- `figures/gp_standardized_feature_distances.png`\n",
        ]
    )
    (EDA_DIR / "GP_PRIOR_EDA_SUMMARY.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
