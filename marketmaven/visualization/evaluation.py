import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from typing import Dict, List


# =========================================================
# Helper: MLflow save + log
# =========================================================
def save_plot(fig, name):
    """Save a Matplotlib figure to MLflow."""
    path = f"marketmaven/mlflow/artifacts/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)


# =========================================================
# 1. TRUE vs PREDICTED
# =========================================================
def plot_true_vs_pred(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    for asset in y_true.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_true[asset], label="True", lw=2)
        ax.plot(y_pred[asset], label="Predicted", lw=2)
        ax.set_title(f"{asset}: True vs Predicted Returns")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_plot(fig, f"true_vs_pred_{asset}")


# =========================================================
# 2. SCATTER: PREDICTED vs TRUE
# =========================================================
def plot_scatter_pooled(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    x = y_pred.values.flatten()
    y = y_true.values.flatten()

    ax.scatter(x, y, alpha=0.3)
    lims = [
        min(x.min(), y.min()),
        max(x.max(), y.max())
    ]
    ax.plot(lims, lims, 'r--', lw=2)
    
    ax.set_title("Pooled: Predicted vs True returns")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Realized")
    ax.grid(True, alpha=0.3)

    save_plot(fig, "pooled_scatter_true_vs_pred")


# =========================================================
# 3. RESIDUALS OVER TIME
# =========================================================
def plot_residuals(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    residuals = y_true - y_pred
    
    for asset in y_true.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(residuals[asset], lw=1.5)
        ax.axhline(0, lw=1, color="black")
        ax.set_title(f"{asset}: Residuals Over Time")
        ax.grid(True, alpha=0.3)
        save_plot(fig, f"residuals_{asset}")


# =========================================================
# 4. INFORMATION COEFFICIENT (IC) OVER TIME
# =========================================================
def compute_ic_series(y_true, y_pred):
    ic_list = []
    for t in range(len(y_true)):
        ic = y_true.iloc[t].corr(y_pred.iloc[t], method="spearman")
        ic_list.append(ic)
    return pd.Series(ic_list, name="IC")


def plot_ic_timeseries(ic_series: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ic_series, label="IC", lw=1.8)
    ax.plot(ic_series.rolling(3).mean(), label="3-period MA", lw=2)
    ax.set_title("Information Coefficient Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "ic_timeseries")


def plot_ic_hist(ic_series: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ic_series.dropna(), bins=20, alpha=0.7)
    ax.set_title("Distribution of Information Coefficient")
    ax.set_xlabel("IC")
    save_plot(fig, "ic_hist")


# =========================================================
# 5. LONG–SHORT RETURNS
# =========================================================
def plot_ls_cumulative(ret: pd.Series, label: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    cumulative = (1 + ret).cumprod()
    ax.plot(cumulative, label=label, lw=2)
    ax.set_title(f"{label}: Long–Short Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, f"ls_cumulative_{label}")


def plot_ls_drawdown(ret: pd.Series, label: str):
    cumulative = (1 + ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(drawdown, lw=2)
    ax.set_title(f"{label}: Long–Short Drawdown")
    ax.grid(True, alpha=0.3)
    save_plot(fig, f"ls_drawdown_{label}")


def plot_rolling_sharpe(ret: pd.Series, label: str, window=12):
    rolling_sharpe = ret.rolling(window).mean() / ret.rolling(window).std()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling_sharpe, lw=2)
    ax.set_title(f"{label}: Rolling Sharpe ({window}-period)")
    ax.grid(True, alpha=0.3)
    save_plot(fig, f"rolling_sharpe_{label}")


# =========================================================
# 6. GP UNCERTAINTY (IF APPLICABLE)
# =========================================================
def plot_gp_uncertainty(y_true, y_pred, y_std):
    # y_pred, y_std should be DataFrames
    for asset in y_true.columns:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(y_true[asset], label="True", color="black", lw=2)
        ax.plot(y_pred[asset], label="Pred Mean", color="blue", lw=2)

        upper = y_pred[asset] + y_std[asset]
        lower = y_pred[asset] - y_std[asset]
        ax.fill_between(np.arange(len(y_pred)), lower, upper,
                        color="blue", alpha=0.2, label="±1 std")

        ax.set_title(f"{asset}: GP Predictive Uncertainty")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_plot(fig, f"gp_uncertainty_{asset}")


def plot_uncertainty_calibration(y_true, y_pred, y_std):
    abs_err = (y_true - y_pred).abs()
    std_vals = y_std

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(std_vals.values.flatten(), abs_err.values.flatten(), alpha=0.3)
    ax.set_xlabel("Predicted Std")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Uncertainty Calibration")
    ax.grid(True, alpha=0.3)

    save_plot(fig, "gp_uncertainty_calibration")
    
def plot_ls_cumulative_compare(ls_gp: pd.Series,
                               ls_mean: pd.Series,
                               ls_ewma: pd.Series):
    """
    Plot long–short cumulative returns for GP, Mean, and EWMA
    on the same figure for direct comparison.
    """

    cumulative_gp   = (1 + ls_gp).cumprod()
    cumulative_mean = (1 + ls_mean).cumprod()
    cumulative_ewma = (1 + ls_ewma).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(cumulative_gp,   label="GP",   lw=2)
    ax.plot(cumulative_mean, label="Mean", lw=2)
    ax.plot(cumulative_ewma, label="EWMA2", lw=2)

    ax.set_title("Long–Short Strategy: Cumulative Returns Comparison")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_plot(fig, "ls_cumulative_comparison")
    

def plot_actual_vs_pred_matrix(true_df, pred_df, asset_cols, save_path):
    # ---- Determine global y range across ALL assets ----
    y_min = true_df.min().min()
    y_max = true_df.max().max()
    pad = 0.05 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    n_assets = len(asset_cols)
    ncols = 4
    nrows = int(np.ceil(n_assets / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3*nrows),
        sharex=True
    )

    axes = axes.flatten()
    x = np.arange(len(true_df))

    for i, asset in enumerate(asset_cols):
        ax = axes[i]

        y_true = true_df[f"{asset}_true"].values
        y_pred = pred_df[f"{asset}_pred"].values

        ax.plot(x, y_true, label="Actual", linewidth=1.5)
        ax.plot(x, y_pred, label="Predicted", linewidth=1.5)

        ax.set_ylim(y_min, y_max)      # <---- SAME SCALE FOR ALL
        ax.set_title(asset)
        ax.grid(True, alpha=0.3)

        if i % ncols == 0:
            ax.set_ylabel("Return")

    # Remove unused
    for j in range(len(asset_cols), len(axes)):
        fig.delaxes(axes[j])

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=12)

    fig.suptitle("Actual vs Predicted — Per ETF (Holdout Periods)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig(save_path, dpi=200)
    plt.close(fig)

