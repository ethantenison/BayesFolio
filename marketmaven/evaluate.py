
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

def evaluate_asset_pricing(y_test: pd.DataFrame, y_pred: pd.DataFrame):
    """
    Evaluate predictive performance using metrics from
    'Empirical Asset Pricing via Ensemble Gaussian Process Regression'.

    Parameters
    ----------
    y_test : pd.DataFrame
        True realized excess returns (n_samples x n_assets).
    y_pred : pd.DataFrame
        Predicted excess returns (n_samples x n_assets).
        Must align with y_test in index and columns.

    Returns
    -------
    metrics : dict
        {
            "R2_pooled": float,
            "R2_avg": float,
            "IC": float
        }
    """
    # Align
    y_true = y_test.copy().reset_index(drop=True)
    y_hat = y_pred.copy().reset_index(drop=True)
    y_hat.columns = y_true.columns  # ensure same order
    
    # Drop rows with all NaN
    valid_idx = ~(y_true.isna().all(axis=1) | y_hat.isna().all(axis=1))
    y_true, y_hat = y_true.loc[valid_idx], y_hat.loc[valid_idx]

    # --- 1. Pooled R2 (aggregate numerator/denominator across all assets+times) ---
    num = ((y_true - y_hat) ** 2).to_numpy().sum()
    denom = ((y_true - y_true.mean()) ** 2).to_numpy().sum()
    R2_pooled = 1 - num / denom if denom > 0 else np.nan

    # --- 2. Average R2 (per asset, then average) ---
    R2_list = []
    for col in y_true.columns:
        yt, yp = y_true[col], y_hat[col]
        mask = yt.notna() & yp.notna()
        if mask.sum() > 1:
            R2_list.append(r2_score(yt[mask], yp[mask]))
    R2_avg = np.nanmean(R2_list) if R2_list else np.nan

    # --- 3. Information Coefficient (cross-sectional Spearman, average over time) ---
    IC_list = []
    for i in range(len(y_true)):
        yt, yp = y_true.iloc[i, :], y_hat.iloc[i, :]
        mask = yt.notna() & yp.notna()
        if mask.sum() > 1:
            IC_list.append(spearmanr(yt[mask], yp[mask]).correlation)
    IC = np.nanmean(IC_list) if IC_list else np.nan

    return {
        "R2_pooled": R2_pooled,
        "R2_avg": R2_avg,
        "IC": IC,
    }
