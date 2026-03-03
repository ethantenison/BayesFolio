from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from sklearn.metrics import r2_score


def evaluate_asset_pricing(y_test: pd.DataFrame, y_pred: pd.DataFrame):
    """
    Evaluate predictive performance using metrics from
    'Empirical Asset Pricing via Ensemble Gaussian Process Regression'.
    I added IC p-value to help determine if good IC was just a fluke.

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
            "IC": float,
            "IC_p_value": float,
            "IR": float,
            "HitRatio": float
        }
    """
    # Align dataframes
    y_true = y_test.copy().reset_index(drop=True)
    y_hat = y_pred.copy().reset_index(drop=True)
    y_hat.columns = y_true.columns  # ensure consistent ordering

    # Drop rows with all NaNs
    valid_idx = ~(y_true.isna().all(axis=1) | y_hat.isna().all(axis=1))
    y_true, y_hat = y_true.loc[valid_idx], y_hat.loc[valid_idx]

    # --- 1. Pooled R2 (aggregate) ---
    num = ((y_true - y_hat) ** 2).to_numpy().sum()
    denom = ((y_true - y_true.mean()) ** 2).to_numpy().sum()
    R2_pooled = 1 - num / denom if denom > 0 else np.nan

    # --- 2. Average R2 (per asset) ---
    R2_list = []
    for col in y_true.columns:
        yt, yp = y_true[col], y_hat[col]
        mask = yt.notna() & yp.notna()
        if mask.sum() > 1:
            R2_list.append(r2_score(yt[mask], yp[mask]))
    R2_avg = np.nanmean(R2_list) if R2_list else np.nan

    # --- 3. Cross-sectional IC per time (Spearman) ---
    IC_list = []
    Hit_list = []
    for i in range(len(y_true)):
        yt, yp = y_true.iloc[i, :], y_hat.iloc[i, :]
        mask = yt.notna() & yp.notna()
        if mask.sum() > 1:
            ic_val, _ = spearmanr(yt[mask], yp[mask], nan_policy="omit")
            IC_list.append(ic_val)

            # Hit ratio for this time step
            hit = (np.sign(yt[mask]) == np.sign(yp[mask])).mean()
            Hit_list.append(hit)

    IC = np.nanmean(IC_list) if IC_list else np.nan
    IC_std = np.nanstd(IC_list, ddof=1) if len(IC_list) > 1 else np.nan

    if len(IC_list) > 1:
        ic_arr = np.asarray(IC_list, dtype=float)
        ic_arr = ic_arr[np.isfinite(ic_arr)]

        if ic_arr.size > 1:
            res = cast(Any, ttest_1samp(ic_arr, 0.0, nan_policy="omit"))

            # Works for both SciPy return styles: object with .pvalue or tuple-like
            p_raw = res.pvalue if hasattr(res, "pvalue") else res[1]

            # If p_raw is a numpy scalar/array, convert safely to Python float
            p_val = float(np.asarray(p_raw).item())
        else:
            p_val = float(np.nan)
    else:
        p_val = float(np.nan)

    IR = IC / IC_std if IC_std and not np.isnan(IC_std) and IC_std > 0 else np.nan

    HitRatio = np.nanmean(Hit_list) if Hit_list else np.nan

    return {
        "R2_pooled": float(R2_pooled),
        "R2_avg": float(R2_avg),
        "IC": float(IC),
        "IC_p_value": float(p_val),
        "IR": float(IR),
        "HitRatio": float(HitRatio),
    }
