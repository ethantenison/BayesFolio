import torch
from scipy.stats import spearmanr
import gpytorch
import numpy as np
import pandas as pd
from scipy.signal import periodogram
import math

def gaussian_nlpd(y_true: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor, eps: float = 1e-12) -> float:
    var = (y_std.clamp_min(eps))**2
    return float(0.5 * ( ((y_true - y_mean)**2) / var + torch.log(2 * torch.pi * var) ).mean())

def mse(y_true: torch.Tensor, y_mean: torch.Tensor) -> float:
    return float(((y_true - y_mean)**2).mean())

def rmse(y_true: torch.Tensor, y_mean: torch.Tensor) -> float:
    return float(torch.sqrt(((y_true - y_mean)**2).mean()))

def r2_score(y_true: torch.Tensor, y_mean: torch.Tensor) -> float:
    y_bar = y_true.mean()
    ss_tot = ((y_true - y_bar)**2).sum()
    ss_res = ((y_true - y_mean)**2).sum()
    return float(1 - ss_res / ss_tot)



def compute_market_return_series(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "y_excess_lead",
):
    """
    Cross-sectional mean excess return per time period.
    """
    ts = (
        df
        .groupby(date_col)[target_col]
        .mean()
        .sort_index()
        .dropna()
    )
    return ts


def compute_periodogram_from_panel(


    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "y_excess_lead",
    freq_per_year: int = 12,  # monthly data
    detrend: bool = True,
):
    """
    Computes the periodogram of the cross-sectional mean return.
    """

    ts = compute_market_return_series(df, date_col, target_col)

    x = np.asarray(ts.values, dtype=np.float64)

    # Optional but recommended: remove mean / trend
    if detrend:
        x = x - np.mean(x)

    freqs, power = periodogram(
        x,
        fs=freq_per_year,      # samples per year
        scaling="density",
        window="hann"
    )

    return pd.DataFrame({
        "frequency": freqs,
        "power": power,
    }), ts
