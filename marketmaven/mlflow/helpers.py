import pandas as pd
import numpy as np
from riskfolio.src.ParamsEstimation import mean_vector
from pydantic import BaseModel, ConfigDict
from marketmaven.models.kernels import MeanF, KernelType
import mlflow

class KernelF(BaseModel):
    typef: KernelType
    featuresf: list[str]
    active_dims_features: list[int]
    smoothnessf: float
    
    model_config = ConfigDict(use_enum_values=True)
    
class KernelT(BaseModel):
    typet: KernelType
    featurest: list[str]
    smoothnesst: float
    active_dims_time: list[int]
    
    model_config = ConfigDict(use_enum_values=True)
    
class MultiTaskConfig(BaseModel):
    num_tasks: int
    mean: MeanF
    rank: int
    scaling: str
    model_config = ConfigDict(use_enum_values=True)

def long_to_panel(y_tensor, I_tensor, asset_names):
    """
    Convert long-format torch vectors into a wide (panel) DataFrame:
    one row per time, one column per asset.
    """
    y_np = y_tensor.cpu().numpy().reshape(-1)
    asset_np = I_tensor.cpu().numpy().reshape(-1).astype(int)

    df = pd.DataFrame({
        "asset": [asset_names[i] for i in asset_np],
        "value": y_np,
    })

    # time index is implicit by occurrence order
    # we assign row numbers for each group of assets
    df["time"] = np.repeat(np.arange(len(df) // len(asset_names)), len(asset_names))

    return df.pivot(index="time", columns="asset", values="value")

def compute_benchmark_panel(y_train, y_test, method="mean"):
    if method == "mean":
        vec = y_train.mean().values
    elif method == "ewma2":
        vec = mean_vector(X=y_train, method=method).values.reshape(-1)
    else:
        raise ValueError
    
    pred = np.tile(vec, (len(y_test), 1))
    return pd.DataFrame(pred, index=y_test.index, columns=y_test.columns)


def r2_os(y_true: pd.DataFrame, y_pred: pd.DataFrame, y_bench: pd.DataFrame):
    """
    Campbell–Thompson out-of-sample R² (R²_OS).Interpretation is that positive means GP is better. 

    Parameters
    ----------
    y_true : pd.DataFrame
        Realized returns. Shape (T × n_assets)
    y_pred : pd.DataFrame
        Model forecasts. Shape (T × n_assets)
    y_bench : pd.DataFrame
        Benchmark forecasts (rolling mean or EWMA2). Shape (T × n_assets)

    Returns
    -------
    dict with:
        - pooled R²_OS
        - average R²_OS across assets
        - per-asset R²_OS
    """

    # --- 1. Residual sum of squares for model and benchmark ---
    ss_model  = ((y_true - y_pred)**2).sum(axis=0)   # per asset
    ss_bench  = ((y_true - y_bench)**2).sum(axis=0)  # per asset

    # --- 2. R²_OS per asset ---
    r2_asset = 1 - ss_model / ss_bench
    r2_asset = r2_asset.replace([np.inf, -np.inf], np.nan)

    # --- 3. Average across assets ---
    r2_avg = r2_asset.mean()

    # --- 4. Pooled (sum errors first, then compute R²) ---
    total_ss_model = ((y_true - y_pred)**2).to_numpy().sum()
    total_ss_bench = ((y_true - y_bench)**2).to_numpy().sum()
    r2_pooled = 1 - total_ss_model / total_ss_bench

    return {
        "R2_OS_pooled": float(r2_pooled),
        "R2_OS_avg": float(r2_avg),
        "R2_OS_per_asset": r2_asset.to_dict()
    }
    
# Flatten before logging
def log_r2_os(prefix, r2_dict):
    flat_metrics = {}

    for k, v in r2_dict.items():
        if isinstance(v, dict):
            # flatten per-asset values
            for asset, val in v.items():
                flat_metrics[f"{prefix}_{asset}"] = float(val)
        else:
            flat_metrics[f"{prefix}_{k}"] = float(v)

    mlflow.log_metrics(flat_metrics)