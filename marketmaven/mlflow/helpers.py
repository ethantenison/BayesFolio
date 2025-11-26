import pandas as pd
import numpy as np
from riskfolio.src.ParamsEstimation import mean_vector
from pydantic import BaseModel, ConfigDict
from marketmaven.models.kernels import MeanF, KernelType
import mlflow
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from gpytorch.constraints import GreaterThan

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


def describe_prior(p: Prior):
    """Return full prior information."""
    d = {"type": p.__class__.__name__}
    for attr in ["loc", "scale", "concentration", "df", "covariance_matrix"]:
        if hasattr(p, attr):
            try:
                d[attr] = float(getattr(p, attr))
            except:
                d[attr] = str(getattr(p, attr))
    return d


def describe_constraint(c: GreaterThan):
    """Return constraint information."""
    d = {"type": c.__class__.__name__}
    for attr in ["lower_bound", "upper_bound", "initial_value"]:
        if hasattr(c, attr):
            d[attr] = getattr(c, attr).item() if hasattr(getattr(c, attr), "item") else getattr(c, attr)
    return d


def describe_kernel(k: Kernel):
    """Recursively describe GPyTorch kernels."""
    out = {"type": k.__class__.__name__}

    # Smoothness for Matern
    if hasattr(k, "nu"):
        out["nu"] = float(k.nu)

    # Periodic kernel
    if hasattr(k, "period_length") and k.period_length is not None:
        try:
            out["period_length"] = k.period_length.item()
        except:
            out["period_length"] = str(k.period_length)

        if hasattr(k, "period_length_prior") and k.period_length_prior is not None:
            out["period_length_prior"] = describe_prior(k.period_length_prior)

    # ARD dims
    if hasattr(k, "ard_num_dims"):
        out["ard_num_dims"] = k.ard_num_dims

    # lengthscale — must check for None
    if hasattr(k, "lengthscale") and k.lengthscale is not None:
        try:
            out["lengthscale"] = k.lengthscale.detach().cpu().numpy().tolist()
        except:
            out["lengthscale"] = str(k.lengthscale)

    # lengthscale prior
    if hasattr(k, "lengthscale_prior") and k.lengthscale_prior is not None:
        out["lengthscale_prior"] = describe_prior(k.lengthscale_prior)

    # lengthscale constraint
    if hasattr(k, "raw_lengthscale_constraint") and k.raw_lengthscale_constraint is not None:
        out["lengthscale_constraint"] = describe_constraint(k.raw_lengthscale_constraint)

    # Additive / Product kernels
    if hasattr(k, "kernels"):
        out["sub_kernels"] = [describe_kernel(sub) for sub in k.kernels]

    # Index kernels
    if hasattr(k, "num_tasks"):
        out["num_tasks"] = k.num_tasks

    if hasattr(k, "rank"):
        out["rank"] = k.rank

    # LKJ prior for task covariance
    if hasattr(k, "task_prior") and k.task_prior is not None:
        out["task_prior"] = describe_prior(k.task_prior)

    return out

def extract_full_gp_config(model):
    """Extract full architecture, priors, hyperparameters, and constraints."""
    cfg = {}

    # ----- Mean module -----
    cfg["mean_module"] = str(model.mean_module)

    # ----- Feature kernel -----
    cfg["covar_module"] = describe_kernel(model.covar_module)

    # ----- Task kernel -----
    cfg["task_covar_module"] = describe_kernel(model.task_covar_module)

    # ----- Likelihood noise priors -----
    if hasattr(model.likelihood, "noise"):
        cfg["noise"] = model.likelihood.noise.tolist()

    if hasattr(model.likelihood, "noise_prior") and model.likelihood.noise_prior is not None:
        cfg["noise_prior"] = describe_prior(model.likelihood.noise_prior)

    if hasattr(model.likelihood, "raw_noise_constraint"):
        cfg["noise_constraint"] = describe_constraint(model.likelihood.raw_noise_constraint)

    # ----- Learned hyperparameters -----
    params = {}
    for k, v in model.named_parameters():
        params[k] = v.detach().cpu().numpy().tolist()
    cfg["learned_parameters"] = params

    return cfg

def model_error_by_time_index(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """
    Compute absolute errors per asset and overall mean absolute error
    """
    # Combine true and predicted into one DataFrame
    combined_df = pd.concat(
        [y_true.add_suffix("_true"),
        y_pred.add_suffix("_pred")],
        axis=1
    )

    # Add error columns for each asset
    for asset in y_true.columns:
        true_col = f"{asset}_true"
        pred_col = f"{asset}_pred"

        combined_df[f"{asset}_abs_error"] = (combined_df[pred_col] - combined_df[true_col]).abs()

    combined_df["mean_abs_error"] = combined_df[[f"{a}_abs_error" for a in y_true.columns]].mean(axis=1)
    # Build the desired column order
    ordered_cols = []

    for asset in y_true.columns:
        ordered_cols.append(f"{asset}_true")
        ordered_cols.append(f"{asset}_pred")
        ordered_cols.append(f"{asset}_abs_error")

    # Append overall metric at the end
    ordered_cols.append("mean_abs_error")

    # Reorder DataFrame
    combined_df = combined_df[ordered_cols]

    return combined_df





