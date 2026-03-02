import pandas as pd
import numpy as np
from riskfolio.src.ParamsEstimation import mean_vector
from pydantic import BaseModel, ConfigDict
from bayesfolio.engine.models.gp.kernels import KernelConfig
from bayesfolio.engine.models.gp.means import MeanF
import mlflow
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior, LKJCovariancePrior
from gpytorch.constraints import GreaterThan
from bayesfolio.engine.models.gp.kernels import KernelArchitectureConfig

def log_kernel_architecture_detailed(kernel_arch: KernelArchitectureConfig, prefix: str = "kernel"):
    """Log kernel architecture with separated block information."""
    # Log global structure
    mlflow.log_param(f"{prefix}.global_structure", kernel_arch.global_structure.value)
    mlflow.log_param(f"{prefix}.interaction_policy", kernel_arch.interaction_policy.value)
    mlflow.log_param(f"{prefix}.num_blocks", len(kernel_arch.blocks))
    
    # Log each block separately
    for i, block in enumerate(kernel_arch.blocks):
        block_prefix = f"{prefix}.block_{i}"
        mlflow.log_param(f"{block_prefix}.variable_type", block.variable_type.value)
        mlflow.log_param(f"{block_prefix}.block_structure", block.block_structure.value)
        mlflow.log_param(f"{block_prefix}.dims", str(block.dims))
        mlflow.log_param(f"{block_prefix}.num_dims", len(block.dims))
        
        # Log base kernel config
        base_kernel = block.base_kernel
        mlflow.log_param(f"{block_prefix}.base_kernel.type", base_kernel.kernel_type.value)
        mlflow.log_param(f"{block_prefix}.base_kernel.ard", base_kernel.ard)
        
        # Log kernel-specific parameters
        if hasattr(base_kernel, 'nu'):
            mlflow.log_param(f"{block_prefix}.base_kernel.nu", base_kernel.nu)
        if hasattr(base_kernel, 'depth'):
            mlflow.log_param(f"{block_prefix}.base_kernel.depth", base_kernel.depth)
    
    # Also save full config as JSON for complete record
    mlflow.log_dict(kernel_arch.model_dump(), f"{prefix}_architecture.json")

    
def log_kernel_to_mlflow(kernel: KernelConfig, prefix: str):
    params = kernel.model_dump()
    prefixed = {f"{prefix}_{k}": v for k, v in params.items()}
    mlflow.log_params(prefixed)
    
def extract_gp_hyperparameters(model):
    params = {}

    # --------------------------------------------------
    # Likelihood (noise)
    # --------------------------------------------------
    if hasattr(model, "likelihood"):
        likelihood = model.likelihood
        noise = getattr(likelihood, "noise", None)
        if noise is not None:
            params["likelihood.noise"] = noise.detach().cpu().numpy().tolist()

    # --------------------------------------------------
    # Mean modules
    # --------------------------------------------------
    if hasattr(model, "mean_module"):
        mean = model.mean_module
        if hasattr(mean, "base_means"):
            for i, m in enumerate(mean.base_means):
                const = getattr(m, "constant", None)
                if const is not None:
                    params[f"mean.base_means.{i}.constant"] = (
                        const.detach().cpu().numpy().tolist()
                    )
        else:
            const = getattr(mean, "constant", None)
            if const is not None:
                params["mean.constant"] = const.detach().cpu().numpy().tolist()

    # --------------------------------------------------
    # Kernel walker
    # --------------------------------------------------
    def walk_kernel(kernel, prefix):
        out = {}

        # Lengthscale
        lengthscale = getattr(kernel, "lengthscale", None)
        if lengthscale is not None:
            out[f"{prefix}.lengthscale"] = (
                lengthscale.detach().cpu().numpy().tolist()
            )

        # Outputscale (ScaleKernel)
        outputscale = getattr(kernel, "outputscale", None)
        if outputscale is not None:
            out[f"{prefix}.outputscale"] = (
                outputscale.detach().cpu().numpy().tolist()
            )

        # Variance (LinearKernel)
        variance = getattr(kernel, "variance", None)
        if variance is not None:
            out[f"{prefix}.variance"] = (
                variance.detach().cpu().numpy().tolist()
            )

        # Composite kernels
        kernels = getattr(kernel, "kernels", None)
        if kernels is not None:
            for i, subkernel in enumerate(kernels):
                out.update(
                    walk_kernel(subkernel, f"{prefix}.kernels.{i}")
                )

        # Wrapped kernels (ScaleKernel, etc.)
        base_kernel = getattr(kernel, "base_kernel", None)
        if base_kernel is not None:
            out.update(
                walk_kernel(base_kernel, f"{prefix}.base_kernel")
            )

        return out

    # Main covariance kernel
    if hasattr(model, "covar_module"):
        params.update(walk_kernel(model.covar_module, "covar_module"))

    # --------------------------------------------------
    # Task covariance (PositiveIndexKernel / IndexKernel)
    # --------------------------------------------------
    if hasattr(model, "task_covar_module"):
        task_kernel = model.task_covar_module

        var = getattr(task_kernel, "var", None)
        if var is not None:
            params["task_covar.var"] = var.detach().cpu().numpy().tolist()

        covar_factor = getattr(task_kernel, "covar_factor", None)
        if covar_factor is not None:
            params["task_covar.covar_factor"] = (
                covar_factor.detach().cpu().numpy().tolist()
            )

    return params


def log_gp_hyperparameters(model, artifact_name="gp_hyperparameters.json"):
    params = extract_gp_hyperparameters(model)
    mlflow.log_dict(params, artifact_name)


def log_gpytorch_state_dict(model, artifact_name="gp_state.json"):
    sd = model.state_dict()
    sd_clean = {k: v.detach().cpu().numpy().tolist() for k, v in sd.items()}
    mlflow.log_dict(sd_clean, artifact_name)


class MultiTaskConfig(BaseModel):
    num_tasks: int
    mean: MeanF | dict[str, MeanF]
    rank: int
    scaling: str
    min_noise: float
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

def describe_task_kernel(tk):
    out = {
        "type": tk.__class__.__name__,
        "num_tasks": tk.num_tasks,
        "rank": tk.raw_covar_factor.shape[-1],
        "target_task_index": tk.target_task_index,
        "unit_scale_for_target": tk.unit_scale_for_target,
    }

    # raw parameters (pre-constraints)
    # out["raw_var"] = tk.raw_var.detach().cpu().numpy().tolist()
    # out["raw_covar_factor"] = tk.raw_covar_factor.detach().cpu().numpy().tolist()

    # transformed parameters
    out["var"] = tk.var.detach().cpu().numpy().tolist()
    out["covar_factor"] = tk.covar_factor.detach().cpu().numpy().tolist()

    # # full covariance & correlation matrix
    # B = tk.covar_matrix.detach().cpu().numpy()
    # out["covar_matrix"] = B.tolist()

    # d = np.sqrt(np.clip(B.diagonal(), 1e-12, None))
    # corr = B / (d[:,None] * d[None,:])
    # out["corr_matrix"] = corr.tolist()

    # --- Priors (including LKJCovariancePrior) -----------------------------
    if hasattr(tk, "task_prior") and tk.task_prior is not None:
        out["task_prior"] = describe_prior(tk.task_prior)


    return out

def describe_prior(p: Prior):
    """Return a JSON-serializable dict describing a GPyTorch Prior."""
    out = {"type": p.__class__.__name__}

    # --- Special case: LKJCovariancePrior ---------------------------------
    if isinstance(p, LKJCovariancePrior):
        # In recent gpytorch, attributes are usually: n (dim), eta, sd_prior
        # but there can be minor version differences, so be defensive.
        if hasattr(p, "n"):
            out["dimension"] = int(p.n)
        if hasattr(p, "eta"):
            out["eta"] = float(p.eta)

        # sd_prior is itself a Prior (e.g. GammaPrior over std dev)
        if hasattr(p, "sd_prior") and p.sd_prior is not None:
            out["sd_prior"] = describe_prior(p.sd_prior)

        return out

    # --- Generic case: other priors ----------------------------------------
    for attr in ["loc", "scale", "concentration", "rate", "df", "covariance_matrix"]:
        if hasattr(p, attr):
            val = getattr(p, attr)
            try:
                out[attr] = float(val)
            except Exception:
                # Could be tensor / array / matrix
                try:
                    out[attr] = val.detach().cpu().numpy().tolist()
                except Exception:
                    out[attr] = str(val)
    return out

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
    cfg["task_covar_module"] = describe_task_kernel(model.task_covar_module)

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
    Compute absolute errors per asset and overall mean absolute error.
    Returns only columns ending in _abs_error plus the mean_abs_error column.
    """
    # Combine true and predicted into one DataFrame
    combined_df = pd.concat(
        [y_true.add_suffix("_true"),
         y_pred.add_suffix("_pred")],
        axis=1
    )

    # Compute abs error for each ETF
    abs_error_cols = []
    for asset in y_true.columns:
        true_col = f"{asset}_true"
        pred_col = f"{asset}_pred"
        err_col = f"{asset}_abs_error"

        combined_df[err_col] = (combined_df[pred_col] - combined_df[true_col]).abs()
        abs_error_cols.append(err_col)

    # Mean absolute error across ETFs
    combined_df["mean_abs_error"] = combined_df[abs_error_cols].mean(axis=1)

    # Return only the error columns + overall MAE
    return combined_df[abs_error_cols + ["mean_abs_error"]]





