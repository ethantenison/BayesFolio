"""
Testing out a GP for time series data.

"""
from astropy import conf
import pandas as pd
import yfinance as yf
import warnings
from marketmaven.asset_prices import build_long_panel
from marketmaven.market_fundamentals import fetch_macro_features
from marketmaven.evaluate import evaluate_asset_pricing
from marketmaven.utils import get_current_date
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from math import sqrt, log
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
torch.set_default_dtype(torch.float32)
from pydantic import BaseModel, Field
from typing import List, Literal
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from typing import Dict, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from itertools import cycle
from marketmaven.configs import TickerConfig, Interval, Horizon
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
import math
import torch
import numpy as np
import pandas as pd
from botorch.models import KroneckerMultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.kernels import MaternKernel, PeriodicKernel, RQKernel, ProductKernel,SpectralMixtureKernel, RBFKernel, ScaleKernel

# MLFlow Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "GP_Model_Experiments"

# Set up MLFlow tracking
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment(EXPERIMENT_NAME)


warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3}'.format

# -------------------------
# 1. Configuration Class
# -------------------------

# Example usage
config = TickerConfig(
    start_date="2016-09-30",
    end_date="2025-09-01",
    interval=Interval.DAILY,
    tickers=[ "ESGD", "ISCF", "VNQ", "SNPE", "VBK"], #"XMMO", "BYLD", leaving out "HYEM", just to reduce space, taking out "IEF" cause it is not correlated, "AVEM","VNQI"
    horizon=Horizon.MONTHLY,
)
#Take away: IEF is highly uncorrelated with the others. 
# ==== CODE ====

df = build_long_panel(config.tickers, config.start_date, config.end_date, horizon=config.horizon)

##### VIX market data #####
#Chatgpt: 🔮 For a 1-month ahead excess returns model, I’d recommend adding vix_ts_level and vix as your
# core features, and optionally vix_ts_chg_1m if you want to capture regime dynamics.
# vix = fetch_vix_term_structure(start=config.start_date, end=config.end_date, freq="BM")
# vix_core = vix[['Date', 'vix', 'vix_ts_level']]


# df = df.merge(vix_core, left_on='date', right_on='Date', how='left')
# df = df.drop(columns=['Date'])

macro_features = fetch_macro_features(start=config.start_date, end=config.end_date)
df = df.merge(macro_features, left_on='date', right_on='Date', how='left')
df = df.drop(columns=['Date'])

# Create separate dataframes for each asset
def create_asset_df(df, asset_ids):
    asset_df_dict = {}
    for asset_id in asset_ids:
        asset_df = df[df['asset_id'] == asset_id].reset_index(drop=True)
        asset_df = asset_df.reset_index()
        asset_df = asset_df[['index', 'date', 'y_excess_lead']]
        asset_df_dict[asset_id] = asset_df
    return asset_df_dict

asset_dict = create_asset_df(df, config.tickers)


def plot_all_assets(asset_dict: dict[str, "pd.DataFrame"]):
    """
    Plots y_excess_lead for each asset on the same line plot, with different colors.

    Args:
        asset_dict: dictionary mapping asset_id -> dataframe 
                    Each dataframe must have 'date' and 'y_excess_lead' columns.
    """
    plt.figure(figsize=(12, 7))

    # cycle through matplotlib default colors
    color_cycler = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for asset_id, asset_df in asset_dict.items():
        color = next(color_cycler)
        plt.plot(
            asset_df["date"], 
            asset_df["y_excess_lead"], 
            label=asset_id, 
            color=color
        )

    plt.xlabel("Date")
    plt.ylabel("Excess Returns (lead)")
    plt.title("Asset Excess Returns Over Time")
    plt.legend(loc="best", ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
plot_all_assets(asset_dict)   

def create_correlation_heatmap(asset_dict: dict[str, pd.DataFrame]) -> None:
    """
    Creates and plots a heatmap of the correlation matrix of y_excess_lead for each asset.

    Args:
        asset_dict: dictionary mapping asset_id -> dataframe 
                    Each dataframe must have 'date' and 'y_excess_lead' columns.
    """
    combined_df = pd.DataFrame()

    for asset_id, asset_df in asset_dict.items():
        temp_df = asset_df[['date', 'y_excess_lead']].copy()
        temp_df.rename(columns={'y_excess_lead': asset_id}, inplace=True)
        if combined_df.empty:
            combined_df = temp_df
        else:
            combined_df = combined_df.merge(temp_df, on='date', how='outer')

    combined_df.set_index('date', inplace=True)
    correlation_matrix = combined_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=0, vmax=1)
    plt.title("Correlation Matrix Heatmap of Assets (y_excess_lead)")
    plt.show()


create_correlation_heatmap(asset_dict)
#Notes for optimization, it is better to include more pure equity or more pure bond funds to get a better spread of correlations.
#blended ones like XMMO and BYLD are more correlated to both sides and make it harder to differentiate.

# X = df[['date', 'vix', 'vix_ts_level']].drop_duplicates().reset_index(drop=True)
# X = X.reset_index()
# X = X[['index', 'vix', 'vix_ts_level']]
X = df[['date',  'vix_ts_level', 'vix_ts_z_12m', 'term_spread',
       'credit_spread', 'credit_spread_chg_1p', 'dxy', 'yc_pc1',
       'yc_pc2', 'yc_pc3']].drop_duplicates().reset_index(drop=True)
X = X.reset_index()

X = X[['index', 'vix_ts_level', 'vix_ts_z_12m', 'term_spread',
       'credit_spread', 'credit_spread_chg_1p', 'dxy', 'yc_pc1',
       'yc_pc2', 'yc_pc3']]

#Create multi column y dataframe
output_names = list(asset_dict.keys())
y = pd.DataFrame()
for name in output_names:
    temp = asset_dict[name][['date', 'y_excess_lead']].copy()
    temp.rename(columns={'y_excess_lead': name}, inplace=True)
    if y.empty:
        y = temp
    else:
        y = y.merge(temp, on='date', how='left')

y = y.iloc[:, 1:]  # drop index column

#fill na values in y dataframe
y = y.fillna(0)

# ML config

class MLConfig(BaseModel):
    hold_out_index: int = Field(3, description="Index to split train/test data")
    model_type: Literal["single", "multi_hadamard", "multi_kronecker"] = Field("multi_hadamard", description="Type of GP model to use")

# Example usage
config = MLConfig(
    hold_out_index = 1,
    model_type = "multi_kronecker"
)

scaler_x = MinMaxScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_train = X_scaled[:-config.hold_out_index]
X_test = X_scaled[-config.hold_out_index:]
y_train = y_scaled[:-config.hold_out_index]
y_test = y_scaled[-config.hold_out_index:]



# Fit scaler on train only
X_train_tensor = torch.Tensor(X_train)
X_test_tensor = torch.Tensor(X_test)

y_train_tensor = torch.Tensor(y_train)
y_test_tensor = torch.Tensor(y_test)

# Botorch multitask
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

device = torch.device("cpu")
dtype = torch.double


# Creating a specialized covar module because periodic kernel is needed:
from botorch.models.transforms import Normalize, Standardize
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)
ard_num_dims = X_train.shape[1]

active_dims_matern = list(range(X_train.shape[-1]))
ard_num_dims = len(active_dims_matern)
lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)

kernel = MaternKernel(
    nu=0.5,
    lengthscale_prior=lengthscale_prior,
    lengthscale_constraint=lengthscale_constraint,
    active_dims=active_dims_matern,
    ard_num_dims=ard_num_dims,
) + PeriodicKernel(
    period_length_prior=LogNormalPrior(loc=0.0, scale=0.5),
    lengthscale_prior=lengthscale_prior,
    lengthscale_constraint=lengthscale_constraint,
    active_dims=active_dims_matern,
    ard_num_dims=ard_num_dims,
)
kernel.kernels[1].initialize(period_length=0.3)

model = KroneckerMultiTaskGP(
    train_X=X_train_tensor,
    train_Y=y_train_tensor,  
    input_transform=None,
    output_transform=Standardize(m=y_train.shape[1]),
    data_covar_module=kernel,
    rank=2,
)
model = model.to(device=device, dtype=dtype)
mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device, dtype)
fit_gpytorch_mll(mll)  # optimizes kernel + likelihood params


model.eval()
model.likelihood.eval()
with torch.no_grad():
    post = model.posterior(X_test_tensor)
    y_pred = post.mean.cpu().numpy()     
    y_var  = post.variance.cpu().numpy()  
    

# Compare predictions to actuals
results = pd.DataFrame(y_pred, columns=[f"{name}_pred" for name in output_names])


summary = evaluate_asset_pricing(pd.DataFrame(y_test), results)
print('Summary of Asset Pricing Metrics:')
print(summary)

def evaluate_multioutput(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    weights: Dict[str, float] = None,
    aggregate: str = "macro",
) -> Tuple[pd.DataFrame, dict]:
    """
    Evaluate multi-output predictions with per-task & aggregated metrics.

    Parameters
    ----------
    y_test : pd.DataFrame
        True values, shape (n_samples, n_tasks). Columns = task names.
    y_pred : pd.DataFrame
        Predicted values, shape (n_samples, n_tasks). Same columns/index as y_test.
    weights : dict[str, float], optional
        Optional task weights for weighted aggregation, e.g. {"ESGD":0.5, "SNPE":0.5}.
        If provided, will be normalized to sum to 1 over intersecting tasks.
    aggregate : {"macro","weighted","flat"}
        - "macro": unweighted average of per-task metrics.
        - "weighted": weighted average using `weights`.
        - "flat": compute metrics on all stacked values (implicitly weights by task scale & sample count).

    Returns
    -------
    per_task : pd.DataFrame
        Index = task names, columns = ["RMSE","MAE","R2","Spearman"].
    summary : dict
        Aggregated scores under keys:
        - "RMSE_agg", "MAE_agg", "R2_agg", "Spearman_agg"
        - "mode": aggregation mode used
        - "n_tasks", "n_samples"
    """
    # Align on intersection of rows/columns (no leakage across misaligned dates/tasks)
    y_true = y_test.copy().reset_index(drop=True)
    common_cols = list(y_test.columns)
    y_hat  = y_pred.copy()
    y_hat.columns = y_true.columns

    # Safety check
    if not y_true.index.equals(y_hat.index):
        # Force align on common index
        common_idx = y_true.index.intersection(y_hat.index)
        y_true = y_true.loc[common_idx]
        y_hat  = y_hat.loc[common_idx]

    tasks = list(common_cols)
    per_task_rows = []

    # Compute per-task metrics with pairwise NaN handling
    for t in tasks:
        yt = y_true[t]
        yp = y_hat[t]
        mask = yt.notna() & pd.notna(yp)
        if mask.sum() == 0:
            per_task_rows.append([t, np.nan, np.nan, np.nan, np.nan])
            continue

        yt_m, yp_m = yt[mask].values, yp[mask].values
        rmse = np.sqrt(mean_squared_error(yt_m, yp_m))
        mae  = mean_absolute_error(yt_m, yp_m)
        r2   = r2_score(yt_m, yp_m) if len(yt_m) > 1 else np.nan
        spr  = spearmanr(yt_m, yp_m, nan_policy="omit").correlation

        per_task_rows.append([t, rmse, mae, r2, spr])

    per_task = pd.DataFrame(per_task_rows, columns=["task","RMSE","MAE","R2","Spearman"]).set_index("task")

    # Aggregations
    agg_mode = aggregate.lower()
    if agg_mode == "flat":
        # Stack all tasks & rows (pairwise NaN drop)
        yt_all = []
        yp_all = []
        for t in tasks:
            m = y_true[t].notna() & y_hat[t].notna()
            yt_all.append(y_true.loc[m, t].values)
            yp_all.append(y_hat.loc[m, t].values)
        yt_all = np.concatenate(yt_all) if yt_all else np.array([])
        yp_all = np.concatenate(yp_all) if yp_all else np.array([])
        if yt_all.size == 0:
            rmse_agg = mae_agg = r2_agg = spr_agg = np.nan
        else:
            rmse_agg = np.sqrt(mean_squared_error(yt_all, yp_all))
            mae_agg  = mean_absolute_error(yt_all, yp_all)
            r2_agg   = r2_score(yt_all, yp_all) if yt_all.size > 1 else np.nan
            spr_agg  = spearmanr(yt_all, yp_all, nan_policy="omit").correlation

    elif agg_mode == "weighted":
        if not weights:
            raise ValueError("aggregate='weighted' requires a weights dict.")
        w = pd.Series(weights, dtype=float).reindex(tasks).fillna(0.0)
        if w.sum() == 0:
            raise ValueError("Provided weights do not overlap tasks or sum to zero.")
        w = w / w.sum()
        rmse_agg = np.nansum(w * per_task["RMSE"])
        mae_agg  = np.nansum(w * per_task["MAE"])
        # For R2 and Spearman, weighted mean is common (but note: not strictly proper)
        r2_agg   = np.nansum(w * per_task["R2"])
        spr_agg  = np.nansum(w * per_task["Spearman"])
    else:  # "macro"
        rmse_agg = per_task["RMSE"].mean()
        mae_agg  = per_task["MAE"].mean()
        r2_agg   = per_task["R2"].mean()
        spr_agg  = per_task["Spearman"].mean()

    summary = {
        "RMSE_agg": rmse_agg,
        "MAE_agg": mae_agg,
        "R2_agg": r2_agg,
        "Spearman_agg": spr_agg,
        "mode": agg_mode,
        "n_tasks": len(tasks),
        "n_samples": len(y_true),
    }
    return per_task, summary

per_task, summary = evaluate_multioutput(y_test, results, aggregate="macro")
print(per_task)
print(summary)


##### rolling forecast
from sklearn.preprocessing import MinMaxScaler
import math
import torch
import numpy as np
import pandas as pd

from botorch.models import KroneckerMultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize

# NEW: imports for priors/constraints/likelihood/kernels
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import MultitaskGaussianLikelihood
def rolling_forecast(X: pd.DataFrame, y: pd.DataFrame, window: int, horizon: int = 1, time_dim: int = 0):
    """
    Rolling forecast with KroneckerMultiTaskGP.
    - Fits MinMaxScaler on X_train only (no leakage).
    - Lets the GP handle Y standardization via outcome_transform=Standardize(...).
    - Uses a custom likelihood with a proper (invertible) positive constraint to avoid
      'Must provide inverse transform to be able to sample from prior.' errors.
    """
    preds, actuals = [], []
    device, dtype = torch.device("cpu"), torch.double

    for start in range(0, len(X) - window - horizon + 1):
        end = start + window

        # Split
        X_train = X.iloc[start:end].values
        y_train = y.iloc[start:end].values
        X_test  = X.iloc[end:end+horizon].values
        y_test  = y.iloc[end:end+horizon].values

        # --- Scale X on train only ---
        scaler_x = MinMaxScaler()
        X_train_scaled = scaler_x.fit_transform(X_train)
        X_test_scaled  = scaler_x.transform(X_test)

        # --- Tensors ---
        X_train_tensor = torch.tensor(X_train_scaled, dtype=dtype, device=device)
        y_train_tensor = torch.tensor(y_train,       dtype=dtype, device=device)
        X_test_tensor  = torch.tensor(X_test_scaled, dtype=dtype, device=device)

        # --- Kernels (Matern across all feats + Periodic on time only is usually better) ---
        n_features = X_train_tensor.shape[-1]
        active_dims_all = list(range(n_features))

        SQRT2 = math.sqrt(2.0)
        SQRT3 = math.sqrt(3.0)
        lengthscale_prior = LogNormalPrior(loc=SQRT2 + math.log(n_features)*0.5, scale=SQRT3)
        lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)

        matern = MaternKernel(
            nu=1.5,  # a bit smoother than 0.5; often works better for monthly returns
            ard_num_dims=n_features,
            active_dims=active_dims_all,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

        # Periodic ONLY on time dim (if you want seasonality)
        T = torch.unique(X_train_tensor[:, time_dim]).numel()
        period_norm = max(12.0 / max(T - 1, 1), 1e-3)  # annual-ish period on minmaxed time
        periodic = PeriodicKernel(
            ard_num_dims=1,
            active_dims=[time_dim],
            period_length_prior=LogNormalPrior(loc=math.log(period_norm), scale=0.5),
            lengthscale_prior=LogNormalPrior(loc=SQRT2 + math.log(1.0)*0.5, scale=SQRT3),
            lengthscale_constraint=GreaterThan(2.5e-2),
        )
        periodic.initialize(period_length=period_norm)

        data_covar_module = matern * periodic  # product usually better than sum for seasonality

        # --- CUSTOM likelihood with invertible constraint (key fix) ---
        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        likelihood = MultitaskGaussianLikelihood(
            num_tasks=y_train.shape[1],
            noise_prior=noise_prior,
            # DO NOT set transform=None here; keep default positive transform
            noise_constraint=GreaterThan(1e-6),  # Softplus-based constraint (invertible)
        )

        # --- Model ---
        model = KroneckerMultiTaskGP(
            train_X=X_train_tensor,
            train_Y=y_train_tensor,
            likelihood=likelihood,                        # << use our likelihood
            outcome_transform=Standardize(m=y_train.shape[1]),  # GP standardizes Y internally
            input_transform=None,
            data_covar_module=data_covar_module,
            rank=2,
        ).to(device=device, dtype=dtype)

        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device, dtype)
        fit_gpytorch_mll(mll)  # will now be able to sample from priors

        # --- Forecast (posterior already unstandardized by the outcome_transform) ---
        model.eval(); model.likelihood.eval()
        with torch.no_grad():
            post = model.posterior(X_test_tensor)
            y_pred = post.mean.cpu().numpy()

        preds.append(y_pred)
        actuals.append(y_test)

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    preds_df   = pd.DataFrame(preds,   columns=[f"{c}_pred" for c in y.columns])
    actuals_df = pd.DataFrame(actuals, columns=y.columns)
    return preds_df, actuals_df



##################Single Task GP######################

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

ard_num_dims = 3
lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)
MIN_INFERRED_NOISE_LEVEL = 1e-4  # Minimum noise level to avoid numerical issues
# We will use the simplest form of GP model, exact inference
class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        kernel = MaternKernel(
            nu=0.5,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint
             ) + PeriodicKernel(
                period_length_prior=LogNormalPrior(loc=0.0, scale=0.5),
                lengthscale_prior=lengthscale_prior,lengthscale_constraint=lengthscale_constraint) #+ LinearKernel()
        kernel_scale = ScaleKernel(kernel)
        kernel_scale.base_kernel.kernels[1].initialize(period_length=1.0)
        self.covar_module = ScaleKernel(kernel_scale)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
likelihood =  GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            initial_value=noise_prior.mode,
        ),
)
model = ExactGPModel(train_x , train_y, likelihood)
model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(MIN_INFERRED_NOISE_LEVEL))

# traing the model
training_iter = 500  # Number of training iterations
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Early stopping variables
best_loss = float("inf")
patience_counter = 0
patience = 10

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f    noise: %.3f' % ( #lengthscalerbf: %.3f periodlength: %.3f
        i + 1, training_iter, loss.item(),
        #model.covar_module.base_kernel.lengthscale.item(),
        #model.covar_module.base_kernel.kernels[0].lengthscale.item(),
        #model.covar_module.base_kernel.kernels[1].period_length.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

    # Early stopping logic
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0  # Reset patience counter if improvement
    else:
        patience_counter += 1  # Increment patience counter if no improvement

    if patience_counter >= patience:
        break  # Stop training early if patience is exceeded

X_full = scaler_X.transform(X)
x_tensor = torch.tensor(X_full, dtype=torch.float32)
y_full = scaler_y.transform(y)
y_tensor = torch.tensor(y_full, dtype=torch.float32).flatten()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = x_tensor
    observed_pred = likelihood(model(test_x))


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x_tensor[:, 0].numpy(), y_tensor.numpy().flatten(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x[:, 0].numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x[:, 0].flatten().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    
    
############## Multitask GP Hadamard    ######################

class MultitaskGPModel(ExactGP):
    """
    Intrinsic Coregionalization Model (ICM) Multi-task GP with Hadamard structure.

    Overview
    --------
    This model implements a multi-task GP where the covariance between two *tasked*
    observations ((x, i), (x', j)) factorizes as a Hadamard (elementwise) product:

        k( (x,i), (x',j) ) = k_X(x, x') * B[i, j]

    • k_X(x, x') is the **data kernel** over inputs (here: Matern(ν=1/2) + Periodic,
      wrapped in ScaleKernel).  
    • B is the **task covariance matrix** learned by an IndexKernel with low rank
      (rank=1 in this example).

    This is the classic **Intrinsic Coregionalization Model (ICM)**. It assumes all
    tasks share the same input kernel shape (lengthscales etc.), while the IndexKernel
    captures how strongly tasks correlate (and their relative marginal variances).

    Why ICM / Hadamard?
    -------------------
    • Simple and data-efficient: few extra parameters, good when tasks are
      moderately to highly correlated and you expect *shared smoothness/periodicity*.
    • Stable & fast: Kronecker-free (no large Kronecker algebra), works well with
      mid-sized panels.  
    • If you need *task-specific* input kernels or multiple latent processes mixed
      per task, consider upgrading to an **LCM** (Linear Coregionalization Model).

    Inputs & Shapes
    ---------------
    We pass a tuple of inputs to the model:
      - x:  Tensor of shape [N, D]   (features / time index, etc.)
      - i:  LongTensor of shape [N, 1] with task IDs in [0, num_tasks-1]
      - y:  Tensor of shape [N]

    For two tasks with aligned inputs, you can stack like:
      full_x = concat([x_task0, x_task1])      -> [2N, D]
      full_i = concat([zeros(N,1), ones(N,1)]) -> [2N, 1]
      full_y = concat([y_task0,   y_task1])    -> [2N]

    Kernels & Priors
    ----------------
    Data kernel:  k_X = Scale(Matern(ν=1/2) + Periodic) wrapped again in ScaleKernel.
    Task kernel:  IndexKernel(num_tasks=2, rank=1) learns B ≽ 0 (task variances +
                  correlations). rank controls the capacity of inter-task structure:
                  rank=1 is ICM with one latent coregionalization component.

    Likelihood
    ----------
    GaussianLikelihood with LogNormal prior and positivity constraint on noise.
    You can:
      • Provide fixed noise (known observation noise), or
      • Infer homoskedastic noise (as done here), or
      • Extend to task-specific noise using MultitaskGaussianLikelihood.

    Training
    --------
    We optimize the Exact Marginal Log-Likelihood (mll) with Adam. Early stopping
    is optional; it’s helpful if the loss plateaus.

    Notes & Tips
    ------------
    • Standardize targets per task (zero mean, unit variance) for stable fits.
    • Normalize inputs (e.g., to [0,1]) so lengthscale priors are well-behaved.
    • If tasks are weakly related or have different smoothness, consider:
        – Increasing IndexKernel rank (>=2), or
        – LCM with multiple latent kernels, or
        – KroneckerMultiTaskGP when inputs are shared on a grid and you want
          exact Kronecker algebra speedups.
    • PeriodicKernel period prior (LogNormal(0, 0.5)) implies a median period ≈ exp(0)=1
      in input units — adjust to match your calendar (e.g., monthly seasonality).

    Returns
    -------
    MultivariateNormal over full stacked observations; predictions for each task
    can be obtained by passing the corresponding (x, i) pairs.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        kernel = MaternKernel(
            nu=0.5,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint
             ) + PeriodicKernel(
                period_length_prior=LogNormalPrior(loc=0.0, scale=0.5),
                lengthscale_prior=lengthscale_prior,lengthscale_constraint=lengthscale_constraint) #+ LinearKernel()
        kernel_scale = ScaleKernel(kernel)
        self.covar_module = ScaleKernel(kernel_scale)
        
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
# initialize likelihood and model
noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
likelihood =  GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            initial_value=noise_prior.mode,
        ),
)

train_i_task1 = torch.full((train_x.shape[0],1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x.shape[0],1), dtype=torch.long, fill_value=1)
full_train_x = torch.cat([train_x, train_x])
full_train_i = torch.cat([train_i_task1, train_i_task2])
full_train_y = torch.cat([train_y, train_y2])
model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)
model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(MIN_INFERRED_NOISE_LEVEL))

# traing the model
training_iter = 500 
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Early stopping variables
best_loss = float("inf")
patience_counter = 0
patience = 10

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(full_train_x, full_train_i)
    loss = -mll(output, full_train_y)
    loss.backward()
    print('Iter %d/500 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()

    # Early stopping logic
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0  # Reset patience counter if improvement
    else:
        patience_counter += 1  # Increment patience counter if no improvement

    if patience_counter >= patience:
        break  # Stop training early if patience is exceeded

# make predictions

# Initialize plots1
model.eval()
likelihood.eval()
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 5))


X_full = scaler_X.transform(X)
x_tensor = torch.tensor(X_full, dtype=torch.float32)
test_i_task1 = torch.full((x_tensor.shape[0],1), dtype=torch.long, fill_value=0)
test_i_task2 = torch.full((x_tensor.shape[0],1), dtype=torch.long, fill_value=1)
y_full = scaler_y.transform(y)
y_full2 = scaler_y2.transform(y2)
y_tensor = torch.tensor(y_full, dtype=torch.float32).flatten()
y_tensor2 = torch.tensor(y_full2, dtype=torch.float32).flatten()


# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_y1 = likelihood(model(x_tensor, test_i_task1))
    observed_pred_y2 = likelihood(model(x_tensor, test_i_task2))
    
# Define plotting function
def ax_plot(ax, train_y, train_x, rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x[:, 0].detach().flatten().numpy(), train_y.detach().numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(train_x[:, 0].detach().flatten().numpy(), rand_var.mean.detach().numpy(), 'b')
    # Shade in confidence
    ax.fill_between(train_x[:, 0].detach().flatten().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)

# Plot both tasks
ax_plot(y1_ax, y_tensor, x_tensor, observed_pred_y1, 'Observed Values (Likelihood)')
ax_plot(y2_ax, y_tensor2, x_tensor, observed_pred_y2, 'Observed Values (Likelihood)')


#calculating results
y_full = scaler_y.transform(y)
y_full2 = scaler_y2.transform(y2)
y1_scaled = pd.DataFrame(y_full, columns=['y1'])
y2_scaled = pd.DataFrame(y_full2, columns=['y2'])
y1_pred = pd.DataFrame(observed_pred_y1.mean.numpy().reshape(-1, 1), columns=['y1_pred'])
y2_pred = pd.DataFrame(observed_pred_y2.mean.numpy().reshape(-1, 1), columns=['y2_pred'])
y_all = pd.concat([y1_scaled, y1_pred, y2_scaled, y2_pred], axis=1)
y_all

##################### Kronecker Multitask GP ######################
class MultitaskGPModel(gpytorch.models.ExactGP):
    """
    Multi-task GP (ICM, Kronecker-structured) with Matern+Periodic data kernel

    What this is
    ------------
    A 2-task Intrinsic Coregionalization Model (ICM) implemented via
    `gpytorch.kernels.MultitaskKernel`. The full covariance factorizes as:

        K( (x, t), (x', t') ) = B[t, t'] ⊗ K_X(x, x')

    where:
      - K_X is the input (data) kernel: Matern(ν=0.5) + Periodic
      - B is a learned task covariance (low-rank + diagonal; controlled by `rank`)
      - ⊗ denotes the Kronecker product

    This is the standard “ICM with Kronecker structure” used in GPyTorch’s
    multitask stack (paired with MultitaskMean and MultitaskGaussianLikelihood).

    When to use this model
    ----------------------
    - Tasks share the *same* input locations (aligned panel): same timestamps or
      feature vectors for every task. This enables Kronecker structure internally.
    - You expect tasks to be correlated (learned by the low-rank coregionalization B).
    - You want per-task noise handled natively (`MultitaskGaussianLikelihood`).

    When NOT to use this model
    --------------------------
    - Tasks are observed on *different* input grids (missing / misaligned data).
      In that case, prefer the Hadamard ICM with `IndexKernel` and stacking (x, task_id).
    - You need different data kernels per task or significantly different smoothness
      per task. Then consider an LCM (Q > 1) or separate single-task GPs.

    Data / Target shapes
    --------------------
    - train_x : Tensor[N, D]
    - train_y : Tensor[N, T]  (T = num_tasks, here 2)
    - likelihood : MultitaskGaussianLikelihood(num_tasks=T)

    Model parts
    -----------
    mean_module:
        MultitaskMean(ConstantMean(), num_tasks=T)
        One mean per task.

    covar_module:
        MultitaskKernel(
            base_kernel = MaternKernel(nu=0.5, ...) + PeriodicKernel(...),
            num_tasks = T,
            rank = 1,
        )
        - `rank` controls the rank of the task mixing (B ≈ W W^T + diag(τ)).
          rank=1 is ICM with a single latent function; increase for richer
          inter-task structure (at computational / statistical cost).

    likelihood:
        MultitaskGaussianLikelihood(num_tasks=T)
        - Learns per-task homoskedastic noise.

    Priors / constraints (recommended)
    ----------------------------------
    - Lengthscale priors for both Matern and Periodic (e.g., LogNormal)
    - Period-length prior for Periodic (e.g., LogNormal centered at a plausible cycle)
    - Outputscale prior / constraints (ScaleKernel) if you find amplitude drifting
    - Consider standardizing inputs to [0,1]^D and targets to zero-mean, unit-variance

    Training tips
    -------------
    - Normalize features and standardize each task column of train_y. Undo at inference.
    - Start with moderate LR (e.g., 0.05–0.1) and 200–500 iters; add early stopping if needed.
    - Watch the learned task covariance B: `model.covar_module.task_covar_module.covar_matrix`
      should become positive-definite with sensible off-diagonals (not all ~0).
    - If tasks are strongly correlated, rank=1 often suffices; if residual correlation remains
      unexplained, try rank=2.
    - If you see underfitting (over-smoothing), check lengthscales (too large) or allow
      a LinearKernel addend for trend; if overfitting, use stronger priors / increase noise.

    Forecasting / inference
    -----------------------
    - At prediction, pass a test_x of shape [M, D]; the model returns a
      MultitaskMultivariateNormal with mean of shape [M, T].
    - You can request only one task’s marginal by indexing the output:
        preds.mean[:, task_idx]
    - If you standardized y, re-scale predictions back to original units.

    Pros vs the Hadamard (IndexKernel) version
    ------------------------------------------
    + Efficient Kronecker algebra on aligned inputs
    + Clean per-task noise via MultitaskGaussianLikelihood
    − Requires aligned inputs across tasks
    − Less flexible for irregular, sparse multi-task panels

    Extending the model
    -------------------
    - Richer inter-task: increase `rank` (ICM→LCM-like as rank grows)
    - Different smoothness regimes: replace Matern ν, add LinearKernel, or use spectral mixtures
    - Seasonality: keep PeriodicKernel; consider multiple periodic components if needed

    References
    ----------
    - ICM / Coregionalization: Bonilla et al., 2008; Álvarez et al., 2012 (GPs for multi-output)
    - GPyTorch MultitaskKernel docs
    """
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        kernel = MaternKernel(
            nu=0.5,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint
             ) + PeriodicKernel(
                period_length_prior=LogNormalPrior(loc=0.0, scale=0.5),
                lengthscale_prior=lengthscale_prior,lengthscale_constraint=lengthscale_constraint) #+ LinearKernel()
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel, num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

train_y_comb = torch.stack([train_y, train_y2], -1)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(train_x, train_y_comb, likelihood)
# model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(MIN_INFERRED_NOISE_LEVEL))

# Find optimal model hyperparameters
model.train()
likelihood.train()
training_iter =500
# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Early stopping variables
best_loss = float("inf")
patience_counter = 0
patience = 10

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y_comb)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    
    # Early stopping logic
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0  # Reset patience counter if improvement
    else:
        patience_counter += 1  # Increment patience counter if no improvement

    if patience_counter >= patience:
        break  # Stop training early if patience is exceeded

    
model.eval()
likelihood.eval()
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 5))

X_full = scaler_X.transform(X)
x_tensor = torch.tensor(X_full, dtype=torch.float32)
y_full = scaler_y.transform(y)
y_full2 = scaler_y2.transform(y2)
y_tensor = torch.tensor(y_full, dtype=torch.float32).flatten()
y_tensor2 = torch.tensor(y_full2, dtype=torch.float32).flatten()
y_tensor_comb = torch.stack([y_tensor, y_tensor2], -1)

# Make predictions
# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(x_tensor))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    
# Plot training data as black stars
y1_ax.plot(x_tensor[:, 0].detach().flatten().numpy(), y_tensor_comb[:, 0].detach().numpy(), 'k*')
# Predictive mean as blue line
y1_ax.plot(x_tensor[:, 0].flatten().numpy(), mean[:, 0].numpy(), 'b')
# Shade in confidence
y1_ax.fill_between(x_tensor[:, 0].flatten().numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
y1_ax.set_ylim([-3, 3])
y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y1_ax.set_title('Observed Values (Likelihood)')

# Plot training data as black stars
y2_ax.plot(x_tensor[:, 0].detach().flatten().numpy(), y_tensor_comb[:, 1].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(x_tensor[:, 0].flatten().numpy(), mean[:, 1].numpy(), 'b')
# Shade in confidence
y2_ax.fill_between(x_tensor[:, 0].flatten().numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
y2_ax.set_ylim([-3, 3])
y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y2_ax.set_title('Observed Values (Likelihood)')

#calculating results
y_full = scaler_y.transform(y)
y_full2 = scaler_y2.transform(y2)
y1_scaled = pd.DataFrame(y_full, columns=['y1'])
y2_scaled = pd.DataFrame(y_full2, columns=['y2'])
y1_pred = pd.DataFrame(mean[:, 0].numpy().reshape(-1, 1), columns=['y1_pred'])
y2_pred = pd.DataFrame(mean[:, 1].numpy().reshape(-1, 1), columns=['y2_pred'])
y_all = pd.concat([y1_scaled, y1_pred, y2_scaled, y2_pred], axis=1)
y_all


# Single task need to be updated. 
#the scale for y1 y1_pred was different but the multitask still beat out the singletask
# Old y2 y2_pred with just singletask
# Obs	Pred
# 0	0.137	0.134
# 1	-0.576	-0.571
# 2	0.145	0.143
# 3	0.188	0.188
# 4	0.509	0.507
# ...	...	...
# 69	-0.624	-0.621
# 70	0.827	0.822
# 71	0.869	0.868
# 72	0.2	0.616
# 73	0.228	0.466

# together #hadamard results 
	y1	y1_pred	y2	y2_pred
0	-0.539	-0.463	0.138	0.0617
1	-0.476	-0.493	-0.575	-0.538
2	0.501	0.46	0.147	0.172
3	0.605	0.54	0.189	0.247
4	0.0689	0.158	0.511	0.429
...	...	...	...	...
69	0.558	0.464	-0.623	-0.532
70	0.757	0.794	0.828	0.773
71	0.338	0.401	0.87	0.809
72	-0.649	0.313	0.201	0.572
73	0.793	0.252	0.14	0.418

#hadamard results with vix, better but need to fix dates actually. 
	y1	y1_pred	y2	y2_pred
0	-0.54	-0.524	0.14	0.124
1	-0.477	-0.476	-0.578	-0.576
2	0.509	0.498	0.148	0.155
3	0.614	0.6	0.191	0.201
4	0.0733	0.0827	0.514	0.504
...	...	...	...	...
70	0.767	0.769	0.834	0.825
71	0.344	0.353	0.877	0.861
72	-0.65	-0.63	0.203	0.184
73	0.648	-0.0952	0.203	0.018
74	-0.127	-0.17	-0.0999	-0.0639


# TAKE AWAY: The multitask is 10% better for y2

### krokecker results 

y1	y1_pred	y2	y2_pred
0	-0.539	-0.489	0.138	-0.423
1	-0.476	-0.477	-0.575	-0.413
2	0.501	0.467	0.147	0.4
3	0.605	0.571	0.189	0.488
4	0.0689	0.113	0.511	0.0952
...	...	...	...	...
69	0.558	0.467	-0.623	0.397
70	0.757	0.761	0.828	0.651
71	0.338	0.386	0.87	0.329
72	-0.649	0.328	0.201	0.279
73	0.793	0.292	0.14	0.248

#Takeaway comparing the Hadamard vs the Krokecker:
# Kronecker Slightly better for y1, Kronecker WAYY better for y2 66% better. 
# Kronecker more realistically displays accurate uncertainty. SNPE does infact vary
# quite a bit more than ESGD. 

#Kronecker results

y1	y1_pred	y2	y2_pred
0	-0.54	0.091	0.14	0.0942
1	-0.477	-0.442	-0.578	-0.557
2	0.509	0.154	0.148	0.157
3	0.614	0.197	0.191	0.217
4	0.0733	0.398	0.514	0.473
...	...	...	...	...
70	0.767	0.698	0.834	0.821
71	0.344	0.695	0.877	0.829
72	-0.65	0.13	0.203	0.15
73	0.648	0.202	0.203	0.232
74	-0.127	0.15	-0.0999	0.17

# After adding in vix Hadamard better


###### Next steps 
"""
- add small Caps
- add byld and emerging market

Review paper "Portfolio Constraints: An Empirical Analysis"
- takeaways: for long-term portfolios, min sample time window of 60 months
they use an out-of-sample window of 12 months
-THey used a rolling window
-Metrics: Sharpe ratio, variance, turnover, certainty-equivalent returns, and short interest.
-Sharpe, vol, turnover, CEQ, sometimes max drawdown, and compare to 1/N and market-cap benchmarks.
-Transaction costs are a huge limiting factor 
Investors should choose models based on objectives:

If minimizing risk → GMV with constraints.

If maximizing efficiency → MSR with TEV or equal-weight bounds.
"""



###### This was the best combination!!!!!

def _fit_and_forecast(X, y, start, window, horizon, time_dim=0, device=torch.device("cpu"), dtype=torch.double):
    """Helper for one rolling step."""

    torch.manual_seed(1)
    np.random.seed(1)
    end = start + window
    X_train = X.iloc[start:end].values
    y_train = y.iloc[start:end].values
    X_test  = X.iloc[end:end+horizon].values
    y_test  = y.iloc[end:end+horizon].values

    # Scale X on train only
    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled  = scaler_x.transform(X_test)

    # Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=dtype, device=device)
    y_train_tensor = torch.tensor(y_train,       dtype=dtype, device=device)
    X_test_tensor  = torch.tensor(X_test_scaled, dtype=dtype, device=device)

    # Kernels
    n_features = X_train_tensor.shape[-1]
    active_dims_all = list(range(n_features))
    SQRT2, SQRT3 = math.sqrt(2.0), math.sqrt(3.0)
    lengthscale_prior = LogNormalPrior(loc=SQRT2 + math.log(n_features)*0.5, scale=SQRT3)
    lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)

    matern = MaternKernel(
        nu=0.5,
        ard_num_dims=n_features,
        active_dims=active_dims_all,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=lengthscale_constraint,
    )
    rq = RQKernel(ard_num_dims=n_features,
        active_dims=active_dims_all,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=lengthscale_constraint)
    # # Periodic on time only
    # T = torch.unique(X_train_tensor[:, time_dim]).numel()
    # period_norm = max(12.0 / max(T - 1, 1), 1e-3)
    # periodic = PeriodicKernel(
    #     ard_num_dims=1,
    #     active_dims=[time_dim],
    #     period_length_prior=LogNormalPrior(loc=math.log(period_norm), scale=0.5),
    #     lengthscale_prior=LogNormalPrior(loc=SQRT2 + math.log(1.0)*0.5, scale=SQRT3),
    #     lengthscale_constraint=GreaterThan(2.5e-2),
    # )
    # periodic.initialize(period_length=period_norm)
        # Replace Periodic with Spectral Mixture (2 components on time dimension)
    sm = SpectralMixtureKernel(
        num_mixtures=8,
        ard_num_dims=1,
        active_dims=[time_dim]
    )
    sm.initialize_from_data(X_train_tensor[:, [time_dim]], y_train_tensor)

    # data_covar_module = matern * periodic
    #data_covar_module = ProductKernel(matern, rq)
    #data_covar_module = matern * rq + periodic

    #data_covar_module = ProductKernel(matern, rq, sm)
    data_covar_module = matern * rq + sm


    # Model
    model = KroneckerMultiTaskGP(
        train_X=X_train_tensor,
        train_Y=y_train_tensor,
        outcome_transform=Standardize(m=y_train.shape[1]),
        input_transform=None,
        data_covar_module=data_covar_module,
        rank=2,
    ).to(device=device, dtype=dtype)

    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device, dtype)
    fit_gpytorch_mll(mll)

    # Forecast
    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(X_test_tensor)
        y_pred = post.mean.cpu().numpy()

    return y_pred, y_test


def rolling_forecast_parallel(X: pd.DataFrame, y: pd.DataFrame, window: int, horizon: int = 1, n_jobs: int = -1):
    """Parallel rolling forecast with joblib."""
    tasks = range(0, len(X) - window - horizon + 1)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_forecast)(X, y, start, window, horizon) for start in tasks
    )

    preds, actuals = zip(*results)
    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    preds_df   = pd.DataFrame(preds,   columns=[f"{c}_pred" for c in y.columns])
    actuals_df = pd.DataFrame(actuals, columns=y.columns)
    return preds_df, actuals_df
    
y_true_all, y_pred_all = rolling_forecast_parallel(X, y, window=60, horizon=1)
summary = evaluate_asset_pricing(y_true_all, y_pred_all)
print('Summary of Asset Pricing Metrics:')
print(summary)


""" 
Interpreting results september 27 from the above

Summary of Asset Pricing Metrics:
{'R2_pooled': np.float64(-6.35103948118288), 'R2_avg': np.float64(-6.799840352064019), 'IC': np.float64(0.29090909090909084)}
⸻

📊 Metrics You Reported
	•	R²_pooled = –6.35
	•	R²_avg = –6.80
	•	IC = 0.291

⸻

🔎 Interpretation

1. R²_pooled and R²_avg (both large negative)
	•	Negative R² means your point predictions for returns are worse than simply predicting the mean.
	•	A value like –6 is very poor in terms of fitting actual magnitudes.
	•	This is common in financial return prediction — absolute returns are dominated by noise, so R² often looks terrible.
	•	Takeaway: Your model struggles to predict levels of returns accurately.

⸻

2. IC = 0.291 (~29%)
	•	This is very strong rank correlation between predicted and realized returns.
	•	In quant finance, an IC of 0.05–0.1 is already considered good.
	•	At 0.29, your model is capturing a highly tradable signal in the ordering of assets, even though the magnitudes are off.

⸻

⚖️ Reconciling the two
	•	Bad R², good IC means:
	•	You can’t predict the exact return size (too noisy).
	•	But you can predict which ETFs will relatively outperform vs. underperform.
	•	This is exactly why many quant funds don’t care about R² and optimize for IC instead.

⸻

✅ Bottom line
	•	IC = 0.291 → excellent signal.
	•	Negative R² → don’t trust the raw return magnitudes, but do trust the rankings.
	•	In practice: build a long-short ETF strategy (long top-ranked, short bottom-ranked) rather than betting on exact return forecasts.

⸻

⚡ If you can sustain IC ~0.29 out-of-sample, that’s institutional-grade predictive power.

Do you want me to show you how to turn these predictions into a portfolio backtest (e.g., top-quintile long vs. bottom-quintile short) so you can see how tradable that IC is?


By dividing the etfs into quintiles you should by the long top quintiles and short the bottom quintiles.
"""


#### Hadamard ####
from sklearn.preprocessing import MinMaxScaler
import math
import torch
import numpy as np
import pandas as pd

from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, RQKernel, SpectralMixtureKernel

def _to_icm_design(X_np, Y_np, task_col_idx=-1):
    """
    X_np: (n, d) features (no task yet)
    Y_np: (n, m) targets (wide, columns=tasks)
    Returns:
      X_icm: (n*m, d+1) with task index in last column
      Y_icm: (n*m, 1)
    """
    n, d = X_np.shape
    m = Y_np.shape[1]
    # repeat X for each task
    X_rep = np.repeat(X_np, m, axis=0)  # (n*m, d)
    # task index column
    task_idx = np.concatenate([np.full(n, t, dtype=np.int64) for t in range(m)]).reshape(-1, 1)
    # stack X and task
    X_icm = np.hstack([X_rep, task_idx])
    # flatten Y column-wise to match the same order
    Y_icm = Y_np.reshape(-1, 1, order="F")  # first all rows of task0, then task1, ...
    return X_icm, Y_icm

def _fit_and_forecast_multitaskgp(
    X, y, start, window, horizon, time_dim=0, device=torch.device("cpu"), dtype=torch.double
):
    # split
    end = start + window
    X_train = X.iloc[start:end].values
    y_train = y.iloc[start:end].values  # (n, m)
    X_test  = X.iloc[end:end+horizon].values
    y_test  = y.iloc[end:end+horizon].values  # (h, m)

    # scale X on train only
    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled  = scaler_x.transform(X_test)

    # ---- Convert to ICM block design ----
    # NOTE: task feature will be appended as last column
    X_icm, Y_icm = _to_icm_design(X_train_scaled, y_train, task_col_idx=-1)

    # tensors
    X_icm_t = torch.tensor(X_icm, dtype=dtype, device=device)
    Y_icm_t = torch.tensor(Y_icm, dtype=dtype, device=device)

    # ----- Kernel over NON-task features (i.e., first d columns) -----
    d = X_train_scaled.shape[1]               # number of non-task features
    task_feature = d                          # last column is task idx

    SQRT2, SQRT3 = math.sqrt(2.0), math.sqrt(3.0)
    lengthscale_prior = LogNormalPrior(loc=SQRT2 + math.log(d)*0.5, scale=SQRT3)
    lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)

    matern = MaternKernel(
        nu=0.5, ard_num_dims=d, active_dims=list(range(d)),
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=lengthscale_constraint,
    )
    rq = RQKernel(
        ard_num_dims=d, active_dims=list(range(d)),
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=lengthscale_constraint,
    )

    # Spectral Mixture on the time feature ONLY (still in the non-task slice)
    # time_dim is the index inside the non-task part (same as your original X columns)
    sm = SpectralMixtureKernel(num_mixtures=8, ard_num_dims=1, active_dims=[time_dim])
    # initialize from data (needs 1D inputs, so slice the time column)
    sm.initialize_from_data(
        train_x=X_icm_t[:, [time_dim]],  # only the time feature
        train_y=Y_icm_t.squeeze(-1)      # 1D target
    )

    data_covar_module = matern * rq + sm


    # model: MultiTaskGP expects train_X shape (n*, d+1) and train_Y shape (n*, 1)
    # outcome_transform standardizes that single output; task structure handled internally
    model = MultiTaskGP(
        train_X=X_icm_t,
        train_Y=Y_icm_t,
        task_feature=task_feature,
        covar_module=data_covar_module,
        outcome_transform=Standardize(m=1),  # single output per row
    ).to(device=device, dtype=dtype)

    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device, dtype)
    fit_gpytorch_mll(mll)

    # ---- Build test design for ALL tasks at the horizon ----
    m = y_train.shape[1]
    # Repeat each X_test row m times and append task indices
    Xtest_rep = np.repeat(X_test_scaled, m, axis=0)                  # (h*m, d)
    t_idx = np.concatenate([np.full(horizon, t) for t in range(m)]).reshape(-1, 1)
    Xtest_icm = np.hstack([Xtest_rep, t_idx])
    Xtest_icm_t = torch.tensor(Xtest_icm, dtype=dtype, device=device)

    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(Xtest_icm_t)   # already unstandardized by outcome_transform
        y_pred_flat = post.mean.cpu().numpy().reshape(-1)  # (h*m,)
    # reshape back to (h, m) in the same order we constructed Xtest_icm (task major)
    y_pred = y_pred_flat.reshape(m, horizon).T  # (h, m)

    return y_pred, y_test


def rolling_forecast_parallel(X: pd.DataFrame, y: pd.DataFrame, window: int, horizon: int = 1, n_jobs: int = -1):
    """Parallel rolling forecast with joblib."""
    tasks = range(0, len(X) - window - horizon + 1)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_forecast_multitaskgp)(X, y, start, window, horizon) for start in tasks
    )

    preds, actuals = zip(*results)
    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    preds_df   = pd.DataFrame(preds,   columns=[f"{c}_pred" for c in y.columns])
    actuals_df = pd.DataFrame(actuals, columns=y.columns)
    return preds_df, actuals_df
    
# !: I'm pretty sure y_true and y_pred are switcheD!    
y_true_all, y_pred_all = rolling_forecast_parallel(X, y, window=60, horizon=1)
summary = evaluate_asset_pricing(y_true_all, y_pred_all)
print('Summary of Asset Pricing Metrics:')
print(summary)


import xgboost as xgb
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)