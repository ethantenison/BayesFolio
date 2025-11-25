from unittest.mock import Base
import mlflow
import pandas as pd
import warnings
import os
from joblib import Parallel, delayed
from pydantic import BaseModel
from sympy import use
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from marketmaven.configs import TickerConfig, Interval, Horizon, CVConfig
from marketmaven.asset_prices import build_long_panel, fetch_etf_features
from marketmaven.market_fundamentals import fetch_enhanced_macro_features
import numpy as np
import torch
from marketmaven.configs import (
    RiskfolioConfig, OptModel, RiskMeasure, Objective, MuEstimator, CovEstimator)
from marketmaven.visualization.eda import correlation_matrix
from marketmaven.gp_data_prep import prepare_multitask_gp_data
from marketmaven.models.cv import rolling_time_splits_multitask
from marketmaven.models.scaling import MultitaskScaler
device = torch.device("cpu")
from marketmaven.models.gp import train_model_hadamard
from math import log, sqrt
from marketmaven.evaluate import evaluate_asset_pricing
from marketmaven.utils import check_equal_occurrences
from pydantic import BaseModel, ConfigDict
from marketmaven.visualization.evaluation import plot_ls_cumulative_compare
from marketmaven.portfolio.helpers import assessing_long_short_performance, long_short_returns
from marketmaven.models.kernels import MeanF, KernelType, initialize_mean, initialize_kernel
from marketmaven.mlflow.helpers import KernelF, KernelT, MultiTaskConfig, long_to_panel, compute_benchmark_panel,r2_os, log_r2_os
warnings.filterwarnings(
    "ignore",
    message=".*torch.sparse.SparseTensor.*is deprecated.*"
)
warnings.filterwarnings("ignore", category=Warning, message=".*not p.d., added jitter.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*not p.d., added jitter.*")
# # MLFlow Configuration
# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# EXPERIMENT_NAME = "December Portfolio Experiments"

# # Set up MLFlow tracking
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("December Portfolio Experiments")


warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3}'.format

############### Experiment Configuration ###############

description = "tracking experiments for dec portfolio assets"


etf_tickers = [
    "SPY", # total US market big cap
    "MGK", # US growth
    "VTV", # US value
    "IJR", # US small cap
    "VNQ", # REIT ETF US centric
    "VNQI", # international REIT ETF
    "VEA", # developed international equity
    "VWO", # AVEM actually is better than VWO but not enough history
    "VSS", # forein small/mid cap
    "BND", # total bond market ETF US centric
    "IEF", # 7-10 year treasury bond ETF US centric
    "BNDX", # total international bond market ETF, USD hedged, but actually developed markets only
    #"IBND", # international corporate bond market ETF unhedged
    #"ISHG", # international high yield bond ETF unhedged
    "LQD", # investment grade bond ETF US centric
    "HYG", # High yield bond ETF US centric 
    "TIP", # Treasury inflation protected securities ETF US centric
    "EMB", # emerging market bond ETF USD hedged
    "EWX", # emerging market small cap dividend ETF
    #"PDBC", # Commodities ETF
    #"BIL", # 1-3 month us treasuries 
    
]


tickers = TickerConfig(
    start_date="2013-06-01",
    end_date="2025-11-02",
    interval=Interval.DAILY,
    tickers=etf_tickers,
    horizon=Horizon.MONTHLY,
    lookback_date="2010-06-01"
)

############### Returns data ###############
return_data = build_long_panel(tickers.tickers, tickers.lookback_date, tickers.end_date, horizon=tickers.horizon)
pivoted_returns = return_data.pivot(index="date", columns="asset_id", values="y_excess_lead")
pivoted_returns = pivoted_returns.dropna().sort_index()
y = (
    pivoted_returns
)
y.columns.name = None
y = y.reset_index()
y.index = y['date']
y = y.drop(columns=['date'])

#figs 
fig = correlation_matrix(pivoted_returns)
fig.show()

############### Factor data ###############
macro_features = fetch_enhanced_macro_features(start=tickers.lookback_date, end=tickers.end_date)
etf_features = fetch_etf_features(tickers.tickers, tickers.lookback_date, tickers.end_date, tickers.horizon)


#combine all factor columns
df = (
    return_data
    .merge(etf_features, on=['asset_id', 'date'], how='left')
    .merge(macro_features, on='date', how='left')
)
df= df[df['date'] > str(tickers.start_date)]
df = df.reset_index(drop=True)

time_col = "date"
y_col = "y_excess_lead"
asset_col = "asset_id"
cols_to_exclude = ['vix', 'log_ret', 'rsp_spy','vix_slope','skew_proxy','hy_spread', 'hy_spread_chg_1m','hy_spread_z_12m', 'yc_pc1', 'yc_pc2',
       'yc_pc3', 'ma_1m', 'ma_3m','overnight_gap','vol_accel']
exclude_set = set(cols_to_exclude) | {asset_col, y_col, time_col}
feature_cols = [c for c in df.columns if c not in exclude_set]
df = df.drop(columns=cols_to_exclude)

all_assets_occur = check_equal_occurrences(df, 'asset_id')
print('Did all assets occur equally often?', all_assets_occur)
df['t_index'] = pd.factorize(df['date'])[0]
cols = df.columns.tolist()       
cols.insert(0, cols.pop(cols.index("t_index")))

######## Visualize Factors ########
df = df[cols]

fig = correlation_matrix(df)
fig.show()

df.hist(bins=30)

############### Data Preparation ###############

cv_config = CVConfig(
    step=1,
    horizon_cv=1,
    embargo=0,
    training_min=136,
)
    
X, I, y, task_map = prepare_multitask_gp_data(
    df,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date", "asset_id"]
)

# 2. Make splits 
splits = rolling_time_splits_multitask(
    df,
    date_col="date",
    asset_col="asset_id",
    train_min=cv_config.training_min,
    step=cv_config.step,
    horizon=cv_config.horizon_cv,
    embargo=cv_config.embargo
)
splits = list(splits)

############### Model Setup ###############
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)



ard_num_dims = len(feature_cols)
active_dims_features = list(range(1, ard_num_dims + 1))

train_idx_1_ahead = list(range(X.shape[0] - len(tickers.tickers)))
test_idx_1_ahead = list(range(X.shape[0] - len(tickers.tickers), X.shape[0]))

kernelf = KernelF(
    typef=KernelType.MATERN_RQ,
    featuresf=feature_cols,
    active_dims_features=active_dims_features,
    smoothnessf=0.5,
)
kernelt = KernelT(
    typet=KernelType.PERIODIC_MATERN,
    featurest=["t_index"],
    smoothnesst=2.5,
    active_dims_time=[0],

)
multiconfig = MultiTaskConfig(
    num_tasks=len(tickers.tickers),
    mean=MeanF.MULTITASK_CONSTANT,
    rank=6,
    scaling="global",
)

def fit_eval_split_mtgp(
    X: torch.Tensor, I: torch.Tensor, y: torch.Tensor,
    train_idx, test_idx,
    scale_y: str = "global", 
    eps: float = 1e-8,
    rank: int = 2
):
    # Slice
    X_tr, I_tr, y_tr = X[train_idx], I[train_idx], y[train_idx]
    X_te, I_te, y_te = X[test_idx], I[test_idx], y[test_idx]

    # Scale
    scaler = MultitaskScaler(scale_y=scale_y, exclude_time_col=False)
    scaler.fit_x(X_tr)
    X_trs = scaler.transform_x(X_tr)
    X_tes = scaler.transform_x(X_te)
    y_trs = scaler.fit_y(y_tr, I_tr)
    X_trs = torch.cat([X_trs, I_tr.to(X_trs.dtype)], dim=-1)
    X_tes = torch.cat([X_tes, I_te.to(X_tes.dtype)], dim=-1)
    X_trs = X_trs.to(device)
    y_trs = y_trs.to(device)
    X_tes = X_tes.to(device)

    # Initialize mean and kernel
    mean_f = initialize_mean(multiconfig.mean, num_tasks=multiconfig.num_tasks, input_size=X.shape[1])

    kernel_f = initialize_kernel(
        kernelf.typef,
        active_dims=kernelf.active_dims_features,
        batch_shape=torch.Size(),
        smoothness=kernelf.smoothnessf)

    n_months = len(np.unique(X_tr.numpy()[:,0]))
    period_length = 12.0 / (n_months - 1)  # slightly more precise
    kernel_t = initialize_kernel(
        kernelt.typet,
        active_dims=kernelt.active_dims_time,
        batch_shape=torch.Size(),
        period_length=period_length,
        smoothness=kernelt.smoothnesst
    )
    
    kernel_total = kernel_f * kernel_t
    
    ####### Train Model #######
    model, likelihood = train_model_hadamard(
        X_trs,
        y_trs,
        rank=multiconfig.rank,
        mean_f=mean_f,
        kernel=kernel_total, 
        visualize=True,
        dtype=torch.float32,
        device=torch.device("cpu"))
    
    ############### Predict ###############
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        f_dist = model(X_tes)
        pred = likelihood(f_dist, X_tes) 

    y_hat = scaler.inverse_y(pred.mean, I_te)
    y_std = scaler.inverse_std(pred.variance.sqrt(), I_te)

    return y_te, y_hat, y_std



    
############### Run ###############
seed = 27

with mlflow.start_run(description="""Evaluating Dec Portfolio Assets with MTGP""") as run:
    mlflow.log_params(tickers.model_dump())
    mlflow.log_params(cv_config.model_dump())
    mlflow.log_params(multiconfig.model_dump())
    mlflow.log_params(kernelf.model_dump())
    mlflow.log_params(kernelt.model_dump())
    mlflow.log_param("seed", seed)
    
    
    torch.manual_seed(seed)

    # ---- MULTITASK GP PARALLEL EVALUATION ----
    results = Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=10
    )(
        delayed(fit_eval_split_mtgp)(
            X, I, y, tr_idx, te_idx,
            rank=multiconfig.rank,
            scale_y=multiconfig.scaling
        )
        for tr_idx, te_idx in splits
    )

    true_list, pred_list, unc_list = zip(*results)

    true = np.vstack(true_list)
    pred = np.vstack(pred_list)
    unc  = np.vstack(unc_list)

    # Convert GP predictions to DataFrames
    asset_cols = tickers.tickers  # list of asset names
    true_df = pd.DataFrame(true, columns=[f"{c}_true" for c in asset_cols])
    preds_df = pd.DataFrame(pred, columns=[f"{c}_pred" for c in asset_cols])
    unc_df = pd.DataFrame(unc, columns=[f"{c}_unc" for c in asset_cols])

    # ---- MTGP EVAL ----
    summary_gp = evaluate_asset_pricing(
        y_test=true_df,
        y_pred=preds_df
    )
    mlflow.log_metrics({f"gp_{k}": v for k, v in summary_gp.items()})

    # ---- BENCHMARK EVALUATIONS ----
    mean_preds = []
    ewma_preds = []
    asset_cols = tickers.tickers

    for i, (tr_idx, te_idx) in enumerate(splits):

        y_tr_tensor = y[tr_idx]
        I_tr_tensor = I[tr_idx]

        # Convert to panel
        y_tr_panel = long_to_panel(y_tr_tensor, I_tr_tensor, asset_cols)  

        # TEST SLICE IS ALREADY STACKED INTO "true"
        y_te_array = y[te_idx]             # shape: (n_assets,)
        y_te_tensor = torch.tensor(y_te_array).unsqueeze(0)  # (1 × n_assets)
        y_te_panel = pd.DataFrame(y_te_tensor.numpy(), columns=asset_cols)

        # ----- Mean benchmark -----
        mean_df = compute_benchmark_panel(
            y_train=y_tr_panel,
            y_test=y_te_panel,
            method="mean"
        )
        mean_preds.append(mean_df.values)

        # ----- EWMA benchmark -----
        ewma_df = compute_benchmark_panel(
            y_train=y_tr_panel,
            y_test=y_te_panel,
            method="ewma2"
        )
        ewma_preds.append(ewma_df.values)


    # Convert lists → arrays
    mean_pred = np.vstack(mean_preds)        # (N_splits × n_assets)
    ewma_pred = np.vstack(ewma_preds)        # (N_splits × n_assets)

    # Ground truth is already stacked correctly from the GP loop
    y_bench_true = true                       # (N_splits × n_assets)

    # Build DataFrames
    true_b_df = pd.DataFrame(y_bench_true, columns=asset_cols)
    mean_df = pd.DataFrame(mean_pred, columns=asset_cols)
    ewma_df = pd.DataFrame(ewma_pred, columns=asset_cols)

    # Evaluate benchmarks
    summary_mean = evaluate_asset_pricing(y_test=true_b_df, y_pred=mean_df)
    summary_ewma = evaluate_asset_pricing(y_test=true_b_df, y_pred=ewma_df)

    # Log metrics
    mlflow.log_metrics({f"historic_{k}": v for k, v in summary_mean.items()})
    mlflow.log_metrics({f"ewma_{k}": v for k, v in summary_ewma.items()})

    true_b_df.to_parquet("marketmaven/mlflow/artifacts/y_test.parquet")
    mlflow.log_artifact("marketmaven/mlflow/artifacts/y_test.parquet")
    preds_df.to_parquet("marketmaven/mlflow/artifacts/y_pred_gp.parquet")
    mlflow.log_artifact("marketmaven/mlflow/artifacts/y_pred_gp.parquet")
    
    # R2_OS calculations
    y_true_norm = true_b_df.rename(columns=lambda c: c.replace("_true", ""))
    y_pred_norm = preds_df.rename(columns=lambda c: c.replace("_pred", ""))
    y_ewma_norm = ewma_df.copy()
    y_mean_norm = mean_df.copy()
    
    r2_gp_vs_mean = r2_os(y_true_norm, y_pred_norm, y_mean_norm)
    r2_gp_vs_ewma = r2_os(y_true_norm, y_pred_norm, y_ewma_norm)
    
    log_r2_os("gp_vs_mean", r2_gp_vs_mean)
    log_r2_os("gp_vs_ewma", r2_gp_vs_ewma)
    
    # Long-short performance
    ls_gp = long_short_returns(y_true=y_true_norm, y_pred=y_pred_norm)
    ls_mean = long_short_returns(y_true=y_true_norm, y_pred=y_mean_norm)
    ls_ewma = long_short_returns(y_true=y_true_norm, y_pred=y_ewma_norm )
    ls_stats_gp = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_pred_norm, label="gp")
    ls_stats_mean = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_mean_norm, label="mean")
    ls_stats_ewma = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_ewma_norm, label="ewma")
    mlflow.log_metrics(ls_stats_gp)
    mlflow.log_metrics(ls_stats_mean)
    mlflow.log_metrics(ls_stats_ewma)
    plot_ls_cumulative_compare(ls_gp, ls_mean, ls_ewma)
    
    