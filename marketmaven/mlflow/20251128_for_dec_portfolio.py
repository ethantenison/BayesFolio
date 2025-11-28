import time
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
from marketmaven.visualization.eda import correlation_matrix, apply_pca_and_replace
from marketmaven.gp_data_prep import prepare_multitask_gp_data
from marketmaven.models.cv import rolling_time_splits_multitask
from marketmaven.models.scaling import MultitaskScaler
device = torch.device("cpu")
from marketmaven.models.gp import train_model_hadamard
from math import log, sqrt
from marketmaven.evaluate import evaluate_asset_pricing
from marketmaven.utils import check_equal_occurrences
from pydantic import BaseModel, ConfigDict
from marketmaven.visualization.evaluation import plot_ls_cumulative_compare, plot_actual_vs_pred_matrix
from marketmaven.portfolio.helpers import assessing_long_short_performance, long_short_returns
from marketmaven.models.kernels import MeanF, KernelType, initialize_mean, initialize_kernel, adaptive_lengthscale_prior
from marketmaven.mlflow.helpers import (
    KernelConfig, MultiTaskConfig, long_to_panel, compute_benchmark_panel, r2_os, log_r2_os,
    extract_full_gp_config,model_error_by_time_index, log_kernel_to_mlflow, log_gpytorch_state_dict
)
import random
import itertools
from gpytorch.priors import LogNormalPrior
warnings.filterwarnings(
    "ignore",
    message=".*torch.sparse.SparseTensor.*is deprecated.*"
)
warnings.filterwarnings("ignore", category=Warning, message=".*not p.d., added jitter.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*not p.d., added jitter.*")
# # MLFlow Configuration
# mlflow ui --backend-store-uri ./mlruns
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

"""
Setting the task matrix eigenvalues to see the effective dimensionality of the asset returns
data = df[['date','asset_id', 'y_target']]

# Pivot the dataframe to go from long to wide
df_wide = data.pivot(index='date', columns='asset_id', values='y_target')

# Reset index to make it a regular dataframe
df_wide = df_wide.reset_index()
df_wide = df_wide.iloc[:, 1:]

cov = np.cov(df_wide.T)
eigvals = np.linalg.eigvalsh(cov)[::-1]
print(eigvals)
print(np.cumsum(eigvals) / np.sum(eigvals))


"""


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
    # # "IBND", # international corporate bond market ETF unhedged
    # # "ISHG", # international high yield bond ETF unhedged
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
# return_data = build_long_panel(tickers.tickers, tickers.lookback_date, tickers.end_date, horizon=tickers.horizon)
# pivoted_returns = return_data.pivot(index="date", columns="asset_id", values="y_excess_lead")
# pivoted_returns = pivoted_returns.dropna().sort_index()
# y = (
#     pivoted_returns
# )
# y.columns.name = None
# y = y.reset_index()
# y.index = y['date']
# y = y.drop(columns=['date'])

# #figs 
# fig = correlation_matrix(pivoted_returns)
# fig.show()

# # ############### Factor data ###############
# macro_features = fetch_enhanced_macro_features(start=tickers.lookback_date, end=tickers.end_date)
# macro_features = macro_features.drop(columns=['vix_ts_level','skew_proxy','vix3m','yc_pc1', 'yc_pc2', 'yc_pc3', 'y10_nominal' ])
# macro_cols = macro_features.columns[1:].tolist()
# etf_features = fetch_etf_features(tickers.tickers, tickers.lookback_date, tickers.end_date, tickers.horizon)
# etf_features = etf_features.drop(columns=['ma_1m','ma_3m','vol_1w', 'price', 'overnight_gap'])


# # #combine all factor columns
# df = (
#     return_data
#     .merge(etf_features, on=['asset_id', 'date'], how='left')
#     .merge(macro_features, on='date', how='left')
# )
# df["y_target"] = df.groupby("asset_id")["y_excess_lead"].shift(-1)
# df["y_lag_1"] = df.groupby("asset_id")["y_target"].shift(1)
# # df["y_lag_2"] = df.groupby("asset_id")["y_target"].shift(2)
# df= df[df['date'] > str("2013-08-28")]
# # Drop last month of each asset (no future return)
# df = df.dropna(subset=["y_target"]).reset_index(drop=True)
# df = df.reset_index(drop=True)

#df.to_csv("marketmaven/datasets/20251127_17tasks_shifted_lagged.csv", index=False)

macro_cols = ['vix', 'vix_ts_chg_1m', 'vix_ts_z_12m',
       'tnote10y', 'tbill3m', 'term_spread', 'credit_spread',
       'credit_spread_chg_1p', 'dxy', 'spy_ret',
       'erp', 'vix_slope', 'rsp_spy',
       'pct_above_50dma', 'hy_spread', 'hy_spread_chg_1m', 'hy_spread_z_12m',
       'oil', 'copper', 'gold', 'schp', 'schp_ret', 'em_fx',
       'gold_crude_ratio', 'breakeven_proxy']

etf_cols = ['mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom',
    'sd_turn', 'ill', 'vol_1m', 'vol_3m',
       'vol_of_vol', 'vol_z', 'vol_accel', 'ma_signal', 'trend_slope',
       'ret_autocorr', 'vol_autocorr', 'ret_skew', 'ret_kurt', 'baspread', 'y_lag_1']

df_raw = pd.read_csv("marketmaven/datasets/20251127_17tasks_shifted_lagged.csv")
df_raw = df_raw.drop(columns=['y_excess_lead', 'log_ret', 'dolvol', 'volume', 'turnover', 'em_fx_ret', 'oil_ret', 'copper_ret','move_proxy','y10_real_proxy',])
df_raw['date'] = pd.to_datetime(df_raw['date'])

time_col = "date"
y_col = "y_target" #"y_excess_lead"
asset_col = "asset_id"
# cols_to_exclude = ['vix', 'log_ret', 'skew_proxy', 'yc_pc1', 'yc_pc2','vix_slope','gold',
#        'yc_pc3', 'ma_1m', 'ma_3m','overnight_gap','vol_accel', 'y10_nominal', 'tnote10y' ,
#        'hy_spread', 'hy_spread_chg_1m','hy_spread_z_12m','sd_turn','volume','rsp_spy', 'price', 'trend_slope'] # 'rsp_spy','dolvol', 'price','turnover', 'baspread',
# exclude_set = set(cols_to_exclude) | {asset_col, y_col, time_col}
etf_macro = etf_cols + macro_cols
# df = df.drop(columns=cols_to_exclude)

all_assets_occur = check_equal_occurrences(df_raw, 'asset_id')
print('Did all assets occur equally often?', all_assets_occur)
droppin_cols = ['date', 'asset_id']
y_col = ['y_target']
col_order  = droppin_cols + etf_macro + y_col
df_raw = df_raw[col_order]
df_raw['t_index'] = pd.factorize(df_raw['date'])[0]
cols = df_raw.columns.tolist()       
cols.insert(0, cols.pop(cols.index("t_index")))

feature_cols = ['t_index'] + etf_macro

# === Build index lists ===

# Time kernel active dims
active_dims_t = [feature_cols.index('t_index')]

# ETF features active dims
active_dims_e = [feature_cols.index(col) for col in etf_cols]

# Macro features active dims
active_dims_m = [feature_cols.index(col) for col in macro_cols]

######## Visualize Factors ########
df = df_raw[cols].copy()

fig = correlation_matrix(df)
fig.show()

df.hist(bins=30)

############### Data Preparation ###############

cv_config = CVConfig(
    step=1,
    horizon_cv=1,
    embargo=0,
    training_min=132,
)
    
X, I, y, task_map = prepare_multitask_gp_data(
    df,
    target_col="y_target",
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


# ard_num_dims = len(feature_cols)
# active_dims_features = list(range(1, ard_num_dims + 1))

train_idx_1_ahead = list(range(X.shape[0] - len(tickers.tickers)))
test_idx_1_ahead = list(range(X.shape[0] - len(tickers.tickers), X.shape[0]))
multiconfig = MultiTaskConfig(
    num_tasks=len(tickers.tickers),
    mean=MeanF.MULTITASK_CONSTANT,
    rank=6,
    scaling="global",
    min_noise=1e-5,
)

def create_kernel_initialization(kernel: KernelConfig, n_months: int):
    # prior_mean = sqrt(kernel.mean_sqrt) + 0.01 * log(len(kernel.active_dims))
    # prior_scale = kernel.std
    #prior = LogNormalPrior(loc=prior_mean, scale=prior_scale)
    prior = adaptive_lengthscale_prior(num_dims=len(kernel.active_dims))
    
    
    # For periodic only basically 
    period_length = 12.0 / (n_months - 1)  # same as your code
    kernel_initialized = initialize_kernel(
        kernel.type,
        active_dims=kernel.active_dims,
        batch_shape=torch.Size(),
        smoothness=kernel.smoothness,
        q=kernel.q,
        prior=prior,
        period_length=period_length
    )

    return kernel_initialized


class KernelSpec(BaseModel):
    type: KernelType
    smoothness: float | None  # None for non-Matern kernels


class ExperimentConfig(BaseModel):
    etf: KernelSpec
    macro: KernelSpec
    time: KernelSpec
   
# Perhaps try  MATERN_LINEAR_RQ
# ETF kernels and their available smoothness options
ETF_KERNEL_GRID = {
    KernelType.EXPO_GAMMA:    [None],
    KernelType.MATERN_RQ:     [0.5],
    KernelType.MATERN_LINEAR: [0.5],
    KernelType.EXPO_RQ:       [None],
}

# Macro kernels
MACRO_KERNEL_GRID = {
    KernelType.RQ_LINEAR:     [None],
    KernelType.MATERN_LINEAR: [1.5, 2.5],
}

# Time kernels
TIME_KERNEL_GRID = {
    KernelType.MATERN:          [1.5],   # Matern 3/2
    KernelType.PERIODIC_MATERN: [2.5],   # Matern 5/2 + periodic
}

def build_experiment_grid():
    experiment_list = []

    for etf_k, etf_s_list in ETF_KERNEL_GRID.items():
        for macro_k, macro_s_list in MACRO_KERNEL_GRID.items():
            for time_k, time_s_list in TIME_KERNEL_GRID.items():
                for etf_s, macro_s, time_s in itertools.product(
                    etf_s_list, macro_s_list, time_s_list
                ):
                    cfg = ExperimentConfig(
                        etf=KernelSpec(type=etf_k, smoothness=etf_s),
                        macro=KernelSpec(type=macro_k, smoothness=macro_s),
                        time=KernelSpec(type=time_k, smoothness=time_s),
                    )
                    experiment_list.append(cfg)

    return experiment_list
    
experiment_grid = build_experiment_grid()
############### Run ###############
seed = 27
# feature_kernel_types = [
#     #KernelType.EXPO_RQ,
#     KernelType.MATERN,
#     #KernelType.MATERN_LINEAR,
#     #KernelType.MATERN_PIECEWISE,
#     #KernelType.MATERN_RQ,
#     #KernelType.MATERN_LINEAR_RQ
# ]
# feature_smoothness = 0.5
# time_kernel_types = [
#     #KernelType.PERIODIC_MATERN,
#     # KernelType.MATERN_LINEAR_PERIODIC,
#     KernelType.MATERN,
#     #KernelType.MATERN_PIECEWISE,
# ]


# etf_kernel_types = [
#     KernelType.MATERN,
# ]
# etf_smoothness_list = [0.5, 1.5, 2.5]
# macro_kernel_types = [
#     KernelType.MATERN,
# ]
# macro_smoothness_list = [1.5, 2.5]
# time_kernel_types = [
#     KernelType.MATERN,
# ]
# time_smoothness_list = [1.5]


# (Optional) group all runs in one MLflow experiment
mlflow.set_experiment("MTGP: Testing variable reduction and different kernel combinations.")


for cfg in experiment_grid:
    run_name = (
        f"etf={cfg.etf.type.value}_s={cfg.etf.smoothness}_"
        f"macro={cfg.macro.type.value}_s={cfg.macro.smoothness}_"
        f"time={cfg.time.type.value}_s={cfg.time.smoothness}"
    )

    with mlflow.start_run(run_name=run_name, description="""Testing different kernel combinations on ETF and macro factors.""") as run:
        # Convert ExperimentConfig → your KernelConfig
        kernel_e = KernelConfig(
            type=cfg.etf.type,
            features=etf_cols,
            active_dims=active_dims_e,
            smoothness=cfg.etf.smoothness or 1.5,
            gamma=1,
            q=1,
        )

        kernel_m = KernelConfig(
            type=cfg.macro.type,
            features=macro_cols,
            active_dims=active_dims_m,
            smoothness=cfg.macro.smoothness or 1.5,
            gamma=1,
            q=1,
        )

        kernel_t = KernelConfig(
            type=cfg.time.type,
            features=[time_col],
            active_dims=active_dims_t,
            smoothness=cfg.time.smoothness or 1.5,
            gamma=1,
            q=1,
        )
        
        mlflow.log_param("seed", seed)
        torch.manual_seed(seed)
        
        def fit_eval_split_mtgp(
            X: torch.Tensor,
            I: torch.Tensor,
            y: torch.Tensor,
            train_idx,
            test_idx,
            multiconfig: MultiTaskConfig,
            kernel_e: KernelConfig,
            kernel_m: KernelConfig,
            kernel_t: KernelConfig,
            scale_y: str = "global",
            eps: float = 1e-8,
        ):
            """
            Fit multitask Hadamard GP on a single train/test split and return:
            - y_true (test),
            - y_hat (predictions),
            - y_std (uncertainty),
            - gp_cfg (serialized GP config dict).
            """
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.use_deterministic_algorithms(True)

            # Slice
            X_tr, I_tr, y_tr = X[train_idx], I[train_idx], y[train_idx]
            X_te, I_te, y_te = X[test_idx], I[test_idx], y[test_idx]

            # ---- Scaling ----
            scaler = MultitaskScaler(scale_y=scale_y, exclude_time_col=False)
            scaler.fit_x(X_tr)
            X_trs = scaler.transform_x(X_tr)
            X_tes = scaler.transform_x(X_te)
            y_trs = scaler.fit_y(y_tr, I_tr)

            # Append task index as last column
            X_trs = torch.cat([X_trs, I_tr.to(X_trs.dtype)], dim=-1)
            X_tes = torch.cat([X_tes, I_te.to(X_tes.dtype)], dim=-1)

            X_trs = X_trs.to(device)
            y_trs = y_trs.to(device)
            X_tes = X_tes.to(device)

            # ---- Mean and Kernels ----
            mean_f = initialize_mean(
                multiconfig.mean,
                num_tasks=multiconfig.num_tasks,
                input_size=X.shape[1],
            )
            
            # Periods
            n_months = len(np.unique(X_tr.numpy()[:, 0]))
            
            # Kernels
            kernele = create_kernel_initialization(kernel_e, n_months)
            kernelm = create_kernel_initialization(kernel_m, n_months)
            kernelt = create_kernel_initialization(kernel_t, n_months)


            kernel_total = (kernele + kernelm) * kernelt

            # ---- Train model ----
            model, likelihood = train_model_hadamard(
                X_trs,
                y_trs,
                rank=multiconfig.rank,
                mean_f=mean_f,
                kernel=kernel_total,
                visualize=True,
                dtype=torch.float32,
                device=torch.device("cpu"),
                min_noise=multiconfig.min_noise,
            )

            model_str = repr(model)

            # ---- Predict ----
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                f_dist = model(X_tes)
                pred = likelihood(f_dist, X_tes)

            y_hat = scaler.inverse_y(pred.mean, I_te)
            y_std = scaler.inverse_std(pred.variance.sqrt(), I_te)

            return y_te, y_hat, y_std, model_str, model

        
        # High-level configs (Pydantic model_dump)
        mlflow.log_params(tickers.model_dump())
        mlflow.log_params(cv_config.model_dump())
        mlflow.log_params(multiconfig.model_dump())
        log_kernel_to_mlflow(kernel_e, "etf")
        log_kernel_to_mlflow(kernel_m, "macro")
        log_kernel_to_mlflow(kernel_t, "time")


        # ---- MULTITASK GP PARALLEL EVALUATION ----
        results = Parallel(
            n_jobs=-1,
            backend="loky",
            verbose=10,
        )(
            delayed(fit_eval_split_mtgp)(
                X, I, y,
                tr_idx, te_idx,
                multiconfig=multiconfig,
                kernel_e=kernel_e,
                kernel_m=kernel_m,
                kernel_t=kernel_t,
                scale_y=multiconfig.scaling,
            )
            for tr_idx, te_idx in splits
        )

        true_list, pred_list, unc_list, model_str_list, model_list = zip(*results)

        # Save last GP config for inspection
        # mlflow.log_dict(gp_cfg_list[-1], "gp_config.json")
        mlflow.log_text(model_str_list[-1], "gp_model.txt")
        log_gpytorch_state_dict(model_list[-1], "gp_state.json")

        true = np.vstack(true_list)
        pred = np.vstack(pred_list)
        unc = np.vstack(unc_list)

        # Convert GP predictions to DataFrames
        asset_cols = list(task_map.keys())  # task_map assumed defined elsewhere
        true_df = pd.DataFrame(true, columns=[f"{c}_true" for c in asset_cols])
        preds_df = pd.DataFrame(pred, columns=[f"{c}_pred" for c in asset_cols])
        unc_df = pd.DataFrame(unc, columns=[f"{c}_unc" for c in asset_cols])
        
        # ---- CONFIDENCE INTERVALS ----
        ci_lower = preds_df.copy()
        ci_upper = preds_df.copy()

        for c in asset_cols:
            ci_lower[f"{c}_lower_95"] = preds_df[f"{c}_pred"] - 1.96 * unc_df[f"{c}_unc"]
            ci_upper[f"{c}_upper_95"] = preds_df[f"{c}_pred"] + 1.96 * unc_df[f"{c}_unc"]

        # Combine lower + upper bands into a single DataFrame
        ci_df = pd.DataFrame()

        for c in asset_cols:
            ci_df[f"{c}_pred"]      = preds_df[f"{c}_pred"]
            ci_df[f"{c}_lower_95"]  = preds_df[f"{c}_pred"] - 1.96 * unc_df[f"{c}_unc"]
            ci_df[f"{c}_upper_95"]  = preds_df[f"{c}_pred"] + 1.96 * unc_df[f"{c}_unc"]

        # Save CI DataFrame as artifact
        ci_path = "marketmaven/mlflow/artifacts/confidence_intervals.csv"
        ci_df.to_csv(ci_path, index=False)
        mlflow.log_artifact(ci_path)
                    
        # ---- MTGP EVAL ----
        summary_gp = evaluate_asset_pricing(
            y_test=true_df,
            y_pred=preds_df
        )

        # ---- BENCHMARK EVALUATIONS ----
        mean_preds = []
        ewma_preds = []
        asset_cols = list(task_map.keys())

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
        
        plot_path = "marketmaven/mlflow/artifacts/actual_vs_pred_matrix.png"

        plot_actual_vs_pred_matrix(
            true_df=true_df,          # already aligned
            pred_df=preds_df,
            asset_cols=asset_cols,
            save_path=plot_path
        )

        mlflow.log_artifact(plot_path)

        # Evaluate benchmarks
        summary_mean = evaluate_asset_pricing(y_test=true_b_df, y_pred=mean_df)
        summary_ewma = evaluate_asset_pricing(y_test=true_b_df, y_pred=ewma_df)

        # Log metrics
        mlflow.log_metrics({f"{k}": v for k, v in summary_gp.items()})
        
        comparison = pd.DataFrame({
        "GP": summary_gp,
        "Mean": summary_mean,
        "EWMA": summary_ewma,
        })
        comparison.to_html("marketmaven/mlflow/artifacts/comparison_table.html")
        mlflow.log_artifact("marketmaven/mlflow/artifacts/comparison_table.html")
        
        # R2_OS calculations
        y_true_norm = true_b_df.rename(columns=lambda c: c.replace("_true", ""))
        y_pred_norm = preds_df.rename(columns=lambda c: c.replace("_pred", ""))
        y_ewma_norm = ewma_df.copy()
        y_mean_norm = mean_df.copy()
        
        compare_error = model_error_by_time_index(y_true_norm, y_pred_norm)
        compare_error.to_html("marketmaven/mlflow/artifacts/compare_error.html")
        mlflow.log_artifact("marketmaven/mlflow/artifacts/compare_error.html")
        
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
        #mlflow.log_metrics(ls_stats_ewma)
        plot_ls_cumulative_compare(ls_gp, ls_mean, ls_ewma)

        
        
        
# for ek in etf_kernel_types:
#     for mk in macro_kernel_types:
#         for tk in time_kernel_types:
#             for sm_t in time_smoothness_list:

#                 # ----- Build kernel configs for this run -----
#                 kernel_e = KernelConfig(
#                     type=ek,
#                     features=etf_cols,
#                     active_dims=active_dims_e,
#                     smoothness=etf_smoothness_list[0],
#                     gamma=1,
#                     q=1,
#                 )
                
#                 kernel_m = KernelConfig(
#                     type=mk,
#                     features=macro_cols,
#                     active_dims=active_dims_m,
#                     smoothness=macro_smoothness_list[0],
#                     gamma=1,
#                     q=1,
#                 )
                
#                 kernel_t = KernelConfig(
#                     type=tk,
#                     features=[time_col],
#                     active_dims=active_dims_t,
#                     smoothness=time_smoothness_list[0],
#                     gamma=1,
#                     q=1,
#                 )

#                 # Nice name per run
#                 run_name = f"ek={ek.value}_mk={mk.value}_tk={tk.value}"

#                 with mlflow.start_run(run_name=run_name, description="Baseline just Matern on time and features.") as run:
#                     # -------- Log configs --------
#                     mlflow.log_param("seed", seed)
#                     torch.manual_seed(seed)
                    
#                     def fit_eval_split_mtgp(
#                         X: torch.Tensor,
#                         I: torch.Tensor,
#                         y: torch.Tensor,
#                         train_idx,
#                         test_idx,
#                         multiconfig: MultiTaskConfig,
#                         kernel_e: KernelConfig,
#                         kernel_m: KernelConfig,
#                         kernel_t: KernelConfig,
#                         scale_y: str = "global",
#                         eps: float = 1e-8,
#                     ):
#                         """
#                         Fit multitask Hadamard GP on a single train/test split and return:
#                         - y_true (test),
#                         - y_hat (predictions),
#                         - y_std (uncertainty),
#                         - gp_cfg (serialized GP config dict).
#                         """
#                         torch.manual_seed(seed)
#                         np.random.seed(seed)
#                         random.seed(seed)

#                         torch.use_deterministic_algorithms(True)

#                         # Slice
#                         X_tr, I_tr, y_tr = X[train_idx], I[train_idx], y[train_idx]
#                         X_te, I_te, y_te = X[test_idx], I[test_idx], y[test_idx]

#                         # ---- Scaling ----
#                         scaler = MultitaskScaler(scale_y=scale_y, exclude_time_col=False)
#                         scaler.fit_x(X_tr)
#                         X_trs = scaler.transform_x(X_tr)
#                         X_tes = scaler.transform_x(X_te)
#                         y_trs = scaler.fit_y(y_tr, I_tr)

#                         # Append task index as last column
#                         X_trs = torch.cat([X_trs, I_tr.to(X_trs.dtype)], dim=-1)
#                         X_tes = torch.cat([X_tes, I_te.to(X_tes.dtype)], dim=-1)

#                         X_trs = X_trs.to(device)
#                         y_trs = y_trs.to(device)
#                         X_tes = X_tes.to(device)

#                         # ---- Mean and Kernels ----
#                         mean_f = initialize_mean(
#                             multiconfig.mean,
#                             num_tasks=multiconfig.num_tasks,
#                             input_size=X.shape[1],
#                         )
                        
#                         # Periods
#                         n_months = len(np.unique(X_tr.numpy()[:, 0]))
                        
#                         # Kernels
#                         kernele = create_kernel_initialization(kernel_e, n_months)
#                         kernelm = create_kernel_initialization(kernel_m, n_months)
#                         kernelt = create_kernel_initialization(kernel_t, n_months)


#                         kernel_total = (kernele + kernelm) * kernelt

#                         # ---- Train model ----
#                         model, likelihood = train_model_hadamard(
#                             X_trs,
#                             y_trs,
#                             rank=multiconfig.rank,
#                             mean_f=mean_f,
#                             kernel=kernel_total,
#                             visualize=True,
#                             dtype=torch.float32,
#                             device=torch.device("cpu"),
#                             min_noise=multiconfig.min_noise,
#                         )

#                         model_str = repr(model)

#                         # ---- Predict ----
#                         model.eval()
#                         likelihood.eval()
#                         with torch.no_grad():
#                             f_dist = model(X_tes)
#                             pred = likelihood(f_dist, X_tes)

#                         y_hat = scaler.inverse_y(pred.mean, I_te)
#                         y_std = scaler.inverse_std(pred.variance.sqrt(), I_te)

#                         return y_te, y_hat, y_std, model_str, model

                    
#                     # High-level configs (Pydantic model_dump)
#                     mlflow.log_params(tickers.model_dump())
#                     mlflow.log_params(cv_config.model_dump())
#                     mlflow.log_params(multiconfig.model_dump())
#                     log_kernel_to_mlflow(kernel_e, "etf")
#                     log_kernel_to_mlflow(kernel_m, "macro")
#                     log_kernel_to_mlflow(kernel_t, "time")


#                     # ---- MULTITASK GP PARALLEL EVALUATION ----
#                     results = Parallel(
#                         n_jobs=-1,
#                         backend="loky",
#                         verbose=10,
#                     )(
#                         delayed(fit_eval_split_mtgp)(
#                             X, I, y,
#                             tr_idx, te_idx,
#                             multiconfig=multiconfig,
#                             kernel_e=kernel_e,
#                             kernel_m=kernel_m,
#                             kernel_t=kernel_t,
#                             scale_y=multiconfig.scaling,
#                         )
#                         for tr_idx, te_idx in splits
#                     )

#                     true_list, pred_list, unc_list, model_str_list, model_list = zip(*results)

#                     # Save last GP config for inspection
#                     # mlflow.log_dict(gp_cfg_list[-1], "gp_config.json")
#                     mlflow.log_text(model_str_list[-1], "gp_model.txt")
#                     log_gpytorch_state_dict(model_list[-1], "gp_state.json")

#                     true = np.vstack(true_list)
#                     pred = np.vstack(pred_list)
#                     unc = np.vstack(unc_list)

#                     # Convert GP predictions to DataFrames
#                     asset_cols = list(task_map.keys())  # task_map assumed defined elsewhere
#                     true_df = pd.DataFrame(true, columns=[f"{c}_true" for c in asset_cols])
#                     preds_df = pd.DataFrame(pred, columns=[f"{c}_pred" for c in asset_cols])
#                     unc_df = pd.DataFrame(unc, columns=[f"{c}_unc" for c in asset_cols])
                    
#                     # ---- CONFIDENCE INTERVALS ----
#                     ci_lower = preds_df.copy()
#                     ci_upper = preds_df.copy()

#                     for c in asset_cols:
#                         ci_lower[f"{c}_lower_95"] = preds_df[f"{c}_pred"] - 1.96 * unc_df[f"{c}_unc"]
#                         ci_upper[f"{c}_upper_95"] = preds_df[f"{c}_pred"] + 1.96 * unc_df[f"{c}_unc"]

#                     # Combine lower + upper bands into a single DataFrame
#                     ci_df = pd.DataFrame()

#                     for c in asset_cols:
#                         ci_df[f"{c}_pred"]      = preds_df[f"{c}_pred"]
#                         ci_df[f"{c}_lower_95"]  = preds_df[f"{c}_pred"] - 1.96 * unc_df[f"{c}_unc"]
#                         ci_df[f"{c}_upper_95"]  = preds_df[f"{c}_pred"] + 1.96 * unc_df[f"{c}_unc"]

#                     # Save CI DataFrame as artifact
#                     ci_path = "marketmaven/mlflow/artifacts/confidence_intervals.csv"
#                     ci_df.to_csv(ci_path, index=False)
#                     mlflow.log_artifact(ci_path)
                                
#                     # ---- MTGP EVAL ----
#                     summary_gp = evaluate_asset_pricing(
#                         y_test=true_df,
#                         y_pred=preds_df
#                     )

#                     # ---- BENCHMARK EVALUATIONS ----
#                     mean_preds = []
#                     ewma_preds = []
#                     asset_cols = list(task_map.keys())

#                     for i, (tr_idx, te_idx) in enumerate(splits):

#                         y_tr_tensor = y[tr_idx]
#                         I_tr_tensor = I[tr_idx]

#                         # Convert to panel
#                         y_tr_panel = long_to_panel(y_tr_tensor, I_tr_tensor, asset_cols)  

#                         # TEST SLICE IS ALREADY STACKED INTO "true"
#                         y_te_array = y[te_idx]             # shape: (n_assets,)
#                         y_te_tensor = torch.tensor(y_te_array).unsqueeze(0)  # (1 × n_assets)
#                         y_te_panel = pd.DataFrame(y_te_tensor.numpy(), columns=asset_cols)

#                         # ----- Mean benchmark -----
#                         mean_df = compute_benchmark_panel(
#                             y_train=y_tr_panel,
#                             y_test=y_te_panel,
#                             method="mean"
#                         )
#                         mean_preds.append(mean_df.values)

#                         # ----- EWMA benchmark -----
#                         ewma_df = compute_benchmark_panel(
#                             y_train=y_tr_panel,
#                             y_test=y_te_panel,
#                             method="ewma2"
#                         )
#                         ewma_preds.append(ewma_df.values)


#                     # Convert lists → arrays
#                     mean_pred = np.vstack(mean_preds)        # (N_splits × n_assets)
#                     ewma_pred = np.vstack(ewma_preds)        # (N_splits × n_assets)

#                     # Ground truth is already stacked correctly from the GP loop
#                     y_bench_true = true                       # (N_splits × n_assets)

#                     # Build DataFrames
#                     true_b_df = pd.DataFrame(y_bench_true, columns=asset_cols)
#                     mean_df = pd.DataFrame(mean_pred, columns=asset_cols)
#                     ewma_df = pd.DataFrame(ewma_pred, columns=asset_cols)
                    
#                     plot_path = "marketmaven/mlflow/artifacts/actual_vs_pred_matrix.png"

#                     plot_actual_vs_pred_matrix(
#                         true_df=true_df,          # already aligned
#                         pred_df=preds_df,
#                         asset_cols=asset_cols,
#                         save_path=plot_path
#                     )

#                     mlflow.log_artifact(plot_path)

#                     # Evaluate benchmarks
#                     summary_mean = evaluate_asset_pricing(y_test=true_b_df, y_pred=mean_df)
#                     summary_ewma = evaluate_asset_pricing(y_test=true_b_df, y_pred=ewma_df)

#                     # Log metrics
#                     mlflow.log_metrics({f"{k}": v for k, v in summary_gp.items()})
                    
#                     comparison = pd.DataFrame({
#                     "GP": summary_gp,
#                     "Mean": summary_mean,
#                     "EWMA": summary_ewma,
#                     })
#                     comparison.to_html("marketmaven/mlflow/artifacts/comparison_table.html")
#                     mlflow.log_artifact("marketmaven/mlflow/artifacts/comparison_table.html")
                    
#                     # R2_OS calculations
#                     y_true_norm = true_b_df.rename(columns=lambda c: c.replace("_true", ""))
#                     y_pred_norm = preds_df.rename(columns=lambda c: c.replace("_pred", ""))
#                     y_ewma_norm = ewma_df.copy()
#                     y_mean_norm = mean_df.copy()
                    
#                     compare_error = model_error_by_time_index(y_true_norm, y_pred_norm)
#                     compare_error.to_html("marketmaven/mlflow/artifacts/compare_error.html")
#                     mlflow.log_artifact("marketmaven/mlflow/artifacts/compare_error.html")
                    
#                     r2_gp_vs_mean = r2_os(y_true_norm, y_pred_norm, y_mean_norm)
#                     r2_gp_vs_ewma = r2_os(y_true_norm, y_pred_norm, y_ewma_norm)
                    
#                     log_r2_os("gp_vs_mean", r2_gp_vs_mean)
#                     log_r2_os("gp_vs_ewma", r2_gp_vs_ewma)
                    
#                     # Long-short performance
#                     ls_gp = long_short_returns(y_true=y_true_norm, y_pred=y_pred_norm)
#                     ls_mean = long_short_returns(y_true=y_true_norm, y_pred=y_mean_norm)
#                     ls_ewma = long_short_returns(y_true=y_true_norm, y_pred=y_ewma_norm )
#                     ls_stats_gp = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_pred_norm, label="gp")
#                     ls_stats_mean = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_mean_norm, label="mean")
#                     ls_stats_ewma = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_ewma_norm, label="ewma")
#                     mlflow.log_metrics(ls_stats_gp)
#                     mlflow.log_metrics(ls_stats_mean)
#                     #mlflow.log_metrics(ls_stats_ewma)
#                     plot_ls_cumulative_compare(ls_gp, ls_mean, ls_ewma)
