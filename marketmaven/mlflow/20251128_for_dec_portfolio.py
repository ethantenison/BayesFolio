from re import sub
import mlflow
import pandas as pd
import warnings
import os
from joblib import Parallel, delayed
from pydantic import BaseModel
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from marketmaven.configs import TickerConfig, Interval, Horizon, CVConfig
from marketmaven.asset_prices import build_long_panel
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
from marketmaven.visualization.evaluation import plot_ls_cumulative_compare, plot_actual_vs_pred_matrix
from marketmaven.portfolio.helpers import assessing_long_short_performance, long_short_returns
from marketmaven.models.kernels import MeanF, KernelType, initialize_mean, initialize_kernel, adaptive_lengthscale_prior
from marketmaven.mlflow.helpers import (
    KernelConfig, MultiTaskConfig, long_to_panel, compute_benchmark_panel, r2_os, log_r2_os,
    model_error_by_time_index, log_kernel_to_mlflow, log_gpytorch_state_dict
)
import random
import itertools
from marketmaven.market_fundamentals import fetch_enhanced_macro_features
from marketmaven.asset_prices import fetch_etf_features
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
    # "AVEM",
    # "ISCF",
    "VWOB",
]


tickers = TickerConfig(
    start_date="2016-11-29",
    end_date="2025-11-29",
    interval=Interval.DAILY,
    tickers=etf_tickers,
    horizon=Horizon.MONTHLY,
    lookback_date="2014-06-01"
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

# # # ############### Factor data ###############
macro_features = fetch_enhanced_macro_features(start=tickers.lookback_date, end=tickers.end_date)
macro_features = macro_features.drop(columns=['vix_ts_level','skew_proxy','vix3m','yc_pc1', 'yc_pc2', 'yc_pc3', 'y10_nominal' ])
macro_cols = macro_features.columns[1:].tolist()
etf_features = fetch_etf_features(tickers.tickers, tickers.lookback_date, tickers.end_date, tickers.horizon)
etf_features = etf_features.drop(columns=['ma_1m','ma_3m','vol_1w', 'price', 'overnight_gap'])

etf_features["ill_adj"] = etf_features["ill"] * 1e9
etf_features["ill_log"] = np.log(etf_features["ill_adj"] + 1)
etf_features["dolvol_adj"] = etf_features["dolvol"] * 1e9
etf_features["dolvol_log"] = np.log(etf_features["dolvol_adj"] + 1)

# optional: very mild global clipping before scaler
etf_features["ill_log"] = etf_features["ill_log"].clip(upper=etf_features["ill_log"].quantile(0.99))
etf_features["dolvol_log"] = etf_features["dolvol_log"].clip(upper=etf_features["dolvol_log"].quantile(0.99))
etf_features = etf_features.drop(columns=['ill', 'ill_adj', 'dolvol', 'dolvol_adj'])
etf_features = etf_features.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)


# #combine all factor columns
# df = (
#     return_data
#     .merge(etf_features, on=['asset_id', 'date'], how='left')
#     .merge(macro_features, on='date', how='left')
# )
df = (
    etf_features
    .merge(macro_features, on='date', how='left')
    .merge(return_data, on=['asset_id', 'date'], how='left')
)
df = df.sort_values(["asset_id", "date"])
df["lag_y_excess_lead"] = (
    df.groupby("asset_id")["y_excess_lead"]
      .shift(1)
)
df["lag2_y_excess_lead"] = (
    df.groupby("asset_id")["y_excess_lead"]
      .shift(2)
)
df= df[df['date'] > str("2016-11-28")]
# df = df.dropna(subset=['y_excess_lead']).reset_index(drop=True)
df = df.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)



etf_cols = ['lag_y_excess_lead','lag2_y_excess_lead', 'mom6m', 'mom12m', 'mom36m', 'chmom',
    "ill_log", 'vol_1m', 'vol_3m',
       'vol_of_vol',"dolvol_log", 'ma_signal', 
       'ret_autocorr', 'baspread'] 

macro_cols = ['vix',
    'tbill3m', 'term_spread', 'credit_spread',
       'credit_spread_chg_1p', 'dxy','spy_ret', 
       'erp', 'rsp_spy',
       'pct_above_50dma', 'hy_spread', 'hy_spread_chg_1m', 'hy_spread_z_12m',
       'oil', 'copper', 'gold', 'schp', 'em_fx',
       'gold_crude_ratio', 'breakeven_proxy']

drop_columns= [ 'volume', 'turnover',
                              'em_fx_ret', 'oil_ret', 'copper_ret','move_proxy', 'sd_turn','vix_ts_z_12m','tnote10y',
                              'vix_ts_chg_1m', 'schp_ret', 'vix_slope','vol_z', 'vol_accel', 'trend_slope','vol_autocorr',
                              'ret_skew', 'ret_kurt','log_ret','mom1m','y10_real_proxy'
                              ]
#df_raw = pd.read_csv("marketmaven/datasets/20251129_14tasks.csv")
df = df.drop(columns=drop_columns)
df['date'] = pd.to_datetime(df['date'])

time_col = "date"
y_col = "y_excess_lead" #"y_excess_lead"
asset_col = "asset_id"
# cols_to_exclude = ['vix', 'log_ret', 'skew_proxy', 'yc_pc1', 'yc_pc2','vix_slope','gold',
#        'yc_pc3', 'ma_1m', 'ma_3m','overnight_gap','vol_accel', 'y10_nominal', 'tnote10y' ,
#        'hy_spread', 'hy_spread_chg_1m','hy_spread_z_12m','sd_turn','volume','rsp_spy', 'price', 'trend_slope'] # 'rsp_spy','dolvol', 'price','turnover', 'baspread',
# exclude_set = set(cols_to_exclude) | {asset_col, y_col, time_col}
etf_macro = etf_cols + macro_cols
# df = df.drop(columns=cols_to_exclude)

all_assets_occur = check_equal_occurrences(df, 'asset_id')
print('Did all assets occur equally often?', all_assets_occur)
droppin_cols = ['date', 'asset_id']
y_col = ['y_excess_lead']
col_order  = droppin_cols + etf_macro + y_col
df = df[col_order]
df['t_index'] = pd.factorize(df['date'])[0]
cols = df.columns.tolist()       
cols.insert(0, cols.pop(cols.index("t_index")))

df = df.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)

feature_cols = ['t_index'] + etf_macro



############################### Game day dataset ##################################

# gd = (
#     etf_features
#     .merge(macro_features, on='date', how='left')
#     .merge(return_data, on=['asset_id', 'date'], how='left')
# )
# gd = gd[gd['date'] > str("2013-08-28")]
# gd = gd.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)


# === Build index lists ===

# Time kernel active dims
active_dims_t = [feature_cols.index('t_index')]

# ETF features active dims
active_dims_e = [feature_cols.index(col) for col in etf_cols]

# Macro features active dims
active_dims_m = [feature_cols.index(col) for col in macro_cols]

######## Visualize Factors ########
df = df[cols].copy()
df.to_csv("marketmaven/datasets/20251130_18tasks.csv", index=False)
df_forecast = df[df['date'] > pd.Timestamp("2025-11-01")]
df_forecast = df_forecast.reset_index(drop=True)
df = df.dropna(subset=['y_excess_lead']).reset_index(drop=True)

fig = correlation_matrix(df)
fig.show()

df.hist(bins=30)

############### Data Preparation ###############

cv_config = CVConfig(
    step=1,
    horizon_cv=1,
    embargo=0,
    training_min=98,
)
    
X, I, y, task_map = prepare_multitask_gp_data(
    df,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date", "asset_id"],
    dtype=torch.float32
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
    rank=7,
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
    rank: int
   
# Perhaps try  MATERN_LINEAR_RQ
# ETF kernels and their available smoothness options
ETF_KERNEL_GRID = { 
    #KernelType.MATERN_LINEAR: [0.5],
    KernelType.MATERN:        [0.5],
    #KernelType.MATERN_RQ:     [0.5],
}



# # Macro kernels
MACRO_KERNEL_GRID = {
    #KernelType.MATERN_LINEAR: [0.5, 1.5],
    KernelType.MATERN_LINEAR_RQ: [0.5],
}


# Time kernels
TIME_KERNEL_GRID = {
    #KernelType.MATERN:          [0.5],   # Matern 3/2
    KernelType.PERIODIC_MATERN: [0.5],   # Matern 5/2 + periodic kinda good for SPY, IJR, MGK, VTV
}

RANK_GRID = [3, 7]

# Best before leaving out extra : etf=matern_s=0.5_macro=maternlinearrq_s=0.5_time=periodicmatern_s=0.5
# periodic smoothness doesn't so much matter, macro smoothness of 0.5 seems better. 

def build_experiment_grid():
    experiment_list = []

    for etf_k, etf_s_list in ETF_KERNEL_GRID.items():
        for macro_k, macro_s_list in MACRO_KERNEL_GRID.items():
            for time_k, time_s_list in TIME_KERNEL_GRID.items():
                for rank in RANK_GRID:   # 👈 LOOP OVER RANKS
                    for etf_s, macro_s, time_s in itertools.product(
                        etf_s_list, macro_s_list, time_s_list
                    ):
                        cfg = ExperimentConfig(
                            etf=KernelSpec(type=etf_k, smoothness=etf_s),
                            macro=KernelSpec(type=macro_k, smoothness=macro_s),
                            time=KernelSpec(type=time_k, smoothness=time_s),
                            rank=rank,                     # 👈 assign here
                        )
                        experiment_list.append(cfg)

    return experiment_list
    
experiment_grid = build_experiment_grid()
############### Run ###############
seed = 27

# (Optional) group all runs in one MLflow experiment
mlflow.set_experiment("Gameday 11/29/2025.")


for cfg in experiment_grid:
    run_name = (
        f"e={cfg.etf.type.value}_{cfg.etf.smoothness}_"
        f"m={cfg.macro.type.value}_{cfg.macro.smoothness}_"
        f"t={cfg.time.type.value}_{cfg.time.smoothness}_"
        f"rank={cfg.rank}"
    )

    with mlflow.start_run(run_name=run_name, description="""Finding the best final model for gameday with per task scaling.""") as run:
        multiconfig = MultiTaskConfig(
            num_tasks=len(tickers.tickers),
            mean=MeanF.MULTITASK_CONSTANT,
            rank=cfg.rank,
            scaling="global",
            min_noise=1e-5,
                )
        
        # Convert ExperimentConfig → your KernelConfig
        kernel_e = KernelConfig(
            type=cfg.etf.type,
            features=etf_cols,
            active_dims=active_dims_e,
            smoothness=cfg.etf.smoothness or 1.5,
            gamma=1.2,
            q=1,
        )

        kernel_m = KernelConfig(
            type=cfg.macro.type,
            features=macro_cols,
            active_dims=active_dims_m,
            smoothness=cfg.macro.smoothness or 1.5,
            gamma=1.2,
            q=1,
        )

        kernel_t = KernelConfig(
            type=cfg.time.type,
            features=[time_col],
            active_dims=active_dims_t,
            smoothness=cfg.time.smoothness or 1.5,
            gamma=1.2,
            q=1,
        )
        
        mlflow.log_param("seed", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.use_deterministic_algorithms(True)
    
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

# def run_single_experiment(cfg, X, I, y, splits):
#     run_name = (
#         f"e={cfg.etf.type.value}_{cfg.etf.smoothness}_"
#         f"m={cfg.macro.type.value}_{cfg.macro.smoothness}_"
#         f"t={cfg.time.type.value}_{cfg.time.smoothness}_"
#         f"rank={cfg.rank}"
#     )


#     with mlflow.start_run(run_name=run_name):
#         multiconfig = MultiTaskConfig(
#             num_tasks=len(tickers.tickers),
#             mean=MeanF.MULTITASK_CONSTANT,
#             rank=cfg.rank,
#             scaling="global",
#             min_noise=1e-5,
#                 )
        
        
#                 # Convert ExperimentConfig → your KernelConfig
#         kernel_e = KernelConfig(
#             type=cfg.etf.type,
#             features=etf_cols,
#             active_dims=active_dims_e,
#             smoothness=cfg.etf.smoothness or 1.5,
#             gamma=1.2,
#             q=1,
#         )

#         kernel_m = KernelConfig(
#             type=cfg.macro.type,
#             features=macro_cols,
#             active_dims=active_dims_m,
#             smoothness=cfg.macro.smoothness or 1.5,
#             gamma=1.2,
#             q=1,
#         )

#         kernel_t = KernelConfig(
#             type=cfg.time.type,
#             features=[time_col],
#             active_dims=active_dims_t,
#             smoothness=cfg.time.smoothness or 1.5,
#             gamma=1.2,
#             q=1,
#         )
        
#         mlflow.log_param("seed", seed)
#         torch.manual_seed(seed)
        
#         def fit_eval_split_mtgp(
#             X: torch.Tensor,
#             I: torch.Tensor,
#             y: torch.Tensor,
#             train_idx,
#             test_idx,
#             multiconfig: MultiTaskConfig,
#             kernel_e: KernelConfig,
#             kernel_m: KernelConfig,
#             kernel_t: KernelConfig,
#             scale_y: str = "global",
#             eps: float = 1e-8,
#         ):
#             """
#             Fit multitask Hadamard GP on a single train/test split and return:
#             - y_true (test),
#             - y_hat (predictions),
#             - y_std (uncertainty),
#             - gp_cfg (serialized GP config dict).
#             """
#             torch.manual_seed(seed)
#             np.random.seed(seed)
#             random.seed(seed)

#             torch.use_deterministic_algorithms(True)

#             # Slice
#             X_tr, I_tr, y_tr = X[train_idx], I[train_idx], y[train_idx]
#             X_te, I_te, y_te = X[test_idx], I[test_idx], y[test_idx]

#             # ---- Scaling ----
#             scaler = MultitaskScaler(scale_y=scale_y, exclude_time_col=False)
#             scaler.fit_x(X_tr)
#             X_trs = scaler.transform_x(X_tr)
#             X_tes = scaler.transform_x(X_te)
#             y_trs = scaler.fit_y(y_tr, I_tr)

#             # Append task index as last column
#             X_trs = torch.cat([X_trs, I_tr.to(X_trs.dtype)], dim=-1)
#             X_tes = torch.cat([X_tes, I_te.to(X_tes.dtype)], dim=-1)

#             X_trs = X_trs.to(device)
#             y_trs = y_trs.to(device)
#             X_tes = X_tes.to(device)

#             # ---- Mean and Kernels ----
#             mean_f = initialize_mean(
#                 multiconfig.mean,
#                 num_tasks=multiconfig.num_tasks,
#                 input_size=X.shape[1],
#             )
            
#             # Periods
#             n_months = len(np.unique(X_tr.numpy()[:, 0]))
            
#             # Kernels
#             kernele = create_kernel_initialization(kernel_e, n_months)
#             kernelm = create_kernel_initialization(kernel_m, n_months)
#             kernelt = create_kernel_initialization(kernel_t, n_months)


#             kernel_total = (kernele + kernelm) * kernelt

#             # ---- Train model ----
#             model, likelihood = train_model_hadamard(
#                 X_trs,
#                 y_trs,
#                 rank=multiconfig.rank,
#                 mean_f=mean_f,
#                 kernel=kernel_total,
#                 visualize=True,
#                 dtype=torch.float32,
#                 device=torch.device("cpu"),
#                 min_noise=multiconfig.min_noise,
#             )

#             model_str = repr(model)

#             # ---- Predict ----
#             model.eval()
#             likelihood.eval()
#             with torch.no_grad():
#                 f_dist = model(X_tes)
#                 pred = likelihood(f_dist, X_tes)

#             y_hat = scaler.inverse_y(pred.mean, I_te)
#             y_std = scaler.inverse_std(pred.variance.sqrt(), I_te)

#             return y_te, y_hat, y_std, model_str, model

        
#         # High-level configs (Pydantic model_dump)
#         mlflow.log_params(tickers.model_dump())
#         mlflow.log_params(cv_config.model_dump())
#         mlflow.log_params(multiconfig.model_dump())
#         log_kernel_to_mlflow(kernel_e, "etf")
#         log_kernel_to_mlflow(kernel_m, "macro")
#         log_kernel_to_mlflow(kernel_t, "time")


#         # ---- MULTITASK GP PARALLEL EVALUATION ----
#         results = Parallel(
#             n_jobs=6,
#             backend="loky",
#             verbose=10,
#         )(
#             delayed(fit_eval_split_mtgp)(
#                 X, I, y,
#                 tr_idx, te_idx,
#                 multiconfig=multiconfig,
#                 kernel_e=kernel_e,
#                 kernel_m=kernel_m,
#                 kernel_t=kernel_t,
#                 scale_y=multiconfig.scaling,
#             )
#             for tr_idx, te_idx in splits
#         )

#         true_list, pred_list, unc_list, model_str_list, model_list = zip(*results)

#         # Save last GP config for inspection
#         # mlflow.log_dict(gp_cfg_list[-1], "gp_config.json")
#         mlflow.log_text(model_str_list[-1], "gp_model.txt")
#         log_gpytorch_state_dict(model_list[-1], "gp_state.json")

#         true = np.vstack(true_list)
#         pred = np.vstack(pred_list)
#         unc = np.vstack(unc_list)

#         # Convert GP predictions to DataFrames
#         asset_cols = list(task_map.keys())  # task_map assumed defined elsewhere
#         true_df = pd.DataFrame(true, columns=[f"{c}_true" for c in asset_cols])
#         preds_df = pd.DataFrame(pred, columns=[f"{c}_pred" for c in asset_cols])
#         unc_df = pd.DataFrame(unc, columns=[f"{c}_unc" for c in asset_cols])
        
#         # ---- CONFIDENCE INTERVALS ----
#         ci_lower = preds_df.copy()
#         ci_upper = preds_df.copy()

#         for c in asset_cols:
#             ci_lower[f"{c}_lower_95"] = preds_df[f"{c}_pred"] - 1.96 * unc_df[f"{c}_unc"]
#             ci_upper[f"{c}_upper_95"] = preds_df[f"{c}_pred"] + 1.96 * unc_df[f"{c}_unc"]

#         # Combine lower + upper bands into a single DataFrame
#         ci_df = pd.DataFrame()

#         for c in asset_cols:
#             ci_df[f"{c}_pred"]      = preds_df[f"{c}_pred"]
#             ci_df[f"{c}_lower_95"]  = preds_df[f"{c}_pred"] - 1.96 * unc_df[f"{c}_unc"]
#             ci_df[f"{c}_upper_95"]  = preds_df[f"{c}_pred"] + 1.96 * unc_df[f"{c}_unc"]

#         # Save CI DataFrame as artifact
#         ci_path = "marketmaven/mlflow/artifacts/confidence_intervals.csv"
#         ci_df.to_csv(ci_path, index=False)
#         mlflow.log_artifact(ci_path)
                    
#         # ---- MTGP EVAL ----
#         summary_gp = evaluate_asset_pricing(
#             y_test=true_df,
#             y_pred=preds_df
#         )

#         # ---- BENCHMARK EVALUATIONS ----
#         mean_preds = []
#         ewma_preds = []
#         asset_cols = list(task_map.keys())

#         for i, (tr_idx, te_idx) in enumerate(splits):

#             y_tr_tensor = y[tr_idx]
#             I_tr_tensor = I[tr_idx]

#             # Convert to panel
#             y_tr_panel = long_to_panel(y_tr_tensor, I_tr_tensor, asset_cols)  

#             # TEST SLICE IS ALREADY STACKED INTO "true"
#             y_te_array = y[te_idx]             # shape: (n_assets,)
#             y_te_tensor = torch.tensor(y_te_array).unsqueeze(0)  # (1 × n_assets)
#             y_te_panel = pd.DataFrame(y_te_tensor.numpy(), columns=asset_cols)

#             # ----- Mean benchmark -----
#             mean_df = compute_benchmark_panel(
#                 y_train=y_tr_panel,
#                 y_test=y_te_panel,
#                 method="mean"
#             )
#             mean_preds.append(mean_df.values)

#             # ----- EWMA benchmark -----
#             ewma_df = compute_benchmark_panel(
#                 y_train=y_tr_panel,
#                 y_test=y_te_panel,
#                 method="ewma2"
#             )
#             ewma_preds.append(ewma_df.values)


#         # Convert lists → arrays
#         mean_pred = np.vstack(mean_preds)        # (N_splits × n_assets)
#         ewma_pred = np.vstack(ewma_preds)        # (N_splits × n_assets)

#         # Ground truth is already stacked correctly from the GP loop
#         y_bench_true = true                       # (N_splits × n_assets)

#         # Build DataFrames
#         true_b_df = pd.DataFrame(y_bench_true, columns=asset_cols)
#         mean_df = pd.DataFrame(mean_pred, columns=asset_cols)
#         ewma_df = pd.DataFrame(ewma_pred, columns=asset_cols)
        
#         plot_path = "marketmaven/mlflow/artifacts/actual_vs_pred_matrix.png"

#         plot_actual_vs_pred_matrix(
#             true_df=true_df,          # already aligned
#             pred_df=preds_df,
#             asset_cols=asset_cols,
#             save_path=plot_path
#         )

#         mlflow.log_artifact(plot_path)

#         # Evaluate benchmarks
#         summary_mean = evaluate_asset_pricing(y_test=true_b_df, y_pred=mean_df)
#         summary_ewma = evaluate_asset_pricing(y_test=true_b_df, y_pred=ewma_df)

#         # Log metrics
#         mlflow.log_metrics({f"{k}": v for k, v in summary_gp.items()})
        
#         comparison = pd.DataFrame({
#         "GP": summary_gp,
#         "Mean": summary_mean,
#         "EWMA": summary_ewma,
#         })
#         comparison.to_html("marketmaven/mlflow/artifacts/comparison_table.html")
#         mlflow.log_artifact("marketmaven/mlflow/artifacts/comparison_table.html")
        
#         # R2_OS calculations
#         y_true_norm = true_b_df.rename(columns=lambda c: c.replace("_true", ""))
#         y_pred_norm = preds_df.rename(columns=lambda c: c.replace("_pred", ""))
#         y_ewma_norm = ewma_df.copy()
#         y_mean_norm = mean_df.copy()
        
#         compare_error = model_error_by_time_index(y_true_norm, y_pred_norm)
#         compare_error.to_html("marketmaven/mlflow/artifacts/compare_error.html")
#         mlflow.log_artifact("marketmaven/mlflow/artifacts/compare_error.html")
        
#         r2_gp_vs_mean = r2_os(y_true_norm, y_pred_norm, y_mean_norm)
#         r2_gp_vs_ewma = r2_os(y_true_norm, y_pred_norm, y_ewma_norm)
        
#         log_r2_os("gp_vs_mean", r2_gp_vs_mean)
#         log_r2_os("gp_vs_ewma", r2_gp_vs_ewma)
        
#         # Long-short performance
#         ls_gp = long_short_returns(y_true=y_true_norm, y_pred=y_pred_norm)
#         ls_mean = long_short_returns(y_true=y_true_norm, y_pred=y_mean_norm)
#         ls_ewma = long_short_returns(y_true=y_true_norm, y_pred=y_ewma_norm )
#         ls_stats_gp = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_pred_norm, label="gp")
#         ls_stats_mean = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_mean_norm, label="mean")
#         ls_stats_ewma = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_ewma_norm, label="ewma")
#         mlflow.log_metrics(ls_stats_gp)
#         mlflow.log_metrics(ls_stats_mean)
#         #mlflow.log_metrics(ls_stats_ewma)
#         plot_ls_cumulative_compare(ls_gp, ls_mean, ls_ewma)
#         return run_name
    
# results = Parallel(
#     n_jobs=2,       # or however many experiments you want in parallel
#     backend="loky",
#     verbose=10,
# )(
#     delayed(run_single_experiment)(cfg, X, I, y, splits)
#     for cfg in experiment_grid
# )


# ######Running predictions 

X, I, y, task_map = prepare_multitask_gp_data(
    df,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date", "asset_id"]
)


multiconfig = MultiTaskConfig(
    num_tasks=len(tickers.tickers),
    mean=MeanF.MULTITASK_CONSTANT,
    rank=3,
    scaling="global",
    min_noise=5e-3,
        )


        # Convert ExperimentConfig → your KernelConfig
kernel_e = KernelConfig(
    type=KernelType.MATERN,
    features=etf_cols,
    active_dims=active_dims_e,
    smoothness=0.5,
    gamma=1.2,
    q=1,
)

kernel_m = KernelConfig(
    type=KernelType.MATERN_LINEAR_RQ,
    features=macro_cols,
    active_dims=active_dims_m,
    smoothness=0.5,
    gamma=1.2,
    q=1,
)

kernel_t = KernelConfig(
    type=KernelType.PERIODIC_MATERN,
    features=[time_col],
    active_dims=active_dims_t,
    smoothness=0.5,
    gamma=1.2,
    q=1,
)


# # ---- Scaling ----
scaler = MultitaskScaler(scale_y="global", exclude_time_col=False)
scaler.fit_x(X)
X_trs = scaler.transform_x(X)
y_trs = scaler.fit_y(y, I)

# Append task index as last column
X_trs = torch.cat([X_trs, I.to(X_trs.dtype)], dim=-1)
X_trs = X_trs.to(device)
y_trs = y_trs.to(device)

# ---- Mean and Kernels ----
mean_f = initialize_mean(
    multiconfig.mean,
    num_tasks=multiconfig.num_tasks,
    input_size=X.shape[1],
)

# Periods
n_months = len(np.unique(X.numpy()[:, 0]))

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



X_t = df_forecast.copy()
X_t["__task_idx__"] = X_t['asset_id'].map(task_map)

# Drop columns not used for training
X_t = X_t.drop(columns=["date", "asset_id"], errors="ignore")
feature_df = X_t.drop(columns=["__task_idx__", "y_excess_lead"], errors="ignore")
x_np = feature_df.to_numpy()
i_np = X_t["__task_idx__"].to_numpy().reshape(-1, 1)

# Convert to tensors
full_train_x = torch.tensor(x_np, dtype=torch.float32)
full_train_i = torch.tensor(i_np, dtype=torch.float32)

X_test = scaler.transform_x(full_train_x)
X_test = torch.cat([X_test, full_train_i.to(X_test.dtype)], dim=-1)
X_test = X_test.to(torch.device("cpu"))

# ---- Predict ----
model.eval()
likelihood.eval()
with torch.no_grad():
    f_dist = model(X_test)
    pred = likelihood(f_dist, X_test)

y_hat = scaler.inverse_y(pred.mean, full_train_i)
y_std = scaler.inverse_std(pred.variance.sqrt(), full_train_i)


asset_cols = list(task_map.keys())  # task_map assumed defined elsewhere
# Reshape y_hat and y_std to match the number of assets in asset_cols
y_hat = y_hat.reshape(-1, len(asset_cols))
y_std = y_std.reshape(-1, len(asset_cols))

preds_df = pd.DataFrame(y_hat, columns=[f"{c}_pred" for c in asset_cols])
unc_df = pd.DataFrame(y_std, columns=[f"{c}_unc" for c in asset_cols])

# Step 1 — rename columns to remove the "_pred" / "_unc" suffix
preds = preds_df.rename(columns=lambda c: c.replace("_pred", ""))
unc = unc_df.rename(columns=lambda c: c.replace("_unc", ""))

# Step 2 — melt both to long format
preds_long = preds.melt(var_name="asset", value_name="prediction")
unc_long = unc.melt(var_name="asset", value_name="uncertainty")

# Step 3 — merge into one long dataframe
df_long = preds_long.merge(unc_long, on="asset")

df_long["score"] = df_long["prediction"] / df_long["uncertainty"]
df_sorted = df_long.sort_values("score", ascending=False)
df_sorted.to_csv("marketmaven/mlflow/artifacts/december_predictions_rank3.csv", index=False)
# df_sorted_r7 = df_sorted.copy()
df_sorted_r3 = df_sorted.copy()
# Task covariance

model.eval()

# Construct dummy inputs 0...num_tasks-1
I = torch.arange(model.num_tasks)

# True covariance matrix
K = model.task_covar_module(I, I).to_dense().detach().cpu()

# Convert to correlation
diag = K.diag().sqrt().clamp_min(1e-12)
corr = K / (diag.unsqueeze(1) * diag.unsqueeze(0))

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#✔ GP task correlations measure co-movement in predictive structure,
#possibly need to use a separate kernel for fixed-income assets. or add MOVE index (you already have a proxy!)

labels = list(task_map.keys())
corr_df = pd.DataFrame(corr.numpy(), index=labels, columns=labels)

plt.figure(figsize=(9, 7))
sns.heatmap(corr_df, cmap="YlGnBu", annot=False, square=True)
plt.title("MTGP Task Correlation Matrix (Correct)")
plt.tight_layout()
plt.show()
# task_cov_df.to_csv("task_covariance.csv")
# mlflow.log_artifact("task_covariance.csv")
# cov_blended = 0.7 * sample_cov + 0.3 * task_cov_df
#preds with rank 7 kernel. 
# BND_pred	BNDX_pred	EMB_pred	EWX_pred	HYG_pred	IEF_pred	IJR_pred	LQD_pred	MGK_pred	SPY_pred	TIP_pred	VEA_pred	VNQ_pred	VNQI_pred	VSS_pred	VTV_pred	VWO_pred	VWOB_pred
# 0	-0.00277	-0.00195	0.00431	0.000639	0.0065	0.00272	0.0143	0.00344	0.00115	0.006	0.00152	0.00942	0.00922	0.00733	0.00833	0.0119	0.0075	0.00304

# #More so trust: LQD, EMB, VTV, HYG, IJR, SPY, VEA, MGK, VWO, VSS
# # emb,hyg, ijr, 

#### Risk free return adjustment
# from marketmaven.market_fundamentals import fetch_rf_daily
# def compute_monthly_rf(rf_daily_cont, horizon=Horizon.MONTHLY):
#     rf_log = rf_daily_cont.resample(horizon).sum()
#     rf_simple = np.exp(rf_log) - 1
#     return rf_simple
# # 1. Get daily RF (same as used to compute excess returns)
# rf_daily_cont = fetch_rf_daily(tickers.lookback_date, tickers.end_date)

# # 2. Convert to monthly RF consistent with your excess-return horizon
# rf_monthly = compute_monthly_rf(rf_daily_cont, horizon=tickers.horizon)
# last_month_rf = float(rf_monthly[-1])

# # 4. Convert predicted excess → predicted actual returns
# df_sorted['prediction'] = df_sorted['prediction'] + last_month_rf

############## Build riskfolio  
import riskfolio as rp
from marketmaven.configs import (
    RiskfolioConfig, OptModel, RiskMeasure, Objective, MuEstimator, CovEstimator)
from IPython.display import display
risk_config = RiskfolioConfig(
    model=OptModel.CLASSIC,
    rm=RiskMeasure.CVaR,
    obj=Objective.SHARPE,
    method_mu=MuEstimator.EWMA2,
    method_cov=CovEstimator.GERBER2,
    nea=8

)

#shrink list of etfs 
final_pivoted_returns = pivoted_returns.drop(columns=['MGK','TIP','BNDX','LQD', 'EMB'])

# Building the portfolio object
port = rp.Portfolio(returns=final_pivoted_returns)
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.
port.assets_stats(method_mu=risk_config.method_mu, method_cov=risk_config.method_cov)

model_risk=risk_config.model # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = risk_config.rm # Risk measure used, this time will be variance
obj = risk_config.obj# Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

port.card = None 
port.nea = risk_config.nea
w = port.optimization(model=model_risk, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

display(w.T)
label = str(risk_config.rm) + " " + str(risk_config.obj) + " Adjusted Basic Portfolio" # Title of point
# Plotting the composition of the portfolio

ax = rp.plot_pie(w=w, title=label, others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)

mu = port.mu # Expected returns
cov = port.cov # Covariance matrix
returns = port.returns # Returns of the assets
points = 50 # Number of points of the frontier
pd.options.display.float_format = '{:.4%}'.format
frontier = port.efficient_frontier(model=model_risk, rm=rm, points=points, rf=rf, hist=hist)

display(frontier.T.head())

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)

ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)

"""
For subsituting returns I should not use these returns from rank7 and just use mean:
BND, BNDX, EMB, LQD, TIP, VNQI, VWOB


From Rank 3 use these instead of rank 7:
EWX, IEF, VEA, VNQ, VSS, VWO

"""


rank3_etfs = ['EWX','IEF','VEA','VNQ','VSS','VWO']
mean_etfs = ['BND','BNDX','VNQI','VWOB'] # 'LQD','TIP''EMB',
rank7_etfs =  ['HYG', 'VTV', 'IJR', 'SPY', ] #'MGK'
df_rank3 = df_sorted_r3[df_sorted_r3['asset'].isin(['EWX', 'IEF', 'VEA', 'VNQ', 'VSS', 'VWO'])]

before_impute = port.mu.copy()
before_impute[['EWX']] = float(df_rank3[df_rank3['asset']=='EWX']['prediction'])
before_impute[['IEF']] = float(df_rank3[df_rank3['asset']=='IEF']['prediction'])
before_impute[['VEA']] = float(df_rank3[df_rank3['asset']=='VEA']['prediction'])
before_impute[['VNQ']] = float(df_rank3[df_rank3['asset']=='VNQ']['prediction'])
before_impute[['VSS']] = float(df_rank3[df_rank3['asset']=='VSS']['prediction'])
before_impute[['VWO']] = float(df_rank3[df_rank3['asset']=='VWO']['prediction'])

#now from rank 7 
before_impute[['HYG']] = float(df_sorted_r7[df_sorted_r7['asset']=='HYG']['prediction'])
before_impute[['VTV']] = float(df_sorted_r7[df_sorted_r7['asset']=='VTV']['prediction'])
before_impute[['IJR']] = float(df_sorted_r7[df_sorted_r7['asset']=='IJR']['prediction'])
before_impute[['SPY']] = float(df_sorted_r7[df_sorted_r7['asset']=='SPY']['prediction'])

port.mu  = before_impute
port.card = None 
port.nea = risk_config.nea

w = port.optimization(model=model_risk, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

display(w.T)
label = str(risk_config.rm) + " " + str(risk_config.obj) + " Adjusted ML Portfolio" # Title of point
# Plotting the composition of the portfolio

ax = rp.plot_pie(w=w, title=label, others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)

# Frontier 
mu = port.mu # Expected returns
cov = port.cov # Covariance matrix
returns = port.returns # Returns of the assets
points = 50 # Number of points of the frontier
pd.options.display.float_format = '{:.4%}'.format
frontier = port.efficient_frontier(model=model_risk, rm=rm, points=points, rf=rf, hist=hist)


display(frontier.T.head())

label = str(risk_config.rm) + " " + str(risk_config.obj) + "MTGP Portfolio" # Title of point
ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)

# Frontier area 
ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)

#All portfolios 

rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
       'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']

w_s = pd.DataFrame([])

for i in rms:
    w = port.optimization(model=model_risk, rm=i, obj=obj, rf=rf, l=l, hist=hist)
    w_s = pd.concat([w_s, w], axis=1)
w_s.columns = rms
w_s.style.format("{:.2%}").background_gradient(cmap='YlGn')

import matplotlib.pyplot as plt

# Plotting a comparison of assets weights for each portfolio

fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)

w_s.plot.bar(ax=ax)


# Pivoting from wide to long
weights_long = pd.melt(w.T, var_name='asset', value_name='weight')

print("Wide DataFrame:")
print(w.T)
print("\nLong DataFrame:")
weights_long