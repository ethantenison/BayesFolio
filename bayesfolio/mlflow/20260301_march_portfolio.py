
import mlflow
import pandas as pd
import warnings
import os
from pathlib import Path
from joblib import Parallel, delayed
from pydantic import BaseModel
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from bayesfolio.schemas.configs.core import TickerConfig, Interval, Horizon, CVConfig
from bayesfolio.data.asset_prices import build_long_panel
import numpy as np
import torch
from bayesfolio.schemas.configs.core import (
    RiskfolioConfig, OptModel, RiskMeasure, Objective, MuEstimator, CovEstimator)
from bayesfolio.visualization.eda import correlation_matrix
from bayesfolio.data.gp_data_prep import prepare_multitask_gp_data
from bayesfolio.engine.models.cv import rolling_time_splits_multitask
from bayesfolio.engine.models.scaling import MultitaskScaler
device = torch.device("cpu")
from bayesfolio.engine.models.gp.multitask import train_model_hadamard
from math import log, sqrt
from bayesfolio.optimization.evaluate import evaluate_asset_pricing
from bayesfolio.utils import check_equal_occurrences
from bayesfolio.visualization.evaluation import plot_ls_cumulative_compare, plot_actual_vs_pred_matrix
from bayesfolio.visualization.variable_importance import xgboost_variable_importance
from bayesfolio.portfolio.helpers import assessing_long_short_performance, long_short_returns,long_short_returns_topk, assess_performance
from bayesfolio.models.old_kernels import KernelType, initialize_kernel, adaptive_lengthscale_prior, KernelConfig
from bayesfolio.engine.models.gp.means import MeanF, initialize_mean
from bayesfolio.mlflow.helpers import (
    MultiTaskConfig, long_to_panel, compute_benchmark_panel, r2_os, log_r2_os,
    model_error_by_time_index, log_kernel_to_mlflow, log_gpytorch_state_dict, log_gp_hyperparameters
)
import riskfolio as rp
import plotly.express as px
import random
import itertools
from bayesfolio.data.market_fundamentals import fetch_enhanced_macro_features
from bayesfolio.data.asset_prices import fetch_etf_features, add_cross_sectional_momentum_rank, cross_sectional_zscore
from gpytorch.kernels import SpectralMixtureKernel
warnings.filterwarnings(
    "ignore",
    message=".*torch.sparse.SparseTensor.*is deprecated.*"
)
from bayesfolio.schemas.configs.core import (
    RiskfolioConfig, OptModel, RiskMeasure, Objective, MuEstimator, CovEstimator)
from IPython.display import display
pd.set_option('display.max_rows', 20)
warnings.filterwarnings("ignore", category=Warning, message=".*not p.d., added jitter.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*not p.d., added jitter.*")
# # MLFlow Configuration
# OLD mlflow ui --backend-store-uri ./mlruns
#mlflow ui --backend-store-uri sqlite:////Users/et/Documents/BayesFolio/mlflow.db
# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "March Portfolio Experiments"

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parents[1]
TRACKING_DB = WORKSPACE_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"
LOCAL_ARTIFACT_DIR = BASE_DIR / "artifacts"
LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# # Set up MLFlow tracking
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)




warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3}'.format

############### Experiment Configuration ###############

description = "tracking experiments for march portfolio assets"

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
    "IJR", # US small cap S&P index 600, more stable
    "IWM", # US small cap Russel index, more volile than IJR
    "VNQ", # REIT ETF US centric
    "VNQI", # international REIT ETF
    "VEA", # developed international equity
    "VWO", # AVEM actually is better than VWO but not enough history
    "VSS", # forein small/mid cap
    "BND", # total bond market ETF US centric
    # "IBND", # international corporate bond market ETF USD hedged
    # "ISHG", # international high yield bond ETF USD hedged
    "IEF", # 7-10 year treasury bond ETF US centric
    "BNDX", # total international bond market ETF, USD hedged, but actually developed markets only
    "LQD", # investment grade bond ETF US centric
    "HYG", # High yield bond ETF US centric 
    #"TIP", # Treasury inflation protected securities ETF US centric
    "EWX", # emerging market small cap ETF
    "VWOB", # Emerging Market Goverment bond 
    #"EMB", # emerging market bond ETF USD hedged
    "HYEM", # emerging market high yield corporate bond ETF USD hedged
    
]

    # # "IBND", # international corporate bond market ETF unhedged
    # # "ISHG", # international high yield bond ETF unhedged
    #"PDBC", # Commodities ETF
    #"BIL", # 1-3 month us treasuries
    # "AVEM",
    # "ISCF",
    #"HYEM", # emerging market high yield corporate bond ETF USD hedged
    
assets_to_drop = [
    'IBND', 'ISHG', 'PDBC', 'BIL', 'EMB', 'TIP',
#"BNDX", "IEF", "LQD", "HYEM", "BND", "HYG", "VWOB", "EWX"
    ]

#remove assets_to_drop from etf_tickers
etf_tickers = [ticker for ticker in etf_tickers if ticker not in assets_to_drop]

tickers = TickerConfig(
    start_date="2016-11-29",
    end_date="2026-02-28",
    interval=Interval.DAILY,
    tickers=etf_tickers,
    horizon=Horizon.MONTHLY,
    lookback_date="2014-07-01"
)

############### Returns data ###############
# return_data = build_long_panel(tickers.tickers, tickers.lookback_date, tickers.end_date, horizon=tickers.horizon)
# return_data.to_csv("20260228_etf_returns.csv", index=False)
return_data = pd.read_csv("20260228_etf_returns.csv")  # Updated to match the new filename
return_data = return_data[~return_data['asset_id'].isin(assets_to_drop)]


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
# macro_features = fetch_enhanced_macro_features(start=tickers.lookback_date, end=tickers.end_date)
# macro_features.to_csv("20260228_macro_features.csv", index=False)
macro_features = pd.read_csv("20260228_macro_features.csv")
macro_features = macro_features.drop(columns=['vix_ts_level','vix3m','yc_pc1', 'yc_pc2', 'yc_pc3', 'y10_nominal' ])
macro_cols = macro_features.columns[1:].tolist()

# etf_features = fetch_etf_features(tickers.tickers, tickers.lookback_date, tickers.end_date, tickers.horizon)
# etf_features.to_csv("20260228_etf_features.csv", index=False)
etf_features = pd.read_csv("20260228_etf_features.csv")
etf_features = etf_features[~etf_features['asset_id'].isin(assets_to_drop)]
etf_features = etf_features.drop(columns=['ma_1m','ma_3m','vol_1w', 'price', 'overnight_gap'])


etf_features["ill_adj"] = etf_features["ill"] * 1e9
etf_features["ill_log"] = np.log(etf_features["ill_adj"] + 1)
etf_features["dolvol_adj"] = etf_features["dolvol"] * 1e9
etf_features["dolvol_log"] = np.log(etf_features["dolvol_adj"] + 1)

# optional: very mild global clipping before scaler
etf_features["ill_log"] = etf_features["ill_log"].clip(upper=etf_features["ill_log"].quantile(0.99))
etf_features["dolvol_log"] = etf_features["dolvol_log"].clip(upper=etf_features["dolvol_log"].quantile(0.99))
etf_features = etf_features.drop(columns=['ill', 'ill_adj', 'dolvol', 'dolvol_adj'])
etf_features = add_cross_sectional_momentum_rank(etf_features, momentum_col='mom12m')
etf_features = etf_features.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)


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

df.to_csv("20260228_return_data.csv", index=False)

##### This is where I got tripped up because you have to manually move the dates. 
df= df[df['date'] > str("2016-11-28")]
df = df[df['date'] < str("2026-02-01")]
df = df.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)


############### Feature Selection ###############
etf_cols = [
 'lag_y_excess_lead',
 'baspread',
 'ret_kurt',
 'chmom',
 'mom12m',
 'mom36m',
 'cs_mom_rank',
 'max_dd_6m',
 'ma_signal',
 'ret_autocorr',
 'vol_z']


macro_cols = ['hy_spread',
               'hy_spread_chg_1m',
                'hy_spread_z_12m',
 'vix_slope',
 'vix_ts_z_12m',
 'vix',
 'spy_flow_z_12m',
 'spy_ret',
 'erp',
 'cpi_yoy',
 'cpi_mom',
 'copper_ret',
 'oil_ret',
 'gold_crude_ratio',
 'pct_above_50dma', 
 'em_fx_ret']


PROTECTED_COLS = [
    "date",
    "asset_id",
    "y_excess_lead",
]

FEATURE_COLS = etf_cols + macro_cols

KEEP_COLS = PROTECTED_COLS + FEATURE_COLS
df = df.loc[:, KEEP_COLS].copy()

df['date'] = pd.to_datetime(df['date'])
time_col = "date"
y_col = "y_excess_lead" #"y_excess_lead"
asset_col = "asset_id"


all_assets_occur = check_equal_occurrences(df, 'asset_id')
print('Did all assets occur equally often?', all_assets_occur)
droppin_cols = ['date', 'asset_id']
y_col = ['y_excess_lead']
col_order  = droppin_cols + FEATURE_COLS + y_col
df = df[col_order]
df['t_index'] = pd.factorize(df['date'])[0]
cols = df.columns.tolist()       
cols.insert(0, cols.pop(cols.index("t_index")))
df = df.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)

feature_cols = ['t_index'] + FEATURE_COLS



############### Preliminary XGBoost Variable Importance ###############
# xgb_df = df.dropna().reset_index(drop=True)
# X = xgb_df.drop(columns=['y_excess_lead', "date"]) # , "asset_id"
# y = xgb_df[['y_excess_lead']]
# X['asset_id'] = X['asset_id'].astype("category")

# global_importance, interaction_fig = xgboost_variable_importance(X, y)
# print("Global Importance:\n", global_importance)

# interaction_fig.show()
# px.bar(pd.DataFrame(global_importance).sort_values(by=0))


############################### Game day dataset ##################################


# === Build index lists ===

# Time kernel active dims
active_dims_t = [feature_cols.index('t_index')]

# ETF features active dims
active_dims_e = [feature_cols.index(col) for col in etf_cols]

# Macro features active dims
active_dims_m = [feature_cols.index(col) for col in macro_cols]

######## Visualize Factors ########
# df = df[cols].copy()
# #df.to_csv("marketmaven/datasets/20260228_tasks.csv", index=False)
# df_forecast = df[df['date'] > pd.Timestamp("2025-12-01")]
# df_forecast = df_forecast.reset_index(drop=True)
# df = df.dropna(subset=['y_excess_lead']).reset_index(drop=True)

# fig = correlation_matrix(df)
# fig.show()

# df.iloc[:, 2:].hist(bins=30, figsize=(20, 15))

############### Data Preparation ###############

cv_config = CVConfig(
    step=1,
    horizon_cv=1,
    embargo=0,
    training_min=104,
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

#take the test splits and grab the actual dates for each test split
test_dates_splits = []
for tr_idx, te_idx in splits:
    test_dates = df.iloc[te_idx]['date'].unique().tolist()
    test_dates_splits.append(test_dates)

risk_config = RiskfolioConfig(
    model=OptModel.CLASSIC,
    rm=RiskMeasure.CVaR,
    obj=Objective.SHARPE,
    method_mu=MuEstimator.EWMA2,
    method_cov=CovEstimator.GERBER2,
    nea=10

)

############### Model Setup ###############
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)


# ard_num_dims = len(feature_cols)
# active_dims_features = list(range(1, ard_num_dims + 1))

train_idx_1_ahead = list(range(X.shape[0] - len(tickers.tickers)))
test_idx_1_ahead = list(range(X.shape[0] - len(tickers.tickers), X.shape[0]))


def create_kernel_initialization(kernel: KernelConfig, n_months: int):
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
        period_length=period_length,
        n_mixtures=kernel.n_mixtures,
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
    mean_f: MeanF | dict
   

#KERNEL GRID SEARCH
ETF_KERNEL_GRID = { 
    KernelType.MATERN:        [0.5],
}


MACRO_KERNEL_GRID = {
    KernelType.MATERN_LINEAR_RQ: [0.5],
}


# Time kernels
TIME_KERNEL_GRID = {
   KernelType.MATERN: [0.5],   # Matern 3/2 good for SPY, IJR, MGK, VTV
   #KernelType.MATERN_LINEAR: [2.5],   # Matern 3/2 good for SPY, IJR, MGK, VTV
   #KernelType.PERIODIC_MATERN: [2.5],   # Matern 5/2 + periodic kinda good for SPY, IJR, MGK, VTV
    #KernelType.EXPONENTIAL_DECAY_MATERN: [1.5, 2.5],   
    # KernelType.EXPONENTIAL_DECAY: [None],

}


RANK_GRID = [5]
MEAN_F_GRID = [MeanF.MULTITASK_ZERO] #etf_mean_spec MeanF.MULTITASK_CONSTANT, 



# Best before leaving out extra : etf=matern_s=0.5_macro=maternlinearrq_s=0.5_time=periodicmatern_s=0.5
# periodic smoothness doesn't so much matter, macro smoothness of 0.5 seems better. 

def build_experiment_grid():
    experiment_list = []

    for etf_k, etf_s_list in ETF_KERNEL_GRID.items():
        for macro_k, macro_s_list in MACRO_KERNEL_GRID.items():
            for time_k, time_s_list in TIME_KERNEL_GRID.items():
                for rank in RANK_GRID:
                    for mean_f in MEAN_F_GRID:        
                        for etf_s, macro_s, time_s in itertools.product(
                            etf_s_list, macro_s_list, time_s_list
                        ):
                            cfg = ExperimentConfig(
                                etf=KernelSpec(type=etf_k, smoothness=etf_s),
                                macro=KernelSpec(type=macro_k, smoothness=macro_s),
                                time=KernelSpec(type=time_k, smoothness=time_s),
                                rank=rank,
                                mean_f=mean_f,         
                            )
                            experiment_list.append(cfg)

    return experiment_list
    
experiment_grid = build_experiment_grid()
############### Run ###############
seed = 27

if mlflow.active_run() is not None:
    mlflow.end_run()

for cfg in experiment_grid:
    run_name = (
        f"e={cfg.etf.type.value}_{cfg.etf.smoothness}_"  
        f"m={cfg.macro.type.value}_{cfg.macro.smoothness}_" 
        f"t={cfg.time.type.value}_{cfg.time.smoothness}_"
        f"r={cfg.rank}"
    )

    with mlflow.start_run(run_name=run_name, description="""Finding the best final model for gameday with per task scaling.""") as run:
        multiconfig = MultiTaskConfig(
            num_tasks=len(tickers.tickers),
            mean=cfg.mean_f,
            rank=cfg.rank,
            scaling="per_task",   
            min_noise=5e-3,
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
            n_mixtures=1,
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
            scale_y: str = "per_task",
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
                mean= multiconfig.mean,
                num_tasks=multiconfig.num_tasks,
                input_size=X.shape[1],
                macro_dims=active_dims_m,
                etf_mean_spec=None,
                task_map=None,
            )
            
            # Periods
            n_months = len(np.unique(X_tr.numpy()[:, 0]))
            
            # Kernels
            kernele = create_kernel_initialization(kernel_e, n_months)
            kernelm = create_kernel_initialization(kernel_m, n_months)
            kernelt = create_kernel_initialization(kernel_t, n_months)


            kernel_total = kernele + kernelm +  (kernelm + kernele) * kernelt + kernele*kernelm # kernelt +

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
                patience=50,
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
        log_gp_hyperparameters(model_list[-1], "gp_hyperparameters.json")

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
        ci_path = LOCAL_ARTIFACT_DIR / "confidence_intervals.csv"
        ci_df.to_csv(ci_path, index=False)
        mlflow.log_artifact(str(ci_path))
                    
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
        
        plot_path = LOCAL_ARTIFACT_DIR / "actual_vs_pred_matrix.png"

        #fix the index to be the dates from the test splits
        date_test_splits = np.array(test_dates_splits)
        date_test_splits = pd.Series(date_test_splits.reshape(-1, ))

        plot_actual_true = true_df.copy()
        plot_pred_df = preds_df.copy()
        plot_actual_true.index = date_test_splits
        plot_pred_df.index = date_test_splits
        
        plot_actual_true.index = pd.to_datetime(plot_actual_true.index)
        plot_pred_df.index = pd.to_datetime(plot_pred_df.index)

        plot_actual_vs_pred_matrix(
            true_df=plot_actual_true,          
            pred_df=plot_pred_df,
            asset_cols=asset_cols,
            save_path=plot_path
        )

        mlflow.log_artifact(str(plot_path))

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
        comparison_path = LOCAL_ARTIFACT_DIR / "comparison_table.html"
        comparison.to_html(comparison_path)
        mlflow.log_artifact(str(comparison_path))
        
        # R2_OS calculations
        y_true_norm = true_b_df.rename(columns=lambda c: c.replace("_true", ""))
        y_pred_norm = preds_df.rename(columns=lambda c: c.replace("_pred", ""))
        y_ewma_norm = ewma_df.copy()
        y_mean_norm = mean_df.copy()
        

        # Save global_importance as a CSV file
        # importance_path = "marketmaven/mlflow/artifacts/global_importance.csv"
        # global_importance.to_csv(importance_path, header=True)
        # mlflow.log_artifact(importance_path)

        # # Optionally, log the global_importance as a dictionary for quick inspection
        # mlflow.log_dict(global_importance.to_dict(), "global_importance.json")
        
        compare_error = model_error_by_time_index(y_true_norm, y_pred_norm)
        compare_error_path = LOCAL_ARTIFACT_DIR / "compare_error.html"
        compare_error.to_html(compare_error_path)
        mlflow.log_artifact(str(compare_error_path))
        
        r2_gp_vs_mean = r2_os(y_true_norm, y_pred_norm, y_mean_norm)
        r2_gp_vs_ewma = r2_os(y_true_norm, y_pred_norm, y_ewma_norm)
        
        log_r2_os("gp_vs_mean", r2_gp_vs_mean)
        
        # Long-short performance
        ls_gp = long_short_returns(y_true=y_true_norm, y_pred=y_pred_norm)
        ls_mean = long_short_returns(y_true=y_true_norm, y_pred=y_mean_norm)
        ls_ewma = long_short_returns(y_true=y_true_norm, y_pred=y_ewma_norm )
        ls_stats_gp = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_pred_norm, label="gp")
        ls_stats_mean = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_mean_norm, label="mean")
        ls_stats_ewma = assessing_long_short_performance(y_true=y_true_norm, y_pred=y_ewma_norm, label="ewma")
        plot_ls_cumulative_compare(ls_gp, ls_mean, ls_ewma, strategy="ls")
        mlflow.log_metrics(ls_stats_gp)
                
        # Top K botom k 
        # Long-short performance
        ls_gpk = long_short_returns_topk(y_true=y_true_norm, y_pred=y_pred_norm, q=0.2)
        ls_meank = long_short_returns_topk(y_true=y_true_norm, y_pred=y_mean_norm, q=0.2)
        ls_ewmak = long_short_returns_topk(y_true=y_true_norm, y_pred=y_ewma_norm, q=0.2 )
        ls_stats_gpk = assess_performance(ls_gpk, label="gp_topk")
        ls_stats_meank = assess_performance(ls_meank, label="mean_topk")
        ls_stats_ewmak = assess_performance(ls_ewmak, label="ewma_topk")
        mlflow.log_metrics(ls_stats_gpk)
        plot_ls_cumulative_compare(ls_gpk, ls_meank, ls_ewmak, strategy="ls_topk")


################################################### Get the predictions. 

full_data = pd.read_csv("20260228_return_data.csv")
full_data = full_data.loc[:, KEEP_COLS].copy()
full_data = full_data.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)
full_data['date'] = pd.to_datetime(full_data['date'])
full_data = full_data[col_order]
full_data['t_index'] = pd.factorize(full_data['date'])[0]
cols = full_data.columns.tolist()       
cols.insert(0, cols.pop(cols.index("t_index")))
full_data = full_data[cols]
full_data = full_data.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)
full_data = full_data[full_data['date'] > pd.Timestamp("2016-11-28")]

full_train_data = full_data[full_data['date'] < pd.Timestamp("2026-02-27")].copy()
full_test_data = full_data[full_data['date'] >= pd.Timestamp("2026-02-27")].copy()

   
X, I, y, task_map = prepare_multitask_gp_data(
    full_train_data,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date", "asset_id"],
    dtype=torch.float32
)


cfg = experiment_grid[0]

# Kernel 
multiconfig = MultiTaskConfig(
    num_tasks=len(tickers.tickers),
    mean=cfg.mean_f,
    rank=cfg.rank,
    scaling="per_task",   
    min_noise=5e-3,
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
    n_mixtures=1,
)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.use_deterministic_algorithms(True)


# # ---- Scaling ----
scaler = MultitaskScaler(scale_y="per_task", exclude_time_col=False)
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
kernel_total = kernele + kernelm + kernelt + (kernelm + kernele) * kernelt

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
    patience=50,
)

model_str = repr(model)

######## Forecasting ###################
full_test_data = full_test_data.reset_index(drop=True)
X_t = full_test_data.copy()
# move t
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


# Saving 
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
df_sorted.to_csv("20260228_gameday_predictions.csv", index=False)



####plotting

##### Task Rank Matrix #####

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


########## TBDDDDDD Build Riskfolio ############


########## Build Riskfolio ############
risk_config = RiskfolioConfig(
    model=OptModel.CLASSIC,
    rm=RiskMeasure.CVaR,
    obj=Objective.SHARPE,
    method_mu=MuEstimator.EWMA2,
    method_cov=CovEstimator.GERBER2,
    nea=10

)

###################################################
# BASIC HISTORICAL PORTFOLIO  vs  GP-POSTERIOR (1M) CVaR PORTFOLIO
# - Train MT Hadamard GP on full universe (including MGK/BND/BNDX)
# - Forecast next-month distribution for ONE rebalance date
# - Sample posterior scenarios and optimize CVaR on those scenarios
# - Exclude MGK/BND/BNDX ONLY at portfolio construction time
###################################################

import numpy as np
import pandas as pd
import torch
import random
import riskfolio as rp
from IPython.display import display

EXCLUDE = {"MGK", "BND", "BNDX"}
device = torch.device("cpu")

# -----------------------------
# 0) Load return history for "basic historical portfolio"
# -----------------------------
return_data = pd.read_csv("20260228_etf_returns.csv")
return_data["date"] = pd.to_datetime(return_data["date"])

pivoted_returns = (
    return_data.pivot(index="date", columns="asset_id", values="y_excess_lead")
    .dropna()
    .sort_index()
)

# exclude assets from portfolio universe (but not necessarily from GP)
final_pivoted_returns = pivoted_returns.drop(columns=[c for c in EXCLUDE if c in pivoted_returns.columns])

# -----------------------------
# 1) BASIC HISTORICAL PORTFOLIO (unchanged idea, but no percent-strings)
# -----------------------------
risk_config = RiskfolioConfig(
    model=OptModel.CLASSIC,
    rm=RiskMeasure.CVaR,
    obj=Objective.SHARPE,
    method_mu=MuEstimator.EWMA2,
    method_cov=CovEstimator.GERBER2,
    nea=10
)

port_hist = rp.Portfolio(returns=final_pivoted_returns)

# use your chosen estimators
port_hist.assets_stats(method_mu=risk_config.method_mu, method_cov=risk_config.method_cov)

model_risk = risk_config.model
rm = risk_config.rm
obj = risk_config.obj
hist = True
rf = 0
l = 0

port_hist.card = None
port_hist.nea = risk_config.nea
port_hist.alpha = 0.20

w_hist = port_hist.optimization(model=model_risk, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

display(w_hist.T)
ax = rp.plot_pie(
    w=w_hist,
    title=f"{rm} {obj} — Basic Historical Portfolio (excl. MGK/BND/BNDX)",
    others=0.05,
    nrow=25,
    cmap="tab20",
    height=6,
    width=10,
    ax=None
)

# (Optional) efficient frontier for basic historical portfolio
mu_hist = port_hist.mu
cov_hist = port_hist.cov
returns_hist = port_hist.returns
frontier_hist = port_hist.efficient_frontier(model=model_risk, rm=rm, points=50, rf=rf, hist=hist)

display(frontier_hist.T.head())
ax = rp.plot_frontier(
    w_frontier=frontier_hist,
    mu=mu_hist,
    cov=cov_hist,
    returns=returns_hist,
    rm=rm,
    rf=rf,
    alpha=port_hist.alpha,
    cmap="viridis",
    w=w_hist,
    label="Historical",
    marker="*",
    s=16,
    c="r",
    height=6,
    width=10,
    ax=None
)

# -----------------------------
# 2) TRAIN MTGP ON full_train_data (includes MGK/BND/BNDX)
# -----------------------------
full_data = pd.read_csv("20260228_return_data.csv")
full_data = full_data.loc[:, KEEP_COLS].copy()
full_data = full_data.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)
full_data["date"] = pd.to_datetime(full_data["date"])

full_data = full_data[col_order]
full_data["t_index"] = pd.factorize(full_data["date"])[0]
cols = full_data.columns.tolist()
cols.insert(0, cols.pop(cols.index("t_index")))
full_data = full_data[cols]
full_data = full_data.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)
full_data = full_data[full_data["date"] > pd.Timestamp("2016-11-28")].copy()

cutoff = pd.Timestamp("2026-02-27")
full_train_data = full_data[full_data["date"] < cutoff].copy()
full_test_data  = full_data[full_data["date"] >= cutoff].copy()

X, I, y, task_map = prepare_multitask_gp_data(
    full_train_data,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date", "asset_id"],
    dtype=torch.float32
)

# deterministic
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.use_deterministic_algorithms(True)

cfg = experiment_grid[0]

multiconfig = MultiTaskConfig(
    num_tasks=len(task_map),
    mean=cfg.mean_f,
    rank=cfg.rank,
    scaling="per_task",
    min_noise=5e-3
)

kernel_e = KernelConfig(
    type=cfg.etf.type,
    features=etf_cols,
    active_dims=active_dims_e,
    smoothness=cfg.etf.smoothness or 1.5,
    gamma=1.2,
    q=1
)

kernel_m = KernelConfig(
    type=cfg.macro.type,
    features=macro_cols,
    active_dims=active_dims_m,
    smoothness=cfg.macro.smoothness or 1.5,
    gamma=1.2,
    q=1
)

kernel_t = KernelConfig(
    type=cfg.time.type,
    features=[time_col],
    active_dims=active_dims_t,
    smoothness=cfg.time.smoothness or 1.5,
    gamma=1.2,
    q=1,
    n_mixtures=1
)

# scaling
scaler = MultitaskScaler(scale_y="per_task", exclude_time_col=False)
scaler.fit_x(X)
X_trs = scaler.transform_x(X)
y_trs = scaler.fit_y(y, I)

# append task index
X_trs = torch.cat([X_trs, I.to(X_trs.dtype)], dim=-1).to(device)
y_trs = y_trs.to(device)

mean_f = initialize_mean(
    multiconfig.mean,
    num_tasks=multiconfig.num_tasks,
    input_size=X.shape[1]
)

n_months = len(np.unique(X.numpy()[:, 0]))
kernele = create_kernel_initialization(kernel_e, n_months)
kernelm = create_kernel_initialization(kernel_m, n_months)
kernelt = create_kernel_initialization(kernel_t, n_months)
kernel_total = kernele + kernelm + kernelt + (kernelm + kernele) * kernelt

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
    patience=50
)

# -----------------------------
# 3) FORECAST ONE REBALANCE DATE -> FULL posterior mean/cov -> sample scenarios
# -----------------------------
full_test_data = full_test_data.reset_index(drop=True)

rebalance_date = full_test_data["date"].min()
X_t = full_test_data[full_test_data["date"] == rebalance_date].copy()

# keep only tasks GP knows about
X_t = X_t[X_t["asset_id"].isin(task_map.keys())].copy()
X_t["task_idx"] = X_t["asset_id"].map(task_map).astype(int)

if X_t["task_idx"].duplicated().any():
    raise ValueError(
        "Forecast batch has duplicate assets for the same date. "
        "Ensure one row per asset at rebalance_date."
    )

# sort by task index to align mean/cov with tickers
X_t = X_t.sort_values("task_idx").reset_index(drop=True)

asset_order_all = X_t["asset_id"].tolist()
task_ids_all    = X_t["task_idx"].to_numpy().tolist()

# build features tensor
X_feat = X_t.drop(columns=["date", "asset_id", "y_excess_lead", "task_idx"], errors="ignore")
X_raw  = torch.tensor(X_feat.to_numpy(), dtype=torch.float32)
I_raw  = torch.tensor(np.array(task_ids_all).reshape(-1, 1), dtype=torch.float32)

# scale x + append task index
X_scaled = scaler.transform_x(X_raw)
X_scaled = torch.cat([X_scaled, I_raw.to(X_scaled.dtype)], dim=-1).to(device)

model.eval()
likelihood.eval()
with torch.no_grad():
    f_dist = model(X_scaled)
    pred = likelihood(f_dist, X_scaled)

m_scaled = pred.mean.detach().cpu()
S_scaled = pred.covariance_matrix.detach().cpu()

# per-asset scaling vectors (supports none/global/per_task)
task_ids_long = [int(t) for t in task_ids_all]
if scaler.scale_y == "none":
    mu_vec = torch.zeros(len(task_ids_long), dtype=m_scaled.dtype)
    sd_vec = torch.ones(len(task_ids_long), dtype=m_scaled.dtype)
elif scaler.scale_y == "global":
    mu_vec = torch.full((len(task_ids_long),), float(scaler.y_mean), dtype=m_scaled.dtype)
    sd_vec = torch.full((len(task_ids_long),), float(scaler.y_std), dtype=m_scaled.dtype)
elif scaler.scale_y == "per_task":
    mu_vec = torch.tensor([float(scaler.y_mean_k.get(t, scaler.global_mu)) for t in task_ids_long], dtype=m_scaled.dtype)
    sd_vec = torch.tensor([float(scaler.y_std_k.get(t, scaler.global_sd)) for t in task_ids_long], dtype=m_scaled.dtype)
else:
    raise ValueError(f"Unknown scaler.scale_y: {scaler.scale_y}")

m = m_scaled * sd_vec + mu_vec
D = torch.diag(sd_vec)
S = D @ S_scaled @ D
S = S + 1e-8 * torch.eye(S.shape[0], dtype=S.dtype)

# exclude portfolio assets after GP prediction
keep_idx = np.array([i for i, a in enumerate(asset_order_all) if a not in EXCLUDE], dtype=int)
asset_order = [asset_order_all[i] for i in keep_idx]

m_sub = m[keep_idx]
S_sub = S[np.ix_(keep_idx, keep_idx)]

# sample posterior scenarios
n_scen = 5000
mv = torch.distributions.MultivariateNormal(m_sub, covariance_matrix=S_sub)
scen = mv.sample((n_scen,)).numpy()

scenario_returns = pd.DataFrame(scen, columns=asset_order)
scenario_returns.to_csv("20260228_gp_posterior_scenarios.csv", index=False)

# -----------------------------
# 4) RISKFOLIO ON GP POSTERIOR SCENARIOS (CVaR uses uncertainty via scenarios)
# -----------------------------
port_gp = rp.Portfolio(returns=scenario_returns)
port_gp.assets_stats(method_mu="hist", method_cov="hist")

port_gp.alpha = 0.20
port_gp.nea = risk_config.nea
port_gp.card = None

w_gp = port_gp.optimization(
    model="Classic",
    rm="CVaR",
    obj="Sharpe",
    rf=0,
    l=0,
    hist=True
)

display(w_gp.T)
ax = rp.plot_pie(
    w=w_gp,
    title=f"{rm} {obj} — GP Posterior Scenarios (1M) (excl. MGK/BND/BNDX)",
    others=0.05,
    nrow=25,
    cmap="tab20",
    height=6,
    width=10,
    ax=None
)

# -----------------------------
# 5) Side-by-side comparison table
# -----------------------------
w_compare = pd.concat(
    [
        w_hist.rename(columns={"weights": "hist_weights"}),
        w_gp.rename(columns={"weights": "gp_weights"})
    ],
    axis=1
).fillna(0.0)

display(w_compare.T)

# Optional: save
w_compare.to_csv("20260228_weights_hist_vs_gp.csv")


################### Determining if the rank is good enough. 
K = model.task_covar_module(I, I).to_dense().detach().cpu()
d = torch.sqrt(torch.diag(K)).clamp_min(1e-12)
Corr = K / (d[:, None] * d[None, :])

eigvals = torch.linalg.eigvalsh(Corr).flip(0)
cum = torch.cumsum(eigvals, dim=0) / eigvals.sum()  # sum is N=18

print(eigvals[:10])
print(cum[:10])

#### determining if per task scaling woudl help
y_panel = full_train_data.pivot(index="date", columns="asset_id", values="y_excess_lead").dropna()
stds = y_panel.std()
ratio = stds.max() / stds.min()
print(ratio, stds.sort_values()) 