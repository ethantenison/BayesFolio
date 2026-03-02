
from re import M
import mlflow
import pandas as pd
import warnings
import os
from joblib import Parallel, delayed
from pydantic import BaseModel
from regex import E
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from bayesfolio.schemas.configs.core import TickerConfig, Interval, Horizon, CVConfig
from bayesfolio.features.asset_prices import build_long_panel
import numpy as np
import torch
from bayesfolio.schemas.configs.core import (
    RiskfolioConfig, OptModel, RiskMeasure, Objective, MuEstimator, CovEstimator)
from bayesfolio.visualization.eda import correlation_matrix
from bayesfolio.features.gp_data_prep import prepare_multitask_gp_data
from bayesfolio.engine.models.cv import rolling_time_splits_multitask
from bayesfolio.engine.models.scaling import MultitaskScaler
device = torch.device("cpu")
from bayesfolio.engine.models.gp.multitask import train_model_hadamard
from math import log, sqrt
from bayesfolio.optimization.evaluate import evaluate_asset_pricing
from bayesfolio.utils import check_equal_occurrences
from bayesfolio.visualization.evaluation import plot_ls_cumulative_compare, plot_actual_vs_pred_matrix
from bayesfolio.visualization.variable_importance import xgboost_variable_importance
from bayesfolio.optimization.portfolio_helpers import assessing_long_short_performance, long_short_returns,long_short_returns_topk, assess_performance
from bayesfolio.engine.models.gp.kernels import (
    InteractionPolicy, KernelArchitectureConfig, KernelBlockConfig, KernelType, KernelVariableType, LinearKernelConfig,
    MaternKernelConfig, BlockStructure,GlobalStructure, ExpoDecayKernelConfig, RQKernelConfig, build_kernel, build_block_kernel
    )
    
from bayesfolio.engine.models.gp.means import MeanF, initialize_mean
from bayesfolio.mlflow.helpers import (
    MultiTaskConfig, long_to_panel, compute_benchmark_panel, r2_os, log_r2_os,
    model_error_by_time_index, log_kernel_architecture_detailed,log_kernel_to_mlflow, log_gpytorch_state_dict, log_gp_hyperparameters
)
import plotly.express as px
import random
import itertools
from bayesfolio.features.market_fundamentals import fetch_enhanced_macro_features
from bayesfolio.features.asset_prices import fetch_etf_features, add_cross_sectional_momentum_rank, cross_sectional_zscore
from gpytorch.kernels import SpectralMixtureKernel
warnings.filterwarnings(
    "ignore",
    message=".*torch.sparse.SparseTensor.*is deprecated.*"
)
from bayesfolio.schemas.configs.core import (
    RiskfolioConfig, OptModel, RiskMeasure, Objective, MuEstimator, CovEstimator)
from IPython.display import display
pd.set_option('display.max_rows', 30)
# Show all columns
pd.set_option('display.max_columns', None)

# Show full column width (don't truncate cell contents)
pd.set_option('display.max_colwidth', None)

# Optional: wider display width
pd.set_option('display.width', None)
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
mlflow.set_experiment("February Portfolio Experiments")


warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3}'.format

############### Experiment Configuration ###############

description = "tracking experiments for february portfolio assets"

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
    'IBND', 'ISHG', 'PDBC', 'BIL', 'EMB', 'TIP', #'IWM', #"BND","BNDX", "MGK"
#"IEF", "LQD", "HYEM",  "HYG", "VWOB", "EWX"
    ]

#remove assets_to_drop from etf_tickers
etf_tickers = [ticker for ticker in etf_tickers if ticker not in assets_to_drop]

tickers = TickerConfig(
    start_date="2016-11-29",
    end_date="2026-01-31",
    interval=Interval.DAILY,
    tickers=etf_tickers,
    horizon=Horizon.MONTHLY,
    lookback_date="2014-07-01"
)

############### Returns data ###############
# return_data = build_long_panel(tickers.tickers, tickers.lookback_date, tickers.end_date, horizon=tickers.horizon)
# return_data.to_csv("20260131_etf_returns.csv", index=False)
return_data = pd.read_csv("20260131_etf_returns.csv")  # Updated to match the new filename
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
# macro_features.to_csv("20260131_macro_features.csv", index=False)
macro_features = pd.read_csv("20260131_macro_features.csv")
macro_features = macro_features.drop(columns=['vix_ts_level','vix3m','yc_pc1', 'yc_pc2', 'yc_pc3', 'y10_nominal' ])
macro_cols = macro_features.columns[1:].tolist()

# etf_features = fetch_etf_features(tickers.tickers, tickers.lookback_date, tickers.end_date, tickers.horizon)
# etf_features.to_csv("20260131_etf_features.csv", index=False)
etf_features = pd.read_csv("20260131_etf_features.csv")
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
df= df[df['date'] > str("2016-11-28")]
df = df[df['date'] < str("2026-01-30")]
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
 'vix', #was shown low in both xgboost and lengthscales 
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

#df_raw = pd.read_csv("marketmaven/datasets/20251129_14tasks.csv")

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



# ############### Preliminary XGBoost Variable Importance ###############
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
df = df[cols].copy()
# df.to_csv("marketmaven/datasets/20260201_tasks.csv", index=False)
# df_forecast = df[df['date'] > pd.Timestamp("2025-12-01")]
# df_forecast = df_forecast.reset_index(drop=True)
df = df.dropna(subset=['y_excess_lead']).reset_index(drop=True)

# fig = correlation_matrix(df)
# fig.show()

# df.iloc[:, 2:].hist(bins=30, figsize=(20, 15))

############### Data Preparation ###############

cv_config = CVConfig(
    step=1,
    horizon_cv=1,
    embargo=0,
    training_min=100,
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



####################New experimental grid search function ####################

# Define grid parameters
RANK_GRID = [4]
MEAN_F_GRID = [MeanF.MULTITASK_ZERO]

# ETF kernel variations
ETF_KERNEL_GRID = {
    'matern_0.5': MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=0.5),
    #'matern_1.5': MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=1.5),
    #'matern_2.5': MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=2.5),
}

# Macro kernel variations (base kernel)
MACRO_BASE_KERNEL_GRID = {
    'matern_0.5': MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=0.5),
    #'matern_1.5': MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=1.5),
}

# Macro modifier options (can include None for no modifier)
MACRO_MODIFIER_GRID = {
    #'none': None,
    'linear': LinearKernelConfig(kernel_type=KernelType.LINEAR, ard=False),
}

# Time kernel variations
TIME_KERNEL_GRID = {
    'matern_0.5': MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=0.5),
    #'matern_1.5': MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=1.5),
    #'expo_decay': ExpoDecayKernelConfig(kernel_type=KernelType.EXPO_DECAY, ard=False),
}

# Global structure and interaction policy options
GLOBAL_STRUCTURE_GRID = [GlobalStructure.HIERARCHICAL] # GlobalStructure.ADDITIVE
INTERACTION_POLICY_GRID = [InteractionPolicy.FULL] # , InteractionPolicy.TEMPORAL_ONLY


class ExperimentConfig(BaseModel):
    etf_kernel_name: str
    macro_base_kernel_name: str
    macro_modifier_name: str
    time_kernel_name: str
    global_structure: GlobalStructure
    interaction_policy: InteractionPolicy
    rank: int
    mean_f: MeanF
    
    def get_run_name(self) -> str:
        """Generate a readable run name for MLflow."""
        return (
            f"etf={self.etf_kernel_name}_"
            f"macro={self.macro_base_kernel_name}+{self.macro_modifier_name}_"
            f"time={self.time_kernel_name}_"
            f"struct={self.global_structure.value}_"
            f"policy={self.interaction_policy.value}_"
            f"rank={self.rank}"
        )
    
    def build_kernel_architecture(self) -> KernelArchitectureConfig:
        """Build the kernel architecture from this config."""
        # Build ETF block
        etf_block = KernelBlockConfig(
            variable_type=KernelVariableType.CONTINUOUS,
            dims=active_dims_e,
            block_structure=BlockStructure.JOINT,
            base_kernel=ETF_KERNEL_GRID[self.etf_kernel_name],
        )
        
        # Build macro base block
        macro_block = KernelBlockConfig(
            variable_type=KernelVariableType.CONTINUOUS,
            dims=active_dims_m,
            block_structure=BlockStructure.JOINT,
            base_kernel=MACRO_BASE_KERNEL_GRID[self.macro_base_kernel_name],
        )
        
        # Build time block
        time_block = KernelBlockConfig(
            variable_type=KernelVariableType.TEMPORAL,
            dims=active_dims_t,
            block_structure=BlockStructure.JOINT,
            base_kernel=TIME_KERNEL_GRID[self.time_kernel_name],
        )
        
        # Build blocks list (conditionally include macro modifier)
        blocks = [etf_block, macro_block]
        
        # Add macro modifier if not 'none'
        print(self.macro_modifier_name)
        if self.macro_modifier_name != 'none':
            macro_modifier_block = KernelBlockConfig(
                variable_type=KernelVariableType.CONTINUOUS,
                dims=active_dims_m,
                block_structure=BlockStructure.JOINT,
                base_kernel=MACRO_MODIFIER_GRID[self.macro_modifier_name],
            )
            blocks.append(macro_modifier_block)
        
        blocks.append(time_block)
        
        return KernelArchitectureConfig(
            blocks=blocks,
            global_structure=self.global_structure,
            interaction_policy=self.interaction_policy,
        )


##### Manual entry for testing #####

# etf_block = KernelBlockConfig(
#     variable_type=KernelVariableType.CONTINUOUS,
#     dims=active_dims_e,
#     block_structure=BlockStructure.JOINT,
#     base_kernel=MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=0.5),
# )

# # Build macro base block
# macro_block = KernelBlockConfig(
#     variable_type=KernelVariableType.CONTINUOUS,
#     dims=active_dims_m,
#     block_structure=BlockStructure.JOINT,
#     base_kernel=MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=0.5),
# )

# # Build time block
# time_block = KernelBlockConfig(
#     variable_type=KernelVariableType.TEMPORAL,
#     dims=active_dims_t,
#     block_structure=BlockStructure.JOINT,
#     base_kernel=MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=0.5),
# )

# blocks = [etf_block, macro_block, time_block]


# kernel_architecture = KernelArchitectureConfig(
#     blocks=blocks,
#     global_structure=GlobalStructure.HIERARCHICAL,
#     interaction_policy=InteractionPolicy.TEMPORAL_ONLY,
# )

# etf_kernel = build_block_kernel(etf_block, batch_shape=torch.Size())
# macro_block = build_block_kernel(macro_block, batch_shape=torch.Size())
# time_block = build_block_kernel(time_block, batch_shape=torch.Size())

# total = etf_kernel + macro_block + etf_kernel*time_block + macro_block*time_block
def build_experiment_grid():
    """Build grid of all experiment configurations."""
    experiment_list = []
    
    for etf_name in ETF_KERNEL_GRID.keys():
        for macro_base_name in MACRO_BASE_KERNEL_GRID.keys():
            for macro_mod_name in MACRO_MODIFIER_GRID.keys():
                for time_name in TIME_KERNEL_GRID.keys():
                    for global_struct in GLOBAL_STRUCTURE_GRID:
                        for interact_policy in INTERACTION_POLICY_GRID:
                            for rank in RANK_GRID:
                                for mean_f in MEAN_F_GRID:
                                    cfg = ExperimentConfig(
                                        etf_kernel_name=etf_name,
                                        macro_base_kernel_name=macro_base_name,
                                        macro_modifier_name=macro_mod_name,
                                        time_kernel_name=time_name,
                                        global_structure=global_struct,
                                        interaction_policy=interact_policy,
                                        rank=rank,
                                        mean_f=mean_f,
                                    )
                                    experiment_list.append(cfg)
    
    return experiment_list


    
experiment_grid = build_experiment_grid()
print(f"Total experiments: {len(experiment_grid)}")


# arc = experiment_grid[0].build_kernel_architecture()
# print('arc:\n', arc)
# kernel = build_kernel(arc, batch_shape=torch.Size())
# print('Kernel built:\n', kernel)
############### Run ###############
seed = 27

# (Optional) group all runs in one MLflow experiment
mlflow.set_experiment("February 2026 portfolio")


# Then in your MLflow loop:
for cfg in experiment_grid:
    run_name = cfg.get_run_name()
    
    with mlflow.start_run(run_name=run_name, description="Grid search experiment") as run:
        # Build kernel architecture from config
        kernel_architecture = cfg.build_kernel_architecture()
        kernel = build_kernel(kernel_architecture, batch_shape=torch.Size())
        
        # Set up multiconfig
        multiconfig = MultiTaskConfig(
            num_tasks=len(tickers.tickers),
            mean=cfg.mean_f,
            rank=cfg.rank,
            scaling="global",   
            min_noise=5e-3,
        )
 
        mlflow.log_params(multiconfig.model_dump())
        log_kernel_architecture_detailed(kernel_architecture, prefix="kernel")
        mlflow.log_params(tickers.model_dump())
        mlflow.log_params(cv_config.model_dump())
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
            kernel,
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
                mean= multiconfig.mean,
                num_tasks=multiconfig.num_tasks,
                input_size=X.shape[1],
                macro_dims=active_dims_m,
                etf_mean_spec=None,
                task_map=None,
            )
            
            
            # ---- Train model ----
            model, likelihood = train_model_hadamard(
                X_trs,
                y_trs,
                rank=multiconfig.rank,
                mean_f=mean_f,
                kernel=kernel,
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
                kernel=kernel,
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
        

        # Save global_importance as a CSV file
        # importance_path = "marketmaven/mlflow/artifacts/global_importance.csv"
        # global_importance.to_csv(importance_path, header=True)
        # mlflow.log_artifact(importance_path)

        # # Optionally, log the global_importance as a dictionary for quick inspection
        # mlflow.log_dict(global_importance.to_dict(), "global_importance.json")
        
        compare_error = model_error_by_time_index(y_true_norm, y_pred_norm)
        compare_error.to_html("marketmaven/mlflow/artifacts/compare_error.html")
        mlflow.log_artifact("marketmaven/mlflow/artifacts/compare_error.html")
        
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
