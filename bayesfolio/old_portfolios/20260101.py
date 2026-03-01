"""

Jan 2026 Portoflio Selections


"""
import pandas as pd
from bayesfolio.configs import TickerConfig, Interval, Horizon, CVConfig
from bayesfolio.asset_prices import build_long_panel
import numpy as np
import torch
from bayesfolio.configs import (
    RiskfolioConfig, OptModel, RiskMeasure, Objective, MuEstimator, CovEstimator)
from bayesfolio.visualization.eda import correlation_matrix
from bayesfolio.gp_data_prep import prepare_multitask_gp_data
from bayesfolio.models.cv import rolling_time_splits_multitask
from bayesfolio.models.scaling import MultitaskScaler
device = torch.device("cpu")
from bayesfolio.models.gp import train_model_hadamard
from bayesfolio.portfolio.helpers import assessing_long_short_performance, long_short_returns,long_short_returns_topk, assess_performance
from bayesfolio.models.kernels import KernelType, initialize_kernel, adaptive_lengthscale_prior,  create_kernel_initialization, KernelConfig
from bayesfolio.models.means import MeanF, initialize_mean
from bayesfolio.mlflow.helpers import (
    MultiTaskConfig, long_to_panel, compute_benchmark_panel, r2_os, log_r2_os,
    model_error_by_time_index, log_kernel_to_mlflow, log_gpytorch_state_dict, log_gp_hyperparameters
)
import random 
import riskfolio as rp
from IPython.display import display
pd.options.display.float_format = '{:.6f}'.format
pd.set_option('display.max_rows', 20)
raw_data = pd.read_csv("marketmaven/datasets/20260102_18tasks.csv")

raw_data['asset_id'] = raw_data['asset_id'].astype("category")
raw_data['date'] = pd.to_datetime(raw_data['date'])



df = raw_data.copy().reset_index(drop=True)


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

time_col = "date"
y_col = "y_excess_lead"  
asset_col = "asset_id"


droppin_cols = ['date', 'asset_id']
y_col = ['y_excess_lead']
col_order  = droppin_cols + FEATURE_COLS + y_col
df = df[col_order]
df['t_index'] = pd.factorize(df['date'])[0]
cols = df.columns.tolist()       
cols.insert(0, cols.pop(cols.index("t_index")))
df = df.sort_values(["date", "asset_id"], ascending=[True, True]).reset_index(drop=True)
df = df[cols].copy()

df_forecast = df[df['date'] > pd.Timestamp("2025-12-01")].copy()
df_forecast = df_forecast.reset_index(drop=True)
df = df.dropna(subset=['y_excess_lead']).reset_index(drop=True)

feature_cols = ['t_index'] + FEATURE_COLS


############################### Game day configs ##################################
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
    "IEF", # 7-10 year treasury bond ETF US centric
    "BNDX", # total international bond market ETF, USD hedged, but actually developed markets only
    "LQD", # investment grade bond ETF US centric
    "HYG", # High yield bond ETF US centric 
    "EWX", # emerging market small cap ETF
    "VWOB", # Emerging Market Goverment bond 
    "HYEM", # emerging market high yield corporate bond ETF USD hedged
    
]

tickers = TickerConfig(
    start_date="2016-11-29",
    end_date="2026-01-01",
    interval=Interval.DAILY,
    tickers=etf_tickers,
    horizon=Horizon.MONTHLY,
    lookback_date="2014-06-01"
)


# === Build index lists ===

# Time kernel active dims
active_dims_t = [feature_cols.index('t_index')]

# ETF features active dims
active_dims_e = [feature_cols.index(col) for col in etf_cols]

# Macro features active dims
active_dims_m = [feature_cols.index(col) for col in macro_cols]

    
X, I, y, task_map = prepare_multitask_gp_data(
    df,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date", "asset_id"],
    dtype=torch.float32
)

multiconfig = MultiTaskConfig(
    num_tasks=len(tickers.tickers),
    mean=MeanF.MULTITASK_ZERO,
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
)

kernel_m = KernelConfig(
    type=KernelType.MATERN_LINEAR_RQ,
    features=macro_cols,
    active_dims=active_dims_m,
    smoothness=0.5,
)

kernel_t = KernelConfig(
    type=KernelType.MATERN,
    features=[time_col],
    active_dims=active_dims_t,
    smoothness=0.5,
)

seed = 27
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.use_deterministic_algorithms(True)

############### Model Setup ###################

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

X_t = df_forecast.copy()
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
#df_sorted.to_csv("marketmaven/old_portfolios/jan_predictions_rank3.csv", index=False)
df_sorted_r3 = df_sorted.copy()


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


########## Build Riskfolio ############
risk_config = RiskfolioConfig(
    model=OptModel.CLASSIC,
    rm=RiskMeasure.CVaR,
    obj=Objective.SHARPE,
    method_mu=MuEstimator.EWMA2,
    method_cov=CovEstimator.GERBER2,
    nea=10

)

return_data = pd.read_csv("20260101_etf_returns.csv")  # Updated to match the new filename
return_data['date'] = pd.to_datetime(return_data['date'])
pivoted_returns = return_data.pivot(index="date", columns="asset_id", values="y_excess_lead")
pivoted_returns = pivoted_returns.dropna().sort_index()

#subset return_data to all time points after 2023-12-31
return_data2 = return_data[return_data['date'] > pd.Timestamp("2023-12-31")]

#shrink list of etfs 
final_pivoted_returns = pivoted_returns.drop(columns=['BNDX','BND', 'MGK', "IJR"])
#IJR didn't do very well last time.....iwm has less uncertainty. 


#results
results = {
    "SPY":  {"prediction": 0.022027, "uncertainty": 0.026897, "score": 0.818950, "rank": 3},
    "IWM":  {"prediction": 0.031506, "uncertainty": 0.039369, "score": 0.800266, "rank": 3},
    "MGK":  {"prediction": 0.030452, "uncertainty": 0.038928, "score": 0.782263, "rank": 3},
    "VTV":  {"prediction": 0.013869, "uncertainty": 0.024678, "score": 0.562002, "rank": 3},
    "VSS":  {"prediction": 0.014364, "uncertainty": 0.025594, "score": 0.561233, "rank": 3},
    "IJR":  {"prediction": 0.021534, "uncertainty": 0.040116, "score": 0.536799, "rank": 3},
    "VEA":  {"prediction": 0.011334, "uncertainty": 0.025098, "score": 0.451584, "rank": 3},
    "EWX":  {"prediction": 0.014188, "uncertainty": 0.031634, "score": 0.448491, "rank": 3},
    "VWO":  {"prediction": 0.011638, "uncertainty": 0.032198, "score": 0.361466, "rank": 3},
    "HYG":  {"prediction": 0.004614, "uncertainty": 0.013830, "score": 0.333641, "rank": 3},
    "HYEM": {"prediction": 0.004413, "uncertainty": 0.019652, "score": 0.224579, "rank": 3},
    "VWOB": {"prediction": 0.004270, "uncertainty": 0.026789, "score": 0.159415, "rank": 3},
    "LQD":  {"prediction": 0.001613, "uncertainty": 0.011668, "score": 0.138236, "rank": 3},
    "BND":  {"prediction": 0.000435, "uncertainty": 0.010553, "score": 0.041243, "rank": 3},
    "VNQI": {"prediction": -0.000872, "uncertainty": 0.030092, "score": -0.028976, "rank": 3},
    "BNDX": {"prediction": -0.000533, "uncertainty": 0.009664, "score": -0.055134, "rank": 3},
    "VNQ":  {"prediction": -0.002890, "uncertainty": 0.032687, "score": -0.088412, "rank": 3},
    "IEF":  {"prediction": -0.002490, "uncertainty": 0.014876, "score": -0.167387, "rank": 3},
}

means = {
    # "BNDX": -0.001134, 
    # "BND": -0.000182,
    "IEF": -0.000722, #might still use prediction 
    "LQD": 0.000434,
    "VNQ": 0.000280,
}

Final_results = {
    "SPY":  {"prediction": 0.022027, "uncertainty": 0.026897, "score": 0.818950, "rank": 3},
    "IWM":  {"prediction": 0.031506, "uncertainty": 0.039369, "score": 0.800266, "rank": 3},
    "VTV":  {"prediction": 0.013869, "uncertainty": 0.024678, "score": 0.562002, "rank": 3},
    "VSS":  {"prediction": 0.014364, "uncertainty": 0.025594, "score": 0.561233, "rank": 3},
    "VEA":  {"prediction": 0.011334, "uncertainty": 0.025098, "score": 0.451584, "rank": 3},
    "EWX":  {"prediction": 0.014188, "uncertainty": 0.031634, "score": 0.448491, "rank": 3},
    "VWO":  {"prediction": 0.011638, "uncertainty": 0.032198, "score": 0.361466, "rank": 3},
    "HYG":  {"prediction": 0.004614, "uncertainty": 0.013830, "score": 0.333641, "rank": 3},
    "HYEM": {"prediction": 0.004413, "uncertainty": 0.019652, "score": 0.224579, "rank": 3},
    "VWOB": {"prediction": 0.004270, "uncertainty": 0.026789, "score": 0.159415, "rank": 3},
    "LQD":  {"prediction": 0.000434, "uncertainty": None, "score": None, "rank": None},
    "VNQI": {"prediction": -0.000872, "uncertainty": 0.030092, "score": -0.028976, "rank": 3},
    "VNQ":  {"prediction": 0.000280, "uncertainty": None, "score": None, "rank": None},
    "IEF":  {"prediction": -0.002490, "uncertainty": 0.014876, "score": -0.167387, "rank": 3},
}

res = pd.DataFrame.from_dict(Final_results, orient='index')
print(res)
################### Without GP predictions ######################


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
port.alpha = 0.2
w = port.optimization(model=model_risk, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

display(w.T)
label = str(risk_config.rm) + " " + str(risk_config.obj) + "Basic Portfolio" # Title of point
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
                      rf=rf, alpha=port.alpha, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)


ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)

################### With GP predictions ######################
pd.options.display.float_format = '{:.6f}'.format
imputing = port.mu.copy()

# Replace values with predictions
for asset, vals in Final_results.items():
    if asset in imputing.columns and vals.get("prediction") is not None:
        imputing.loc[:, asset] = vals["prediction"]


port.card = None 
port.nea = risk_config.nea
port.mu = imputing
port.alpha = 0.20

w = port.optimization(model=model_risk, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

display(w.T)
label = str(risk_config.rm) + " " + str(risk_config.obj) + " Adjusted ML Portfolio" # Title of point
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
                      rf=rf, alpha=port.alpha, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)


ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)

weights_series = w.T.loc['weights']
res['weights'] = weights_series

res.to_csv("marketmaven/old_portfolios/jan_portfolio_weights.csv", index=False)