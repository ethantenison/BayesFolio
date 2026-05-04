from __future__ import annotations

from datetime import date

import pandas as pd
import riskfolio as rp
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.kernels.positive_index import PositiveIndexKernel
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from IPython.display import display

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.core.settings import (
    CovEstimator,
    Horizon,
    Interval,
    MuEstimator,
    Objective,
    OptModel,
    RiskfolioConfig,
    RiskMeasure,
)
from bayesfolio.engine.features import build_features_dataset, make_default_feature_providers
from bayesfolio.engine.features.gp_data_prep import prepare_multitask_gp_data_with_task_feature
from bayesfolio.engine.forecast.gp.multitask_builder import (
    BlockStructure,
    CovarModuleConfig,
    GlobalStructure,
    InteractionPolicy,
    KernelBlockConfig,
    KernelBlockRole,
    KernelInteractionConfig,
    LengthscalePolicy,
    LengthscalePolicyConfig,
    LinearKernelComponentConfig,
    MaternKernelComponentConfig,
    MeanKind,
    MeanModuleConfig,
    RQKernelComponentConfig,
    build_multitask_gp,
)
from bayesfolio.io import ParquetArtifactStore

ETF_TICKERS = [
    "SPY",  # total US market big cap
    "MGK",  # US growth
    "VTV",  # US value
    "IJR",  # US small cap S&P index 600, more stable
    "IWM",  # US small cap Russel index, more volile than IJR
    "VNQ",  # REIT ETF US centric
    "VNQI",  # international REIT ETF
    "VEA",  # developed international equity
    "VWO",  # AVEM actually is better than VWO but not enough history
    "VSS",  # forein small/mid cap
    "BND",  # total bond market ETF US centric
    "IEF",  # 7-10 year treasury bond ETF US centric
    "BNDX",  # total international bond market ETF, USD hedged, but actually developed markets only
    "LQD",  # investment grade bond ETF US centric
    "HYG",  # High yield bond ETF US centric
    "EWX",  # emerging market small cap ETF
    "VWOB",  # Emerging Market Goverment bond
    "HYEM",  # emerging market high yield corporate bond ETF USD hedged
]

DROP_ASSETS: list[str] = []
EXCLUDE_FROM_SCENARIOS: set[str] = set()
N_POSTERIOR_SCENARIOS = 5000

LOOKBACK_DATE = date(2019, 2, 1)
START_DATE = date(2021, 2, 1)
END_DATE = date(2026, 5, 1)

SELECTED_ETF_COLS = [
    "baspread",
    "ret_kurt",
    "chmom",
    "mom12m",
    "mom36m",
    "cs_mom_rank",
    "max_dd_6m",
    "ma_signal",
    "ret_autocorr",
    "vol_z",
]

SELECTED_MACRO_COLS = [
    "hy_spread",
    "hy_spread_chg_1m",
    "hy_spread_z_12m",
    "vix_slope",
    "vix_ts_z_12m",
    "vix",
    "spy_flow_z_12m",
    "spy_ret",
    "erp",
    "cpi_yoy",
    "cpi_mom",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
]


command = BuildFeaturesDatasetCommand.model_validate(
    {
        "schema": "bayesfolio.features_dataset.command",
        "tickers": ETF_TICKERS,
        "drop_assets": DROP_ASSETS,
        "lookback_date": LOOKBACK_DATE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "interval": Interval.DAILY,
        "horizon": Horizon.MONTHLY,
        "etf_cols": SELECTED_ETF_COLS,
        "macro_cols": SELECTED_MACRO_COLS,
        "drop_etf_cols": [],
        "drop_macro_cols": [],
        "clip_quantile": 0.99,
        "seed": 27,
        "artifact_name": "portfolio_etf_macro_features_2026_05.parquet",
        "include_unlabeled_tail": True,
    }
)

providers = make_default_feature_providers(
    cache_root="artifacts/cache",
)

artifact_store = ParquetArtifactStore(
    base_dir="artifacts/features",
)

result = build_features_dataset(
    command=command,
    providers=providers,
    artifact_store=artifact_store,
)

print("Artifact URI:", result.artifact.uri)
print("Rows:", result.artifact.row_count)
print("Columns:", result.artifact.column_count)
print("Diagnostics:")
for note in result.diagnostics:
    print(" -", note)

features_df = pd.read_parquet(result.artifact.uri)
print(features_df.shape)
print(features_df.columns.tolist())
print(features_df.head())


features_df = pd.read_parquet(result.artifact.uri)

KEEP_COLS = [
    "t_index",
    "date",
    "asset_id",
    "lag_y_excess_lead",
    "baspread",
    "ret_kurt",
    "chmom",
    "mom12m",
    "mom36m",
    "cs_mom_rank",
    "max_dd_6m",
    "ma_signal",
    "ret_autocorr",
    "vol_z",
    "hy_spread",
    "hy_spread_chg_1m",
    "hy_spread_z_12m",
    "vix_slope",
    "vix_ts_z_12m",
    "vix",
    "spy_flow_z_12m",
    "spy_ret",
    "erp",
    "cpi_yoy",
    "cpi_mom",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
    "y_excess_lead",
]

features_df = features_df.loc[:, KEEP_COLS].copy()

################# Building GP #################
# 1) Exact feature groups
time_cols = ["t_index"]

# Put lag target in ETF block so all features are assigned exactly once.
etf_cols = [
    "lag_y_excess_lead",
    "baspread",
    "ret_kurt",
    "chmom",
    "mom12m",
    "mom36m",
    "cs_mom_rank",
    "max_dd_6m",
    "ma_signal",
    "ret_autocorr",
    "vol_z",
]

macro_cols = [
    "hy_spread",
    "hy_spread_chg_1m",
    "hy_spread_z_12m",
    "vix_slope",
    "vix_ts_z_12m",
    "vix",
    "spy_flow_z_12m",
    "spy_ret",
    "erp",
    "cpi_yoy",
    "cpi_mom",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
]

input_columns = [*time_cols, *etf_cols, *macro_cols]

# 2) Train-only frame (no unlabeled tail rows)
train_df_minus_today = (
    features_df[[*input_columns, "asset_id", "y_excess_lead"]].dropna(subset=["y_excess_lead"]).copy()
)
train_df = features_df[[*input_columns, "asset_id", "y_excess_lead"]].copy()

# 3) Prepare tensors (task index appended as last column)
full_x_raw, full_y, task_map = prepare_multitask_gp_data_with_task_feature(
    train_df,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=[],
    dtype=torch.float32,
)

# 4) Min-max normalize non-task columns (same pattern as workflow.py)
full_x = full_x_raw.clone()
non_task_dim_count = len(input_columns)
mins = full_x[:, :non_task_dim_count].amin(dim=0)
maxs = full_x[:, :non_task_dim_count].amax(dim=0)
ranges = (maxs - mins).clamp_min(1e-12)
full_x[:, :non_task_dim_count] = (full_x[:, :non_task_dim_count] - mins) / ranges

#### separting the test is shouldn't be done like this after transforming but ni modo. it is only one.
train_x = full_x[:-18, :]
train_y = full_y[:-18, :]
test_x = full_x[-18:, :]
test_y = full_y[-18:, :]

# 5) Build exact dims
idx = {name: i for i, name in enumerate(input_columns)}
time_dims = [idx[c] for c in time_cols]
etf_dims = [idx[c] for c in etf_cols]
macro_dims = [idx[c] for c in macro_cols]

# 6) Exact kernel structure:
# matern0.5_time
# + matern0.5_etf
# + (matern0.5 + rq + linear)_macro
# + time*etf + time*macro + macro*etf
covar_config = CovarModuleConfig(
    blocks=[
        KernelBlockConfig(
            name="time",
            variable_type=KernelBlockRole.TIME,
            components=[
                MaternKernelComponentConfig(
                    dims=time_dims,
                    matern_nu=0.5,
                    ard=True,
                    use_outputscale=True,
                    lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                )
            ],
            block_structure=BlockStructure.ADDITIVE,
            use_outputscale=False,
        ),
        KernelBlockConfig(
            name="etf",
            variable_type=KernelBlockRole.ETF,
            components=[
                MaternKernelComponentConfig(
                    dims=etf_dims,
                    matern_nu=0.5,
                    ard=True,
                    use_outputscale=True,
                    lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                )
            ],
            block_structure=BlockStructure.ADDITIVE,
            use_outputscale=False,
        ),
        KernelBlockConfig(
            name="macro",
            variable_type=KernelBlockRole.MACRO,
            components=[
                MaternKernelComponentConfig(
                    dims=macro_dims,
                    matern_nu=0.5,
                    ard=True,
                    use_outputscale=True,
                    lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                ),
                RQKernelComponentConfig(
                    dims=macro_dims,
                    ard=True,
                    use_outputscale=False,
                    lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                ),
                LinearKernelComponentConfig(
                    dims=macro_dims,
                    use_outputscale=False,
                ),
            ],
            block_structure=BlockStructure.ADDITIVE,
            use_outputscale=True,
        ),
    ],
    global_structure=GlobalStructure.HIERARCHICAL,
    interaction_policy=InteractionPolicy.CUSTOM,
    custom_interactions=[
        KernelInteractionConfig(blocks=["time", "etf"], name="time_x_etf", use_outputscale=True),
        KernelInteractionConfig(blocks=["time", "macro"], name="time_x_macro", use_outputscale=True),
        KernelInteractionConfig(blocks=["macro", "etf"], name="macro_x_etf", use_outputscale=True),
    ],
)

# 7) Outcome transform (task-stratified standardization)
task_feature_idx = train_x.shape[-1] - 1
all_task_values = train_x[:, task_feature_idx].to(torch.long).unique(sorted=True)
outcome_transform = StratifiedStandardize(
    stratification_idx=task_feature_idx,
    all_task_values=all_task_values,
    observed_task_values=train_x[:, task_feature_idx].to(torch.long),
    batch_shape=train_y.shape[:-2],
)

# 8) Build + fit model (deterministic seed)
torch.manual_seed(27)

model = build_multitask_gp(
    train_X=train_x,
    train_Y=train_y,
    task_feature=-1,
    covar_config=covar_config,
    mean_config=MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT),
    rank=5,
    min_inferred_noise_level=5e-3,
    outcome_transform=outcome_transform,
    input_transform=None,
)

model.train()
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

model.eval()
model.likelihood.eval()

with torch.no_grad():
    posterior = model.posterior(test_x, observation_noise=True)
    pred_mean = posterior.mean.squeeze(-1)
    pred_var = posterior.variance.squeeze(-1).clamp_min(0.0)
    pred_std = pred_var.sqrt()
    scenario_samples = posterior.rsample(torch.Size([N_POSTERIOR_SCENARIOS])).squeeze(-1).detach().cpu()

    print(pred_mean.shape)
    print(pred_std.shape)

task_idx_to_asset = {task_idx: asset for asset, task_idx in task_map.items()}
task_ids = test_x[:, -1].to(torch.long).detach().cpu().tolist()
asset_cols = [task_idx_to_asset[task_idx] for task_idx in task_ids]

# Predictive scenarios are already on the original target scale because
# model.posterior(..., observation_noise=True) untransforms the posterior.
keep_idx = [index for index, asset in enumerate(asset_cols) if asset not in EXCLUDE_FROM_SCENARIOS]
scenario_assets = [asset_cols[index] for index in keep_idx]
scenario_returns = pd.DataFrame(scenario_samples[:, keep_idx].numpy(), columns=scenario_assets)
scenario_returns.to_csv("20260501_gp_posterior_scenarios.csv", index=False)

# Reshape y_hat and y_std to match the number of assets in asset_cols
y_hat = pred_mean.numpy().reshape(-1, len(asset_cols))
y_std = pred_std.numpy().reshape(-1, len(asset_cols))

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
df_sorted.to_csv("20260501_gameday_predictions.csv", index=False)


####riskfolio


EXCLUDE = {"MGK", "BND", "BNDX"}

# exclude assets from portfolio universe (but not necessarily from GP)
final_scenarios = scenario_returns.drop(columns=[c for c in EXCLUDE if c in scenario_returns.columns])

risk_config = RiskfolioConfig(
    model=OptModel.CLASSIC,
    rm=RiskMeasure.CVaR,
    obj=Objective.SHARPE,
    method_mu=MuEstimator.HIST,
    method_cov=CovEstimator.HIST,
    nea=12,
    rf=0.0,
    ra=0.5,
    method_kurt=None,
    hist=True,
    upperlng=0.35,
)

port_gp = rp.Portfolio(returns=final_scenarios)
port_gp.assets_stats(method_mu=risk_config.method_mu, method_cov=risk_config.method_cov)

port_gp.alpha = 0.20
port_gp.nea = risk_config.nea
port_gp.card = None

w_gp = port_gp.optimization(
    model=risk_config.model, rm=risk_config.rm, obj=risk_config.obj, rf=int(risk_config.rf), l=0, hist=True
)

display(w_gp.T)
ax = rp.plot_pie(
    w=w_gp,
    title=f"{risk_config.rm} {risk_config.obj} — GP Posterior Scenarios (1M) (excl. MGK/BND/BNDX)",
    others=0.05,
    nrow=25,
    cmap="tab20",
    height=6,
    width=10,
    ax=None,
)


#### Determining rank quality


task_kernel = next(kernel for kernel in model.covar_module.kernels if isinstance(kernel, PositiveIndexKernel))

probe = torch.zeros(
    model.num_tasks,
    train_x.shape[-1],
    dtype=train_x.dtype,
    device=train_x.device,
)
probe[:, model._task_feature] = torch.arange(
    model.num_tasks,
    dtype=train_x.dtype,
    device=train_x.device,
)

K = task_kernel(probe).to_dense().detach().cpu()

d = torch.sqrt(torch.diag(K)).clamp_min(1e-12)
corr = K / (d[:, None] * d[None, :])

eigvals = torch.linalg.eigvalsh(corr).flip(0)
cum = torch.cumsum(eigvals, dim=0) / eigvals.sum()

print(eigvals[:10])
print(cum[:10])

###It shows by how much the latent structure is accounted for by each
