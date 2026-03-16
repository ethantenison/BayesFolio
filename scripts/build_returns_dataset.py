# ruff: noqa
"""Build ETF returns and features dataset using the current BayesFolio API.

This script fetches ETF excess return data and a full merged feature panel
(ETF + macro predictors) for a configurable ticker universe, using the current
bayesfolio.contracts / io / engine.features stack.

Usage:
    poetry run python scratch/build_returns_dataset.py

Outputs (printed + available for downstream use):
    returns_wide  : pd.DataFrame  date × asset_id, y_excess_lead (decimal, 0.02 = 2%)
    features_df   : pd.DataFrame  full merged panel ready for GP modelling

Extend below the "GP section placeholder" comment to add your GP method.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine.features import (
    FeatureProviders,
    build_features_dataset,
    build_long_panel,
    make_default_feature_providers,
    prepare_multitask_gp_data_with_task_feature,
)
from bayesfolio.engine.forecast import (
    build_gp_interpretation_report,
    display_gp_interpretation_report,
    render_gp_interpretation_report,
)
from bayesfolio.engine.forecast.gp.multitask_builder import (
    CovarModuleConfig,
    KernelComponentConfig,
    KernelKind,
    KernelTermConfig,
    LengthscalePolicy,
    LengthscalePolicyConfig,
    MeanKind,
    MeanModuleConfig,
    build_multitask_gp,
)
from bayesfolio.io import (
    ParquetArtifactStore,
    ReturnsProvider,
)

# ---------------------------------------------------------------------------
# Universe configuration
# ---------------------------------------------------------------------------

ETF_TICKERS: list[str] = [
    "SPY",  # total US market large cap
    # "MGK",  # US growth
    "VTV",  # US value
    "IJR",  # US small cap (S&P 600)
    # "IWM",  # US small cap (Russell)
    # "VNQ",  # US REIT
    # "VNQI",  # international REIT
    # $"VEA",  # developed international equity
    # "VWO",  # emerging market equity
    # "VSS",  # foreign small/mid cap
    # "BND",  # total US bond market
    # "IEF",  # 7-10 yr treasury
    # "BNDX",  # total international bond (USD hedged)
    # "LQD",  # investment grade corporate bonds
    # "HYG",  # high yield bonds
    # "EWX",  # emerging market small cap
    # "VWOB",  # emerging market government bonds
    # "HYEM",  # emerging market high yield corporate bonds
]

# Assets to exclude from the portfolio universe.
# Keep in this list to omit from model inputs and labels.
DROP_ASSETS: list[str] = []

# Date range
LOOKBACK_DATE = date(2019, 7, 1)  # Earliest history for feature engineering
START_DATE = date(2021, 11, 29)  # First row in the output panel
END_DATE = date(2026, 2, 28)  # Last row in the output panel

# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

# Macro columns to drop (low-signal or redundant)
DROP_MACRO_COLS: list[str] = [
    "vix_ts_level",
    "vix3m",
    "yc_pc1",
    "yc_pc2",
    "yc_pc3",
    "y10_nominal",
    "de10y",
    "jp10y",
    "uk10y",
    "cn10y",
]

# ETF feature columns to drop
DROP_ETF_COLS: list[str] = [
    "ma_1m",
    "ma_3m",
    "vol_1w",
    "price",
    "overnight_gap",
    "lag2_y_excess_lead",
    "mom1m",
    "mom6m",
    "vol_1m",
    "vol_3m",
    "vol_of_vol",
    "vol_accel",
    "ret_skew",
    "ill_log",
    "dolvol_log",
    "log_ret",
    "sd_turn",
    "turnover",
    "volume",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_providers() -> FeatureProviders:
    """Construct the provider container with local disk caching.

    Returns:
        FeatureProviders: Container holding returns, ETF features, and macro
            feature providers, each pointing to the shared artifact cache.
    """
    return make_default_feature_providers()


def build_returns_wide(
    tickers: list[str],
    start: date,
    end: date,
    horizon: Horizon = Horizon.MONTHLY,
) -> pd.DataFrame:
    """Fetch excess returns and pivot to wide format (date × asset_id).

    Pulls from the local cache when available; falls back to a network fetch.
    Returns are in **decimal units** (0.02 = 2%).

    For portfolio optimization based on historical returns only. ML returns should use
    the full feature panel with aligned labels (build_full_feature_panel) to avoid lookahead
    and ensure proper date filtering.

    Args:
        tickers: List of ETF ticker symbols.
        start: Lookback start date (inclusive), used as ISO string internally.
        end: End date (inclusive), used as ISO string internally.
        horizon: Return aggregation horizon.

    Returns:
        pd.DataFrame: Wide-format excess returns, shape (n_periods, n_tickers).
            Index is a DatetimeIndex sorted ascending. Columns are ticker symbols.
            Rows where *all* values are NaN are dropped.
    """
    provider = ReturnsProvider(
        fetcher=build_long_panel,
        cache_dir="artifacts/cache/returns",
    )
    returns_long: pd.DataFrame = provider.get_y_excess_lead_long(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        horizon=horizon,
    )
    returns_wide = (
        returns_long.pivot(index="date", columns="asset_id", values="y_excess_lead").sort_index().dropna(how="all")
    )
    returns_wide.columns.name = None
    return returns_wide


def build_full_feature_panel() -> pd.DataFrame:
    """Build and cache the full merged feature panel (ETF + macro + labels).

    Runs the complete end-to-end pipeline:
      - Fetches and caches returns, ETF features, and macro features.
      - Applies log-liquidity engineering, cross-sectional momentum ranking,
        date filtering, and lag alignment.
      - Persists the result to parquet under artifacts/features/.

    Returns:
        pd.DataFrame: The full merged feature panel, loaded from the saved
            parquet artifact. Columns include ``date``, ``asset_id``, all
            feature columns, and ``y_excess_lead`` (decimal). Returns are in
            **decimal units** (0.02 = 2%).
    """
    command = BuildFeaturesDatasetCommand(
        tickers=ETF_TICKERS,
        drop_assets=DROP_ASSETS,
        lookback_date=LOOKBACK_DATE,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=Interval.DAILY,
        horizon=Horizon.MONTHLY,
        drop_macro_cols=DROP_MACRO_COLS,
        drop_etf_cols=DROP_ETF_COLS,
        clip_quantile=0.99,
        seed=27,
        artifact_name="march_2026_features.parquet",
    )

    providers = make_providers()
    artifact_store = ParquetArtifactStore(base_dir="artifacts/features")

    result = build_features_dataset(
        command=command,
        providers=providers,
        artifact_store=artifact_store,
    )

    print(f"[build] Artifact saved: {result.artifact.uri}")
    print(f"[build] Shape: {result.artifact.row_count} rows × {result.artifact.column_count} cols")
    for note in result.diagnostics:
        print(f"[build] Note: {note}")

    return pd.read_parquet(result.artifact.uri)


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
# --- Quick returns-only path (wide format; good for riskfolio / benchmarks) ---
# print("Fetching returns (wide format)...")
# returns_wide = build_returns_wide(
#     tickers=ETF_TICKERS,
#     start=LOOKBACK_DATE,
#     end=END_DATE,
# )
# print(f"returns_wide shape: {returns_wide.shape}")
# print(returns_wide.tail())

# --- Full feature panel (needed for GP modelling) ---
print("\nBuilding full feature panel...")
features_df = build_full_feature_panel()
print(f"features_df shape: {features_df.shape}")
print(features_df.dtypes)
print(features_df.tail())

# --- GP section placeholder ---
# TODO: Add GP data preparation and model training below.

# ---------------------------------------------------------------------------
# GP configuration
# ---------------------------------------------------------------------------

# Quick switch for the default lengthscale policy used in the base Matern block.
# Use BOTORCH_STANDARD for BoTorch's fixed lower bound or ADAPTIVE for the
# dimension-scaled lower bound used in the older research code.
GP_LENGTHSCALE_POLICY = LengthscalePolicy.ADAPTIVE

# Default kernel layout for the current script: one Matern block across all
# non-task features.
USE_SPLIT_KERNEL_BLOCKS_EXAMPLE = False

X, y, task_map = prepare_multitask_gp_data_with_task_feature(
    features_df,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date"],
    dtype=torch.float64,
)

task_feature = -1  # or the actual column index holding the task id
all_task_values = X[:, task_feature].to(torch.long).unique(sorted=True)
task_idx = X.shape[-1] - 1
non_task_indices = [i for i in range(X.shape[-1]) if i != task_idx]

Xn = X.clone()

mins = X[:, non_task_indices].amin(dim=0)
maxs = X[:, non_task_indices].amax(dim=0)
ranges = (maxs - mins).clamp_min(1e-12)

Xn[:, non_task_indices] = (X[:, non_task_indices] - mins) / ranges

outcome_transform = StratifiedStandardize(
    stratification_idx=task_feature,
    all_task_values=all_task_values,
    observed_task_values=X[:, task_feature].to(torch.long),
    batch_shape=y.shape[:-2],  # usually torch.Size([])
)

# Default single-block covariance over all non-task features.
covar_config = CovarModuleConfig(
    terms=[
        KernelTermConfig(
            components=[
                KernelComponentConfig(
                    kind=KernelKind.MATERN,
                    dims=non_task_indices,
                    ard=True,
                    matern_nu=2.5,
                    use_outputscale=True,
                    lengthscale_policy=LengthscalePolicyConfig(
                        policy=GP_LENGTHSCALE_POLICY,
                    ),
                )
            ]
        )
    ]
)

# Example split-block layout for future use.
# Replace these placeholder index lists with the actual normalized feature
# indices for your time, ETF, and macro feature groups.
if USE_SPLIT_KERNEL_BLOCKS_EXAMPLE:
    time_feature_indices: list[int] = []
    etf_feature_indices: list[int] = []
    macro_feature_indices: list[int] = []

    covar_config = CovarModuleConfig(
        terms=[
            KernelTermConfig(
                components=[
                    KernelComponentConfig(
                        kind=KernelKind.PERIODIC,
                        dims=time_feature_indices,
                        ard=False,
                        use_outputscale=True,
                    )
                ]
            ),
            KernelTermConfig(
                components=[
                    KernelComponentConfig(
                        kind=KernelKind.MATERN,
                        dims=etf_feature_indices,
                        ard=True,
                        matern_nu=2.5,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(
                            policy=GP_LENGTHSCALE_POLICY,
                        ),
                    )
                ]
            ),
            KernelTermConfig(
                components=[
                    KernelComponentConfig(
                        kind=KernelKind.MATERN,
                        dims=macro_feature_indices,
                        ard=True,
                        matern_nu=2.5,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(
                            policy=GP_LENGTHSCALE_POLICY,
                        ),
                    )
                ]
            ),
        ]
    )

mean_config = MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT)

model = build_multitask_gp(
    train_X=Xn,
    train_Y=y,
    task_feature=-1,
    covar_config=covar_config,
    mean_config=mean_config,
    outcome_transform=outcome_transform,
    input_transform=None,
    rank=1,
)

model.train()
likelihood = model.likelihood  # MultiTaskGP has a likelihood attribute
mll = ExactMarginalLogLikelihood(likelihood, model)

fit_gpytorch_mll(mll)  # fits model hyperparameters

# Removing features led to great speed up.

# ---- Predict ----
model.eval()
likelihood.eval()
with torch.no_grad():
    f_dist = model(Xn)
    pred = likelihood(f_dist, Xn)

from IPython.display import display

report = build_gp_interpretation_report(
    df=features_df,
    model=model,
    target_column="y_excess_lead",
    task_column="asset_id",
)

rendered_report = render_gp_interpretation_report(report)

display(rendered_report["summary_display"])
display(rendered_report["notes_display"])
display(rendered_report["feature_summary"])
display(rendered_report["task_correlation_figure"])
display_gp_interpretation_report(rendered_report)
