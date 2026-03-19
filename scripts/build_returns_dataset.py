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

from collections.abc import Mapping
from datetime import date
from typing import cast
from uuid import uuid4

import pandas as pd
from regex import R
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.kernels import AdditiveKernel, Kernel, ProductKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from IPython.display import HTML, Markdown, display

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
    BlockStructure,
    CovarModuleConfig,
    GlobalStructure,
    InteractionPolicy,
    KernelBlockConfig,
    KernelBlockRole,
    LengthscalePolicy,
    LengthscalePolicyConfig,
    LinearKernelComponentConfig,
    MaternKernelComponentConfig,
    MeanKind,
    MeanModuleConfig,
    PeriodicKernelComponentConfig,
    build_multitask_gp,
    RQKernelComponentConfig,
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

# Block-based feature selection
# Keep blocks instead of manually listing columns to drop.
FEATURE_BLOCK_COLUMNS: dict[str, list[str]] = {
    "time": [
        "t_index",
    ],
    "etf": [
        "mom12m",
        "mom36m",
        "chmom",
        "vol_z",
        "ma_signal",
        "ma_regime",
        "trend_slope",
        "ret_autocorr",
        "vol_autocorr",
        "ret_kurt",
        "baspread",
        "max_dd_3m",
        "max_dd_6m",
        "cs_mom_rank",
        "lag_y_excess_lead",
    ],
    "macro": [
        "vix",
        "vix_ts_chg_1m",
        "vix_ts_z_12m",
        "tnote10y",
        "tbill3m",
        "term_spread",
        "credit_spread",
        "credit_spread_chg_1p",
        "dxy",
        "spy_ret",
        "erp",
        "skew_proxy",
        "move_proxy",
        "vix_slope",
        "rsp_spy",
        "rsp_spy_roc_1m",
        "spy_flow_z_12m",
        "dealer_gamma_proxy",
        "pct_above_50dma",
        "hy_spread",
        "hy_spread_chg_1m",
        "hy_spread_z_12m",
        "oil",
        "copper",
        "gold",
        "schp",
        "schp_ret",
        "em_fx",
        "em_fx_ret",
        "oil_ret",
        "copper_ret",
        "gold_crude_ratio",
        "y10_real_proxy",
        "breakeven_proxy",
        "cpiaucsl",
        "cpi_yoy",
        "cpi_mom",
    ],
}

# Choose which blocks to include in GP inputs.
SELECTED_FEATURE_BLOCKS: tuple[str, ...] = ("time", "etf", "macro")

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
            "drop_macro_cols": [],
            "drop_etf_cols": [],
            "clip_quantile": 0.99,
            "seed": 27,
            "artifact_name": "march_2026_features.parquet",
        }
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


def get_gp_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str,
    asset_col: str,
    drop_cols: list[str],
) -> list[str]:
    """Return feature column names in the exact order used by GP data prep.

    Args:
        df: Input frame containing features, target, and asset/task id.
        target_col: Name of the prediction target column.
        asset_col: Name of the asset/task column.
        drop_cols: Columns dropped before tensor conversion.

    Returns:
        list[str]: Ordered non-task feature columns used in ``X`` before the
            appended task-index column.
    """
    df_proc = df.drop(columns=drop_cols + [asset_col], errors="ignore")
    feature_df = df_proc.drop(columns=[target_col, "__task_idx__"], errors="ignore")
    return feature_df.columns.tolist()


def build_feature_index_groups_from_blocks(
    feature_columns: list[str],
    *,
    block_columns: dict[str, list[str]],
    blocks: tuple[str, ...] = ("time", "etf", "macro"),
) -> dict[str, list[int]]:
    """Build feature index groups directly from explicit block columns.

    Args:
        feature_columns: Ordered non-task feature columns used by the GP model.
        block_columns: Mapping of block name to explicit feature names.
        blocks: Block names to derive indices for.

    Returns:
        dict[str, list[int]]: Block-to-index mapping in ``feature_columns`` order.
    """
    groups: dict[str, list[int]] = {block: [] for block in blocks}
    feature_lookup = {name: idx for idx, name in enumerate(feature_columns)}
    for block in blocks:
        for column_name in block_columns.get(block, []):
            idx = feature_lookup.get(column_name)
            if idx is not None:
                groups[block].append(idx)
    return groups


def select_feature_blocks(
    df: pd.DataFrame,
    *,
    selected_blocks: tuple[str, ...],
    block_columns: dict[str, list[str]],
    required_columns: tuple[str, ...] = ("date", "asset_id", "y_excess_lead"),
) -> pd.DataFrame:
    """Select feature columns by named blocks and keep required columns.

    Args:
        df: Full feature panel.
        selected_blocks: Block names to include.
        block_columns: Mapping from block name to column names.
        required_columns: Columns always retained.

    Returns:
        pd.DataFrame: A reduced panel preserving original column order.
    """
    unknown_blocks = [name for name in selected_blocks if name not in block_columns]
    if unknown_blocks:
        raise ValueError(f"Unknown feature blocks requested: {unknown_blocks}")

    keep_columns: set[str] = set(required_columns)
    for block_name in selected_blocks:
        keep_columns.update(block_columns[block_name])

    selected_existing = [col for col in df.columns if col in keep_columns]
    return df[selected_existing].copy()


def _extract_active_dims(kernel: Kernel) -> list[int]:
    """Return active dimensions from a kernel if available."""
    dims = getattr(kernel, "active_dims", None)
    if dims is None:
        return []
    if hasattr(dims, "tolist"):
        return [int(v) for v in dims.tolist()]
    return [int(v) for v in dims]


def _infer_block_name(
    *,
    kernel: Kernel,
    dims: list[int],
    block_indices: Mapping[str, list[int]],
) -> str:
    """Infer a block label from kernel active dimensions."""
    kernel_name = type(kernel).__name__
    if "PositiveIndexKernel" in kernel_name:
        return "task"
    if not dims:
        return "all"

    dim_set = set(dims)
    matching: list[str] = []
    for block_name, indices in block_indices.items():
        if not indices:
            continue
        block_set = set(indices)
        if dim_set.issubset(block_set):
            matching.append(block_name)
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        return "+".join(sorted(matching))
    return "custom"


def _unwrap_scale(kernel: Kernel) -> Kernel:
    """Unwrap ScaleKernel layers to get the semantic base kernel."""
    current = kernel
    while isinstance(current, ScaleKernel):
        current = current.base_kernel
    return current


def build_kernel_expression(
    kernel: Kernel,
    *,
    block_indices: Mapping[str, list[int]],
) -> str:
    """Build a readable symbolic expression for a covariance kernel tree."""

    def visit(node: Kernel) -> tuple[str, bool]:
        if isinstance(node, ScaleKernel):
            return visit(node.base_kernel)

        if isinstance(node, AdditiveKernel):
            children = [visit(cast(Kernel, c)) for c in node.kernels]
            return " + ".join(part for part, _ in children), True

        if isinstance(node, ProductKernel):
            children = [visit(cast(Kernel, c)) for c in node.kernels]
            rendered: list[str] = []
            for text, is_additive in children:
                rendered.append(f"({text})" if is_additive else text)
            return " * ".join(rendered), False

        base = _unwrap_scale(node)
        dims = _extract_active_dims(base)
        block = _infer_block_name(kernel=base, dims=dims, block_indices=block_indices)
        kernel_kind = type(base).__name__.replace("Kernel", "")
        return f"Kernel_{block}[{kernel_kind}]", False

    expression, _ = visit(kernel)
    return expression


def build_kernel_mermaid(
    kernel: Kernel,
    *,
    block_indices: Mapping[str, list[int]],
) -> str:
    """Create a Mermaid flowchart showing additive/product kernel structure."""
    lines: list[str] = ["flowchart TD"]
    next_id = 0

    def new_id() -> str:
        nonlocal next_id
        next_id += 1
        return f"k{next_id}"

    def class_for_kernel(kernel_name: str) -> str:
        name = kernel_name.lower()
        if "matern" in name:
            return "kt_matern"
        if "periodic" in name:
            return "kt_periodic"
        if "positiveindex" in name:
            return "kt_index"
        if "rbf" in name:
            return "kt_rbf"
        if "rq" in name:
            return "kt_rq"
        if "linear" in name:
            return "kt_linear"
        return "kt_other"

    def add_node(node: Kernel, parent_id: str | None = None) -> str:
        if isinstance(node, ScaleKernel):
            return add_node(node.base_kernel, parent_id)

        node_id = new_id()

        if isinstance(node, AdditiveKernel):
            lines.append(f'{node_id}["Additive (+)"]:::op_add')
            if parent_id is not None:
                lines.append(f"{parent_id} --> {node_id}")
            for child in node.kernels:
                add_node(cast(Kernel, child), node_id)
            return node_id

        if isinstance(node, ProductKernel):
            lines.append(f'{node_id}["Product (*)"]:::op_mul')
            if parent_id is not None:
                lines.append(f"{parent_id} --> {node_id}")
            for child in node.kernels:
                add_node(cast(Kernel, child), node_id)
            return node_id

        base = _unwrap_scale(node)
        dims = _extract_active_dims(base)
        block_name = _infer_block_name(kernel=base, dims=dims, block_indices=block_indices)
        kernel_name = type(base).__name__
        label = f"Kernel_{block_name}<br/>{kernel_name}"
        lines.append(f'{node_id}["{label}"]:::{class_for_kernel(kernel_name)}')
        if parent_id is not None:
            lines.append(f"{parent_id} --> {node_id}")
        return node_id

    root_id = new_id()
    lines.append(f'{root_id}["Covariance Kernel"]:::op_root')
    add_node(kernel, root_id)

    lines.extend(
        [
            "classDef op_root fill:#2f2f2f,color:#ffffff,stroke:#1b1b1b,stroke-width:1px;",
            "classDef op_add fill:#f2f2f2,color:#111111,stroke:#666666,stroke-width:1px;",
            "classDef op_mul fill:#e6e6e6,color:#111111,stroke:#666666,stroke-width:1px;",
            "classDef kt_matern fill:#4c78a8,color:#ffffff,stroke:#2f4f6f,stroke-width:1px;",
            "classDef kt_periodic fill:#59a14f,color:#ffffff,stroke:#2f5f2b,stroke-width:1px;",
            "classDef kt_index fill:#e15759,color:#ffffff,stroke:#8a2e2f,stroke-width:1px;",
            "classDef kt_rbf fill:#f28e2b,color:#ffffff,stroke:#8c4f12,stroke-width:1px;",
            "classDef kt_rq fill:#76b7b2,color:#ffffff,stroke:#3e6461,stroke-width:1px;",
            "classDef kt_linear fill:#edc948,color:#111111,stroke:#8a7423,stroke-width:1px;",
            "classDef kt_other fill:#bab0ab,color:#111111,stroke:#6a625f,stroke-width:1px;",
        ]
    )
    return "\n".join(lines)


def render_mermaid_in_notebook(mermaid_markup: str) -> None:
    """Render Mermaid diagram markup directly in notebook output."""
    container_id = f"mermaid-{uuid4().hex}"
    escaped_markup = mermaid_markup.replace("`", "\\`")
    display(
        HTML(
            f"""
<div id=\"{container_id}\" class=\"mermaid\">{escaped_markup}</div>
<script type=\"module\">
    import mermaid from \"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs\";
    mermaid.initialize({{ startOnLoad: true }});
    const el = document.getElementById(\"{container_id}\");
    if (el) {{
        mermaid.run({{ nodes: [el] }});
    }}
</script>
"""
        )
    )


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
features_df_full = build_full_feature_panel()
features_df = select_feature_blocks(
    features_df_full,
    selected_blocks=SELECTED_FEATURE_BLOCKS,
    block_columns=FEATURE_BLOCK_COLUMNS,
)
print(f"features_df_full shape: {features_df_full.shape}")
print(f"features_df_selected shape: {features_df.shape}")
print(f"selected blocks: {SELECTED_FEATURE_BLOCKS}")
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
GP_MIN_INFERRED_NOISE_LEVEL = 5e-3

# Default kernel layout for the current script: one Matern block across all
# non-task features.
USE_SPLIT_KERNEL_BLOCKS_EXAMPLE = True

X, y, task_map = prepare_multitask_gp_data_with_task_feature(
    features_df,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date"],
    dtype=torch.float64,
)

feature_columns = get_gp_feature_columns(
    features_df,
    target_col="y_excess_lead",
    asset_col="asset_id",
    drop_cols=["date"],
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
    blocks=[
        KernelBlockConfig(
            name="features",
            variable_type=KernelBlockRole.GENERIC,
            components=[
                MaternKernelComponentConfig(
                    dims=non_task_indices,
                    ard=True,
                    matern_nu=2.5,
                    use_outputscale=True,
                    lengthscale_policy=LengthscalePolicyConfig(
                        policy=GP_LENGTHSCALE_POLICY,
                    ),
                )
            ],
            block_structure=BlockStructure.ADDITIVE,
            use_outputscale=False,
        )
    ],
    global_structure=GlobalStructure.ADDITIVE,
)

# Example split-block layout for future use.
if USE_SPLIT_KERNEL_BLOCKS_EXAMPLE:
    feature_groups = build_feature_index_groups_from_blocks(
        feature_columns,
        block_columns=FEATURE_BLOCK_COLUMNS,
        blocks=("time", "etf", "macro"),
    )

    time_feature_indices = feature_groups["time"]
    etf_feature_indices = feature_groups["etf"]
    macro_feature_indices = feature_groups["macro"]

    print(
        "[gp] split kernel groups "
        f"time={len(time_feature_indices)}, "
        f"etf={len(etf_feature_indices)}, "
        f"macro={len(macro_feature_indices)}"
    )

    covar_config = CovarModuleConfig(
        blocks=[
            KernelBlockConfig(
                name="time",
                variable_type=KernelBlockRole.TIME,
                components=[
                    MaternKernelComponentConfig(
                        dims=time_feature_indices,
                        ard=False,
                        matern_nu=0.5,
                        use_outputscale=False,
                        lengthscale_policy=LengthscalePolicyConfig(
                            policy=GP_LENGTHSCALE_POLICY,
                        ),
                    )
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=True,
            ),
            KernelBlockConfig(
                name="etf",
                variable_type=KernelBlockRole.ETF,
                components=[
                    MaternKernelComponentConfig(
                        dims=etf_feature_indices,
                        ard=True,
                        matern_nu=0.5,
                        use_outputscale=False,
                        lengthscale_policy=LengthscalePolicyConfig(
                            policy=GP_LENGTHSCALE_POLICY,
                        ),
                    )
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=True,
            ),
            KernelBlockConfig(
                name="macro",
                variable_type=KernelBlockRole.MACRO,
                components=[
                    MaternKernelComponentConfig(
                        dims=macro_feature_indices,
                        ard=True,
                        matern_nu=0.5,
                        use_outputscale=False,
                        lengthscale_policy=LengthscalePolicyConfig(
                            policy=GP_LENGTHSCALE_POLICY,
                        ),
                    ),
                    LinearKernelComponentConfig(
                        dims=macro_feature_indices,
                        use_outputscale=False,
                    ),
                    RQKernelComponentConfig(
                        dims=macro_feature_indices,
                        ard=True,
                        use_outputscale=False,
                        lengthscale_policy=LengthscalePolicyConfig(
                            policy=GP_LENGTHSCALE_POLICY,
                        ),
                    ),
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=True,
            ),
        ],
        global_structure=GlobalStructure.HIERARCHICAL,
        interaction_policy=InteractionPolicy.SPARSE,
    )

kernel_block_indices = build_feature_index_groups_from_blocks(
    feature_columns,
    block_columns=FEATURE_BLOCK_COLUMNS,
    blocks=("time", "etf", "macro"),
)

mean_config = MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT)

model = build_multitask_gp(
    train_X=Xn,
    train_Y=y,
    task_feature=-1,
    covar_config=covar_config,
    mean_config=mean_config,
    min_inferred_noise_level=GP_MIN_INFERRED_NOISE_LEVEL,
    outcome_transform=outcome_transform,
    input_transform=None,
    rank=1,
)

kernel_expression = build_kernel_expression(
    model.covar_module,
    block_indices=kernel_block_indices,
)
kernel_mermaid = build_kernel_mermaid(
    model.covar_module,
    block_indices=kernel_block_indices,
)

print("\n[gp] covariance expression:")
print(kernel_expression)
print("\n[gp] mermaid diagram markup:")
print(kernel_mermaid)
render_mermaid_in_notebook(kernel_mermaid)

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
