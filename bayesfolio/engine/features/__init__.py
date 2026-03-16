"""Data ingestion and feature preparation APIs.

This package owns feature fetch/engineering helpers, dataset assembly, and the
canonical provider wiring used by BayesFolio feature-building workflows.
"""

from bayesfolio.engine.features.asset_prices import (
    add_cross_sectional_momentum_rank,
    build_long_panel,
    cross_sectional_zscore,
    fetch_etf_features,
)
from bayesfolio.engine.features.dataset_builder import FeatureProviders, build_features_dataset
from bayesfolio.engine.features.gp_data_prep import (
    prepare_multitask_gp_data,
    prepare_multitask_gp_data_with_task_feature,
)
from bayesfolio.engine.features.market_fundamentals import (
    fetch_core_global_macro,
    fetch_enhanced_macro_features,
    fetch_macro_features,
)
from bayesfolio.engine.features.providers import make_default_feature_providers
from bayesfolio.engine.features.universe_loader import build_universe_snapshot

__all__ = [
    "build_universe_snapshot",
    "build_features_dataset",
    "FeatureProviders",
    "add_cross_sectional_momentum_rank",
    "build_long_panel",
    "cross_sectional_zscore",
    "fetch_etf_features",
    "prepare_multitask_gp_data",
    "prepare_multitask_gp_data_with_task_feature",
    "fetch_core_global_macro",
    "fetch_enhanced_macro_features",
    "fetch_macro_features",
    "make_default_feature_providers",
]
