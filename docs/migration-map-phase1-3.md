# BayesFolio Reorganization Migration Map (Phases 1-3)

This document records module moves completed during the package reorganization.

## Phase 4 package polish

- Added explicit package export surfaces:
	- `bayesfolio.__all__`
	- `bayesfolio.features.__all__`
	- `bayesfolio.optimization.__all__`
	- `bayesfolio.schemas.__all__`
	- `bayesfolio.schemas.configs.__all__`
- Added top-level migration reference in `README.md` under "Migration Notes".

## Core module moves

- `bayesfolio.asset_prices` -> `bayesfolio.features.asset_prices`
- `bayesfolio.market_fundamentals` -> `bayesfolio.features.market_fundamentals`
- `bayesfolio.gp_data_prep` -> `bayesfolio.features.gp_data_prep`
- `bayesfolio.evaluate` -> `bayesfolio.optimization.evaluate`
- `bayesfolio.backtesting` -> `bayesfolio.optimization.backtesting`
- `bayesfolio.configs` -> `bayesfolio.schemas.configs.core`

## Legacy module moves

- `bayesfolio.models.old_kernels` -> `bayesfolio.ml.legacy.old_kernels`
- `bayesfolio.models.old_gp` -> `bayesfolio.ml.legacy.old_gp`
- `bayesfolio.kernels` -> `bayesfolio.ml.legacy.root_kernels`

## Experiment and artifact relocation

- Dated MLflow runner scripts moved from `bayesfolio/mlflow/*.py` to `experiments/mlflow/*.py`
- Dated CSV snapshots moved from `bayesfolio/mlflow/2026*.csv` to `experiments/mlflow/data_snapshots/`
- Legacy nested artifact directory moved from `bayesfolio/mlflow/mlflow/artifacts` to `experiments/mlflow/legacy_store/nested_artifacts`
- Legacy run-store directory moved from `bayesfolio/mlflow/mlruns` to `experiments/mlflow/legacy_store/mlruns`

## Notes

- No compatibility shims were added.
- Repository-wide imports were updated to new paths in active scripts and experiments.
- If an old import path appears in a local notebook cache, restart the kernel and rerun imports.
