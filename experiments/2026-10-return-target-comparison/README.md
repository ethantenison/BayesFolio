# October 2026 Return-Target Comparison

Status: Completed on 2026-10-11

## Question

Should the monthly BayesFolio multitask GP predict adjusted total excess return,
price-only excess return, or separately modeled price and distribution
components that are recombined before portfolio optimization?

## Design

The experiment holds the accepted July 2026 multitask-GP architecture and
common covariates fixed while comparing three target formulations:

1. `total_return`: adjusted total excess return.
2. `price_only`: distribution-unadjusted price excess return.
3. `decomposed`: the same price model plus a separate distribution-contribution
   multitask GP; independent posterior draws are added before optimization.

The compact universe spans the requested asset classes:

- `SPY`: U.S. large-cap equity
- `IJR`: U.S. small-cap equity
- `IEF`: U.S. fixed income
- `VEA`: developed international equity
- `BNDX`: developed international fixed income

The evaluation uses the established 24 realized monthly windows from 2024-05-31
through 2026-04-30. Each window uses an expanding training set beginning with
the frozen March 2021 feature panel. All prediction and portfolio results are
evaluated against realized **total** excess return; price-only native-target
metrics are also retained to distinguish an easier statistical target from the
actual investment payoff.

## Outputs

The runner writes a replayable input panel and durable JSON evidence:

- `manifest.json`
- `gp_config.json`
- `prediction_results.json`
- `uncertainty_results.json`
- `portfolio_results.json`
- `input_panel.parquet`
- `market_data_snapshot.csv`
- detailed prediction, interval, weight, return, and fit-diagnostic CSV files

Generated exploratory runs belong under `outputs/`. The selected final run is
copied to `evidence/` so the decision record can be versioned.

## Result

The selected evidence supports retaining adjusted total excess return as the
production target. It produced the best prediction RMSE/MAE, the best proper
uncertainty scores, and the highest cumulative return among the GP setups.
Price-only return was worse across the primary comparisons. Decomposition
improved Sharpe ratio and drawdown, but had lower return, weaker prediction and
uncertainty scores, and higher turnover. Equal weight remained the strongest
risk-adjusted benchmark in this deliberately small diagnostic universe.

See [`decision_log.md`](decision_log.md) for the full comparison and caveats,
and `evidence/20261011_three_target_comparison_final/` for the reproducible
manifest and result files.

## Run

Initial data construction and experiment:

```bash
uv run --locked python experiments/2026-10-return-target-comparison/run_return_target_comparison.py \
  --output-dir experiments/2026-10-return-target-comparison/evidence/20261011_three_target_comparison_final
```

Offline replay from the frozen panel:

```bash
uv run --locked python experiments/2026-10-return-target-comparison/run_return_target_comparison.py \
  --input-panel experiments/2026-10-return-target-comparison/evidence/20261011_three_target_comparison_final/input_panel.parquet \
  --output-dir /tmp/bayesfolio-return-target-replay
```
