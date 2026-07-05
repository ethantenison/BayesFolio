# June 2026 Portfolio Optimization Walk-Forward

This experiment tests the end-to-end monthly portfolio loop over 12 rebalance
dates. It compares:

- `gp_scenarios_riskfolio`: multitask GP posterior scenarios optimized with the
  June Riskfolio setup.
- `historical_y_ewma2_riskfolio`: historical `y_excess_lead` returns optimized
  with Riskfolio using EWMA2 expected returns/covariance, Sharpe objective, and
  CVaR risk.
- `equal_weight`: equal-weight monthly rebalance baseline.

`MGK` and `BND` are included in the GP training/task universe as helper assets
but excluded from final portfolio weights and realized portfolio performance.

The script writes a run manifest, monthly weights/returns, metric summaries,
prediction diagnostics, posterior scenario snapshots, and equity/drawdown plots
to a versioned run directory under `outputs/runs/`.
