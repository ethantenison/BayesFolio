# Portfolio Optimization Walk-Forward Report

Run directory: `/Users/et/Documents/BayesFolio/experiments/2026-06-portfolio-optimization/outputs/runs/20260616_gp_scenario_portfolio`
Rebalances: `12` monthly windows from `2025-05-30` to `2026-04-30`.
Starting value: `$10,000`.

## Strategy Summary

| strategy                     | n_rebalances | cumulative_return | cagr   | annualized_vol | sharpe | max_drawdown | terminal_value | mean_monthly_return | hit_rate | avg_turnover | max_weight | mean_ic | median_ic |
| ---------------------------- | ------------ | ----------------- | ------ | -------------- | ------ | ------------ | -------------- | ------------------- | -------- | ------------ | ---------- | ------- | --------- |
| gp_scenarios_riskfolio       | 12.0000      | 0.1898            | 0.1898 | 0.0971         | 1.9554 | -0.0556      | 11897.8692     | 0.0150              | 0.8333   | 0.2288       | 0.1892     | 0.3708  | 0.3941    |
| historical_y_ewma2_riskfolio | 12.0000      | 0.1809            | 0.1809 | 0.0996         | 1.8158 | -0.0611      | 11809.3518     | 0.0144              | 0.7500   | 0.1351       | 0.1719     |         |           |
| equal_weight                 | 12.0000      | 0.1493            | 0.1493 | 0.0882         | 1.6933 | -0.0533      | 11493.1155     | 0.0120              | 0.7500   | 0.0833       | 0.0625     |         |           |

## Notes

- `MGK` and `BND` were included in GP fitting and scenario generation, then excluded from final weights.
- `gp_scenarios_riskfolio` uses the last multitask GP configuration: positive beta task covariance, lengthscale-only time modulation, rank 5.
- `historical_y_ewma2_riskfolio` ignores GP predictions and optimizes directly on historical `y_excess_lead` with EWMA2, Sharpe, and CVaR.
- `equal_weight` is the additional baseline requested for portfolio-run comparison.
- No transaction costs, taxes, slippage, or liquidity filters are applied in this first end-to-end check.
