# Portfolio Optimization Walk-Forward Report

Run directory: `/Users/et/Documents/BayesFolio/experiments/2026-06-portfolio-optimization/outputs/runs/20260616_signed_lkj_eta_2_lengthscale_floor_002_portfolio`
Rebalances: `12` monthly windows from `2025-05-30` to `2026-04-30`.
Starting value: `$10,000`.

## Strategy Summary

| strategy                     | n_rebalances | cumulative_return | cagr   | annualized_vol | sharpe | max_drawdown | terminal_value | mean_monthly_return | hit_rate | avg_turnover | max_weight | mean_ic | median_ic |
| ---------------------------- | ------------ | ----------------- | ------ | -------------- | ------ | ------------ | -------------- | ------------------- | -------- | ------------ | ---------- | ------- | --------- |
| gp_scenarios_riskfolio       | 12.0000      | 0.1997            | 0.1997 | 0.0948         | 2.1064 | -0.0473      | 11996.6522     | 0.0157              | 0.7500   | 0.3435       | 0.1862     | 0.2559  | 0.3574    |
| historical_y_ewma2_riskfolio | 12.0000      | 0.1809            | 0.1809 | 0.0996         | 1.8158 | -0.0611      | 11809.3518     | 0.0144              | 0.7500   | 0.1351       | 0.1719     |         |           |
| equal_weight                 | 12.0000      | 0.1493            | 0.1493 | 0.0882         | 1.6933 | -0.0533      | 11493.1155     | 0.0120              | 0.7500   | 0.0833       | 0.0625     |         |           |

## Notes

- `MGK` and `BND` were included in GP fitting and scenario generation, then excluded from final weights.
- `gp_scenarios_riskfolio` experiment variant: `signed_lkj_eta_2_lengthscale_floor`. Control is positive beta task covariance, lengthscale-only time modulation, rank 5.
- `historical_y_ewma2_riskfolio` ignores GP predictions and optimizes directly on historical `y_excess_lead` with EWMA2, Sharpe, and CVaR.
- `equal_weight` is the additional baseline requested for portfolio-run comparison.
- IRA context: turnover is tracked as a stability diagnostic, not as a tax-cost veto.
- No transaction costs, taxes, slippage, or liquidity filters are applied in this end-to-end check.
