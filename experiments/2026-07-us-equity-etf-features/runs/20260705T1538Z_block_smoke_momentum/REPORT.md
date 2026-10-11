# Portfolio Optimization Walk-Forward Report

Run directory: `/Users/et/Documents/BayesFolio/experiments/2026-07-us-equity-etf-features/runs/20260705T1538Z_block_smoke_momentum`
Portfolio construction dates: `1` monthly_bme windows from `2026-06-30` to `2026-06-30` (`1` realized).
Starting value: `$10,000`.

## Strategy Summary

| strategy                             | n_rebalances | cumulative_return | cagr    | annualized_vol | sharpe | max_drawdown | terminal_value | mean_period_return | mean_monthly_return | hit_rate | avg_turnover | max_weight | mean_ic | median_ic |
| ------------------------------------ | ------------ | ----------------- | ------- | -------------- | ------ | ------------ | -------------- | ------------------ | ------------------- | -------- | ------------ | ---------- | ------- | --------- |
| gp_scenarios_riskfolio               | 1.0000       | -0.0027           | -0.0320 | 0.0000         |        | 0.0000       | 9972.9044      | -0.0027            | -0.0027             | 0.0000   | 1.0000       | 0.3500     | -1.0000 | -1.0000   |
| historical_y_ewma2_gerber1_riskfolio | 1.0000       | -0.0014           | -0.0172 | 0.0000         |        | 0.0000       | 9985.5905      | -0.0014            | -0.0014             | 0.0000   | 1.0000       | 0.3500     |         |           |
| equal_weight                         | 1.0000       | -0.0021           | -0.0244 | 0.0000         |        | 0.0000       | 9979.4562      | -0.0021            | -0.0021             | 0.0000   | 1.0000       | 0.2500     |         |           |
| schwab_moderate_aggressive_static    | 1.0000       | -0.0018           | -0.0216 | 0.0000         |        | 0.0000       | 9981.8464      | -0.0018            | -0.0018             | 0.0000   | 1.0000       | 0.4500     |         |           |

## Notes

- `MGK` and `BND` were included in GP fitting and scenario generation, then excluded from final weights.
- Training history: `48` rolling calendar months before each window.
- `gp_scenarios_riskfolio` experiment variant: `signed_lkj_eta_2`. Control is positive beta task covariance, lengthscale-only time modulation, rank 5.
- `historical_y_ewma2_gerber1_riskfolio` ignores GP predictions and optimizes directly on historical `y_excess_lead`, Sharpe, and CVaR.
- `equal_weight` is the additional baseline requested for portfolio-run comparison.
- `schwab_moderate_aggressive_static` is a fixed target-weight benchmark using an 80/15/5 stock/bond/cash Schwab-style Moderate Aggressive allocation. The 5% cash sleeve is written as `CASH` with zero realized return.
- IRA context: turnover is tracked as a stability diagnostic, not as a tax-cost veto.
- No transaction costs, taxes, slippage, or liquidity filters are applied in this end-to-end check.
