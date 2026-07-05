# Portfolio Optimization Walk-Forward Report

Run directory: `/Users/et/Documents/BayesFolio/experiments/2026-07-us-equity-etf-features/runs/20260705T1513Z_first_round_signed_lkj_noise_floor_rank2_3w`
Portfolio construction dates: `3` monthly_bme windows from `2026-04-30` to `2026-06-30` (`3` realized).
Starting value: `$10,000`.

## Strategy Summary

| strategy                             | n_rebalances | cumulative_return | cagr   | annualized_vol | sharpe | max_drawdown | terminal_value | mean_period_return | mean_monthly_return | hit_rate | avg_turnover | max_weight | mean_ic | median_ic |
| ------------------------------------ | ------------ | ----------------- | ------ | -------------- | ------ | ------------ | -------------- | ------------------ | ------------------- | -------- | ------------ | ---------- | ------- | --------- |
| gp_scenarios_riskfolio               | 3.0000       | 0.0521            | 0.2253 | 0.0875         | 2.5766 | -0.0027      | 10521.1675     | 0.0174             | 0.0174              | 0.6667   | 0.4815       | 0.3500     | 0.1333  | 0.4000    |
| historical_y_ewma2_gerber1_riskfolio | 3.0000       | 0.0418            | 0.1780 | 0.0958         | 1.8587 | -0.0106      | 10418.1347     | 0.0141             | 0.0141              | 0.3333   | 0.3509       | 0.3500     |         |           |
| equal_weight                         | 3.0000       | 0.0499            | 0.2151 | 0.0796         | 2.7035 | -0.0021      | 10499.2220     | 0.0166             | 0.0166              | 0.6667   | 0.3333       | 0.2500     |         |           |
| schwab_moderate_aggressive_static    | 3.0000       | 0.0426            | 0.1818 | 0.0761         | 2.3893 | -0.0026      | 10426.4537     | 0.0143             | 0.0143              | 0.3333   | 0.3333       | 0.4500     |         |           |

## Notes

- `MGK` and `BND` were included in GP fitting and scenario generation, then excluded from final weights.
- Training history: `48` rolling calendar months before each window.
- `gp_scenarios_riskfolio` experiment variant: `signed_lkj_eta_2_task_noise_floor`. Control is positive beta task covariance, lengthscale-only time modulation, rank 5.
- `historical_y_ewma2_gerber1_riskfolio` ignores GP predictions and optimizes directly on historical `y_excess_lead`, Sharpe, and CVaR.
- `equal_weight` is the additional baseline requested for portfolio-run comparison.
- `schwab_moderate_aggressive_static` is a fixed target-weight benchmark using an 80/15/5 stock/bond/cash Schwab-style Moderate Aggressive allocation. The 5% cash sleeve is written as `CASH` with zero realized return.
- IRA context: turnover is tracked as a stability diagnostic, not as a tax-cost veto.
- No transaction costs, taxes, slippage, or liquidity filters are applied in this end-to-end check.
