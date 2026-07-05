# Portfolio Optimization Walk-Forward Report

Run directory: `/Users/et/Documents/BayesFolio/experiments/2026-07-us-equity-etf-features/runs/20260705T1540Z_block_volatility_regime_signed_lkj_rank2_6w`
Portfolio construction dates: `6` monthly_bme windows from `2026-01-30` to `2026-06-30` (`6` realized).
Starting value: `$10,000`.

## Strategy Summary

| strategy                             | n_rebalances | cumulative_return | cagr   | annualized_vol | sharpe | max_drawdown | terminal_value | mean_period_return | mean_monthly_return | hit_rate | avg_turnover | max_weight | mean_ic | median_ic |
| ------------------------------------ | ------------ | ----------------- | ------ | -------------- | ------ | ------------ | -------------- | ------------------ | ------------------- | -------- | ------------ | ---------- | ------- | --------- |
| gp_scenarios_riskfolio               | 6.0000       | 0.0631            | 0.1302 | 0.1615         | 0.8061 | -0.0520      | 10631.1507     | 0.0113             | 0.0113              | 0.3333   | 0.2647       | 0.3500     | 0.1667  | 0.3000    |
| historical_y_ewma2_gerber1_riskfolio | 6.0000       | 0.0698            | 0.1445 | 0.1584         | 0.9126 | -0.0520      | 10698.3446     | 0.0123             | 0.0123              | 0.3333   | 0.2875       | 0.3500     |         |           |
| equal_weight                         | 6.0000       | 0.0908            | 0.1899 | 0.1681         | 1.1295 | -0.0522      | 10908.2386     | 0.0157             | 0.0157              | 0.5000   | 0.1667       | 0.2500     |         |           |
| schwab_moderate_aggressive_static    | 6.0000       | 0.0790            | 0.1643 | 0.1569         | 1.0470 | -0.0496      | 10790.1161     | 0.0138             | 0.0138              | 0.3333   | 0.1667       | 0.4500     |         |           |

## Notes

- `MGK` and `BND` were included in GP fitting and scenario generation, then excluded from final weights.
- Training history: `48` rolling calendar months before each window.
- `gp_scenarios_riskfolio` experiment variant: `signed_lkj_eta_2`. Control is positive beta task covariance, lengthscale-only time modulation, rank 5.
- `historical_y_ewma2_gerber1_riskfolio` ignores GP predictions and optimizes directly on historical `y_excess_lead`, Sharpe, and CVaR.
- `equal_weight` is the additional baseline requested for portfolio-run comparison.
- `schwab_moderate_aggressive_static` is a fixed target-weight benchmark using an 80/15/5 stock/bond/cash Schwab-style Moderate Aggressive allocation. The 5% cash sleeve is written as `CASH` with zero realized return.
- IRA context: turnover is tracked as a stability diagnostic, not as a tax-cost veto.
- No transaction costs, taxes, slippage, or liquidity filters are applied in this end-to-end check.
