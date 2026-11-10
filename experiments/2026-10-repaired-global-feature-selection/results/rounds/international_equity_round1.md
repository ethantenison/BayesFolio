# International Equity round 1

Cumulative configurations: 149 -> 161.

Fixed-reference hypervolume: 0.515212263 -> 0.595004502.

Stalled: False; two consecutive stalled rounds: False.

| configuration                     | hypothesis_name                              | base_configuration                | category                            |    delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:----------------------------------|:---------------------------------------------|:----------------------------------|:------------------------------------|------------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| international_equity_72b455cbe8ce | high_sharpe_without_gold_oil_ratio           | international_equity_0d2237db58e8 | exploitation_pruning                | -0.05       |  0.0546595  |    -0.0444782  |                   -1 |              0 |               0 |                  1 | False                 |
| international_equity_2c45dc291f0a | high_sharpe_without_dxy_level                | international_equity_0d2237db58e8 | exploitation_pruning                | -0.0194444  | -0.0112352  |    -0.00361942 |                   -1 |              1 |               2 |                  1 | False                 |
| international_equity_baaff7ca24fa | high_sharpe_without_hy_spread_change         | international_equity_0d2237db58e8 | exploitation_credit_pruning         | -0.0166667  | -0.0244685  |     0.00878711 |                   -1 |              1 |               2 |                  2 | True                  |
| international_equity_c0d7d8de042b | high_sharpe_with_reduced_tips_representation | international_equity_0d2237db58e8 | exploitation_block_pruning          | -0.05       | -0.0238363  |    -0.0413254  |                   -2 |              1 |               2 |                  0 | False                 |
| international_equity_c8b9d5ec7552 | high_ic_without_six_month_momentum           | international_equity_c6c6899c3500 | exploitation_local_ablation         | -0.0444444  | -0.0257688  |     0.00330429 |                   -1 |              0 |               2 |                  2 | False                 |
| international_equity_dd465a0a6793 | risk_front_without_vix_slope_change          | international_equity_d761551f5d77 | exploitation_pruning                |  0.0611111  | -0.0302593  |    -0.0148509  |                   -1 |              3 |               3 |                  1 | True                  |
| international_equity_4c2b72c2c055 | risk_front_without_tlt_volatility_proxy      | international_equity_d761551f5d77 | exploitation_pruning                |  0.0222222  | -0.0259872  |     0.0176641  |                   -1 |              3 |               3 |                  2 | True                  |
| international_equity_5fead71b4d35 | risk_front_without_absolute_trend_slope      | international_equity_d761551f5d77 | exploitation_local_ablation         |  0.0194444  | -0.0191818  |    -0.0355545  |                   -1 |              3 |               2 |                  0 | False                 |
| international_equity_7b43e57c804c | risk_front_plus_oil_return_only              | international_equity_d761551f5d77 | frontier_bridge                     | -0.00277778 | -0.0420277  |     0.00139462 |                    1 |              2 |               3 |                  1 | True                  |
| international_equity_c8293843148d | risk_front_with_alternative_local_basis      | international_equity_d761551f5d77 | exploration_representation_transfer | -0.105556   | -0.00160938 |     0.00566025 |                    0 |              0 |               2 |                  2 | False                 |
| international_equity_abc793f72858 | risk_front_plus_etf_realized_volatility      | international_equity_d761551f5d77 | exploration_uncertainty_information | -0.025      |  0.0093177  |     0.0228598  |                    1 |              1 |               0 |                  3 | True                  |
| international_equity_c6bca2241a06 | risk_front_plus_recent_excess_return_lag     | international_equity_d761551f5d77 | exploration_conditional_lag_value   | -0.105556   |  0.00570434 |    -0.0690828  |                    1 |              0 |               2 |                  0 | False                 |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
