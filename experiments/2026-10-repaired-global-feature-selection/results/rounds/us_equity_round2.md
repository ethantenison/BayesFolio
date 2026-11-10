# Us Equity round 2

Cumulative configurations: 23 -> 35.

Fixed-reference hypervolume: 0.708220358 -> 0.834919079.

Stalled: False; two consecutive stalled rounds: False.

| configuration          | hypothesis_name                               | base_configuration     | category                             |     delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:-----------------------|:----------------------------------------------|:-----------------------|:-------------------------------------|-------------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| us_equity_6bba78566e06 | rank_free_without_six_month_momentum          | us_equity_5fba6b4b201b | exploitation_local_pruning           | -0.0361111   |  0.029669   |    -0.0788839  |                   -1 |              1 |               0 |                  1 | False                 |
| us_equity_d0a35b28f60b | rank_free_without_trend_slope                 | us_equity_5fba6b4b201b | exploitation_local_pruning           |  0.0222222   |  0.012175   |    -0.0737405  |                   -1 |              2 |               2 |                  1 | True                  |
| us_equity_f9ec66acb71f | rank_free_without_skip_momentum               | us_equity_5fba6b4b201b | exploitation_local_pruning           | -0.136111    |  0.00135009 |    -0.185842   |                   -1 |              0 |               2 |                  0 | False                 |
| us_equity_fc5807bf3a73 | rank_free_plus_lag                            | us_equity_5fba6b4b201b | conditional_lag_bridge               | -0.122222    |  0.00578478 |    -0.173576   |                    1 |              0 |               1 |                  0 | False                 |
| us_equity_617e8bd4bada | rank_free_without_credit_change               | us_equity_5fba6b4b201b | exploitation_macro_pruning           |  7.40149e-17 |  0.0821465  |    -0.136163   |                   -1 |              1 |               0 |                  0 | False                 |
| us_equity_23a11358aeee | rank_free_credit_level_instead_of_change      | us_equity_5fba6b4b201b | exploration_credit_representation    |  0.075       |  0.125351   |    -0.00882964 |                    0 |              3 |               0 |                  1 | True                  |
| us_equity_07841223ab03 | rank_free_credit_level_instead_of_zscore      | us_equity_5fba6b4b201b | exploration_credit_representation    |  0.00833333  |  0.328586   |    -0.0922172  |                    0 |              2 |               0 |                  0 | False                 |
| us_equity_46cc53338e30 | rank_free_monthly_instead_of_annual_inflation | us_equity_5fba6b4b201b | exploration_inflation_representation | -0.191667    | -0.00076959 |    -0.151084   |                    0 |              0 |               2 |                  0 | True                  |
| us_equity_35ccd91a15a2 | rank_free_nominal_yield_instead_of_curve      | us_equity_5fba6b4b201b | exploration_rate_representation      | -0.280556    |  0.192327   |    -0.360143   |                    0 |              0 |               0 |                  0 | False                 |
| us_equity_ee8ad358e762 | rank_free_without_six_month_momentum_plus_lag | us_equity_5fba6b4b201b | exploration_local_substitution       | -0.158333    |  0.0313197  |    -0.206734   |                    0 |              0 |               0 |                  0 | False                 |
| us_equity_85d409a7acad | rank_free_plus_local_volatility               | us_equity_5fba6b4b201b | exploration_uncertainty_information  | -0.0722222   |  0.026193   |    -0.0237382  |                    1 |              0 |               0 |                  0 | False                 |
| us_equity_51b0f3f6c657 | rank_free_plus_market_flow                    | us_equity_5fba6b4b201b | exploration_cross_family_transfer    | -0.0416667   |  0.0281182  |    -0.217738   |                    1 |              1 |               0 |                  0 | False                 |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
