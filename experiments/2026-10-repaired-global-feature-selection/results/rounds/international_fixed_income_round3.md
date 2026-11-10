# International Fixed Income round 3

Cumulative configurations: 113 -> 125.

Fixed-reference hypervolume: 1.004573515 -> 1.057468608.

Stalled: False; two consecutive stalled rounds: False.

| configuration                           | hypothesis_name                                             | base_configuration                      | category                            |   delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:----------------------------------------|:------------------------------------------------------------|:----------------------------------------|:------------------------------------|-----------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| international_fixed_income_373dc9719ca5 | two_horizon_no_hy_without_flow                              | international_fixed_income_12f65dc2d4e4 | exploitation_combined_pruning       | -0.0555556 | -0.0359705  |    -0.0192567  |                   -1 |              1 |               3 |                  0 | True                  |
| international_fixed_income_ade5ca6498cf | two_horizon_no_hy_without_long_momentum                     | international_fixed_income_12f65dc2d4e4 | exploitation_local_pruning          |  0.0208333 | -0.0189949  |    -0.0216823  |                   -1 |              1 |               3 |                  0 | True                  |
| international_fixed_income_4606efdcb93b | two_horizon_no_hy_without_annual_momentum                   | international_fixed_income_12f65dc2d4e4 | exploration_local_pruning           | -0.111111  | -0.00971355 |    -0.0323868  |                   -1 |              0 |               2 |                  0 | False                 |
| international_fixed_income_172c138a3bfd | two_horizon_no_hy_without_monthly_inflation                 | international_fixed_income_12f65dc2d4e4 | exploitation_macro_pruning          | -0.0277778 | -0.0120138  |    -0.0311174  |                   -1 |              1 |               3 |                  0 | False                 |
| international_fixed_income_7de592f7ab53 | two_horizon_no_hy_without_annual_inflation                  | international_fixed_income_12f65dc2d4e4 | exploration_macro_interaction       | -0.0972222 | -0.00759373 |    -0.0209585  |                   -1 |              1 |               3 |                  0 | False                 |
| international_fixed_income_4539c580d85a | two_horizon_no_hy_without_breadth_proxy                     | international_fixed_income_12f65dc2d4e4 | exploration_market_pruning          | -0.0555556 | -0.0180537  |    -0.0154729  |                   -1 |              1 |               2 |                  0 | False                 |
| international_fixed_income_27e3b8210acc | two_horizon_no_hy_without_market_return_proxy               | international_fixed_income_12f65dc2d4e4 | exploration_market_pruning          | -0.180556  | -0.00210324 |    -0.0280441  |                   -1 |              0 |               2 |                  0 | False                 |
| international_fixed_income_ed4f030d1f2e | two_horizon_no_hy_without_vix_level                         | international_fixed_income_12f65dc2d4e4 | exploration_risk_pruning            | -0.0138889 | -0.0228628  |    -0.0118691  |                   -1 |              1 |               3 |                  1 | True                  |
| international_fixed_income_fea7048e758e | two_horizon_no_hy_without_vix_slope                         | international_fixed_income_12f65dc2d4e4 | exploration_risk_pruning            | -0.0555556 | -0.0196047  |    -0.0191288  |                   -1 |              1 |               3 |                  0 | False                 |
| international_fixed_income_e0dc2db9f1bd | two_horizon_no_hy_without_vix_slope_zscore                  | international_fixed_income_12f65dc2d4e4 | exploration_risk_interaction        | -0.0347222 |  0.0319554  |    -0.0232003  |                   -1 |              1 |               0 |                  1 | False                 |
| international_fixed_income_89264c7e34df | two_horizon_no_hy_with_skip_annual_momentum                 | international_fixed_income_12f65dc2d4e4 | exploration_shared_representation   | -0.180556  |  0.00868478 |    -0.0249993  |                    0 |              0 |               0 |                  0 | False                 |
| international_fixed_income_e9e05d05fc25 | density_no_hy_no_flow_no_acceleration_without_long_momentum | international_fixed_income_8ff6f583776b | exploitation_density_branch_pruning |  0.111111  | -0.00920323 |     0.00346514 |                   -1 |              3 |               3 |                  3 | True                  |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
