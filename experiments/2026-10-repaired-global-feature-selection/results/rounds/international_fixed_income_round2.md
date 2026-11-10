# International Fixed Income round 2

Cumulative configurations: 101 -> 113.

Fixed-reference hypervolume: 0.945671955 -> 1.004573515.

Stalled: False; two consecutive stalled rounds: False.

| configuration                           | hypothesis_name                    | base_configuration                      | category                        |   delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:----------------------------------------|:-----------------------------------|:----------------------------------------|:--------------------------------|-----------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| international_fixed_income_9f5f8487e05e | no_hy_without_flow                 | international_fixed_income_775c0e6b8de9 | exploitation_combined_pruning   | -0.0625    | -0.00135787 |    0.00549819  |                   -1 |              0 |               2 |                  2 | True                  |
| international_fixed_income_20bd7258ae05 | no_hy_without_global_rank          | international_fixed_income_775c0e6b8de9 | exploitation_combined_pruning   | -0.0486111 |  0.0138159  |    0.00422703  |                   -1 |              1 |               0 |                  3 | False                 |
| international_fixed_income_ac22b1ea6705 | no_hy_without_acceleration         | international_fixed_income_775c0e6b8de9 | exploitation_combined_pruning   | -0.0486111 |  0.0231112  |   -0.00335734  |                   -1 |              0 |               0 |                  2 | False                 |
| international_fixed_income_526bcca833f6 | no_hy_without_long_momentum        | international_fixed_income_775c0e6b8de9 | exploitation_local_pruning      |  0         |  0.0115726  |    0.00639233  |                   -1 |              2 |               1 |                  2 | True                  |
| international_fixed_income_cc8af3b7a3a4 | no_hy_without_flow_or_rank         | international_fixed_income_775c0e6b8de9 | exploration_pruning_interaction | -0.0833333 |  0.00678593 |    0.00273632  |                   -2 |              0 |               1 |                  2 | False                 |
| international_fixed_income_8ff6f583776b | no_hy_without_flow_or_acceleration | international_fixed_income_775c0e6b8de9 | exploration_pruning_interaction | -0.0972222 | -0.00421793 |   -0.00185304  |                   -2 |              1 |               2 |                  1 | True                  |
| international_fixed_income_12f65dc2d4e4 | no_hy_without_rank_or_acceleration | international_fixed_income_775c0e6b8de9 | exploration_pruning_interaction |  0.0416667 |  0.0223096  |    0.0296673   |                   -2 |              3 |               0 |                  3 | True                  |
| international_fixed_income_396e6586950a | no_hy_without_monthly_inflation    | international_fixed_income_775c0e6b8de9 | exploration_macro_pruning       |  0.0138889 |  0.0265788  |    0.000480455 |                   -1 |              2 |               0 |                  2 | False                 |
| international_fixed_income_242615071b6b | no_hy_without_em_fx                | international_fixed_income_775c0e6b8de9 | exploration_fx_pruning          | -0.229167  |  0.015527   |   -0.0210037   |                   -1 |              0 |               1 |                  0 | False                 |
| international_fixed_income_2000432e3373 | no_hy_without_oil                  | international_fixed_income_775c0e6b8de9 | exploration_commodity_pruning   | -0.0694444 |  0.0367996  |   -0.00650669  |                   -1 |              0 |               0 |                  1 | False                 |
| international_fixed_income_4e8c828f6d80 | no_hy_plus_lag                     | international_fixed_income_775c0e6b8de9 | conditional_lag_addition        | -0.159722  |  0.0281333  |   -0.0177358   |                    1 |              0 |               0 |                  0 | False                 |
| international_fixed_income_8a22a6c8c795 | hy_level_only_without_flow         | international_fixed_income_b62ac17a6009 | frontier_bridge                 | -0.0625    | -0.00743746 |    0.00563668  |                   -1 |              1 |               2 |                  2 | True                  |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
