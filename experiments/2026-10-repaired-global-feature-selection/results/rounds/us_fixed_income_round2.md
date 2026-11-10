# Us Fixed Income round 2

Cumulative configurations: 81 -> 93.

Fixed-reference hypervolume: 0.431890306 -> 0.483386142.

Stalled: False; two consecutive stalled rounds: False.

| configuration                | hypothesis_name                               | base_configuration           | category                          |    delta_ic |    delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:-----------------------------|:----------------------------------------------|:-----------------------------|:----------------------------------|------------:|-------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| us_fixed_income_553096eb50bd | rank_free_credit_without_lag                  | us_fixed_income_f081d2651e89 | conditional_lag_ablation          | -0.0611111  | -0.0220647   |   -0.0440922   |                   -1 |              0 |               3 |                  0 | False                 |
| us_fixed_income_d048ca81b59f | rank_free_credit_without_long_momentum        | us_fixed_income_f081d2651e89 | exploitation_combined_pruning     | -0.0388889  | -0.0110397   |   -0.0215522   |                   -1 |              1 |               3 |                  1 | False                 |
| us_fixed_income_c2fef69db5f6 | rank_free_credit_without_acceleration         | us_fixed_income_f081d2651e89 | exploration_local_pruning         |  0.00277778 | -0.00491281  |    0.0262144   |                   -1 |              2 |               2 |                  1 | True                  |
| us_fixed_income_59f981f1532f | rank_free_credit_without_lag_or_long_momentum | us_fixed_income_f081d2651e89 | exploration_interaction           | -0.0583333  | -0.0271177   |   -0.0536801   |                   -2 |              0 |               2 |                  0 | False                 |
| us_fixed_income_843c0e47174a | pruned_no_lag_rate_plus_credit_zscore         | us_fixed_income_982e0458d34f | frontier_bridge                   |  0.0277778  | -0.0439071   |   -0.0123913   |                    1 |              3 |               2 |                  1 | False                 |
| us_fixed_income_a21c49aac2a0 | pruned_no_lag_rate_plus_local_volatility      | us_fixed_income_982e0458d34f | frontier_bridge                   | -0.00833333 |  0.0250339   |    0.00807701  |                    1 |              1 |               1 |                  2 | True                  |
| us_fixed_income_f7fc90961e04 | pruned_rate_plus_lag                          | us_fixed_income_982e0458d34f | conditional_lag_addition          | -0.0611111  | -0.0162967   |   -0.0995546   |                    1 |              0 |               3 |                  0 | False                 |
| us_fixed_income_1104d9eef71f | pruned_no_lag_rate_without_acceleration       | us_fixed_income_982e0458d34f | exploitation_local_pruning        |  0.0333333  | -0.0100897   |    0.00511073  |                   -1 |              3 |               1 |                  2 | False                 |
| us_fixed_income_d642e6ed3a18 | pruned_no_lag_rate_without_ma_signal          | us_fixed_income_982e0458d34f | exploitation_local_pruning        | -0.0222222  |  0.0225455   |   -0.0104978   |                   -1 |              1 |               1 |                  1 | False                 |
| us_fixed_income_4eba1636f6c5 | reduced_curve_density_without_lag             | us_fixed_income_ba63874f87e3 | conditional_lag_ablation          |  0.0722222  |  0.000967491 |    0.0585213   |                   -1 |              3 |               1 |                  3 | True                  |
| us_fixed_income_4d815a6b9483 | reduced_curve_density_without_autocorrelation | us_fixed_income_ba63874f87e3 | exploitation_local_pruning        |  0.025      |  0.0041496   |   -0.000504803 |                   -1 |              3 |               0 |                  1 | False                 |
| us_fixed_income_89293fc088d7 | pruned_no_lag_rate_plus_oil                   | us_fixed_income_982e0458d34f | exploration_cross_family_transfer |  0.0416667  | -0.155328    |    0.00556273  |                    1 |              3 |               3 |                  1 | True                  |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
