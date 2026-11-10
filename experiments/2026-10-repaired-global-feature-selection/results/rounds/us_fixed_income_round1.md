# Us Fixed Income round 1

Cumulative configurations: 69 -> 81.

Fixed-reference hypervolume: 0.360575223 -> 0.431890306.

Stalled: False; two consecutive stalled rounds: False.

| configuration                | hypothesis_name                                     | base_configuration           | category                                   |    delta_ic |    delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:-----------------------------|:----------------------------------------------------|:-----------------------------|:-------------------------------------------|------------:|-------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| us_fixed_income_0f309a3be154 | credit_sharpe_parent_without_lag                    | us_fixed_income_cacbbd05e688 | exploitation_conditional_lag_ablation      |  0.0166667  |  0.0273296   |    0.0106556   |                   -1 |              1 |               0 |                  2 | False                 |
| us_fixed_income_e6c66c10f20f | credit_sharpe_parent_without_long_momentum          | us_fixed_income_cacbbd05e688 | exploitation_local_ablation                | -0.0194444  |  0.00683566  |    0.000498936 |                   -1 |              1 |               1 |                  1 | True                  |
| us_fixed_income_f081d2651e89 | credit_sharpe_parent_without_global_rank            | us_fixed_income_cacbbd05e688 | exploitation_local_ablation                |  0.0972222  |  0.0252409   |    0.0362293   |                   -1 |              3 |               0 |                  3 | True                  |
| us_fixed_income_df7179f43dbc | credit_parent_with_shared_equity_local_basis        | us_fixed_income_cacbbd05e688 | exploration_shared_representation          | -0.0444444  |  0.012246    |   -0.0639063   |                   -2 |              1 |               1 |                  1 | False                 |
| us_fixed_income_a96aca07ab89 | no_lag_rate_parent_without_one_month_tips_return    | us_fixed_income_2a1e5cf172bf | exploitation_rate_pruning                  | -0.380556   |  0.0602295   |   -0.341407    |                   -1 |              0 |               1 |                  0 | False                 |
| us_fixed_income_408c8fb664f6 | no_lag_rate_parent_without_three_month_tips_proxy   | us_fixed_income_2a1e5cf172bf | exploitation_rate_pruning                  | -0.225      | -0.0376113   |   -0.198304    |                   -1 |              0 |               2 |                  0 | False                 |
| us_fixed_income_71e68626a46e | no_lag_rate_parent_without_nominal_yield            | us_fixed_income_2a1e5cf172bf | exploitation_rate_ablation                 | -0.361111   |  0.0854728   |   -0.0949224   |                   -1 |              0 |               0 |                  0 | False                 |
| us_fixed_income_982e0458d34f | no_lag_rate_parent_without_return_autocorrelation   | us_fixed_income_2a1e5cf172bf | exploitation_local_pruning                 |  0.00833333 | -0.045617    |    0.0150686   |                   -1 |              2 |               3 |                  3 | True                  |
| us_fixed_income_be1385fc1c2f | no_lag_rate_parent_plus_hy_zscore                   | us_fixed_income_2a1e5cf172bf | frontier_bridge                            |  0.00833333 | -0.128795    |   -0.0729019   |                    1 |              2 |               3 |                  0 | True                  |
| us_fixed_income_37d53e9ed186 | no_lag_rate_parent_plus_local_realized_volatility   | us_fixed_income_2a1e5cf172bf | exploration_uncertainty_information        |  0.0333333  | -0.0238914   |   -0.0154118   |                    1 |              2 |               2 |                  1 | True                  |
| us_fixed_income_a0a64f885cee | density_anchor_without_duplicate_nominal_yield_name | us_fixed_income_e449a29b091f | exploitation_exact_definition_redundancy   |  0.075      | -0.000175115 |    0.0148216   |                   -1 |              3 |               1 |                  2 | False                 |
| us_fixed_income_ba63874f87e3 | density_anchor_with_minimal_equivalent_curve_basis  | us_fixed_income_e449a29b091f | exploitation_representation_simplification |  0.158333   | -0.00769602  |    0.0233164   |                   -2 |              3 |               1 |                  2 | True                  |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
