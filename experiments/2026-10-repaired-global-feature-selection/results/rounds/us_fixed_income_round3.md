# Us Fixed Income round 3

Cumulative configurations: 93 -> 105.

Fixed-reference hypervolume: 0.483386142 -> 0.544561035.

Stalled: False; two consecutive stalled rounds: False.

| configuration                | hypothesis_name                         | base_configuration           | category                                |   delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:-----------------------------|:----------------------------------------|:-----------------------------|:----------------------------------------|-----------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| us_fixed_income_b4b6de59299e | oil_rate_without_acceleration           | us_fixed_income_89293fc088d7 | exploitation_local_pruning              | -0.0138889 | -0.00953853 |    -0.0317428  |                   -1 |              1 |               3 |                  1 | True                  |
| us_fixed_income_78c395992b67 | oil_rate_plus_lag                       | us_fixed_income_89293fc088d7 | conditional_lag_addition                |  0.0388889 |  0.0302988  |     0.0116993  |                    1 |              3 |               0 |                  2 | True                  |
| us_fixed_income_bfbfdcc83885 | oil_rate_plus_credit_zscore             | us_fixed_income_89293fc088d7 | frontier_bridge                         | -0.025     |  0.042302   |    -0.0405365  |                    1 |              1 |               0 |                  0 | False                 |
| us_fixed_income_e9e58816c053 | oil_rate_without_three_month_tips_proxy | us_fixed_income_89293fc088d7 | exploration_rate_pruning                | -0.183333  |  0.0377743  |    -0.20834    |                   -1 |              0 |               0 |                  0 | False                 |
| us_fixed_income_4ec4ea3a4985 | oil_rate_without_one_month_tips_return  | us_fixed_income_89293fc088d7 | exploration_rate_pruning                | -0.377778  |  0.247371   |    -0.231766   |                   -1 |              0 |               0 |                  0 | False                 |
| us_fixed_income_5f3d435c615e | oil_rate_without_nominal_yield          | us_fixed_income_89293fc088d7 | exploration_rate_interaction            | -0.263889  |  0.0716174  |    -0.0794122  |                   -1 |              0 |               0 |                  0 | False                 |
| us_fixed_income_3199c073b636 | oil_rate_with_skip_annual_momentum      | us_fixed_income_89293fc088d7 | exploration_shared_local_representation | -0.0555556 | -0.0174803  |    -0.0407907  |                    0 |              0 |               2 |                  1 | True                  |
| us_fixed_income_ec28732828b9 | five_input_credit_without_lag           | us_fixed_income_c2fef69db5f6 | conditional_lag_ablation                | -0.0722222 |  0.0167008  |    -0.0576837  |                   -1 |              0 |               1 |                  0 | False                 |
| us_fixed_income_4eaa93896507 | five_input_credit_without_long_momentum | us_fixed_income_c2fef69db5f6 | exploitation_local_pruning              | -0.0833333 | -0.0321753  |    -0.0373958  |                   -1 |              0 |               3 |                  0 | False                 |
| us_fixed_income_1847dce24fe4 | five_input_credit_plus_oil              | us_fixed_income_c2fef69db5f6 | frontier_bridge                         |  0.0333333 | -0.0369768  |    -0.122097   |                    1 |              3 |               3 |                  0 | False                 |
| us_fixed_income_4e8b02eef7f0 | five_input_credit_plus_local_volatility | us_fixed_income_c2fef69db5f6 | exploration_uncertainty_information     | -0.0277778 | -0.00683106 |    -0.0234939  |                    1 |              1 |               2 |                  1 | True                  |
| us_fixed_income_efb8a2a0fb12 | no_lag_reduced_curve_density_plus_oil   | us_fixed_income_4eba1636f6c5 | frontier_bridge                         |  0.0361111 |  0.0136552  |     0.00490213 |                    1 |              3 |               1 |                  1 | False                 |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
