# Global round 2

Cumulative configurations: 13 -> 19.

Fixed-reference hypervolume: 0.877450650 -> 0.970672195.

Stalled: False; two consecutive stalled rounds: False.

| configuration       | hypothesis_name                                   | base_configuration   | category                                      |    delta_ic |    delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:--------------------|:--------------------------------------------------|:---------------------|:----------------------------------------------|------------:|-------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| global_52d9b7e07091 | lagged_density_front_without_global_rank          | global_43cd2137e85f  | frontier_bridge_successful_change_combination | -0.00396825 | -0.0121541   |      0.0579747 |                   -1 |              1 |               3 |                  2 | True                  |
| global_c0bc5df949cf | rank_free_front_without_unskipped_annual_momentum | global_15b18df4a34b  | exploration_local_representation_interaction  |  0.00877976 | -0.00412019  |     -0.0215613 |                   -1 |              3 |               2 |                  0 | True                  |
| global_4a6348639308 | rank_free_front_without_trend_slope               | global_15b18df4a34b  | exploitation_local_pruning                    | -0.00223214 |  0.00314381  |     -0.0052801 |                   -1 |              1 |               1 |                  1 | False                 |
| global_a8a6b8394d6e | rank_free_front_without_vix_level                 | global_15b18df4a34b  | exploration_risk_representation               |  0.00585317 |  0.0037894   |      0.0979986 |                   -1 |              2 |               1 |                  3 | True                  |
| global_973f696b2e34 | rank_free_front_without_oil_return                | global_15b18df4a34b  | exploration_commodity_representation          | -0.0159226  |  0.0550621   |     -0.190954  |                   -1 |              0 |               0 |                  0 | False                 |
| global_d062f4d835aa | rank_free_front_plus_absolute_credit_level        | global_15b18df4a34b  | frontier_bridge_conditional_credit_transfer   |  0.00749008 | -0.000455686 |     -0.017516  |                    1 |              2 |               1 |                  1 | True                  |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
