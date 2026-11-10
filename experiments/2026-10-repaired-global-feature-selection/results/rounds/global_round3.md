# Global round 3

Cumulative configurations: 19 -> 25.

Fixed-reference hypervolume: 0.970672195 -> 1.016778459.

Stalled: False; two consecutive stalled rounds: False.

| configuration       | hypothesis_name                                               | base_configuration   | category                                         |    delta_ic |    delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:--------------------|:--------------------------------------------------------------|:---------------------|:-------------------------------------------------|------------:|-------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| global_9e7175a3ef4a | no_vix_front_without_unskipped_annual_momentum                | global_a8a6b8394d6e  | frontier_bridge_combined_successful_removals     |  0.00243056 | -0.00525935  |     -0.0231742 |                   -1 |              2 |               2 |                  2 | True                  |
| global_ac1e9b719b57 | lagged_density_front_without_vix_level                        | global_52d9b7e07091  | frontier_bridge_risk_lag_interaction             |  0.00352183 |  0.00727111  |      0.0588536 |                   -1 |              2 |               0 |                  3 | False                 |
| global_33e57e1c2370 | lagged_density_front_without_vix_or_unskipped_annual_momentum | global_52d9b7e07091  | exploration_temporal_representation_substitution |  0.00679563 |  0.00103736  |      0.0962278 |                   -2 |              2 |               1 |                  3 | True                  |
| global_da43a19fc3dd | no_vix_front_without_dealer_proxy                             | global_a8a6b8394d6e  | exploration_conditional_risk_pruning             | -0.0361111  | -0.0331739   |     -0.186521  |                   -1 |              0 |               3 |                  0 | True                  |
| global_c86bac72391c | no_vix_front_without_normalized_credit_state                  | global_a8a6b8394d6e  | exploration_shared_credit_importance             | -0.0777778  |  0.0746321   |     -0.269194  |                   -1 |              0 |               0 |                  0 | False                 |
| global_fdb5fb8b385f | absolute_credit_front_without_vix_level                       | global_d062f4d835aa  | frontier_bridge_credit_risk_representation       |  0.00391865 |  0.000913051 |      0.0759481 |                   -1 |              2 |               1 |                  2 | True                  |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
