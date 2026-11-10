# Global round 1

Cumulative configurations: 7 -> 13.

Fixed-reference hypervolume: 0.822258382 -> 0.877450650.

Stalled: False; two consecutive stalled rounds: False.

| configuration       | hypothesis_name                                                 | base_configuration   | category                                   |    delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:--------------------|:----------------------------------------------------------------|:---------------------|:-------------------------------------------|------------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| global_15b18df4a34b | international_exception_front_without_global_rank               | global_389a69848aa0  | exploitation_local_pruning                 |  0.00818452 | -0.0139989  |     0.0448512  |                   -1 |              2 |               2 |                  2 | True                  |
| global_20d835629365 | international_exception_front_without_unskipped_annual_momentum | global_389a69848aa0  | exploration_shared_momentum_representation | -0.00193452 | -0.00149662 |    -0.00218445 |                   -1 |              1 |               2 |                  0 | False                 |
| global_67eeea779f2d | international_exception_front_without_gold_oil_ratio            | global_389a69848aa0  | exploration_commodity_interaction          | -0.0514881  |  0.0185754  |    -0.0844866  |                   -1 |              0 |               0 |                  0 | False                 |
| global_43cd2137e85f | international_exception_front_plus_lag                          | global_389a69848aa0  | exploration_conditional_lag_transfer       |  0.00362103 | -0.0203889  |     0.0433993  |                    1 |              3 |               3 |                  2 | True                  |
| global_d019284686e8 | ten_input_density_core_plus_absolute_credit_level               | global_b48732adb186  | exploration_cross_family_credit_transfer   |  0.0566964  |  0.017501   |     0.00780526 |                    1 |              3 |               0 |                  2 | False                 |
| global_300876dbdd56 | ten_input_density_core_plus_dealer_proxy                        | global_b48732adb186  | frontier_bridge_single_risk_axis           | -0.0172123  |  0.0397245  |     0.0154444  |                    1 |              0 |               0 |                  2 | False                 |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
