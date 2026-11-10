# Global round 0

Cumulative configurations: 1 -> 7.

Fixed-reference hypervolume: 0.262731421 -> 0.822258382.

Stalled: False; two consecutive stalled rounds: False.

| configuration       | hypothesis_name                                     | base_configuration   | category                           |    delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:--------------------|:----------------------------------------------------|:---------------------|:-----------------------------------|------------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| global_b48732adb186 | shared_mechanism_core_10                            | global_c55e49933d10  | shared_core_hypothesis             |  0.0957341  |  -0.0968794 |      0.160488  |                  -17 |              3 |               3 |                  2 | True                  |
| global_3745593ecc26 | literal_selected_family_union_29                    | global_c55e49933d10  | family_union_hypothesis            | -0.0123016  |   0.0346109 |      0.0506617 |                    2 |              0 |               0 |                  2 | False                 |
| global_9d4dc5c2a19f | evidence_reduced_union_22                           | global_c55e49933d10  | reduced_union_hypothesis           |  0.0734127  |  -0.0162219 |      0.169089  |                   -5 |              3 |               2 |                  2 | False                 |
| global_2b66b7fc9390 | shared_core_plus_fixed_income_exceptions_14         | global_c55e49933d10  | fixed_income_exception_hypothesis  |  0.00550595 |   0.0475662 |      0.16058   |                  -13 |              2 |               0 |                  3 | False                 |
| global_389a69848aa0 | shared_core_plus_international_risk_exceptions_15   | global_c55e49933d10  | international_exception_hypothesis |  0.149653   |  -0.0772898 |      0.297159  |                  -12 |              3 |               3 |                  3 | True                  |
| global_1b4304ffe9c1 | shared_core_plus_fixed_income_exceptions_and_lag_15 | global_c55e49933d10  | matched_global_lag_contrast        |  0.0128968  |  -0.0168379 |      0.1712    |                  -12 |              2 |               3 |                  3 | False                 |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
