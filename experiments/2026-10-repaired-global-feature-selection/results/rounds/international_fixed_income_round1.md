# International Fixed Income round 1

Cumulative configurations: 89 -> 101.

Fixed-reference hypervolume: 0.740084338 -> 0.945671955.

Stalled: False; two consecutive stalled rounds: False.

| configuration                           | hypothesis_name                          | base_configuration                      | category                          |    delta_ic |    delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:----------------------------------------|:-----------------------------------------|:----------------------------------------|:----------------------------------|------------:|-------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| international_fixed_income_d34c5bbac26b | strong_ic_without_hy_level               | international_fixed_income_b1352653865d | exploitation_credit_ablation      | -0.00694444 | -0.0177858   |   -0.00403141  |                   -1 |              0 |               3 |                  1 | False                 |
| international_fixed_income_b62ac17a6009 | strong_ic_without_hy_zscore              | international_fixed_income_b1352653865d | exploitation_credit_ablation      |  0.0833333  | -0.0339058   |   -0.00327533  |                   -1 |              3 |               3 |                  1 | True                  |
| international_fixed_income_bb60018b724a | strong_ic_without_cpi_year_change        | international_fixed_income_b1352653865d | exploitation_macro_ablation       | -0.0277778  |  0.000764224 |   -0.00341774  |                   -1 |              0 |               1 |                  2 | False                 |
| international_fixed_income_9f739d5371ef | strong_ic_without_spy_flow               | international_fixed_income_b1352653865d | exploitation_macro_ablation       | -0.0555556  | -0.00610509  |    0.0101725   |                   -1 |              0 |               3 |                  2 | True                  |
| international_fixed_income_82ab37de30d5 | strong_ic_without_long_momentum          | international_fixed_income_b1352653865d | exploitation_local_ablation       | -0.0277778  | -0.00670966  |   -0.000784983 |                   -1 |              1 |               2 |                  1 | False                 |
| international_fixed_income_841b345780aa | strong_ic_without_momentum_acceleration  | international_fixed_income_b1352653865d | exploitation_local_ablation       | -0.0277778  |  0.0110477   |    0.00675669  |                   -1 |              0 |               1 |                  2 | True                  |
| international_fixed_income_e97db2a4d393 | strong_ic_without_vix_slope_zscore       | international_fixed_income_b1352653865d | exploitation_risk_ablation        | -0.0694444  |  0.0249943   |   -0.00255752  |                   -1 |              0 |               0 |                  1 | False                 |
| international_fixed_income_115dcc914f64 | strong_ic_without_cross_sectional_rank   | international_fixed_income_b1352653865d | exploitation_local_ablation       | -0.0763889  | -0.00114292  |    0.011235    |                   -1 |              0 |               2 |                  3 | True                  |
| international_fixed_income_fdc132170a8f | strong_ic_with_vix_level_only            | international_fixed_income_b1352653865d | exploitation_block_pruning        | -0.152778   |  0.0330745   |   -0.00303296  |                   -2 |              0 |               0 |                  0 | False                 |
| international_fixed_income_775c0e6b8de9 | strong_ic_without_both_hy_features       | international_fixed_income_b1352653865d | exploitation_block_ablation       |  0.0555556  | -0.0399427   |    0.000394911 |                   -2 |              2 |               3 |                  2 | True                  |
| international_fixed_income_55f9567cb5fe | strong_ic_plus_nominal_yield_and_curve   | international_fixed_income_b1352653865d | exploration_rate_state            | -0.118056   |  0.0491051   |    0.00620903  |                    2 |              0 |               0 |                  2 | False                 |
| international_fixed_income_f9bcef18e066 | strong_ic_with_shared_equity_local_basis | international_fixed_income_b1352653865d | exploration_shared_representation | -0.125      | -0.00671374  |   -0.0100591   |                    0 |              0 |               2 |                  1 | False                 |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
