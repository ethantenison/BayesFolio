# Us Equity round 1

Cumulative configurations: 11 -> 23.

Fixed-reference hypervolume: 0.664053736 -> 0.708220358.

Stalled: False; two consecutive stalled rounds: False.

| configuration          | hypothesis_name                             | base_configuration     | category                            |   delta_ic |    delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:-----------------------|:--------------------------------------------|:-----------------------|:------------------------------------|-----------:|-------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| us_equity_0fe495c66f9e | compact_without_mom6m                       | us_equity_c10464d00ae6 | exploitation_local_ablation         | -0.0722222 | -0.00784516  |      0.0223575 |                   -1 |              0 |               1 |                  2 | False                 |
| us_equity_1ee67d8f4013 | compact_without_skip_month_momentum         | us_equity_c10464d00ae6 | exploitation_local_ablation         | -0.0694444 |  0.00416042  |     -0.0100653 |                   -1 |              0 |               1 |                  1 | False                 |
| us_equity_5fba6b4b201b | compact_without_cross_sectional_rank        | us_equity_c10464d00ae6 | exploitation_local_ablation         | -0.025     | -0.0261027   |      0.0698697 |                   -1 |              1 |               2 |                  3 | True                  |
| us_equity_fd58ef4445a1 | compact_without_trend_slope                 | us_equity_c10464d00ae6 | exploitation_local_ablation         | -0.0611111 | -0.0165002   |     -0.0606433 |                   -1 |              0 |               2 |                  0 | False                 |
| us_equity_0d4f4d5f8c96 | compact_without_cpi_year_change             | us_equity_c10464d00ae6 | exploitation_macro_ablation         | -0.186111  | -0.000123781 |     -0.0205325 |                   -1 |              0 |               1 |                  1 | False                 |
| us_equity_a4c0679af605 | compact_without_hy_change                   | us_equity_c10464d00ae6 | exploitation_macro_ablation         | -0.05      |  0.0658847   |     -0.0300515 |                   -1 |              0 |               0 |                  0 | False                 |
| us_equity_38ae0afec701 | compact_without_hy_zscore                   | us_equity_c10464d00ae6 | exploitation_macro_ablation         | -0.122222  |  0.326219    |     -0.096178  |                   -1 |              0 |               0 |                  1 | False                 |
| us_equity_867ae082cead | compact_without_curve_spread                | us_equity_c10464d00ae6 | exploitation_macro_ablation         | -0.147222  |  0.296533    |     -0.221657  |                   -1 |              0 |               0 |                  0 | False                 |
| us_equity_e7028305d052 | compact_plus_recent_return_lag              | us_equity_c10464d00ae6 | exploration_conditional_lag         | -0.108333  | -0.0530079   |     -0.0962974 |                    1 |              0 |               3 |                  0 | True                  |
| us_equity_27c180061051 | compact_plus_local_realized_volatility      | us_equity_c10464d00ae6 | exploration_uncertainty_information | -0.0277778 | -0.0117073   |     -0.0242006 |                    1 |              0 |               2 |                  1 | False                 |
| us_equity_7ebdf64a59ff | compact_plus_oil_return                     | us_equity_c10464d00ae6 | frontier_bridge                     | -0.0416667 | -0.00822449  |     -0.036788  |                    1 |              0 |               2 |                  0 | False                 |
| us_equity_a9609614fc7e | compact_macro_with_transferable_local_basis | us_equity_c10464d00ae6 | exploration_shared_representation   | -0.075     |  0.0201827   |     -0.0634464 |                    0 |              1 |               1 |                  0 | False                 |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
