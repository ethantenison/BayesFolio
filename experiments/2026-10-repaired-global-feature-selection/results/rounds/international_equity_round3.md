# International Equity round 3

Cumulative configurations: 173 -> 185.

Fixed-reference hypervolume: 0.620432258 -> 0.701799493.

Stalled: False; two consecutive stalled rounds: False.

| configuration                     | hypothesis_name                                 | base_configuration                | category                             |    delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:----------------------------------|:------------------------------------------------|:----------------------------------|:-------------------------------------|------------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| international_equity_66ef3bfb80d7 | compact_oil_without_six_month_momentum          | international_equity_41d789e7d4d7 | exploitation_local_pruning           |  0.0333333  | -0.0437865  |    -0.00311063 |                   -1 |              3 |               3 |                  1 | True                  |
| international_equity_af9e3885f064 | compact_oil_without_rank                        | international_equity_41d789e7d4d7 | exploration_local_interaction        |  0.0166667  | -0.0390821  |    -0.0334335  |                   -1 |              3 |               3 |                  1 | False                 |
| international_equity_ef2048f76023 | compact_oil_without_trend                       | international_equity_41d789e7d4d7 | exploration_local_interaction        |  0.00833333 | -0.0162015  |    -0.0695929  |                   -1 |              2 |               2 |                  0 | False                 |
| international_equity_60e0b820d655 | compact_oil_without_vix_level                   | international_equity_41d789e7d4d7 | exploration_risk_pruning             | -0.0222222  |  0.0074809  |    -0.0446822  |                   -1 |              1 |               1 |                  0 | False                 |
| international_equity_29cd1c88fc05 | compact_oil_without_vix_slope_zscore            | international_equity_41d789e7d4d7 | exploration_risk_pruning             | -0.0944444  |  0.109355   |    -0.0822814  |                   -1 |              0 |               0 |                  0 | False                 |
| international_equity_9b13070209b1 | compact_oil_without_vix_slope_level             | international_equity_41d789e7d4d7 | exploration_risk_pruning             |  0.0611111  | -0.042883   |    -0.011085   |                   -1 |              3 |               3 |                  1 | True                  |
| international_equity_3c896548ff79 | compact_oil_plus_annual_inflation               | international_equity_41d789e7d4d7 | exploration_cross_family_transfer    |  0.0361111  | -0.0438967  |    -0.053543   |                    1 |              2 |               3 |                  0 | True                  |
| international_equity_326475b1b82b | compact_oil_plus_credit_zscore                  | international_equity_41d789e7d4d7 | exploration_cross_family_transfer    | -0.0333333  | -0.00873623 |    -0.141212   |                    1 |              0 |               3 |                  0 | False                 |
| international_equity_f7933f4054ed | broad_without_hy_or_credit_change               | international_equity_54880a9ae051 | exploitation_combined_credit_pruning |  0.0305556  | -0.0155971  |     0.0180234  |                   -1 |              3 |               2 |                  3 | True                  |
| international_equity_d6633d86241b | broad_no_hy_without_gold_oil_ratio              | international_equity_54880a9ae051 | exploration_commodity_interaction    | -0.0611111  |  0.0785086  |    -0.0559289  |                   -1 |              0 |               0 |                  0 | False                 |
| international_equity_2848c1d07c0e | broad_no_hy_without_monthly_inflation_or_copper | international_equity_54880a9ae051 | exploration_macro_block_pruning      | -0.133333   | -0.0403806  |    -0.13718    |                   -2 |              0 |               3 |                  0 | False                 |
| international_equity_d0bc2acfc464 | broad_no_hy_without_tips_proxy_block            | international_equity_54880a9ae051 | exploration_rate_block_pruning       | -0.075      | -0.058976   |    -0.0433069  |                   -3 |              0 |               3 |                  1 | True                  |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
