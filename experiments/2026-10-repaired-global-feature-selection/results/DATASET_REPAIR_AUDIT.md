# Repaired dataset validation

Dataset: `artifacts/v1/global.parquet`.

SHA-256: `24b3c84a46921b8f7e7523479b1b28946444857872a6888715af3200f6fa6592`.

All 15 tasks have exactly 108 complete monthly rows: 84 initial training dates and 24 scored dates.
The original decimal excess-return targets are bitwise unchanged; predictors are lagged exactly once.

## Aligned cache repairs

| column               |   compared_cells |   changed_cells |   old_missing |   old_zero |   missing_to_valid |   zero_to_nonzero |   new_missing |   new_zero |
|:---------------------|-----------------:|----------------:|--------------:|-----------:|-------------------:|------------------:|--------------:|-----------:|
| baspread             |             1620 |              15 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| ret_kurt             |             1620 |            1611 |             0 |         30 |                  0 |                30 |             0 |          0 |
| chmom                |             1620 |            1553 |             0 |        180 |                  0 |               180 |             0 |          0 |
| mom6m                |             1620 |            1474 |             0 |         75 |                  0 |                75 |             0 |          0 |
| mom12m               |             1620 |            1484 |             0 |        180 |                  0 |               180 |             0 |          0 |
| mom12m_skip1m        |             1620 |            1514 |           345 |        180 |                345 |               180 |             0 |          0 |
| mom36m               |             1620 |            1515 |             0 |        540 |                  0 |               540 |             0 |          0 |
| max_dd_3m            |             1620 |            1513 |             0 |         75 |                  0 |                75 |             0 |          0 |
| max_dd_6m            |             1620 |            1487 |             0 |        165 |                  0 |               165 |             0 |          0 |
| ma_signal            |             1620 |            1528 |             0 |         30 |                  0 |                30 |             0 |          0 |
| ret_autocorr         |             1620 |            1603 |             0 |         15 |                  0 |                15 |             0 |          0 |
| vol_z                |             1620 |              45 |             0 |         30 |                  0 |                30 |             0 |          0 |
| trend_slope          |             1620 |            1566 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| vix                  |             1620 |               0 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| vix_ts_z_12p         |             1620 |             165 |             0 |        165 |                  0 |               165 |             0 |          0 |
| tnote10y             |             1620 |               0 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| tbill3m              |             1620 |               0 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| term_spread          |             1620 |               0 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| credit_spread        |             1620 |            1620 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| credit_spread_chg_1p |             1620 |            1620 |             0 |         15 |                  0 |                15 |             0 |          0 |
| dxy                  |             1620 |               0 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| spy_ret              |             1620 |            1620 |             0 |         45 |                  0 |                45 |             0 |          0 |
| erp                  |             1620 |            1620 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| move_proxy           |             1620 |            1620 |             0 |         45 |                  0 |                45 |             0 |          0 |
| vix_slope            |             1620 |               0 |             0 |          0 |                  0 |                 0 |             0 |          0 |
| rsp_spy_roc_1p       |             1620 |            1605 |             0 |         15 |                  0 |                15 |             0 |          0 |
| spy_flow_z_12p       |             1620 |             945 |             0 |        165 |                  0 |               165 |             0 |          0 |
| pct_above_50dma      |             1620 |            1620 |             0 |         75 |                  0 |                75 |             0 |          0 |
| hy_spread            |             1620 |            1290 |             0 |       1290 |                  0 |              1290 |             0 |          0 |
| hy_spread_chg_1p     |             1620 |            1290 |             0 |       1290 |                  0 |              1290 |             0 |          0 |
| hy_spread_z_12p      |             1620 |            1290 |             0 |       1290 |                  0 |              1290 |             0 |          0 |
| y10_nominal          |             1620 |            1605 |             0 |         15 |                  0 |                15 |             0 |          0 |
| schp_ret             |             1620 |            1620 |             0 |         45 |                  0 |                45 |             0 |          0 |
| em_fx_ret            |             1620 |            1620 |             0 |         15 |                  0 |                15 |             0 |         30 |
| oil_ret              |             1620 |            1620 |             0 |         30 |                  0 |                30 |             0 |          0 |
| copper_ret           |             1620 |            1620 |             0 |         30 |                  0 |                30 |             0 |          0 |
| gold_crude_ratio     |             1620 |            1620 |             0 |         15 |                  0 |                15 |             0 |          0 |
| y10_real_proxy       |             1620 |            1620 |             0 |         45 |                  0 |                45 |             0 |          0 |
| breakeven_proxy      |             1620 |            1620 |             0 |         60 |                  0 |                60 |             0 |          0 |
| cpi_chg_12p          |             1620 |             255 |             0 |        180 |                  0 |               180 |             0 |          0 |
| cpi_chg_1p           |             1620 |              30 |             0 |         15 |                  0 |                15 |             0 |          0 |

## Inflation availability

{
  "carried_signal_months": {
    "cpi_chg_12p": [
      "2025-10-31"
    ],
    "cpi_chg_1p": [
      "2025-10-31",
      "2025-11-28"
    ]
  },
  "missing_source_months": [
    "2025-10-31"
  ],
  "policy": "Last valid transformed signal, maximum two months; no level imputation"
}

## Exact-index HY source checks

{
  "independent": {
    "max_absolute_difference": 0.0,
    "overlap_observations": 6322
  },
  "live_fred": {
    "max_absolute_difference": 0.0,
    "overlap_observations": 605
  }
}

## Interpretation limits

- New dataset uses freshly frozen adjusted market sources.
- Source revisions, repair, ranks, and history changes are not separable effects.
- All original experiment data and results remain untouched.
- Latest-vintage CPI; not a vintage-controlled forecasting claim.

No fit is authorised by this audit alone; the active study separately authorises bounded execution.
