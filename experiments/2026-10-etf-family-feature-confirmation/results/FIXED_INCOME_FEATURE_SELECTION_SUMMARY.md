# Fixed-Income ETF Feature Search

This report consolidates the U.S. and international fixed-income searches across block screening,
targeted ablations, feature-level add/remove tests, and final combination refinement.

The Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher conventional annualized
Sharpe. Cumulative return and RMSE are recorded only; turnover is excluded. All results use cached data,
seeds `27,127,227`, 24 common monthly scored windows, rank 2, maxiter 15, 256 scenarios, and `t_index`.

## U.S. fixed income

Universe: `BND, IEF, LQD, HYG`. Evaluated `31` unique configurations and `93` valid fits.

### Final Pareto front

| configuration                      |   mean_ic |   mean_gaussian_nll |   sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only |
|:-----------------------------------|----------:|--------------------:|---------:|------------------------------------:|-----------------------:|
| USFI-TREND                         |  0.147222 |           -2.673565 | 0.429514 |                            0.034860 |               0.015503 |
| USFI-COMPACT-NO-MOM12M             |  0.216667 |           -2.658920 | 0.429309 |                            0.035102 |               0.015927 |
| USFI-COMPACT-NO-CS-MOM-RANK        |  0.175000 |           -2.671665 | 0.454434 |                            0.037021 |               0.015595 |
| USFI-COMPACT+RET-AUTOCORR          |  0.213889 |           -2.627451 | 0.429903 |                            0.035265 |               0.015871 |
| USFI-COMPACT+VOL-Z                 |  0.230556 |           -2.632968 | 0.427207 |                            0.035214 |               0.015833 |
| USFI-COMPACT-NO-MOM12+RET-AUTOCORR |  0.222222 |           -2.653782 | 0.416432 |                            0.033936 |               0.015779 |
| USFI-COMPACT-NO-CS+RET-AUTOCORR    |  0.211111 |           -2.657938 | 0.458606 |                            0.037691 |               0.015742 |
| USFI-COMPACT-NO-MOM12-NO-CS        |  0.191667 |           -2.668034 | 0.422049 |                            0.034851 |               0.015789 |

### Hypervolume progression

| round                  |   unique_configurations |   hypervolume |
|:-----------------------|------------------------:|--------------:|
| block_screen           |                       7 |   0.173644663 |
| targeted_ablation      |                      14 |   0.242330509 |
| feature_refinement     |                      24 |   0.276936663 |
| combination_refinement |                      31 |   0.280640676 |

### Promotion set

- **primary — USFI-COMPACT-NO-CS+RET-AUTOCORR**: IC `0.211111`, NLL `-2.657938`, Sharpe `0.458606`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, ma_signal, ret_autocorr`.
- **challenger — USFI-COMPACT+VOL-Z**: IC `0.230556`, NLL `-2.632968`, Sharpe `0.427207`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, cs_mom_rank, ma_signal, vol_z`.
- **calibration_anchor — USFI-TREND**: IC `0.147222`, NLL `-2.673565`, Sharpe `0.429514`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, mom36m, cs_mom_rank, ma_signal`.

## International fixed income

Universe: `BNDX, VWOB, HYEM`. Evaluated `48` unique configurations and `144` valid fits.

### Final Pareto front

| configuration                                            |   mean_ic |   mean_gaussian_nll |   sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only |
|:---------------------------------------------------------|----------:|--------------------:|---------:|------------------------------------:|-----------------------:|
| INTFI-TREND+EQUITY-RISK                                  |  0.062500 |           -2.631626 | 0.850382 |                            0.068254 |               0.017250 |
| INTFI-TREND+COMMODITY-RET                                | -0.083333 |           -2.710271 | 0.819405 |                            0.065428 |               0.015601 |
| INTFI-TREND+MARKET-RISK                                  |  0.159722 |           -2.688399 | 0.847268 |                            0.067871 |               0.015314 |
| INTFI-TREND+GLOBAL                                       |  0.229167 |           -2.649867 | 0.849766 |                            0.068428 |               0.013867 |
| INTFI-ALL-MACRO-NO-HY-SPREAD-CHG-1M                      |  0.277778 |           -2.605438 | 0.860532 |                            0.069081 |               0.015558 |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-CPI-MOM                  |  0.277778 |           -2.602178 | 0.862439 |                            0.069168 |               0.015353 |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-COPPER                   |  0.354167 |           -2.601980 | 0.870139 |                            0.069877 |               0.015702 |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-ERP-NO-CPI-MOM-NO-COPPER |  0.388889 |           -2.605973 | 0.859886 |                            0.069001 |               0.015244 |

### Hypervolume progression

| round                  |   unique_configurations |   hypervolume |
|:-----------------------|------------------------:|--------------:|
| block_screen           |                       7 |   0.558858315 |
| targeted_ablation      |                      15 |   0.558861053 |
| feature_refinement     |                      36 |   0.698328598 |
| combination_refinement |                      48 |   0.894649816 |

### Promotion set

- **primary — INTFI-ALL-MACRO-NO-HY-CHANGE-NO-COPPER**: IC `0.354167`, NLL `-2.601980`, Sharpe `0.870139`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, mom36m, cs_mom_rank, ma_signal, hy_spread, hy_spread_z_12m, vix_slope, vix_ts_z_12m, vix, spy_flow_z_12m, spy_ret, erp, cpi_yoy, cpi_mom, oil_ret, gold_crude_ratio, pct_above_50dma, em_fx_ret`.
- **challenger — INTFI-ALL-MACRO-NO-HY-CHANGE-NO-ERP-NO-CPI-MOM-NO-COPPER**: IC `0.388889`, NLL `-2.605973`, Sharpe `0.859886`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, mom36m, cs_mom_rank, ma_signal, hy_spread, hy_spread_z_12m, vix_slope, vix_ts_z_12m, vix, spy_flow_z_12m, spy_ret, cpi_yoy, oil_ret, gold_crude_ratio, pct_above_50dma, em_fx_ret`.
- **calibration_anchor — INTFI-TREND+MARKET-RISK**: IC `0.159722`, NLL `-2.688399`, Sharpe `0.847268`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, mom36m, cs_mom_rank, ma_signal, vix_slope, vix_ts_z_12m, vix, spy_flow_z_12m, spy_ret, erp, pct_above_50dma`.

## Findings and recommendation

- U.S. fixed income favored local time-series features. Credit/inflation and broader macro additions
  were noise at this screening budget. Removing `mom36m` opened the useful region; removing
  `cs_mom_rank` and adding `ret_autocorr` produced the best Sharpe-oriented balance.
- International fixed income required macro context. The broad macro model became materially better
  after removing `hy_spread_chg_1m` and `copper_ret`; also removing `erp` and `cpi_mom` maximized IC
  while retaining a strong Sharpe and slightly better NLL.
- `INTFI-TREND+COMMODITY-RET` remains mathematically nondominated only because it has the best NLL;
  its negative IC means it contributes no hypervolume above the fixed zero reference and is not promoted.
- Promote the primary and challenger for each family to longer-history/full-budget confirmation.
  Keep the calibration anchor in that confirmation so selection remains genuinely multi-objective.
  Do not replace production inputs from this screen alone.

## Validation and limitations

- Every reported configuration passed exact universe, date, seed, input-column, and model-control checks.
- The search is conditional on the cached 2013-08 through 2026-04 data and the 24 scored months from
  2024-05 through 2026-04. It is evidence-directed rather than exhaustive over every feature subset.
- The next confirmation should preserve a common date panel and use the production optimization budget
  before any production promotion decision.
