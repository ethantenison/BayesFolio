# U.S. Equity Feature EDA Summary
Artifacts analyzed:
- `BME`: `experiments/2026-07-us-equity-etf-features/artifacts/us_equity_full_feature_candidates_20260705T135534Z_bme.parquet`
- `3W-FRI`: `experiments/2026-07-us-equity-etf-features/artifacts/us_equity_full_feature_candidates_20260705T135547Z_3w_fri.parquet`

## Target Summary
- `BME`: 320 labeled rows, mean `0.74%`, std `5.17%`, positive share `57.81%`.
- `3W-FRI`: 465 labeled rows, mean `0.55%`, std `4.52%`, positive share `59.14%`.

Largest absolute per-asset target means:
- `3W-FRI` `MGK`: mean `0.83%`, std `5.46%`, positive share `58.06%`.
- `3W-FRI` `SPY`: mean `0.67%`, std `3.91%`, positive share `60.22%`.
- `3W-FRI` `VTV`: mean `0.56%`, std `3.18%`, positive share `60.22%`.
- `BME` `MGK`: mean `1.10%`, std `5.92%`, positive share `59.38%`.
- `BME` `SPY`: mean `0.91%`, std `4.44%`, positive share `59.38%`.
- `BME` `VTV`: mean `0.77%`, std `3.93%`, positive share `59.38%`.

## Feature Quality
- `BME`: 44 features, max missing rate `0.0000`, constant features `[]`.
- `3W-FRI`: 44 features, max missing rate `0.0000`, constant features `[]`.

## Strongest Linear Associations To Target

`BME` top absolute correlations:
- `spy_ret`: corr `0.212`, n `320`
- `lag_y_excess_lead`: corr `-0.188`, n `320`
- `hy_spread`: corr `0.171`, n `320`
- `cpi_chg_12p`: corr `-0.165`, n `320`
- `hy_spread_z_12p`: corr `0.154`, n `320`
- `vol_accel`: corr `0.134`, n `320`
- `hy_spread_chg_1p`: corr `0.121`, n `320`
- `vix`: corr `-0.116`, n `320`
- `tbill3m`: corr `0.113`, n `320`
- `term_spread`: corr `-0.112`, n `320`

`3W-FRI` top absolute correlations:
- `hy_spread_z_12p`: corr `0.274`, n `465`
- `lag2_y_excess_lead`: corr `-0.221`, n `465`
- `short_reversal_1m`: corr `0.208`, n `465`
- `vol_z`: corr `0.204`, n `465`
- `vix_slope`: corr `-0.184`, n `465`
- `vix`: corr `0.184`, n `465`
- `turnover`: corr `0.171`, n `465`
- `trend_slope`: corr `-0.168`, n `465`
- `cpi_chg_12p`: corr `-0.157`, n `465`
- `vix_ts_z_12p`: corr `-0.146`, n `465`

## Visual Readout
- Target distributions are wide relative to average returns; this is a ranking/noisy-signal problem, not a low-noise regression problem.
- The feature-target correlations are modest, as expected for one-period ETF excess returns. Strong claims should require walk-forward ranking and portfolio diagnostics.
- HY spread features now vary cleanly in both horizons after the source-grid fix.
- Feature-correlation heatmaps should be used to choose compact blocks; do not feed all correlated trend/volatility variants into the first GP run.

## Files
- `experiments/2026-07-us-equity-etf-features/eda/figures/feature_correlation_heatmap_3w_fri.png`
- `experiments/2026-07-us-equity-etf-features/eda/figures/feature_correlation_heatmap_bme.png`
- `experiments/2026-07-us-equity-etf-features/eda/figures/feature_std_by_block_horizon.png`
- `experiments/2026-07-us-equity-etf-features/eda/figures/hy_spread_horizon_alignment.png`
- `experiments/2026-07-us-equity-etf-features/eda/figures/target_distribution_by_asset_horizon.png`
- `experiments/2026-07-us-equity-etf-features/eda/figures/top_feature_target_correlations.png`
