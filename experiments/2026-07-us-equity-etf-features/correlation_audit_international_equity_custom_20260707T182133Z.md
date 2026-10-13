# ETF Family Candidate Feature Correlation Audit
Artifact: `/Users/et/Documents/BayesFolio/experiments/2026-07-us-equity-etf-features/artifacts/international_equity_custom_feature_candidates_20260707T182133Z.parquet`
Family: `international_equity` / `VEA, VWO, VSS, EWX`. Horizon: `BME`.
ETF-local and macro predictors were rebuilt live for this family so newly added columns and horizon-aligned macro transformations are included. Return labels used local cache.

## Availability
- Available audited features: `56`
- Missing from artifact: `[]`

## Perfect Correlations
- `vix_slope` vs `vix_ts_level`: corr `1.000000`, n `256`

## High Correlations Worth Watching
- `tnote10y` vs `y10_nominal`: corr `0.9995`, n `256`
- `vix_slope` vs `skew_proxy`: corr `0.9995`, n `256`
- `vix_ts_level` vs `skew_proxy`: corr `0.9995`, n `256`

## Constant Features
- `de10y`
- `jp10y`
- `uk10y`
- `cn10y`
