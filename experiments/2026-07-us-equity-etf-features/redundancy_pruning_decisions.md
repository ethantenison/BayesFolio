# U.S. Equity Redundancy And Scaling Decisions

This note records experiment-scope decisions after the GP-focused EDA. It does
not delete BayesFolio provider features globally.

## Feature Pair Decisions

Use `us_equity_compact_pruned_v1` as the first-pass GP/MTGP feature set.

Drop from the first-pass feature set:

- `short_reversal_1m` when `lag2_y_excess_lead` is present. After the builder's
  one-period predictor shift, `short_reversal_1m` is mechanically close to the
  negative of the two-period lagged target, especially on BME.
- `mom12m`; keep `mom12m_skip1m`. The skip-month definition is the more
  canonical equity momentum signal and is 0.96 correlated with plain 12-month
  momentum in both horizons.
- `baspread` from the first pass. In this U.S. ETF family it mostly behaves like
  another volatility proxy: corr with `vol_1m` is about 0.95.
- `vol_1w`, `downside_vol_1m`, and `max_dd_6m` from the first pass. Keep
  `vol_1m`, `vol_ratio_1m_3m`, and `max_dd_3m` as a smaller volatility/drawdown
  representation.
- `ma_regime` when `ma_signal`/`trend_slope` are available. Keep trend state
  compact initially.
- `tbill3m` from the first pass. Keep `term_spread` to represent the rate-curve
  state; `tbill3m` is strongly negatively correlated with `term_spread` in this
  sample.

Keep but isolate in ablations:

- `short_reversal_1m` in `us_equity_no_target_lag_reversal_ablation`, where
  explicit target-history lags are removed.
- Broader volatility variants if the compact volatility block underperforms.
- `tbill3m` or `erp` only in a later rate-level sensitivity block.

## IJR And IWM

Do not silently drop one before the first run. They are near-duplicate tasks
(`IJR`/`IWM` target correlation is about 0.97 in both horizons), but this is an
MTGP task-structure question rather than a feature-pruning question.

Run two task-universe variants:

- `full_style_size`: `SPY`, `MGK`, `VTV`, `IJR`, `IWM`.
- `compact_one_small_cap`: `SPY`, `MGK`, `VTV`, `IWM`.

If the full family mostly learns a duplicated small-cap task and weakens
style/size separation, use the compact one-small-cap universe for the main
search. If the duplicate improves calibration without collapsing forecasts,
keep both.

## BoTorch Input Scaling And Outliers

For BoTorch-style GP runs, keep using input normalization and output
standardization, but do not feed raw extreme-valued features directly into
min-max scaling.

Recommended preprocessing inside each walk-forward training window:

1. Fit clipping thresholds on the training window only.
2. Winsorize heavy-tailed features before min-max normalization.
3. Fit BoTorch `Normalize(d)` or equivalent min-max transform on the clipped
   training window only.
4. Fit output `Standardize(m)` on the training targets only.
5. Apply the stored train-window transforms to validation/scoring rows.

Initial clipping policy:

- Bounded breadth/risk features such as `pct_above_50dma`: keep min-max scaling,
  but verify the support is bounded and not malformed.
- Heavy-tailed ETF-local features such as `ill_log`, `vol_accel`,
  `vol_of_vol`, and `ret_kurt`: winsorize at train-window 1st/99th percentiles
  before min-max scaling.
- Return and macro level/change features: start with train-window 1st/99th
  percentile clipping, then test sensitivity if a feature looks important.

This avoids one crisis observation setting the full min-max range and compressing
the normal regime into a narrow part of the unit cube.

## Cache Decision

The legacy shared `artifacts/cache/returns/returns_3w_fri.parquet` file allowed
different 3W anchor phases to live in the same cache. The returns provider now
uses anchor-specific cache filenames for `3W-FRI`, and the old interleaved cache
has been archived under this experiment's `cache_backups/` directory.
