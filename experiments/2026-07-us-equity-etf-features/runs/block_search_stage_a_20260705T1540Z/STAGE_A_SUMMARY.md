# Stage A Input Block Search Summary

Scope: 12 feature blocks/interactions, first-round U.S. equity family (`SPY`, `MGK`, `VTV`, `IWM`), monthly BME, 6 recent scored windows (`2026-01-30` to `2026-06-30`), signed LKJ MTGP rank 2, BoTorch input normalization, standardized outputs, 15 optimizer iterations, 512 posterior scenarios.

## Leaderboard

| feature_set               | n_features | rmse   | mean_gaussian_nll | coverage_80 | coverage_95 | gp_mean_ic | gp_cum_return | excess_vs_equal_weight | gp_avg_turnover |
| ------------------------- | ---------- | ------ | ----------------- | ----------- | ----------- | ---------- | ------------- | ---------------------- | --------------- |
| liquidity_plus_volatility | 8          | 0.0546 | -1.4935           | 0.7500      | 0.8750      | 0.0000     | 0.0632        | -0.0276                | 0.3666          |
| target_history            | 2          | 0.0552 | -1.4805           | 0.7083      | 0.8750      | 0.1000     | 0.0706        | -0.0203                | 0.2306          |
| target_plus_volatility    | 7          | 0.0553 | -1.4211           | 0.7500      | 0.8750      | 0.0667     | 0.0701        | -0.0207                | 0.2116          |
| volatility_regime         | 5          | 0.0558 | -1.4022           | 0.7083      | 0.8750      | 0.1667     | 0.0631        | -0.0277                | 0.2647          |
| momentum_plus_volatility  | 9          | 0.0558 | -1.3965           | 0.7083      | 0.8750      | 0.1000     | 0.0701        | -0.0208                | 0.2225          |
| target_plus_momentum      | 6          | 0.0568 | -1.3436           | 0.7083      | 0.8750      | -0.3667    | 0.0651        | -0.0257                | 0.2598          |
| momentum_trend            | 4          | 0.0584 | -1.3252           | 0.6667      | 0.7917      | -0.1667    | 0.0686        | -0.0222                | 0.3187          |
| target_plus_macro         | 10         | 0.0571 | -1.2774           | 0.7500      | 0.8333      | 0.3000     | 0.0972        | 0.0064                 | 0.3078          |
| momentum_plus_macro       | 12         | 0.0600 | -1.2557           | 0.7917      | 0.8333      | 0.5000     | 0.1229        | 0.0320                 | 0.3239          |
| liquidity                 | 3          | 0.0589 | -1.2408           | 0.7083      | 0.7500      | -0.2333    | 0.0789        | -0.0119                | 0.4293          |
| macro_risk                | 8          | 0.0597 | -1.1099           | 0.7917      | 0.8750      | 0.3333     | 0.1122        | 0.0214                 | 0.3030          |
| compact_pruned            | 22         | 0.0591 | -1.0743           | 0.7917      | 0.8333      | 0.5000     | 0.0953        | 0.0045                 | 0.2785          |

## Best By Objective

- Best uncertainty-aware forecast NLL: `liquidity_plus_volatility` (`-1.4935`).
- Best mean IC: `momentum_plus_macro` (`0.5000`).
- Best portfolio cumulative return: `momentum_plus_macro` (`0.1229`).

## Initial Readout

- Macro/risk information is the strongest standalone block by IC and portfolio return in this screen, but not by NLL.
- `momentum_plus_macro` is the strongest interaction by IC and cumulative return, and it beats the compact combined set on NLL, IC, and return in this small screen.
- `liquidity_plus_volatility` is the best NLL block, but its IC and portfolio return are weak, so it looks more like an uncertainty-calibration candidate than a ranking/portfolio candidate.
- Pure momentum and target-plus-momentum look weak here; that argues against momentum being useful without regime context.
- The compact pruned full set is competitive but not dominant, which suggests some features may still add noise or interact poorly at this optimizer budget.
- Coverage is very high across blocks, so uncertainty may be conservative; inspect NLL and standardized residual scale rather than coverage alone.

## Caveats

- This is a 6-window screen, not enough for final selection.
- Maxiter is intentionally modest; rerun shortlisted blocks with a larger optimizer budget before making claims.
- Some macro-heavy runs emitted small Cholesky jitter warnings; this is not fatal, but kernel conditioning remains a diagnostic to watch.
- Portfolio return is included as a downstream check, but forecast NLL/RMSE/IC should drive feature-block promotion at this stage.

## Figures

- `figures/block_leaderboard_metrics.png`
- `figures/per_etf_rmse_heatmap.png`
- `figures/per_etf_nll_heatmap.png`
- `figures/prediction_dispersion_by_block.png`
- `figures/coverage_calibration_heatmap.png`
- `figures/ic_by_date_heatmap.png`
- `figures/top_minus_bottom_spread.png`
- `figures/equity_curves_top_blocks.png`
