# Positive Beta June Structure: Lengthscale-Only

## Run

- Manifest: `manifest.json`
- Command: `experiments/2026-06-task-covariance/run_task_covariance_rollforward.py --variants positive_beta_prior --time-modulation-mode lengthscale_only --maxiter 50 --run-id 20260615_positive_beta_june_structure_lengthscale_only`
- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_06.parquet`
- Feature hash: `3e63653c8deba3dd2ef412e22399ea31f14fb262c0ca37dc05272b33a5b1e3f5`
- Git SHA at run: `e18ce604057a3113bc7b7baf8cf3d4a88750afbc`
- Scored windows: `2025-05-30` through `2026-04-30`
- Live prediction date: `2026-05-29`

## Question

Use the current June candidate structure for the monthly multitask GP:

- `positive_beta_prior`
- positive task covariance kernel
- lengthscale-only time modulation
- no target winsorization
- 50 optimizer iterations

This was a production-candidate check, not a full model-grid rerun.

## Metrics

| Metric | Value |
|---|---:|
| IC | `0.367` |
| IC p-value | `0.00096` |
| IR | `1.288` |
| Hit ratio | `60.2%` |
| Top/bottom 5 Sharpe | `3.205` |
| Top/bottom 5 max drawdown | `-0.13%` |
| Residual IC | `0.201` |
| Residual IC p-value | `0.151` |
| Residual top/bottom 5 Sharpe | `1.925` |

## Live June Readout

Top raw scores:

| Asset | Predicted excess | Score |
|---|---:|---:|
| `MGK` | `2.04%` | `0.376` |
| `SPY` | `1.11%` | `0.253` |
| `EWX` | `0.38%` | `0.105` |
| `HYG` | `0.17%` | `0.081` |
| `VTV` | `0.32%` | `0.074` |

Top residual scores:

| Asset | Predicted residual | Residual score |
|---|---:|---:|
| `IEF` | `0.38%` | `0.198` |
| `BND` | `0.26%` | `0.163` |
| `MGK` | `0.83%` | `0.154` |
| `BNDX` | `0.19%` | `0.143` |
| `LQD` | `0.35%` | `0.142` |

## Takeaways

The lengthscale-only model remains the best current direction. Prior ablations showed that
outputscale-only and both-time-modulation variants weaken the ranking signal. The lengthscale-only
run keeps strong raw IC and top/bottom signal.

Amplitude is still the core unresolved issue. The posterior means are conservative: earlier
SPY/VEA diagnostics showed low prediction amplitude versus realized movement, even when ranking
metrics were useful. Removing or restructuring component outputscales increased movement in some
ablations, but mostly damaged IC and residual performance.

The current diagnosis is:

- learned component amplitudes/outputscales can collapse or effectively suppress pieces of the
  kernel;
- learned time-varying effective lengthscales can become very small on transformed inputs,
  especially at evaluation dates;
- despite that, the time-varying lengthscale is picking up useful rank signal;
- the extra time-varying outputscale wrapper appears harmful for this data/model shape;
- posterior uncertainty remains wide, so the model recognizes outcome volatility but does not move
  posterior means aggressively.

## Caveats

- This run has no plot artifacts.
- It ran only `positive_beta_prior`, not the full variant grid.
- The worktree was dirty at run start.
- Residualized IC is weaker and not statistically significant, so the raw top/bottom result should
  not be overclaimed.
- This is model-quality evidence, not a final portfolio allocation backtest.

## Next Check

Run fake-data recovery before another broad grid:

1. keep the real feature matrix and task IDs;
2. simulate known monthly-return-scale latent signal;
3. fit the current lengthscale-only model;
4. verify whether posterior means recover amplitude and rank order.

If fake-data recovery fails, the architecture or priors are too restrictive. If it succeeds, the
real data/features are likely the limiting source of posterior mean amplitude.
