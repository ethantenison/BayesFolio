# Outputscale Structure Ablation

## Question

Do flat SPY/VEA multitask-GP predictions come from the current component outputscale structure?

## Run

- Script: `experiments/2026-06-task-covariance/run_outputscale_structure_ablation.py`
- Output path: `experiments/2026-06-task-covariance/outputs/runs/20260614_outputscale_structure_ablation`
- Variant: `positive_beta_prior`
- Time settings: `lengthscale_only`, `neither`
- Scale structures: `component_scales`, `no_component_scales`, `global_scale`
- Max optimizer iterations: 50
- Scored windows: 12

Artifacts:

- `manifest.json`
- `variant_summary.csv`
- `window_predictions.csv`
- `hyperparameter_diagnostics.csv`
- `plots/spy_outputscale_structure_ablation.png`
- `plots/vea_outputscale_structure_ablation.png`

## Result

The outputscale structure is involved, but removing component outputscales is not the fix.

The current structure, `component_scales_lengthscale_only`, remains best on decision-relevant ranking metrics:

| Variant | IC | Top-bottom Sharpe | Residual IC | Residual top-bottom Sharpe |
|---|---:|---:|---:|---:|
| component_scales_lengthscale_only | 0.375 | 2.982 | 0.190 | 2.055 |
| component_scales_neither | 0.316 | 2.033 | 0.155 | 1.307 |
| no_component_scales_lengthscale_only | 0.124 | 0.889 | -0.014 | 0.267 |
| no_component_scales_neither | 0.141 | 1.200 | -0.007 | -0.028 |
| global_scale_lengthscale_only | 0.115 | 1.172 | -0.305 | -2.418 |
| global_scale_neither | 0.135 | 0.397 | -0.019 | -0.264 |

Amplitude improves somewhat in some ablations, but the added movement is mostly not useful signal:

| Slice | Variant | Pred/true std | Residual pred/true std | Residual corr |
|---|---|---:|---:|---:|
| SPY/VEA | no_component_scales_neither | 0.240 | 0.244 | 0.105 |
| SPY/VEA | no_component_scales_lengthscale_only | 0.237 | 0.237 | 0.053 |
| SPY/VEA | component_scales_neither | 0.226 | 0.217 | 0.281 |
| SPY/VEA | component_scales_lengthscale_only | 0.122 | 0.111 | 0.483 |
| SPY/VEA | global_scale_lengthscale_only | 0.081 | 0.052 | -0.278 |

## Interpretation

The flatness is not solved by removing outputscale kernels.

The current model is conservative, but its smaller movement is better aligned with returns. Removing component scales lets predictions move more, but mostly erases IC and residual performance. A single global outputscale performs worst on residual signal.

The stronger diagnosis is posterior shrinkage plus time/kernel structure:

- `component_scales_lengthscale_only` preserves the best rank signal but has low amplitude.
- The learned time-varying lengthscale is very small on transformed inputs, especially at eval dates.
- ETF-only and some interaction components are effectively suppressed by learned outputscales.
- The model uncertainty remains wide, so the model recognizes outcome volatility but refuses to move posterior means strongly.

## Critic Pass

This run only covers `positive_beta_prior`. It answers the outputscale-structure question for the most relevant variant, but it does not prove the same ordering for signed task kernels.

The run also uses real-data roll-forward only. It does not test whether the architecture can recover known simulated signal amplitude. If fake-data recovery fails, the architecture or priors are broken. If fake-data recovery succeeds, the real data or transformations are the likely culprit.

## Next Move

Run fake-data recovery for this architecture before another broad model grid.

Minimum useful simulation:

1. Keep the real feature matrix and task IDs.
2. Simulate a known latent function with monthly return-scale amplitude.
3. Add realistic noise.
4. Fit the current component-scale lengthscale-only model.
5. Check whether posterior means recover amplitude and rank signal.

This separates "model cannot express amplitude" from "real data does not support amplitude."
