# Positive Beta Prior With Both Time Modulations And Outputscale Floor/Prior

## Question

Can the poor amplitude of the `positive_beta_prior` model with both time-varying lengthscale and time-varying outputscale be improved by forcing component outputscales away from zero?

## Runs Compared

| Label | Run path | Time modulation | Outputscale adjustment |
|---|---|---|---|
| `both` | `20260614_positive_beta_both_time_modulation` | lengthscale + outputscale | none |
| `both_floor_prior` | `20260614_positive_beta_both_outputscale_floor_prior` | lengthscale + outputscale | component `ScaleKernel.outputscale > 0.01`, LogNormal median `0.05`, sigma `0.75` |
| `lengthscale_only` | `20260613_lengthscale_only_full` | lengthscale only | none |
| `neither` | `20260614_positive_beta_no_time_modulation` | none | none |

## Result

The floor/prior does not fix the `both` model. It makes residual performance worse.

| Run | IC | Top-bottom Sharpe | Residual IC | Residual Sharpe |
|---|---:|---:|---:|---:|
| `both` | 0.270 | 1.860 | -0.006 | 0.040 |
| `both_floor_prior` | 0.250 | 2.156 | -0.139 | -1.305 |
| `lengthscale_only` | 0.367 | 3.205 | 0.201 | 1.925 |
| `neither` | 0.320 | 2.341 | 0.189 | 1.897 |

SPY/VEA amplitude:

| Run | Pred/true std | Residual pred/true std | Corr | Residual corr |
|---|---:|---:|---:|---:|
| `both` | 0.073 | 0.054 | -0.111 | 0.016 |
| `both_floor_prior` | 0.065 | 0.039 | -0.105 | 0.053 |
| `lengthscale_only` | 0.101 | 0.090 | 0.362 | 0.500 |
| `neither` | 0.225 | 0.215 | 0.247 | 0.291 |

## Learned Outputscales

The constrained/prior run did push component outputscales above the floor:

| Component | Mean learned outputscale |
|---|---:|
| `time` | 0.043 |
| `etf` | 0.014 |
| `macro_matern` | 0.056 |
| `macro_rq` | 0.027 |
| `macro_linear` | 0.017 |
| `time_x_etf` | 0.080 |
| `time_x_macro` | 0.284 |
| `macro_x_etf` | 0.016 |

But those larger component outputscales did not produce better posterior-mean amplitude. The likely bottleneck is the interaction between the extra time-varying outputscale wrapper, the time-varying lengthscale warp, and posterior shrinkage.

## Interpretation

The `both` wrapper remains the wrong direction for this model.

Adding a component outputscale floor and prior:

- prevents component scales from collapsing below 0.01;
- does not increase SPY/VEA amplitude;
- worsens residual IC and residual top-bottom returns;
- leaves the model visually flat.

This weakens the hypothesis that small component outputscales are the main problem. The extra time-varying outputscale wrapper appears to suppress useful residual signal in a way that component-scale priors do not repair.

## Caveat

This experiment adjusted ordinary component `ScaleKernel.outputscale` parameters. It did not add priors or constraints directly to the custom time-varying outputscale wrapper's raw bias/slope parameters. That wrapper is still a separate suspect.

## Recommendation

Do not use the `both` wrapper for now.

Keep `lengthscale_only` as the current reference model, but run fake-data recovery next. If the model cannot recover known simulated amplitude under `lengthscale_only`, the architecture/priors are too restrictive. If it can recover fake amplitude, the real data or feature/target construction is limiting posterior mean movement.
