# Post-selection predictive distribution diagnostics

Numerical diagnostics only. Configuration and November weights were frozen first and remain unchanged. No fit, optimization, covariance repair or recalibration was performed.

All53 outcomes and the development41/confirmation12 subsets were previously inspected. NLL uses natural-log Gaussian densities in decimal discounted-excess return units; negative NLL is valid. RMSE is in percentage points. Lower NLL/RMSE are descriptive diagnostics, not a new selection rule.

## Frozen finalists: all53 months, eligible14 tasks

| Policy | Joint NLL/asset | Marginal NLL | RMSE(pp) | Portfolio mean z² | Portfolio95 coverage | EW NLL |
|---|---:|---:|---:|---:|---:|---:|
| `policy_c5f34924606d3b69f1bf` | -2.3971 | -1.7706 | 4.021 | 3.494 | 66.0% | -0.8137 |
| `policy_56a0b06b63cc63c2913b` | -2.3904 | -1.7812 | 4.011 | 3.425 | 64.2% | -0.8557 |
| `policy_b71ce3fe84ee69f58247` **selected** | -2.3732 | -1.7473 | 4.128 | 2.674 | 83.0% | -0.7510 |
| `policy_1ecb16e803d529c446d5` | -2.3727 | -1.7479 | 4.132 | 2.574 | 83.0% | -0.7517 |
| `policy_269c851205829f549e8e` | -2.3721 | -1.7466 | 4.133 | 2.842 | 83.0% | -0.7481 |
| `policy_bffebed65d9d74291d0a` | -2.5812 | -1.9141 | 4.135 | 1.329 | 84.9% | -1.9662 |
| `policy_ffda294836ec0b34898d` | -2.3882 | -1.7688 | 4.023 | 3.208 | 73.6% | -0.8660 |
| `policy_e88e9221e7d2bdd5631c` | -2.3900 | -1.7730 | 4.018 | 3.215 | 75.5% | -0.8746 |
| `policy_91b4b9ebe70878b95ef8` | -2.3971 | -1.7706 | 4.021 | 2.734 | 79.2% | -0.8137 |
| `policy_87884c01c87dcdace8a6` | -2.3711 | -1.7455 | 4.136 | 2.656 | 83.0% | -0.7521 |

## Expert diagnostics: seed-pooled forecasts, no allocation

Joint15 and joint14 scores use different task sets and should be compared within their own columns. Every expert retained all15 GP tasks; the14-task view simply excludesMGK for a matched comparison.

| Expert | Joint15 NLL/asset | Joint14 NLL/asset | Marginal15 NLL | RMSE15(pp) | Joint15 mean Mahalanobis/d |
|---|---:|---:|---:|---:|---:|
| `broad_r2_i15_exp` | -2.3533 | -2.4012 | -1.7301 | 4.180 | 1.039 |
| `broad_roll48_r2_i15` | -2.2803 | -2.3308 | -1.7666 | 4.375 | 0.973 |
| `compact_r1_i15_exp` | -2.1418 | -2.1779 | -1.6425 | 4.624 | 1.063 |
| `compact_r2_i15_exp` | -2.2538 | -2.2974 | -1.5890 | 4.459 | 1.100 |

## Selected configuration across temporal blocks

| Block | Months | Joint NLL/asset | Portfolio mean z² | Portfolio95 coverage | EW mean z² |
|---|---:|---:|---:|---:|---:|
| development41 | 41 | -2.3190 | 2.311 | 82.9% | 5.421 |
| confirmation12 | 12 | -2.5585 | 3.915 | 83.3% | 3.943 |
| full53 | 53 | -2.3732 | 2.674 | 83.0% | 5.087 |

## Interpretation limits

- Every block is retrospective and previously inspected; these scores do not change the frozen decision.
- Joint NLL combines logdet and mean-error terms; it scores a Gaussian distribution, not covariance alone.
- Covariance recipes pool operational second moments, omitting between-model disagreement covariance.
- Marginal/portfolio squared standardized errors target1 and nominal95 coverage targets0.95 only under the stated predictive Gaussian assumptions. These are aggregate diagnostics, not recalibration.
- Portfolio total-return errors/standard deviations multiply by the same(1+RF); standardized errors and coverage are unchanged. Reported NLL remains in discounted-excess units.
- Each month supplies one residual vector. Offdiagonal/error-outer-product discrepancies cannot identify true time-varying covariance; allocation performance is not proof of covariance accuracy.
- Finalist scores use only14eligible tasks and exclude the MGK historical placeholder principal row/column. Separate15-task expert scores use full GP observed covariance and includeMGK as a prediction task.
- Fixed eligible-EW projection uses identical14-ETF weights across all candidates; allocated projections use each candidate's saved raw weights and are not interchangeable distribution comparisons.
- Coverage from12 confirmation months is coarse; seeds are not independent market evidence.

The JSON includes every temporal block, monthly NLL/z²/coverage values, per-ETF metrics, offdiagonal discrepancies and source/decision hashes. Decision files matched before and after.
