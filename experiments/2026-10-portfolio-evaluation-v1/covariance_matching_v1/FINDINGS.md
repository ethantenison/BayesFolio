# Covariance matching with fixed prediction blends

Matching covariance weights to prediction weights was mathematically valid, but did not materially improve portfolio results for either leading prediction recipe. The existing 50/50 covariance control remained numerically highest on Sharpe, with differences too small to establish a meaningful advantage over the matched methods.

The main finding is about the allocator: covariance enters only through a hard variance ceiling, not a continuous risk penalty in the objective. That ceiling was active in just one of 72 seed-months for each 50/50 control. Once the ceiling is satisfied, changing covariance usually cannot change the rank-maximizing portfolio. This comparison therefore reveals little about which covariance would be preferable under a different allocation objective or a tighter risk ceiling; neither was changed here.

## Frozen predictions and unchanged portfolio policy

Two prediction recipes were held fixed throughout. Fractions below are compact-model forecast weights, with the remainder supplied by the broad model.

| Family | Sharpe prediction recipe | Ranking and Sharpe compromise recipe |
|---|---:|---:|
| U.S. equities | 50% | 37.5% |
| U.S. fixed income | 37.5% | 25% |
| International equities | 0% | 0% |
| International fixed income | 25% | 6.25% |

Both source GPs forecast all 15 ETFs jointly; the broad model has 14 non-time inputs and the compact model five. Both retain `t_index`. Rank remained 2, maxiter 15, and the cached forecast origins remained May 2024 through April 2026, with targets in the following month. Seeds were 27, 127 and 227. There were no new GP fits or feature searches.

The full 15-ETF allocator remained long-only, fully invested, rank-score maximizing, and subject to at least seven effective assets and the exact same monthly absolute variance ceiling. That ceiling comes from the expanding training-only historical equal-weight portfolio. There were no ETF caps, CVaR, fixed 80/20 capital split, fallback weights or budget relaxations. All returns are gross because costs were not supplied.

Pooled ranking IC was exactly unchanged across covariance methods: 0.348909 for the Sharpe recipe and 0.356597 for the compromise. Predicted means, allocation scores and RMSE were also verified unchanged. Covariance comparisons cannot improve these mean-only forecast metrics when the means are fixed.

## Covariance methods

Twenty cases covered ten methods for each frozen prediction recipe. Every case completed all three seeds and 24 months: 1,440 portfolios in total.

The seven primary methods were broad-only covariance, compact-only covariance, arithmetic 50/50 covariance, family matching, variance-only matching, correlation-only matching, and a fully shared latent-shock matrix-root construction. Two additional latent-sharing levels were sensitivity cases. An independence assumption was a diagnostic only, excluded from winner selection.

Let A be the diagonal matrix of per-ETF compact prediction fractions, E = I − A, and H the arithmetic 50/50 covariance.

### Family matching

The family gate was:

`G = sqrt(E) Σ_broad sqrt(E) + sqrt(A) Σ_compact sqrt(A)`.

Within each family it applies the same broad/compact weights to covariance as to predictions. Cross-family covariance uses geometric products of the two ETFs' fractions. The sum of congruence transformations preserves positive semidefiniteness, and every actual output passed strict positive-definiteness checks.

This construction can attenuate cross-family covariance solely because their weights differ, even if the two source covariance matrices were identical. Validity alone does not establish that the attenuation represents actual market dependence.

Two controls separated its effects:

- **Variance-only matching:** use the diagonal variances of G, but preserve H's correlation matrix.
- **Correlation-only matching:** use G's correlation matrix, but preserve H's diagonal variances.

These are deterministic covariance constructions based only on saved forecasts, not calibration fitted to realized errors.

### Shared latent shocks

With symmetric positive-definite matrix square roots S_broad and S_compact, define L = E S_broad and M = A S_compact. The covariance was:

`Σ_rho = L Lᵀ + M Mᵀ + rho (L Mᵀ + M Lᵀ)`.

This corresponds to a declared joint Gaussian coupling with standardized latent-shock cross-covariance rho × I. Rho = 1, 0.9 and 0.5 were considered; rho = 0 was the independence diagnostic. Rho is a hypothesized latent dependence parameter, not a measured correlation between the models' forecast errors. In particular, rho = 1 does not assert that every pair of observed model errors is perfectly correlated.

The fully shared construction preserves identical source covariance matrices under any family weights. None of these methods estimates the unavailable cross-model forecast-error covariance, and none is claimed to be the posterior of a newly fitted single GP. No mixture-of-models disagreement term is added, because this is a coupled linear uncertainty construction rather than a random model-selection mixture.

## Portfolio results

Sharpe below is conventional arithmetic-excess Sharpe, averaged across the three seeds. The variance ratio evaluates each candidate covariance on the same frozen 50/50-control holdings, so it isolates estimated-risk changes from changes in portfolio weights.

| Covariance | Sharpe recipe Sharpe | Compromise recipe Sharpe | Fixed control portfolio variance ratio for Sharpe recipe |
|---|---:|---:|---:|
| Broad only | 1.896759 | 1.882789 | 0.9423 |
| Compact only | 1.894624 | 1.880338 | 1.0577 |
| 50/50 control | **1.897252** | **1.883108** | 1.0000 |
| Family matched | 1.897221 | 1.883076 | 0.9039 |
| Match variances only | 1.897221 | 1.883076 | 0.9663 |
| Match correlations only | 1.897221 | 1.883076 | 0.9354 |
| Fully shared matrix-root construction | 1.897221 | 1.883076 | 0.9623 |

The 50/50 control's cumulative returns were 60.2632% and 60.1996% for the two prediction recipes. Family matching returned 60.2597% and 60.1961%. The difference was approximately 0.0035 percentage points of total return and 0.00003 Sharpe—not a meaningful portfolio gain or loss.

The matched methods changed material weights in only one seed-month relative to the controls: seed 227 at the April 30, 2025 forecast origin. The maximum holding change was approximately 1.41 percentage points. Broad-only covariance changed weights in three seed-months, and compact-only covariance in two. Every accepted portfolio retained the original constraints.

Family matching lowered estimated variance of the frozen control holdings by about 9.6% for the Sharpe recipe and 8.2% for the compromise. The variance-only control lowered it by about 3.4% and 4.0%. These are changes in **predicted variance**, not measured reductions in realized portfolio risk. Because the variance ceiling ceased to bind, all these matched constructions produced almost identical allocations.

![Covariance sensitivity and portfolio Sharpe changes](outputs/covariance_sensitivity.png)

The independence diagnostic reduced predicted control-portfolio variance by approximately 35.7% and 31.7%, yet still produced essentially the same portfolio results. That is not evidence that independent errors are correct: it shows how a strong uncertainty assumption can change the risk estimate without affecting an allocator whose ceiling is inactive.

## Distribution scores did not select one universal winner

For the Sharpe prediction recipe, the following scores illustrate the disagreement between the complete 15-dimensional density and one fixed reference portfolio. Lower NLL is better in both columns; absolute values differ because their dimensions differ.

| Covariance | Joint predictive NLL | Equal-weight portfolio NLL | EW error RMS divided by predicted SD RMS |
|---|---:|---:|---:|
| Broad only | −39.0486 | **−2.1235** | 1.4452 |
| 50/50 control | −38.8312 | −2.1036 | 1.4087 |
| Family matched | −38.9250 | −2.0451 | 1.4957 |
| Match variances only | −38.9617 | −2.1166 | 1.4261 |
| Independence diagnostic | **−39.8772** | −1.6335 | 1.8157 |

Family matching improved joint NLL relative to 50/50 but worsened the fixed equal-weight portfolio's NLL and CRPS. The independence diagnostic made this divergence particularly visible: its joint NLL improved most, while equal-weight NLL, CRPS and interval coverage deteriorated. For the Sharpe recipe its nominal 95% equal-weight coverage fell from 84.7% to 80.6%; for the compromise it fell from 86.1% to 79.2%.

Variance-only matching modestly improved both joint NLL and equal-weight NLL versus 50/50, and had the lowest equal-weight CRPS in this comparison, although the CRPS differences were tiny. Broad-only covariance had the lowest equal-weight NLL for both prediction recipes. Thus, matching variances while retaining the original correlations is worth retaining as a diagnostic candidate; matching the entire covariance did not dominate.

The proper interpretation is not that joint NLL is intrinsically invalid. It evaluates the full distribution in all ETF directions, whereas equal-weight scoring evaluates one portfolio direction. Their empirical rankings can disagree, especially on a short reused sample. Neither should be substituted for the other when assessing portfolio-relevant uncertainty.

All original fixed-reference probes, asset-level diagnostics and adaptive-portfolio diagnostics were preserved. No uncertainty recalibration was fitted, and the comparisons do not establish reliable coverage from only 24 market months.

## Takeaways

1. **Covariance weights do not need to mechanically match prediction weights.** We tested valid matching constructions, and they did not materially improve this allocator's return or Sharpe.
2. **Keep the 50/50 covariance as the unchanged portfolio benchmark, not as a proven optimal covariance forecast.** Broad-only was nearly tied. Variance-only matching has some favorable distribution-score behavior but no meaningful portfolio benefit here.
3. **The current allocation objective is insensitive to covariance once its hard risk ceiling is satisfied.** Near-identical Sharpe does not establish that the covariances are equally accurate or would be interchangeable under a different risk-sensitive objective.
4. **Do not adopt the independence diagnostic because it has lower joint NLL.** It assumes away shared uncertainty and performed worse on fixed equal-weight distribution scores. Its role was to expose that failure mode, not supply a winner.

All comparisons reuse the same forecast window after previous adaptive searches. The three seeds are model initializations on one market history, not 72 independent market months. No production model, portfolio policy or investment allocation was changed.

## Artifacts and verification

The [all-method table](outputs/all_configurations.md), [complete comparison and risk diagnostics](outputs/comparison.json), [fixed-reference scores](outputs/fixed_reference_scores.md), and [frozen contract](outputs/contract.json) preserve the definitions and results. Every case has JSON and NPZ files containing all seeds, monthly forecasts, complete covariance matrices and raw holdings.

The audit independently recomputed return, conventional Sharpe, volatility and drawdown from raw holdings and returns; checked covariance positive definiteness, original absolute budgets, effective counts and strict accepted solver statuses; verified means/scores/IC remained fixed; and replayed both 50/50 controls against their saved results. All 1,440 portfolios passed. No allocations failed, no inaccurate weights were accepted, and no covariance repair, clipping or fallback weights were used.

The new and related regression suites passed 53 tests, including matrix identities, matching endpoints, dependence assumptions, covariance/component controls and worker isolation. Ruff and the public-docstring gate passed. Editor Pylance diagnostics were not run. Prior sealed source and result hashes verified unchanged. All new work is isolated in this folder.

Verification from the repository root:

```sh
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/covariance_matching_v1
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/covariance_matching_v1 experiments/2026-10-portfolio-evaluation-v1/blend_interactions_v2 experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1 experiments/2026-10-portfolio-evaluation-v1/joint_source_hybrids_v2 experiments/2026-10-portfolio-evaluation-v1/minimum_seven_hybrid_v1 experiments/2026-10-portfolio-evaluation-v1/allocation_numerics_v2
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/covariance_matching_v1/run_covariances.py --verify
```

`run_covariances.py --execute` resumes completed immutable case outputs; `outputs/analyze_results.py` reproduces fixed-reference summaries and the plot without new allocations. To discard this research later, preserve desired artifacts and remove only this new folder; no production rollback is required.
