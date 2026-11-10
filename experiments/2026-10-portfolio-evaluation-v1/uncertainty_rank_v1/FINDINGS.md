# Uncertainty aware ranking results

Gaussian expected ranks did not improve the current portfolio strategy on the reused evaluation window. The original hard ranks had higher conventional Sharpe and return for both frozen prediction recipes. Expected ranks changed holdings in every seed-month but produced almost the same realized volatility as the controls. Unlike the preceding continuous risk penalty experiment, this was not a useful exchange of return for materially lower volatility.

All eight cases and 576 monthly portfolios completed successfully, with no numerical retries, fallback weights, constraint changes or GP refits. Both original hard-rank controls reproduced their previous raw holdings exactly. Retain hard ranks as the current research signal; the tested uncertainty transformation is not a demonstrated improvement.

## What changed and what stayed fixed

The primary new score was expected rank divided by the 15-ETF count:

`score_i = (1 + sum(j != i, P(Y_i > Y_j))) / 15`

`P(Y_i > Y_j) = Phi((mu_i - mu_j) / sqrt(var_i + var_j - 2 * cov_ij))`.

The uncertainty is the complete observed next-month return covariance, including likelihood noise. The probabilities describe realized outperformance, not confidence in the order of latent expected returns. Blended means and the existing 50/50 GP covariance define a Gaussian hypothesis, not the posterior of a newly fitted single GP. The probabilities are model-implied and were not recalibrated.

The two frozen mean recipes were the same prior Sharpe leader and prior ranking and Sharpe compromise. Their compact fractions, in family order U.S. equities, international equities, U.S. fixed income and international fixed income, remained `[0.5, 0, 0.375, 0.25]` and `[0.375, 0, 0.25, 0.0625]`.

Each recipe had four registered score methods:

1. Hard ranks of the means, the primary control.
2. Full Gaussian expected ranks, used directly, the primary new hypothesis.
3. Expected ranks with cross-ETF covariance removed from the score calculation, a diagnostic.
4. Hard re-ranking of full Gaussian expected scores, a diagnostic separating ordering changes from score spacing.

For every case, allocation risk still used exactly the same full 50/50 GP covariance. The original strict allocator maximized score-weighted holdings with only its existing 1e-8 concentration tie-breaker. No continuous risk penalty was added. Portfolios remained long-only and fully invested, with effective assets at least seven and the unchanged training-only historical equal-weight variance ceiling. There were no ETF position caps, CVaR, fixed equity allocation, covariance repairs, turnover objective, fallback weights or budget relaxation.

Both underlying source GPs still forecast all 15 ETFs jointly, retaining rank 2, maxiter 15, `t_index` and the initial 84-month training window. Seeds were 27, 127 and 227. All 24 forecast origins from May 2024 through April 2026 and their following-month targets were retained. No GP fitting or input-variable selection occurred.

## Portfolio and ranking results

Return is gross cumulative total return over the 24 target months. Volatility is annualized arithmetic-excess volatility. All figures average three seeds; IC refers to the actual allocation score rather than the unchanged raw predictive means.

| Prediction recipe | Score method | Conventional Sharpe | Cumulative return | Volatility | Decision score IC |
|---|---|---:|---:|---:|---:|
| Prior Sharpe leader | Hard ranks | **1.897252** | **60.26%** | 10.56% | 0.348909 |
| Prior Sharpe leader | Expected ranks with full covariance | 1.841785 | 58.44% | 10.56% | 0.347966 |
| Prior Sharpe leader | Expected ranks with diagonal uncertainty | 1.841038 | 58.58% | 10.59% | 0.349851 |
| Prior Sharpe leader | Hard re-ranking of expected ranks | 1.870926 | 58.96% | 10.49% | 0.347966 |
| Prior ranking and Sharpe compromise | Hard ranks | **1.883108** | **60.20%** | 10.63% | **0.356597** |
| Prior ranking and Sharpe compromise | Expected ranks with full covariance | 1.818237 | 58.05% | 10.63% | 0.346925 |
| Prior ranking and Sharpe compromise | Expected ranks with diagonal uncertainty | 1.816109 | 58.14% | 10.66% | 0.349355 |
| Prior ranking and Sharpe compromise | Hard re-ranking of expected ranks | 1.857105 | 58.74% | 10.53% | 0.346925 |

The primary expected-rank method lost 0.055467 Sharpe and 1.83 percentage points of cumulative return for the prior Sharpe recipe. For the compromise it lost 0.064872 Sharpe and 2.15 percentage points of cumulative return. Every expected-rank variant had lower Sharpe than its matched hard-rank control in each of the three seeds. Those seeds share one market history, so this consistency is not independent validation.

The prior Sharpe recipe's maximum drawdown changed from -6.36% to -6.43% under full expected ranks. There was no compensating improvement in drawdown. Effective assets stayed at seven to solver tolerance. The variance ceiling remained active in only one seed-month for every case, just as for the controls.

![Ranking scores and conventional Sharpe under the unchanged allocator](outputs/expected_rank_comparison.png)

![Mean individual seed portfolio wealth for the four score methods](outputs/expected_rank_wealth.png)

## Ordering and score spacing both mattered

For the prior Sharpe recipe, expected ranks changed the cross-ETF ordering in 70 of 72 seed-months; for the compromise, in 71. The mean absolute rank-position change was only 0.45 and 0.53 places per ETF, respectively, with a maximum move of four positions. These were frequent small changes, not a complete reversal of the rankings.

Full expected scores changed holdings materially in all 72 seed-months. The average one-way difference from the control holdings was 8.36% of capital for the prior Sharpe recipe and 8.93% for the compromise. This is a distance between two simultaneous portfolios, not measured turnover across rebalance dates or a transaction-cost objective. The largest individual holding differences were 7.31 and 7.10 percentage points.

Hard re-ranking of expected scores preserves exactly the same ordering and decision IC as direct expected scores, but replaces their uneven gaps with equally spaced rank positions. That improved Sharpe from 1.841785 to 1.870926 for the prior Sharpe recipe and from 1.818237 to 1.857105 for the compromise. Both still trailed their original controls. On this sample, the changed ordering was not an improvement, and using the new relative score gaps made allocation performance worse still.

The raw expected-score spreads averaged only 33.8% and 32.8% of the hard-rank spreads. Uniform compression alone is not the explanation: an affine rescaling of a linear score objective leaves its optimizer unchanged apart from the tiny concentration tie-breaker. The material differences arise from changed ordering and nonuniform spacing, as the reranking diagnostic demonstrates.

Average equity exposure moved from 80.94% to 78.82% for the prior Sharpe recipe and from 81.24% to 78.80% for the compromise. These modest capital shifts and the changed ETF weights did not reduce realized volatility meaningfully.

## Family ranking effects were not uniform

These are within-family ranking metrics of the globally computed allocation scores, not results from new family-only GPs or separately optimized family portfolios.

| ETF family | Sharpe recipe hard ranks IC | Sharpe recipe full expected IC | Compromise hard ranks IC | Compromise full expected IC |
|---|---:|---:|---:|---:|
| U.S. equities | 0.180556 | 0.088889 | 0.163889 | 0.063889 |
| International equities | 0.141667 | 0.175000 | 0.141667 | 0.172222 |
| U.S. fixed income | 0.405556 | 0.383333 | 0.436111 | 0.402778 |
| International fixed income | 0.166667 | 0.173611 | 0.145833 | 0.152778 |

Expected-rank adjustments modestly improved international-family IC point estimates while worsening U.S. equity and U.S. fixed-income ordering. That helps explain why a globally near-unchanged pooled IC can conceal different family behavior. It does not establish that another family-specific blending search would succeed, and no such search was added.

This experiment cannot diagnose the original U.S. equity return-forecast collapse: its predictive means were deliberately frozen. It shows that the current uncertainty-aware ranking transformation did not repair that problem.

## Pairwise probabilities did not provide an allocation advantage

The full and diagonal Gaussian score hypotheses had the following pairwise losses. Lower is better. The same 105 unordered ETF pairs were scored once per seed-month, with no observed ties.

| Prediction recipe | Ranking uncertainty | Pairwise Brier loss | Pairwise log loss |
|---|---|---:|---:|
| Prior Sharpe leader | Full covariance | 0.232643 | 0.660526 |
| Prior Sharpe leader | Diagonal uncertainty | 0.232288 | 0.658719 |
| Prior ranking and Sharpe compromise | Full covariance | 0.231910 | 0.658691 |
| Prior ranking and Sharpe compromise | Diagonal uncertainty | 0.231675 | 0.657299 |

For context, assigning every pair a 50/50 probability gives Brier loss 0.25 and log loss 0.693147. The Gaussian probabilities beat that uninformative reference here, but including cross-ETF covariance did not improve these pairwise losses relative to the diagonal score ablation. Differences were small and the pairs are correlated.

These losses describe the frozen probability hypothesis; allocating from expected ranks does not itself improve that hypothesis. Hard ordinal ranks do not claim certainty probabilities, so their ordering should not be judged as if every rank position were an asserted probability of one.

Raw-mean IC, RMSE and the complete joint predictive NLL stayed exactly fixed across all score methods within each recipe. Joint NLL remained -38.831202 for the prior Sharpe recipe and -38.849306 for the compromise. Original fixed-reference distribution scores and adaptive portfolio uncertainty diagnostics were also retained. The diagonal score ablation did not replace the Gaussian covariance used for those portfolio distribution evaluations.

## Takeaways and limits

1. Keep the existing hard ranks as the current research default. Neither continuous risk penalties nor this expected-rank transformation established a Sharpe improvement on the tested window.
2. Do not promote direct expected scores or their reranked version. The diagnostics explain the loss but do not supply a winning replacement.
3. Do not infer that multitask uncertainty or covariance has no value. This test concerns one uncalibrated Gaussian ranking hypothesis and one allocator. It does show that adding uncertainty to the signal is not automatically an improvement.
4. Avoid another broad search over ranking transformations merely to force a win on these months. The stronger evidence remains the performance of the existing hard-rank benchmark; model-family heterogeneity and covariance quality remain separate open questions.

The experiment reused the same 24 already-examined market months. Three seeds and thousands of ETF-pair events are not independent histories. Expected-rank utility concerns relative realized performance, not directly expected monetary return or portfolio Sharpe. The results do not establish whether a differently estimated or calibrated uncertainty model would help, and no recalibration was fitted. No production allocator, model or real portfolio was changed.

## Artifacts and verification

The [full eight-case table](outputs/all_configurations.md), [comparison JSON](outputs/comparison.json), [paired changes and family exposures](outputs/paired_comparisons.json), [registered contract](outputs/contract.json) and [audit](outputs/audit.json) contain the results. Every case has JSON and NPZ evidence with complete means, allocation covariance, pairwise probabilities, difference variances, expected scores, actual allocation coefficients and raw holdings. Fixed-reference and adaptive distribution metrics remain in each case JSON.

All 576 raw portfolios passed independent checks of original constraints and realized performance. All accepted solver statuses were strictly optimal, without retries or substitute weights. Both hard-rank controls replayed exactly. The audit checked unchanged means and allocation covariance, pairwise complements, rank totals and bounds, recomputed scores, complete monthly decision IC and protected input hashes. All prior sealed evidence remained unchanged.

The combined suite passed 89 tests, including 21 new tests. Synthetic coverage includes an independent 200,000-draw Gaussian simulation, common-shock cancellation, return-unit and asset-order invariance, the zero-noise limit, ties, covariance ablation, unchanged worker allocator and stable log-CDF scoring without probability clipping. Ruff and formatting passed. An experiment-specific AST test checked module and public-symbol documentation; editor Pylance diagnostics were not run.

Eight process workers used one BLAS thread each. Allocation, scoring and audit took about 2.8 seconds, excluding tests, chart generation and final sealing. No GP fitting time was incurred. The local report and figures follow the existing experiment format, with all work isolated in this new folder.

Verification from the repository root:

```sh
.venv/bin/ruff check --no-force-exclude experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1
.venv/bin/ruff format --check --no-force-exclude experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache uv run --locked --no-sync pytest -q experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1 experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1 experiments/2026-10-portfolio-evaluation-v1/covariance_matching_v1 experiments/2026-10-portfolio-evaluation-v1/blend_interactions_v2 experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1 experiments/2026-10-portfolio-evaluation-v1/joint_source_hybrids_v2 experiments/2026-10-portfolio-evaluation-v1/minimum_seven_hybrid_v1 experiments/2026-10-portfolio-evaluation-v1/allocation_numerics_v2
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1/run_uncertainty_rank.py --verify
```

Rollback needs no production change. Preserve desired evidence before discarding only this new experiment directory.
