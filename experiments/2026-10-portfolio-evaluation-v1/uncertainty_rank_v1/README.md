# Uncertainty aware expected rank experiment

This cached comparison asks whether the GP distribution improves the ranking signal itself. It does not repeat the continuous risk penalty experiment. The existing hard ranks are compared with Gaussian expected ranks while the portfolio allocator, allocation covariance, predictive means and portfolio constraints remain unchanged.

## Registered comparison

For ETF i, `expected_rank_i = 1 + sum(j != i, P(Y_i > Y_j))`, with `P(Y_i > Y_j) = Phi((mu_i - mu_j) / sqrt(var_i + var_j - 2 * cov_ij))`. Allocation scores are expected ranks divided by 15. This preserves the original hard-rank normalization and its zero-uncertainty limit; no probabilities are clipped, no covariance is repaired, and no temperature or calibration is fitted. Self-comparisons never count as wins.

The distribution concerns next-month observed returns, including likelihood noise. It is not a posterior probability that one latent expected return is greater than another. Blended means paired with 50/50 GP covariance define the same Gaussian hypothesis used previously, not a newly fitted single GP posterior. The probabilities are model-implied, not established calibrated probabilities. Expected scores use mean gaps relative to uncertainty, so they are not purely ordinal signals.

Two frozen prediction recipes are used. Compact fractions are listed in the order U.S. equities, international equities, U.S. fixed income and international fixed income:

| Recipe | U.S. equity | International equity | U.S. fixed income | International fixed income |
|---|---:|---:|---:|---:|
| Prior Sharpe leader | 50% | 0% | 37.5% | 25% |
| Prior ranking and Sharpe compromise | 37.5% | 0% | 25% | 6.25% |

Each recipe has four score variants, eight cases total:

1. Hard ranks of means, the primary replay control.
2. Full Gaussian expected ranks used directly, the primary new hypothesis.
3. Expected ranks computed without cross-ETF covariance, a diagnostic of dependence information. Allocation risk still uses the full 50/50 GP covariance.
4. Hard re-ranking of full Gaussian expected scores, a diagnostic separating changed ordering from changed score spacing.

There is no adaptive second round, score blending search, covariance search, GP refit, risk penalty, recalibration, or minimum-variance rank-floor experiment. The primary comparison is expected ranks versus hard ranks for each fixed recipe, not retrospective selection across additional configurations.

## Fixed allocator and evaluation

The same strict allocator maximizes `scores @ weights - 1e-8 * sum(weights**2)`, subject to long-only fully invested holdings, effective assets at least seven and the unchanged absolute training-only historical equal-weight variance ceiling. The allocation covariance is exactly the same arithmetic 50/50 GP matrix in every case. There are no ETF caps, CVaR, fixed equity allocation, fallback weights or constraint relaxation.

The source models forecast all 15 ETFs jointly, retain GP rank 2, maxiter 15 and `t_index`, and start with 84 training months. The forecast origins are May 2024 through April 2026; targets are next-month returns. All seeds 27, 127 and 227 and all 24 months are retained. The two hard-rank controls must replay before launching the other cases.

Record conventional Sharpe, compounded return, realized volatility, drawdown, portfolio uncertainty diagnostics and fixed-reference distribution scores. Report decision-score IC separately from the unchanged raw-mean IC, RMSE and joint NLL. Expected scores can change ranking IC even though the underlying return means and density stay fixed. Save all pairwise probabilities and difference variances, ordering changes, score-spread changes and raw holdings.

Pairwise event Brier and log losses compare the full and diagonal Gaussian ranking hypotheses after decisions. Each of the 105 unordered pairs is counted once, in fixed ETF-axis order. The event is strictly `Y_i > Y_j`; observed ties are counted and treated as false for that event. The hard-mean ordering's Brier loss is a deterministic reference, not a calibrated probability forecast. Pairwise log loss is computed from log-CDFs without probability clipping.

The same 24 market months have been reused repeatedly. Three seeds and correlated ETF pairs are not independent histories. Returns are gross because costs are unspecified. Results are exploratory and do not authorize production promotion or real trades.

## Verification and rollback

Synthetic tests cover the Gaussian formula, pairwise complements, zero-noise limit, ties, unit and permutation invariance, Monte Carlo expected ranks, strict invalid-input rejection and unchanged worker allocator. Independent audits check all raw holdings, exact allocation covariance, scores and identities, complete months, strict solver statuses and protected source hashes. Process workers use at most cores minus two, capped at 15 and at the eight cases, with one BLAS thread each.

Run from the repository root:

```sh
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1/run_uncertainty_rank.py --execute
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1/analyze_uncertainty_rank.py
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1/run_uncertainty_rank.py --seal
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1/run_uncertainty_rank.py --verify
```

Completed checkpoints are immutable and resumable. All changes stay in this new folder. Preserve desired artifacts before discarding only this experiment; no production or prior-study rollback is required.
