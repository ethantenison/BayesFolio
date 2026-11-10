# Rank and risk allocation results

Continuous risk penalties made the multitask GP covariance influence holdings throughout the evaluation window, rather than only when a hard ceiling bound. None of the positive penalties tested beat the previous rank allocator's conventional Sharpe. Mild penalties preserved nearly all ranking preference with a small performance cost; stronger penalties lowered realized volatility and drawdown but sacrificed more return and Sharpe.

The comparison completed 56 cases and 4,032 monthly portfolios with no failed or substituted allocations and no GP refits. The same qualitative result appeared for both frozen prediction recipes. Keep the prior allocator as the performance benchmark. The new allocator provides an explicit risk tradeoff, not an established performance upgrade.

## Objective and fixed policy

For each month, the new allocator maximized:

`s @ w - lambda * (w @ covariance @ w / B_t) - 1e-8 * sum(w**2)`.

The scores `s` are the existing normalized ranks of blended predictive means. They are preferences, not predicted returns or probabilities. The final term is the original tiny concentration tie-breaker. `B_t` is the same expanding, training-only historical equal-weight variance used in the previous experiment. It remains both a hard variance ceiling and a common normalization for the penalty; it is not recomputed separately from each candidate covariance.

Lambda values were registered before evaluation: 0, 0.05, 0.1, 0.25, 0.5, 1 and 2. Four risk sources were compared: arithmetic 50/50 GP covariance, broad-model GP covariance, historical covariance, and a diagonal-only 50/50 GP matrix preserving its individual variances. The two prior leading prediction blends were frozen:

| Prediction recipe | Compact fraction for U.S. equities | International equities | U.S. fixed income | International fixed income |
|---|---:|---:|---:|---:|
| Prior Sharpe leader | 50% | 0% | 37.5% | 25% |
| Prior ranking and Sharpe compromise | 37.5% | 0% | 25% | 6.25% |

Both underlying models forecast all 15 ETFs jointly. The broad model uses 14 non-time inputs and the compact model five; both retain `t_index`, GP rank 2 and maxiter 15. This experiment changed neither inputs nor GP training. Forecast origins remain May 2024 through April 2026, with targets in the following month, seeds 27, 127 and 227, and an initial 84-month training window.

Every portfolio remains long-only, fully invested, with effective assets at least seven and the original absolute variance ceiling. There are no ETF caps, CVaR, fixed equity allocation, turnover objective, covariance repairs, budget relaxations or fallback weights. Returns are gross because transaction costs were not supplied. No uncertainty recalibration was fitted.

## Main performance comparison

These rows use the prior Sharpe prediction recipe and the same 50/50 GP covariance. Return is cumulative over the 24 target months; volatility is annualized arithmetic-excess volatility. Statistics average all three seeds. Drawdowns are signed, so a value closer to zero represents a smaller loss.

| Lambda | Conventional Sharpe | Cumulative return | Annualized volatility | Maximum drawdown | Rank advantage retained |
|---:|---:|---:|---:|---:|---:|
| 0, prior objective | 1.897252 | 60.26% | 10.56% | -6.36% | 100.00% |
| 0.05 | 1.893381 | 59.89% | 10.52% | -6.33% | 99.92% |
| 0.10 | 1.888121 | 59.41% | 10.47% | -6.31% | 99.66% |
| 0.25 | 1.866308 | 57.24% | 10.20% | -6.20% | 97.58% |
| 0.50 | 1.780331 | 49.73% | 9.25% | -5.92% | 87.36% |
| 1.00 | 1.569894 | 36.44% | 7.39% | -5.22% | 57.67% |
| 2.00 | 1.328502 | 26.71% | 5.84% | -4.28% | 19.89% |

Rank advantage retained is the fraction of the same covariance source's zero-penalty score advantage over equal weight that remains. It is not IC, expected return or the fraction of correct ETF rankings. Negative retention is possible when risk aversion selects a portfolio with a lower score than equal weight.

At lambda 0.25, holdings changed in all 72 seed-months even though the variance ceiling bound in none. At lambda zero the ceiling bound in only one seed-month. This resolves the previous experiment's mechanical limitation: risk now influences the objective below the ceiling.

Increasing the penalty moved capital from equities into fixed income. For the 50/50 GP source, average equity exposure fell from 80.94% at lambda zero to 75.31% at 0.25, 65.10% at 0.5 and 46.59% at 1. These are optimizer outputs, not imposed sleeve allocations. Their lower volatility came with a substantial opportunity cost during this particular market window.

The prior ranking and Sharpe compromise recipe followed the same pattern: Sharpe was 1.883108 at zero, 1.859210 at 0.25, 1.777374 at 0.5 and 1.565432 at 1. No positive penalty for either prediction recipe and any of the four covariance sources exceeded the overall prior leader's Sharpe.

![Conventional Sharpe and realized volatility at each registered risk penalty](outputs/rank_risk_tradeoffs.png)

## Covariance now changes allocations but has no universal winner

Broad-model GP covariance produced higher Sharpe than the 50/50 GP source at every positive tested lambda for both prediction recipes. Differences were small at mild penalties and larger at stronger penalties. For the Sharpe prediction recipe:

| Covariance source | Sharpe at lambda 0.25 | Volatility at lambda 0.25 | Sharpe at lambda 1 | Volatility at lambda 1 |
|---|---:|---:|---:|---:|
| 50/50 GP | 1.866308 | 10.20% | 1.569894 | 7.39% |
| Broad GP | 1.871040 | 10.20% | 1.624665 | 7.45% |
| Historical | 1.733115 | 7.13% | 0.950484 | 4.55% |
| GP variances only | 1.867837 | 10.42% | 1.718618 | 9.20% |

The diagonal-only control sometimes preserved more Sharpe at strong penalties, but it also took substantially more realized risk. Removing covariance made the allocator overlook shared ETF exposure. Its high-penalty Sharpe is not proof that off-diagonal dependence should be discarded.

The same caution applies to GP versus historical covariance. At identical lambda, GP portfolios generally had higher Sharpe, but historical covariance estimated materially higher risk and produced much more conservative allocations. Its zero-penalty ceiling bound in 63 of 72 seed-months, versus one for 50/50 GP. Thus an identical absolute ceiling does not imply identical realized risk across the risk estimators.

One descriptive near-matched-risk comparison illustrates the limitation. Historical covariance at lambda zero produced 7.448% volatility and Sharpe 1.703950, while broad GP at lambda 1 produced 7.447% volatility and Sharpe 1.624665. That does not establish a historical-risk winner either: selecting this comparison uses realized outcomes. It does show why higher Sharpe at the same lambda cannot by itself establish better covariance accuracy.

## Joint NLL and portfolio distribution scores remain distinct

Forecast means, rank scores, RMSE and IC stayed fixed within each prediction recipe. Pooled IC remains 0.348909 for the prior Sharpe recipe and 0.356597 for the compromise. Joint NLL also stayed exactly fixed across lambda for each recipe and covariance source. Changing allocations cannot improve these fixed forecast metrics.

For the Sharpe prediction recipe, fixed equal-weight distribution diagnostics were:

| Covariance | Joint NLL | Equal-weight NLL | Equal-weight CRPS | Nominal 95% coverage | Error RMS divided by predicted SD RMS |
|---|---:|---:|---:|---:|---:|
| 50/50 GP | -38.8312 | -2.1036 | 0.011685 | 84.72% | 1.4087 |
| Broad GP | -39.0486 | -2.1235 | 0.011697 | 84.72% | 1.4452 |
| Historical | -41.1988 | -2.3230 | 0.012234 | 100.00% | 0.6804 |
| GP variances only | -33.9519 | 0.0772 | 0.012524 | 61.11% | 2.5403 |

Lower NLL and CRPS are better; the absolute NLL values in the two columns are not comparable because their dimensions differ. The historical covariance had the best joint and equal-weight NLL here, but not the best CRPS or portfolio Sharpe. Its equal-weight intervals were much wider. The GP sources underestimated equal-weight forecast-error scale on this sample, whereas the historical source overestimated it. Removing off-diagonal covariance made equal-weight uncertainty estimates substantially worse.

Joint NLL remains a proper score of the full Gaussian hypothesis, not a direct ranking score or investment objective. The latest result does not justify calling it irrelevant: it shows that favorable joint NLL does not mechanically translate into higher portfolio Sharpe, even when covariance actively affects allocation. Four risk hypotheses on two fixed mean recipes and one short market history are insufficient to establish a general NLL-to-performance relationship.

All existing fixed-reference probes and adaptive portfolio uncertainty diagnostics are retained. No calibration model or scale adjustment was fitted.

## Decisions and remaining questions

1. Keep the previous rank allocator as the current Sharpe benchmark. There is no demonstrated Sharpe gain from adding a continuous risk penalty on this grid.
2. Keep the new rank-risk allocator available for an explicit lower-risk preference. A mild penalty preserves most performance, but its volatility reduction is also small. Strong penalties are a material asset-allocation decision, not a free improvement.
3. Retain both broad GP and 50/50 GP covariance as research controls. Broad GP performed somewhat better at positive penalties; that is not sufficient for a production change.
4. Do not remove cross-ETF covariance because a diagonal allocation has high Sharpe. The diagonal source materially understates fixed-portfolio uncertainty and generally takes more realized risk at the same penalty.
5. Before claiming any risk estimator is superior for investment purposes, separate covariance scale from correlation structure and compare comparable risk tradeoffs. This experiment reveals that issue but does not fit calibration or add a new search.

The remaining policy question is the desired return-versus-risk tradeoff. Maximizing Sharpe on this reused sample favors the old objective. Choosing a lower volatility target could justify a different point, but it should be an explicit preference rather than a new retrospective winner rule. No production model, allocator or real portfolio was changed.

All 24 market months were reused after previous adaptive searches. The three seeds are GP initialization replicates, not 72 independent market observations. These results remain exploratory and do not resolve untouched-period performance or covariance calibration.

## Verification and artifacts

The [complete 56-case table](outputs/all_configurations.md), [machine-readable comparison](outputs/comparison.json), [paired effects and family exposures](outputs/paired_comparisons.json), [fixed-reference scores](outputs/fixed_reference_summary.json), [audit](outputs/audit.json) and [registered contract](outputs/contract.json) preserve the definitions and results. Every case includes JSON and NPZ evidence with complete means, covariances, rank coefficients and raw holdings.

Both original 50/50 zero-penalty controls replayed with maximum holding differences below 1.9e-13. The independent audit checked all 4,032 raw portfolios, exact common forecasts, positive-definite covariance, unchanged absolute budgets, effective count, solver statuses, utility algebra, and realized return, conventional Sharpe, volatility and drawdown. Predicted risk and rank score decreased along each monthly lambda frontier within numerical tolerance. Two individual solves needed a numerical retry; both finished with strict optimal status on the original problem. No inaccurate or substitute weights were accepted.

The 68-test combined suite passed, including 15 new tests. Ruff lint and formatting passed, and an experiment-specific AST test checked module and public-symbol docstrings because the normal repository docstring gate excludes experiments. Editor Pylance diagnostics were unavailable. All protected prior inputs and output hashes verified unchanged.

The runner used 14 processes, available cores minus two, with one BLAS thread per worker. Evaluation and audit took about 4.4 seconds, excluding tests, charts and final sealing. No GP fitting time was incurred. The write-page guidance kept this report in the existing local findings format rather than creating an unrelated cloud artifact.

Verification from the repository root:

```sh
.venv/bin/ruff check --no-force-exclude experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1
.venv/bin/ruff format --check --no-force-exclude experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache uv run --locked --no-sync pytest -q experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1 experiments/2026-10-portfolio-evaluation-v1/covariance_matching_v1 experiments/2026-10-portfolio-evaluation-v1/blend_interactions_v2 experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1 experiments/2026-10-portfolio-evaluation-v1/joint_source_hybrids_v2 experiments/2026-10-portfolio-evaluation-v1/minimum_seven_hybrid_v1 experiments/2026-10-portfolio-evaluation-v1/allocation_numerics_v2
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1/run_rank_risk.py --verify
```

All work is isolated in this new folder. Rollback requires no production changes: preserve desired evidence before discarding only this experiment directory.
