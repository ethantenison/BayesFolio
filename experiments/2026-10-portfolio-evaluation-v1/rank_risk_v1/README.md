# Rank and risk portfolio allocation experiment

This cached comparison tests whether the multitask GP covariance helps portfolio construction when risk continuously affects the allocation objective. The previous allocator used covariance only in a hard ceiling that rarely bound. Both leading prediction recipes and all original portfolio constraints remain fixed.

## Registered comparison

The objective is `rank_score @ weights - lambda * (weights @ covariance @ weights / B_t) - 1e-8 * sum(weights**2)`. The tiny final term is the unchanged concentration tie-breaker. `B_t` is the same expanding, training-only historical equal-weight portfolio variance for every candidate. It is both the original hard ceiling and a common dimensionless normalization, never a candidate-specific rescaling.

Lambda values are fixed before running: 0, 0.05, 0.1, 0.25, 0.5, 1 and 2. The four covariance sources are arithmetic 50/50 GP, broad-model GP, training-only historical, and diagonal-only 50/50 GP. The diagonal control preserves individual variances but removes cross-ETF covariance. All means are unchanged; pairing blended means with a covariance hypothesis is not a newly fitted single GP posterior.

The two prediction recipes use compact fractions in this order: U.S. equity, international equity, U.S. fixed income, international fixed income.

| Recipe | U.S. equity | International equity | U.S. fixed income | International fixed income |
|---|---:|---:|---:|---:|
| Prior Sharpe leader | 50% | 0% | 37.5% | 25% |
| Prior ranking and Sharpe compromise | 37.5% | 0% | 25% | 6.25% |

The 56 cases cover seeds 27, 127 and 227 and all 24 forecast origins from May 2024 through April 2026. Source GPs retain rank 2, maxiter 15, `t_index`, all 15 ETF tasks and an initial 84-month training window. No new fits or data repairs occur. The allocator remains long-only and fully invested, with minimum effective assets seven, the unchanged variance ceiling, no ETF position caps, no CVaR, no fixed equity allocation and no fallback weights. No uncertainty recalibration or turnover objective is added.

## Evaluation and verification

The zero-penalty 50/50 controls must reproduce both prior portfolios before the rest of the grid launches. Every result saves complete forecasts, covariance matrices, raw weights, solver statuses, forecast scores, fixed-reference distribution scores, adaptive portfolio uncertainty diagnostics, and portfolio performance. Failed months cannot be omitted or filled.

Primary comparisons are conventional Sharpe, compounded return, realized volatility, drawdown and retained rank-score advantage. Rank retention measures the fraction of a covariance source's zero-penalty score advantage over equal weight that remains; it can be negative and is not return, IC, or a confidence probability. Forecast IC and NLL cannot change with lambda when the forecast distribution is fixed. Joint NLL remains a diagnostic, not a portfolio objective.

Increasing lambda should decrease predicted normalized variance and rank score along each monthly efficient frontier, up to numerical tolerance and the tiny tie-breaker. Audits recompute holdings constraints and portfolio statistics independently, verify utility algebra, check frozen forecasts, and verify protected input hashes. Independent worker processes use at most available cores minus two, capped at 15, with one BLAS thread per worker. All monthly allocations are completed before cross-case aggregation.

All results are gross because costs are unspecified. The 24 market months have already been examined repeatedly; three seeds are initialization replicates, not independent market histories. Reported leaders are exploratory, not untouched validation or production promotion.

## Commands and rollback

Run from the repository root:

```sh
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1/run_rank_risk.py --execute
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1/analyze_rank_risk.py
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1/run_rank_risk.py --seal
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1/run_rank_risk.py --verify
```

Completed case checkpoints are immutable and resumable. Preserve desired artifacts before discarding only this new experiment folder; no production or prior-study rollback is required.
