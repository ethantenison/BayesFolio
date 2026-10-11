# Return-Target Comparison Decision

## Decision

Retain adjusted total excess return as the production multitask-GP target.
Do not replace it with price-only return. Treat decomposition as a useful
research direction, not as a production promotion from this experiment.

## Evidence

The experiment used five representative ETFs (`SPY`, `IJR`, `IEF`, `VEA`, and
`BNDX`) and the established 24 monthly walk-forward decisions from 2024-05-31
through 2026-04-30. Training was expanding-window from March 2021. All
decision metrics and portfolio realizations used adjusted total excess return,
even when a model was trained on a different target.

### Prediction quality

| Target | RMSE | MAE | Bias | Direction accuracy | IC |
|---|---:|---:|---:|---:|---:|
| Total return | 0.03517 | 0.02506 | -0.00536 | 0.508 | 0.1125 |
| Decomposed | 0.03611 | 0.02545 | -0.00443 | 0.500 | 0.0708 |
| Price only | 0.03650 | 0.02628 | -0.00695 | 0.467 | 0.0917 |

The block Bayesian bootstrap assigns a 98.9% posterior probability that total
return has lower RMSE than decomposition and a 99.9% probability that it has
lower RMSE than price-only. The MAE evidence is weaker against decomposition
(87.6%) but decisive against price-only (100%).

The distribution component itself is directionally predictable: 80.8%
direction accuracy and IC 0.208 (p = 0.040). Its pooled R-squared is negative,
however, so this result supports further component research rather than a
standalone magnitude forecast claim.

### Uncertainty quality

| Target | NLPD | CRPS | 80% coverage | 80% interval score | Standardized residual SD |
|---|---:|---:|---:|---:|---:|
| Total return | -2.1518 | 0.01798 | 0.875 | 0.11489 | 0.791 |
| Decomposed | -2.1188 | 0.01834 | 0.892 | 0.11913 | 0.812 |
| Price only | -2.1055 | 0.01867 | 0.842 | 0.11985 | 0.828 |

Total return has the best proper scores. The posterior probability that its
NLPD is better is 100% against both alternatives; for CRPS it is 99.8% against
decomposition and 100% against price-only. All three models are somewhat
overdispersed: standardized residual SD is below one and nominal interval
coverage is generally high.

### Portfolio quality

| Setup | Cumulative return | Annualized return | Sharpe | Max drawdown | Average turnover |
|---|---:|---:|---:|---:|---:|
| Total return GP | 18.75% | 8.97% | 0.814 | -10.57% | 18.82% |
| Decomposed GP | 17.09% | 8.21% | 0.893 | -6.93% | 31.23% |
| Price-only GP | 15.03% | 7.25% | 0.702 | -10.21% | 25.72% |
| Equal weight | 18.45% | 8.83% | 1.048 | -5.39% | 0.91% |

Total return produced the best GP cumulative return and the lowest GP turnover.
Decomposition delivered a better Sharpe ratio and drawdown but lower return and
substantially higher turnover. The probability that total return has the higher
mean monthly return is 70.4% versus decomposition and 93.6% versus price-only;
both 95% credible intervals cross zero, so the portfolio ranking is not
decision-grade on 24 months alone. Equal weight had the best risk-adjusted
result, underscoring that this compact universe is a diagnostic experiment and
not evidence for production allocation changes.

## Caveats

- The decomposed posterior adds independent price and distribution draws. A
  future test should model their residual dependence or use a joint likelihood.
- Riskfolio reported infeasibility in several windows. The maintained adapter
  does not expose solver status and may fall back to equal weights. Exact
  equal-weight windows are listed in `portfolio_results.json`.
- No transaction costs were charged. This favors the higher-turnover
  decomposed setup.
- The results do not establish behavior for the full BayesFolio ETF universe.
- Cholesky stabilization added `1e-8` diagonal jitter, consistent with the
  established regression workflow.

## Reproducibility

An offline replay from the frozen input panel reproduced the GP configuration,
all three result JSON files, detailed prediction and uncertainty rows, portfolio
weights, and input panel byte-for-byte. Fit diagnostics matched except for the
wall-clock `fit_seconds` field.

See the selected evidence directory for the manifest, GP configuration,
frozen inputs, detailed rows, and result JSON files:
`evidence/20261011_three_target_comparison_final/`.
