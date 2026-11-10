# What the v2 investigation established

All 15 ETFs remained in every GP. MGK was excluded only from the 14-asset investment axis, and all target weights were recomputed. All model windows/features stayed fixed; maxiter15 was unchanged. Twelve months is the evaluation-block length.

## Allocation and covariance

The selected policy has NE floor 3, risk multiplier 1.0, no ETF cap and no equity cap. Lower concentration floors were useful only alongside the tested risk constraint and predictions; this is not a universal claim that fewer assets are better.
Against the matched three-model NE7 control, full53 CAGR changed from 10.62% to 14.64%, cumulative return from 56.16% to 82.87%, and maximum drawdown from -11.47% to -9.02%.
Covariance can affect this allocator when its variance constraint binds. The selected November allocation is risk-bound; earlier inactive-ceiling results did not establish that covariance was useless. The final covariance recipe remains 50/50 broad-expanding/compact-expanding. Historical covariance and tighter ceilings supplied defensive frontier points with lower growth, rather than a universal winner. This is allocation evidence, not an independent proof of covariance accuracy.

## Predictions and GP variants

The selected fixed family blend is almost entirely compact for U.S. equity, mostly broad expanding for international equity, and mostly broad rolling48 for fixed income. This supports conditional family-dependent ensemble specialization, not intrinsic causal input importance. Every expert still trains on all15 tasks.
Compact rank1 was compared after complete static-blend retuning and source/covariance routing controls. A rank1 defensive tradeoff reached the development frontier but was not the selected policy. The chosen experts all remain rank2/maxiter15. No documented kernel/prior malfunction supported an additional proposal; low amplitude and hitting the intentionally fixed iteration limit were not treated as failures. No unauthorized feature, window, rank or fitting-budget search was performed.

The selected full53 eligible14 ranking IC is 0.135, versus 0.145 for the matched reference. Portfolio growth/drawdown improved despite the lower whole-universe IC, which demonstrates why IC was supporting evidence rather than a veto.

## Chronology and limits

The latest12 return was 19.51%, versus 14.49% for the matched reference. Its latest12 drawdown was -8.06%, versus -6.42%: recent drawdown was worse, within the frozen3pp tolerance. Full-path drawdown improvement must not be misrepresented as improvement in every period.
Four eligible final Pareto policies reached the fixed aspiration distance0; the frozen annual-stability/Sharpe/IC/simplicity chain resolved selection. This is an explicit operational preference, not identification of a uniquely optimal return/risk point.
All histories were inspected before this campaign. The latest12 was locked procedurally until the shortlist froze, but is not pristine holdout evidence. The 16 unmatched rank1 exploratory leaves were preserved and uniformly excluded from promotion/rank attribution. No month or failed portfolio was substituted.

## Costs and tail-risk evidence

Gross selected cumulative return is 82.87%; the illustrative10bps-per-traded-dollar path is 69.80%. The full report compares matched costs for controls and both standing benchmarks. The narrow gross Schwab edge does not survive that10bps cost assumption.
The fixed November target has a static historical stress-path drawdown of -30.03%. Its hypothetical common-equity shock losses are -18.88% and -28.88%. This is not the dynamic GP backtest, a forecast or a crash probability. Excluding MGK did not create a crash-protected portfolio.
DISTRIBUTION_DIAGNOSTICS.md shows substantial directional uncertainty mismatch: selected full53 portfolio mean standardized squared error2.674 and nominal95% coverage83.0%. Joint average task diagnostics can conceal portfolio-direction errors. No recalibration was fitted and no postconfirmation retuning followed these diagnostics.

## Finite experiment ledger

| Round | Preserved cases including prior rounds/controls | Hypervolume | Best fixed-anchor distance |
|---|---:|---:|---:|
| 1 | 222 | 0.01439273 | 0.559621 |
| 2 | 470 | 0.01711893 | 0.000000 |
| 3 | 631 | 0.01833863 | 0.000000 |
| 4 | 876 | 0.01849538 | 0.000000 |

Four adaptive rounds reached the round ceiling after the required tracks were assessed.874 new complete configurations and2controls were preserved;10finalists were confirmed.Only6new fits were needed because compatible forecast caches already existed. Budgets were ceilings, not targets. All independent gates and deterministic reproduction still had to pass before completion.
