# November 2026 decision contract

Activated October31,2026. User's latest caps supersede the earlier draft:
two batches of at most48 new configurations,128 monthly fit attempts including
retries, at most14 single-thread workers in one fit pool, shortlist at most15.
Final deadline November1 18:00 America/Chicago; exploration stops at16:30.
No requirement to exhaust a budget or manufacture a challenger victory.

## Immutable decision and investment boundaries

Write only this versioned folder. No commits/pushes/trades/account changes,
production changes, or edits to previous datasets/source/results. The October31
logical as-of is Saturday; market origin is Friday October30 (monthly BME),
predicting November. Record both dates; do not invent a Saturday market bar.

Use the approved15 assets and order, IWM not IJR, no REIT/commodity targets.
Keep t_index and exclude raw HY level/change inputs. Rank2/maxiter15, original
signed-LKJ construction and transforms. Three experts: broad14 expanding,
compact5 expanding, broad14 rolling48. Covariance always50/50 broad/compact,
including likelihood noise; no third covariance or fitted recalibration.

Apply static family prediction weights, then average means and covariance
equally across seeds27/127/227 and allocate one portfolio. Every historical
candidate uses this exact pool-first policy. Hard rank is unchanged: long-only,
fully invested, NE>=7, no position cap, original causal expanding LedoitWolf
historical equal-weight variance ceiling. No CVaR, continuous risk penalty,
fallback portfolios, positive-return floor, clipping, covariance repair,
constraint relaxation, or date-dependent blend selection.

## Evidence, candidates, and frozen scoring protocol

Before new allocation scoring or fresh October downloads, save PREREGISTRATION.json
and its SHA256 with the complete candidate menu, dates, blocks, and gates below.
Old inspected results may inform interpretation; no untouched-holdout claim.

Five mandatory controls: incumbent Sharpe leader, approved ranking compromise,
and the three prior48-month Sharpe/return/rank prediction recipes, all with the
same fixed50/50 covariance. Batch1 contains only predefined stable alternatives:
25/50/75% shrinkage of each48 challenger toward the incumbent; pair/triple
averages of the48 challengers; shrunk averages; incumbent/compromise average;
uniform-three-expert and uniform broad/compact averages; and25/50% rolling48
substitution in USFI only, international FI only, or both FI families.
Deduplicate exact recipes. Maximum48 new batch1 recipes.

Batch2 is not required. It may be used only for a documented specific mechanism
unresolved by the first batch, with a new frozen menu before scoring, never
merely because no challenger won. No new GP feature/kernel search is planned.

Historical common ledger:52 forecast origins May2022--August2026, realized
June2022--September2026. Report earlier24 and later28 separately. Six disjoint
blocks by origin index: [0,12),[12,24),[24,31),[31,38),[38,45),[45,52).
All data were previously inspected; new selection remains retrospective.

Primary objectives: conventional annualized Sharpe, compounded total return,
monthly pooled rank IC. Also report within-family IC, exact within/between-family
rank-covariance decomposition, family arithmetic-return contributions, exposures,
drawdown, volatility, and leave-one-month-out influence. NLL diagnostic only.

Conservative operational promotion gates (not significance tests):
1. Across52 months: Sharpe at least incumbent+0.05, IC at least incumbent+0.01,
   cumulative return no lower than incumbent (numerical tolerance1e-10).
2. Later28: Sharpe no lower than incumbent-0.05; IC no lower than incumbent-0.01;
   cumulative return no lower than incumbent-0.01 (one percentage point).
3. Mean arithmetic excess-return advantage nonnegative in at least4/6 blocks.
4. Positive full-period Sharpe advantage after removing each individual month.
5. Maximum drawdown no more than0.01 worse; no family's mean IC more than0.05
   below incumbent. All allocations complete and valid; no dropped months.

Shortlist includes all five controls and at most ten additional alternatives,
ordered by worst normalized gate slack then Sharpe/IC/return, with no new fits
needed. Only candidates clearing EVERY gate may replace incumbent. Among those,
remove Pareto-dominated candidates on Sharpe/return/IC and choose the smallest
coefficient distance from incumbent; break ties by Sharpe, IC, return, then ID.
If no candidate clears every gate, select incumbent. No arbitrary blended
performance score decides promotion. Freeze the historical decision before
scoring October's newly appended realized return. That one month is a reported
diagnostic, never the sole promotion trigger or reason to enlarge the search.

## Data and fits

Use a separate October extension with snapshots and availability audits. Preserve
all112 existing dates/cells. Preserve September's already-frozen predictors,
add its now-known October-return label, and append October live predictors with
unknown November target. Frozen September CPI/HY snapshots are available before
the October cutoff and sufficient under the established extra predictor lag.
No fresh post-cutoff macro revisions may contaminate this decision.

Fit only missing September30 historical and October30 live forecasts for the
three shared experts and three seeds: expected18 monthly fits. Preserve one
isolated directory per attempt and full optimizer telemetry/observed covariance.
One shared pool<=14; all retries count toward128. Final November live targets
remain NaN and can never enter fitting. Budgets use every available pre-origin
target (112 dates at September origin,113 at October), not rolling48 history.

## Deliverables and completion

DECISION.md, CONFIG.json, NOVEMBER_WEIGHTS.csv (all15, including zero weights),
VALIDATION.json, comparison tables/plot, source/data/code hashes and fit ledger,
saved forecasts/covariance/raw solver outputs, STATUS.json and one reproducible
verify command. Independent raw-weight checks and original conic parity required.
Explicitly distinguish pooled moments from averaged seed-specific portfolios.
Document Friday-close to Monday-November2 execution gap and gross-cost limits.

Completion requires fresh validated data, finite moments, strictly accepted raw
weights satisfying all constraints, reproduced weights and rankings, unchanged
protected inputs, and delivered report. Correctness/source blockers stop the
recommendation, not solved by stale data or old weights. Retaining the incumbent
configuration is valid success; reusing prior-month weights on solver failure
is not. No trades are executed.
