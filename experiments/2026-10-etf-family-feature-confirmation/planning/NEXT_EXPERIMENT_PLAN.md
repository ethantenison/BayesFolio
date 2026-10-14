# Next U.S. ETF Feature Experiment Batch

## Status and provenance

This is a planning artifact only. It was produced by the current ChatGPT/Codex
session after the local planning model failed to emit the required files. No new
GP fit was launched while producing this plan. The machine-readable companion is
`next_experiment_candidates.json`, and both artifacts are bound to the frozen
Step-3 evidence by SHA-256.

The incumbent comparison set contains the completed 24-window, three-seed results
for U-AC, U-C005, U-C009, and U-ALL-MATCH. Its fixed-reference hypervolume is
`0.6102258300032914`, using the raw-objective reference point
`mean_ic=0`, `mean_gaussian_nll=0`, and `sharpe=0`.

## Objective contract

This remains a genuine Pareto problem. There is no scalar score and no metric is a
guardrail:

- maximize mean IC;
- minimize mean Gaussian NLL;
- maximize conventional annualized Sharpe;
- record cumulative return and RMSE as diagnostics only.

Every comparison keeps `t_index`, SPY/MGK/VTV/IWM, the same 24 scored months, the
same frozen local feature artifact, seeds 27/127/227, rank 2, maxiter 15, and 256
posterior scenarios. The batch uses cached data only.

## Batch design

The next batch contains four candidates, or 12 deterministic fits. It deliberately
balances two exploitation candidates, one bridge candidate, and one exploration
candidate. The exploitation candidates refine the two most useful regions already
seen: U-ALL-MATCH's Sharpe and U-C005's IC. The bridge tests whether a small set of
breadth/flow/inflation-update variables can transfer Sharpe behavior into the
simpler U-C005 model. The exploration candidate revisits the previously credible
target-history feature family on the longer common panel.

### Priority 1: `all_match_redundancy_pruned` (exploitation)

Parent: U-ALL-MATCH.

Replace plain 12-month momentum with skip-month momentum, replace six-month maximum
drawdown with the three-month version, and remove the bid-ask-spread proxy. Keep the
remaining U-ALL-MATCH variables and model specification unchanged. The hypothesis
is that removing redundant/noisy inputs can lower NLL without sacrificing the
parent's leading Sharpe.

### Priority 2: `c005_three_way` (exploitation)

Parent: U-C005.

Keep the U-C005 feature set and model controls, then add the runner's time x ETF x
macro three-way interaction. Earlier 12-window evidence found that this interaction
improved Gaussian NLL without materially damaging the portfolio path at maxiter 15.
This is the cleanest direct confirmation of that result on the current seed panel.

### Priority 3: `c005_breadth_flow_bridge` (bridge)

Parents: U-C005 and U-ALL-MATCH.

Keep the U-C005 model and its existing predictors, then add `spy_flow_z_12p`, `erp`,
and `cpi_chg_1p`. These cached variables are present in U-ALL-MATCH, whose main
frontier contribution is Sharpe. This candidate tests a narrow transfer hypothesis
instead of importing the entire higher-dimensional input block.

### Priority 4: `target_macro_revisit` (exploration)

Parent: U-C005.

Swap the four momentum/trend ETF variables for the two lagged target variables,
while keeping the U-C005 macro block and model specification. A prior broad search
found this family competitive on IC and NLL, but it has not received the current
24-window, three-seed confirmation.

## Deterministic execution contract

For each candidate, invoke
`experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py`
with the ordered `exact_runner_args` from the companion JSON. Replace `{seed}` with
27, 127, and 227, and replace `{candidate_id}` with the candidate ID. Run the
candidates in the listed priority order. Do not select, drop, or alter a seed after
seeing intermediate metrics.

Before the first fit, verify that every `cached_columns_required` entry exists in
the frozen parquet, the 24 scored dates are unchanged, and the Step-3 evidence hash
still matches. After each candidate's three runs, aggregate it with the same code
used for Step 3 and recompute the combined Pareto front and fixed-reference
hypervolume against the unchanged reference point.

## Promotion and stopping rules

A candidate is promoted only if its three-seed aggregate is nondominated after it
is added to the incumbent set and the resulting fixed-reference hypervolume is
strictly greater than `0.6102258300032914`. Diagnostics cannot promote or reject a
candidate.

Stopping is deterministic:

1. Stop before fitting if evidence hashes, cached columns, scored dates, or fixed
   controls fail validation.
2. Complete all three seeds for a started candidate unless a run has a technical
   failure; do not make decisions from a partial seed panel.
3. Stop the batch on a reproducibility or aggregation failure and repair the
   deterministic pipeline before continuing.
4. After all valid candidates are aggregated, stop this round if none increases
   hypervolume. New hypotheses must then be generated from the complete round,
   rather than expanding the current batch opportunistically.

## Risks and interpretation

- The pruning candidate may remove useful tail/liquidity state information along
  with redundancy.
- The three-way term may improve density only by adding unstable complexity.
- The bridge variables are correlated macro-state proxies, so any gain may be
  seed-sensitive or fail to transfer cleanly.
- Lagged target variables may improve density and IC while weakening Sharpe.

These are falsifiable hypotheses, not expected winners. All three objectives remain
visible in the Pareto decision, including outcomes where a candidate trades one
objective for another.

## Deferred ideas

Robust predictive likelihoods, higher optimizer iteration counts, structured macro
splits, time-index ablations, a calibrated proposal optimizer, and new holdout
discipline remain outside this batch. They should not be mixed into these feature
comparisons because each would change a separate experimental dimension.
