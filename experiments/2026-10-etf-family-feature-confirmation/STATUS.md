# ETF Family Feature Confirmation Status

Milestone: the first ChatGPT-planned follow-up batch is complete. All 12 new
fits passed deterministic validation, and one candidate expanded the existing
three-objective Pareto frontier.

## Summary

- The committed Step-3 baseline remains the comparison set: U-C005, U-C009,
  and U-ALL-MATCH are nondominated, with fixed-reference hypervolume
  `0.6102258300032914`.
- ChatGPT/Codex replaced the failed local-model planning turn and produced four
  validator-approved candidates: two exploitation, one bridge, and one
  exploration candidate.
- All four candidates completed seeds `27`, `127`, and `227` on the frozen
  24-window panel with SPY/MGK/VTV/IWM, `t_index`, rank 2, maxiter 15, and 256
  scenarios.
- `all_match_redundancy_pruned` is the only promoted candidate. Adding it to
  the incumbents raises hypervolume to `0.6106353266869455`, an increase of
  `0.000409496683654087`.

## Findings

Three-seed candidate aggregates are:

- `all_match_redundancy_pruned`: IC `0.213889`, NLL `-1.664957`, Sharpe
  `1.335338`; nondominated; promoted;
- `c005_three_way`: IC `0.263889`, NLL `-1.673605`, Sharpe `1.168122`;
  dominated by an incumbent;
- `c005_breadth_flow_bridge`: IC `0.172222`, NLL `-1.597127`, Sharpe
  `1.133363`; dominated;
- `target_macro_revisit`: IC `0.255556`, NLL `-1.643259`, Sharpe `1.185022`;
  dominated by an incumbent.

The pruning result is narrow but directionally coherent: compared with
U-ALL-MATCH, it improves mean NLL from `-1.658456` to `-1.664957` and Sharpe
from `1.334647` to `1.335338`, while IC falls from `0.227778` to `0.213889`.
That tradeoff creates a distinct Pareto point rather than a universal winner.

The three-way interaction did not reproduce its earlier advantage on the
current confirmation panel. The bridge variables added noise across all three
objectives, and the target-lag family did not beat U-C005 after full three-seed
confirmation.

## Impact

The evidence now supports a more compact version of the all-input model region:
replace plain 12-month momentum with skip-month momentum, replace six-month
maximum drawdown with the three-month measure, and omit the spread proxy. The
gain is small, so this is a Pareto-front expansion rather than grounds for a
production-model replacement.

The unsuccessful candidates are useful negative evidence. They should not be
rerun unchanged in the next hypothesis round.

## Proposed Fix

Preserve `all_match_redundancy_pruned` on the active frontier and use it as the
parent for any next exploitation proposal. Keep U-C005 and U-C009 as the IC and
NLL anchors. Any next batch should make one coherent change at a time and remain
bound to the same cached panel and objective contract.

Patch/result scope:

- deterministic next-batch summarizer and promotion logic;
- 12 validated candidate runs;
- run-level objectives, candidate aggregates, combined Pareto front,
  fixed-reference hypervolume, and validation artifacts;
- no production-model or deployment changes.

Rollback: revert the result commit. Generated run directories and execution
snapshots remain ignored local artifacts.

## Verification

```bash
.venv/bin/ruff check \
  experiments/2026-10-etf-family-feature-confirmation/summarize_next_experiment_batch.py \
  experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py
.venv/bin/ruff format --check \
  experiments/2026-10-etf-family-feature-confirmation/summarize_next_experiment_batch.py \
  experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py
.venv/bin/pytest -q \
  experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py
.venv/bin/python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_next_experiment_batch.py
```

Expected result: 12/12 runs valid, promoted candidates
`['all_match_redundancy_pruned']`, and combined hypervolume
`0.6106353266869455`.

## Open Questions

The next hypothesis round has not been preregistered. Work on a calibrated
proposal optimizer and new holdout discipline remains deferred as requested.
