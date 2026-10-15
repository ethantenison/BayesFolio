# ETF Family Feature Confirmation Status

Milestone: the U.S. follow-up batch and both international equity stages are
complete. The international workflow produced 24 new validated fits: 12 for
the four-way confirmation and 12 for targeted macro ablations.

## Summary

- The international task universe is `VEA`, `VWO`, `VSS`, and `EWX`. Every run
  uses the same cached 24-window panel, seeds `27`, `127`, and `227`, `t_index`,
  rank 2, maxiter 15, and 256 scenarios.
- The corrected momentum-plus-breadth/flow candidate (`I-BREADTH`) dominates
  momentum-plus-market, momentum-only, and the compute-matched all-input model
  on all three aggregate objectives.
- Adding FX, breakeven inflation, or credit to `I-BREADTH` is dominated by the
  unchanged baseline. Adding oil and gold/crude improves mean NLL slightly but
  gives up IC and Sharpe, so it joins rather than replaces the Pareto frontier.
- Return and RMSE are recorded only. Turnover is absent from the evaluation and
  selection outputs.

## Findings

The international confirmation aggregates are:

- `I-BREADTH`: IC `0.136111`, NLL `-1.861688`, Sharpe `1.340510`;
- `I-MARKET`: IC `0.113889`, NLL `-1.522946`, Sharpe `1.265790`;
- `I-MOM`: IC `0.025000`, NLL `-1.730624`, Sharpe `1.178977`;
- `I-ALL-MATCH`: IC `0.005556`, NLL `-1.802055`, Sharpe `1.168111`.

`I-BREADTH` is the sole confirmation-stage nondominated configuration. The
fixed-reference hypervolume is `0.33968051082932643`.

The targeted ablation aggregates, compared with the same `I-BREADTH` runs, are:

- `I-BREADTH+FX`: IC `0.050000`, NLL `-1.745432`, Sharpe `1.253096`;
- `I-BREADTH+INFLATION`: IC `0.050000`, NLL `-1.753976`, Sharpe `1.201915`;
- `I-BREADTH+CREDIT`: IC `0.072222`, NLL `-1.796184`, Sharpe `1.263735`;
- `I-BREADTH+COMMODITIES`: IC `0.027778`, NLL `-1.867543`, Sharpe `1.213128`.

The commodities block improves NLL by `0.005855` relative to `I-BREADTH`, but
reduces IC by `0.108333` and Sharpe by `0.127382`. The resulting two-point
frontier has hypervolume `0.3398777952385972`, an increase of only
`0.00019728440927077`. This is a calibration-versus-ranking/portfolio tradeoff,
not evidence that the commodities block is an overall winner.

The preceding U.S. result remains unchanged: `all_match_redundancy_pruned` is
the only promoted follow-up candidate, expanding the U.S. frontier by
`0.000409496683654087`.

## Impact

For the international equity family, the evidence supports the corrected
momentum-plus-breadth/flow feature set as the primary compact model. The broad
all-input model adds noise and is dominated. FX, inflation, and credit should
not be added in the tested forms. The oil/gold block should be retained only as
an explicit calibration-oriented Pareto alternative pending a later selection
policy or holdout test.

No production model or deployment configuration was changed.

## Proposed Fix

Use `I-BREADTH` as the international equity default candidate for subsequent
confirmation or integration work. Preserve `I-BREADTH+COMMODITIES` as a
research alternative, not an automatic replacement. Do not rerun the three
dominated macro blocks unchanged.

The next family-level work is the deferred sequence from the plan: investment-
grade bonds/Treasuries, high-yield and emerging-market debt, REITs, then
commodities. The calibrated proposal optimizer and new holdout discipline also
remain deferred as requested.

Rollback: revert the international result commit. Generated run directories
and execution snapshots remain ignored local artifacts.

## Verification

```bash
.venv/bin/ruff check \
  experiments/2026-10-etf-family-feature-confirmation/build_cached_international_confirmation_artifact.py \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_international_experiments.py \
  experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
  experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py
.venv/bin/ruff format --check \
  experiments/2026-10-etf-family-feature-confirmation/build_cached_international_confirmation_artifact.py \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_international_experiments.py \
  experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
  experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py
.venv/bin/pytest -q \
  experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py
.venv/bin/python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
  --config experiments/2026-10-etf-family-feature-confirmation/international_confirmation_config.json \
  --seeds 27,127,227
.venv/bin/python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
  --config experiments/2026-10-etf-family-feature-confirmation/international_targeted_ablation_config.json \
  --seeds 27,127,227
```

Expected result: 24/24 newly executed international runs and all three reused
`I-BREADTH` comparison runs validate; the confirmation frontier is
`I-BREADTH`, and the targeted frontier is `I-BREADTH` plus
`I-BREADTH+COMMODITIES`.

## Open Questions

No question blocks this milestone. A later decision is still needed on whether
the small NLL-only commodities tradeoff merits a separate holdout confirmation
or should remain research-only.
