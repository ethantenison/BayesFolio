# ETF Family Feature Confirmation Status

Milestone: the U.S. follow-up batch and both international equity stages are
complete. The international workflow produced 30 new validated fits: 12 for
the four-way confirmation, 12 for targeted macro ablations, and three for the
oil-only follow-up plus three for the gold/crude-ratio-only follow-up.

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
- Adding `oil_ret` alone raises IC materially and Sharpe slightly, but worsens
  NLL. It therefore creates a distinct ranking-oriented Pareto point.
- Adding `gold_crude_ratio` alone is dominated by `I-BREADTH` on all three
  objectives. The paired block's NLL gain is therefore a joint-model effect.
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

The oil-only follow-up produced IC `0.200000`, NLL `-1.781851`, and Sharpe
`1.342357`. Relative to `I-BREADTH`, oil alone improves IC by `0.063889` and
Sharpe by `0.001847`, while worsening NLL by `0.079837`. Relative to the paired
oil plus gold/crude block, oil alone improves IC and Sharpe substantially but
gives up the paired block's small calibration advantage. All three
configurations are nondominated, with fixed-reference hypervolume
`0.49314030242413337`.

The completed ratio-only follow-up produced IC `0.091667`, NLL `-1.782716`,
and Sharpe `1.278213`. `I-BREADTH` is better on all three objectives, so the
ratio-only candidate is dominated. Because neither oil-only nor ratio-only
reproduces the paired block's NLL improvement, that calibration gain appears
only in their joint specification. This is consistent with an interaction or
joint regularization effect rather than a useful standalone ratio effect.

The preceding U.S. result remains unchanged: `all_match_redundancy_pruned` is
the only promoted follow-up candidate, expanding the U.S. frontier by
`0.000409496683654087`.

## Impact

For the international equity family, the evidence supports the corrected
momentum-plus-breadth/flow feature set as the primary compact model. The broad
all-input model adds noise and is dominated. FX, inflation, and credit should
not be added in the tested forms. The oil/gold block should be retained only as
an explicit calibration-oriented Pareto alternative pending a later selection
policy or holdout test. Oil alone is the strongest tested international model
for IC and narrowly the strongest for Sharpe, but its worse NLL prevents it
from replacing the better-calibrated `I-BREADTH` configuration outright.

No production model or deployment configuration was changed.

## Proposed Fix

Use `I-BREADTH` as the international equity default candidate for subsequent
confirmation or integration work. Preserve `I-BREADTH+COMMODITIES` as a
calibration-oriented alternative and `I-BREADTH+OIL` as a ranking-oriented
alternative; neither is an automatic replacement. Do not rerun the three
dominated macro blocks or the ratio-only candidate unchanged.

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
.venv/bin/python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
  --config experiments/2026-10-etf-family-feature-confirmation/international_oil_only_config.json \
  --seeds 27,127,227
.venv/bin/python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
  --config experiments/2026-10-etf-family-feature-confirmation/international_ratio_only_config.json \
  --seeds 27,127,227
```

Expected result: 30/30 newly executed international runs and all reused
`I-BREADTH` comparison runs validate; the confirmation frontier is
`I-BREADTH`, and the oil follow-up frontier contains `I-BREADTH`,
`I-BREADTH+COMMODITIES`, and `I-BREADTH+OIL`. The ratio-only candidate is
dominated.

## Open Questions

No question blocks this milestone. A later decision is still needed on whether
the three Pareto alternatives merit a separate holdout confirmation.
