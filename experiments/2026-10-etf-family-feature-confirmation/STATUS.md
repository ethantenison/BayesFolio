# ETF Family Feature Confirmation Status

Milestone: the U.S. follow-up batch and both international equity stages are
complete. The international workflow produced 69 new validated fits: 12 for
the four-way confirmation, 12 for targeted macro ablations, and three for the
oil-only follow-up, three for the gold/crude-ratio-only follow-up, plus 21 for
the oil-centered backward-selection and FX batch and 18 for the second
oil-interaction refinement.

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
- Removing `erp` from `I-BREADTH+OIL` improves IC and NLL relative to oil-only
  while giving up little Sharpe. Removing `spy_ret` instead produces the best
  aggregate Sharpe and another IC gain, with materially worse NLL.
- The DXY and EM-FX additions are dominated, as are the oil candidates that
  remove `rsp_spy_roc_1p` or `pct_above_50dma`.
- Removing both `erp` and `spy_flow_z_12p` from the oil model improves all
  three objectives over every earlier candidate. Removing `erp` without oil
  remains nondominated because it has the highest aggregate Sharpe.
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

The oil-centered backward-selection and FX batch produced:

- `I-BREADTH+OIL DROP SPY-FLOW`: IC `0.188889`, NLL `-1.865049`, Sharpe `1.299568`;
- `I-BREADTH+OIL DROP ERP`: IC `0.241667`, NLL `-1.841202`, Sharpe `1.333582`;
- `I-BREADTH+OIL DROP RSP-SPY`: IC `0.002778`, NLL `-1.589168`, Sharpe `1.209956`;
- `I-BREADTH+OIL DROP PCT-ABOVE`: IC `0.172222`, NLL `-1.775647`, Sharpe `1.275120`;
- `I-BREADTH+OIL DROP SPY-RET`: IC `0.219444`, NLL `-1.692986`, Sharpe `1.353966`;
- `I-BREADTH+EM-FX`: IC `0.055556`, NLL `-1.831303`, Sharpe `1.252026`;
- `I-BREADTH+DXY`: IC `0.136111`, NLL `-1.786251`, Sharpe `1.248828`.

The expanded frontier hypervolume is `0.6072236686688807`, up
`0.1140833662447475` from the prior three-point frontier. Considered alone
against that prior frontier, dropping `erp` adds `0.10736477108342524` of
hypervolume, dropping `spy_ret` adds `0.04850206342543795`, and dropping
`spy_flow_z_12p` adds `0.006187686575504336`. The other four candidates add
none. The final nondominated set has six points: the prior three plus those
three backward eliminations.

The second oil-interaction refinement produced:

- `I-BREADTH DROP ERP`: IC `0.213889`, NLL `-1.781311`, Sharpe `1.395827`;
- `I-BREADTH+OIL DROP ERP+SPY-RET`: IC `0.161111`, NLL `-1.654478`, Sharpe `1.351071`;
- `I-BREADTH+OIL DROP ERP+SPY-FLOW`: IC `0.252778`, NLL `-1.917061`, Sharpe `1.382449`;
- `I-BREADTH+OIL DROP ERP+SPY-RET+SPY-FLOW`: IC `0.075000`, NLL `-1.699727`, Sharpe `1.301757`;
- `I-BREADTH+COMMODITIES DROP ERP`: IC `0.130556`, NLL `-1.903091`, Sharpe `1.273767`;
- `I-BREADTH DROP SPY-RET`: IC `0.022222`, NLL `-1.755382`, Sharpe `1.303634`.

The new oil/drop-ERP/drop-flow candidate dominates the entire preceding
six-point frontier. Relative to `I-BREADTH+OIL DROP ERP`, it improves IC by
`0.011111`, improves NLL by `0.075859`, and improves Sharpe by `0.048867`.
The no-oil/drop-ERP candidate is the only other final Pareto point because its
Sharpe is `0.013378` higher. The fixed-reference hypervolume increased from
`0.6072236686688807` to `0.6750184934043326`, a gain of
`0.0677948247354519`.

The factorial attribution is conditional rather than additive. Dropping
`spy_ret` only helped when ERP remained present; combining both removals was
dominated. Likewise, dropping flow helped strongly when `spy_ret` remained,
while the three-variable removal was dominated. The oil-plus-ratio/drop-ERP
candidate improved calibration but was dominated by the oil-only/drop-ERP/
drop-flow candidate.

The preceding U.S. result remains unchanged: `all_match_redundancy_pruned` is
the only promoted follow-up candidate, expanding the U.S. frontier by
`0.000409496683654087`.

## Impact

For the international equity family, the corrected momentum-plus-breadth/flow
feature set was a productive compact starting point. The broad all-input model
adds noise and is dominated. FX, inflation, and credit should not be added in
the tested forms. The oil/drop-ERP/drop-flow specification is now the
strongest tested model for both IC and NLL and also improves Sharpe over every
earlier candidate. The no-oil/drop-ERP specification has the highest Sharpe,
so the final in-sample discovery frontier contains those two configurations.

No production model or deployment configuration was changed.

## Proposed Fix

Advance `I-BREADTH+OIL DROP ERP+SPY-FLOW` as the primary international
confirmation candidate and `I-BREADTH DROP ERP` as the higher-Sharpe
alternative. Retain `spy_ret`, `rsp_spy_roc_1p`, and `pct_above_50dma` in the
primary candidate. Do not continue combinatorial optimization on this same
24-date discovery panel without the deferred holdout discipline; doing so
would increasingly optimize to the evaluation sample.

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
.venv/bin/python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
  --config experiments/2026-10-etf-family-feature-confirmation/international_oil_backward_selection_config.json \
  --seeds 27,127,227
.venv/bin/python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
  --config experiments/2026-10-etf-family-feature-confirmation/international_oil_interaction_refinement_config.json \
  --seeds 27,127,227
```

Expected result: 69/69 newly executed international runs and all reused
comparison runs validate; the confirmation frontier is
`I-BREADTH`, and the oil follow-up frontier contains `I-BREADTH`,
`I-BREADTH+COMMODITIES`, and `I-BREADTH+OIL`. The ratio-only candidate is
dominated. The expanded oil-centered frontier contains those three plus the
drop-SPY-flow, drop-ERP, and drop-SPY-return configurations.
The second refinement reduces the final frontier to `I-BREADTH DROP ERP` and
`I-BREADTH+OIL DROP ERP+SPY-FLOW`.

## Open Questions

No question blocks this milestone. The deferred holdout design should compare
the two final Pareto candidates before either changes production behavior.
