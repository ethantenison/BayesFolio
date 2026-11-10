# Historical-scope correction: macro-only baselines

Review against the recovered GOAL_CONTRACT.md found an unauthorized inventory
restriction: five full historical manifests had been excluded solely because
their ETF-local feature block was empty. The goal did not require a local input,
and the existing wrapper already supports omitting an empty block naturally.

The five source manifests collapse to two distinct historical masks: one U.S.
equity and one international-equity baseline. All retain t_index and the same
eight macro inputs:

vix, vix_slope, spy_ret, pct_above_50dma, hy_spread_chg_1p,
hy_spread_z_12p, cpi_chg_12p, term_spread.

Each source's original manifest hash, complete twelve-month prediction coverage,
finite forecasts, exact family and validated feature names were rechecked. The
new audit `historical_macro_only_audit.json` records all five source references
and the two deduplicated repaired-mask IDs. Original scope reports remain
preserved as as-of evidence; their local-block exclusions are superseded here.

Two supplemental historical batches are registered:

- plans/00_us_equity_macro_only_historical_extension.json
- plans/00_international_equity_macro_only_historical_extension.json

Each requires seeds 27/127/227 on the same repaired data, 84 initial months and
24 expanding-window score dates, with the same fixed kernel-construction rule,
likelihood and portfolio settings. No zero placeholder or family-masked kernel
is introduced. The construction naturally omits the unavailable local block and
its interactions. No old performance metric is reused.

`candidate_masks.py` is a compatibility adapter: ordinary local-feature candidate
objects remain byte-equivalent; legitimate empty blocks are now admitted.
Validation still rejects unaudited masks disguised as historical baselines.
Execution-critical fit sources, frozen data, old plans/fingerprints and all
adaptive caps remain unchanged. These are two previously omitted historical
baselines, not a fourth family refinement round or an extra adaptive global batch.

Run them only within the total fifteen-fit cap. Once the active global batch has
at most three remaining jobs, two disjoint family dispatchers with at most three
workers each can use free capacity: at most nine fits total. Otherwise wait for
the global dispatcher. Never start another unfiltered dispatcher against a live
batch. Refresh cumulative family/global evidence after all jobs complete, and
interpret whether the restored baselines change fronts or global hypotheses.

Previously frozen family-round and global-round-zero comparisons remain valid
as-of comparisons of their actual candidate sets, but must not be presented as
having included these two masks. The final inventory and report must reconcile them.

## Executed result

Both three-seed batches completed successfully. Scoped dispatcher sessions 3745
(U.S. equity) and 7126 (international equity) exited zero. Every run passed the
normal source/data/configuration/seed/date and 84+k expanding-training checks.

| Family / new mask | IC | NLL | Sharpe | An existing strict aggregate dominator |
| --- | ---: | ---: | ---: | --- |
| U.S. equity / us_equity_cb0ff7bcfbc2 | 0.072222 | -1.758769 | 1.349784 | us_equity_5fba6b4b201b |
| International equity / international_equity_32e6597d0b79 | 0.077778 | -1.785446 | 1.273037 | international_equity_66ef3bfb80d7 |

Neither adds a family Pareto point under the adopted aggregate three-objective
rule. This is evidence about these exact macro-only masks, not proof that every
ETF-local feature is individually necessary. The corrected historical inventory
has 320 family masks plus the joint all-input control (321 configurations/963
seed runs). All prior adaptive-round comparisons stay as-of records of their
original candidate sets; future cumulative reports include these baselines.
