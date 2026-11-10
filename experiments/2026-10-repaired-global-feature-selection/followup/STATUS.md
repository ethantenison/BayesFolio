# Joint follow up complete

## Latest checkpoint

Both dispatcher sessions (39907 and 32115) exited zero. All thirty-six follow-up
seed runs completed; the final native process check found no surviving children.
The fifteen-worker ceiling was respected. Do not restart completed dispatchers.

The full comparison independently validated all 37 joint masks / 111 seed runs.
Hypervolume progressed 1.016778459 -> 1.187618053 -> 1.228693994: batch two added
3.46 percent, and the two follow-ups added 20.84 percent over the original front.
Three configurations remain nondominated:

- global_e6c41f2020fd: fourteen inputs, highest IC; primary research lead.
- global_4397c0b6828e: fifteen inputs, highest mean Sharpe, IC/NLL cost.
- global_bb1b8f9500f2: twelve inputs, best NLL, substantial Sharpe cost.

Read results/all_joint/RESEARCH_COMPARISON.md and TABLES.md. Independent audit
passed 111 runs, with 506 source/evidence hashes and no unresolved items. All
three frontier masks retain lagged y and t_index. Family effects remain mixed;
there is no universal family winner or production promotion. Original data,
code and completed-study final evidence passed unchanged. No commit or push.
Tests: 71 passed; Ruff and new public-docstring checks passed.

No additional fits remain authorized under this completed two-batch campaign.
Reproduce reports with compare_joint_followup.py; do not edit fit-critical
sources or contract.json, which are fingerprinted into the completed runs.

The older notes below are historical as-of launch records, not current state.

## Historical launch checkpoint (superseded)

The user has now explicitly requested launch of batch two and a comparison of
all joint results. First batch is still live in session 39907/PID 21360, with
fifteen active fits and approximately twenty of twenty-four windows in the
latest sampled check. Wait for its remaining jobs and deterministic audit;
then write INTERPRETATION_BATCH1.md, register batch two and execute it. Do not
ask for another approval or exceed fifteen concurrent fit workers.

The new compare_joint_followup.py will audit and compare all original twenty-five
and follow-up twelve masks (37 configurations/111 seeds), writing
results/all_joint/. Its five synthetic coverage tests, Ruff and docstrings passed.
The comparator is not execution-fingerprinted and was added without changing
followup_workflow.py, contract.json, any fit source or existing run fingerprints.

Dispatcher session 39907, native PID 21360. Native inspection confirmed fifteen
live fit children after all six deterministic preflights passed. There are
eighteen required seed runs; the remaining three wait for worker capacity.
Recheck the live session/process before restarting. Never duplicate a dispatcher.

| Addition to the thirteen-input lead | New configuration | Seeds |
| --- | --- | --- |
| hy_spread | global_a306a45ffc49 | 27, 127, 227 |
| hy_spread_chg_1p | global_d77edcc67c1d | 27, 127, 227 |
| spy_ret | global_e6c41f2020fd | 27, 127, 227 |
| cpi_chg_1p | global_ccfa2775bb52 | 27, 127, 227 |
| schp_ret | global_dc10aef417bc | 27, 127, 227 |
| y10_nominal | global_6a602a86c2b6 | 27, 127, 227 |

Each mask has fourteen non-time inputs plus t_index. The same frozen v2 data,
84 initial training months, 24 expanding score dates and GP/portfolio controls
apply. The adapter and contract are now execution-fingerprinted: do not edit
them while runs are live. Original completed-study fingerprints were rechecked
unchanged. Verification before launch: 66 tests passed, Ruff and the public-
docstring gate passed. No production, Git commit or push changes were made.

The deterministic dispatcher will validate all outputs and produce
results/batch1/ with cumulative fronts, paired changes, family diagnostics and
an independent arithmetic audit. There is no scientific result until complete
three-seed output passes. Status JSON uses queued until each child completes;
that label alone does not distinguish a running worker from a waiting job.

The second six-mask shortlist is authorized in contract.json but is not yet
registered or launched. After reviewing full batch-one results, hosted Codex
must write INTERPRETATION_BATCH1.md and explain the next selection before using
--register-second. The current dispatcher does not perform that reasoning turn.

Resume only after verifying no dispatcher/worker survives:

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/followup_workflow.py --execute --workers 15
```

The completed original study remains under its original cap and reports.
Do not run its dispatcher in parallel. This follow-up has at most two batches
of six new masks, with no new optimizer/holdout or production promotion.
