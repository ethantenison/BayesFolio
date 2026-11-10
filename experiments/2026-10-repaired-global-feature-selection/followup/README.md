# Joint feature transfer follow up

Both batches are complete. The final comparison covers 37 masks / 111 seeds
and leaves three Pareto points. Read
[the research comparison](results/all_joint/RESEARCH_COMPARISON.md) for the
recommended SPY-proxy lead, Sharpe and density alternatives, family exceptions
and exact inputs; [TABLES.md](results/all_joint/TABLES.md) contains all 37 results.
No further fits or production promotion are part of this completed campaign.

This separately authorized campaign preserves the completed original study and
its three-round cap. It uses the same frozen v2 panel, targets, 84 initial months,
24 expanding score dates, three seeds, kernel and portfolio controls. t_index
stays in every mask. IC, marginal Gaussian NLL and conventional Sharpe remain
the three objectives; return/RMSE are diagnostics, turnover excluded.

Batch one adds each of HY level, HY change, SPY proxy, monthly CPI, SCHP return
and nominal yield individually to global_33e57e1c2370. Each has fourteen non-time
inputs and seeds 27, 127 and 227: eighteen fits. The exact twelve authorized
operations and two-batch/six-mask limits are in contract.json.

After all first-batch outputs pass independent arithmetic and family checks,
hosted Codex records INTERPRETATION_BATCH1.md before registering the follow-up
pair additions and pruning/substitution candidates. No local model or API key is
used. The dispatcher never invents a plan or silently registers batch two.

The small adapter redirects the existing runner's output root and adds its own
source/contract to signatures inside a reversible scope. Frozen feature pointers,
the GP implementation, original plans, old fingerprints and final reports remain
unchanged. Evidence snapshots and all new outputs live here. An exclusive lock
prevents duplicate follow-up dispatchers; at most fifteen fit workers use one
Torch/scenario worker each, with the existing BLAS limits. The original completed
full-window global smoke/control is revalidated rather than refitted; every new
mask receives the existing runner's deterministic preflight before any fit starts.

From the repository root:

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/followup_workflow.py --prepare --check
.venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_followup_workflow.py
.venv/bin/ruff check experiments/2026-10-repaired-global-feature-selection/followup_workflow.py experiments/2026-10-repaired-global-feature-selection/test_followup_workflow.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/followup_workflow.py --execute --workers 15
```

The last command resumes only registered exact-signature work. Once hosted
review is saved, --register-second --check registers batch two; the same execute
command then skips verified batch-one fits and runs the new batch. Do not launch
the old dispatcher in parallel or change fit-critical follow-up sources mid-run.

The reporter combines frozen original controls with new results, retains the
full Pareto front/fixed hypervolume reference, records exact paired objective
effects, family IC/NLL/coverage/allocation, and independently reconstructs each
new seed's forecasts-times-weights returns, conventional Sharpe and NLL. These
remain adaptive research on reused dates, not untouched confirmation or promotion.

## Compare all joint models after both batches

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/compare_joint_followup.py
```

This independent closing command requires both batches complete. It revalidates
the original twenty-five masks and twelve follow-up masks, all 111 seed outputs,
signatures, actual training lengths and published objective/family values. It
reconstructs returns from saved weights and labels and recomputes conventional
Sharpe, rank IC, marginal Gaussian NLL and calibration independently. Its output
is results/all_joint/: every configuration, the full Pareto front, exact inputs,
paired feature contexts, seed/family/monthly uncertainty and source hashes.
TABLES.md keeps the full results; the accompanying hosted interpretation explains
tradeoffs without changing the original study's final reports or controls.
