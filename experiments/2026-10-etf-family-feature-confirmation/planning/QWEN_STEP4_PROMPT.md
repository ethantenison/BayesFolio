# Qwen step 4: plan the next deterministic experiment batch

Use xhigh reasoning for this single planning turn. Step 3 is complete and validated. Your goal is to propose the next
small deterministic batch most likely to improve the three-objective Pareto frontier and its fixed-reference
hypervolume. Treat this as sequential multi-objective experimental design inspired by Bayesian optimization, not as a
license to invent measured effects.

## First actions

1. Read `/Users/et/Documents/BayesFolio/experiments/2026-10-etf-family-feature-confirmation/planning/step4_evidence.json` and verify its SHA-256 is `c3d37dc58815e57c1bb50eb90ce8f3636cadd39dcc2156cbfe2b4985852ca661`.
2. Immediately create `/Users/et/Documents/BayesFolio/experiments/2026-10-etf-family-feature-confirmation/planning/NEXT_EXPERIMENT_PLAN.md` with a
   checkpoint containing the current frontier, current hypervolume, and the exploitation/exploration questions you
   will investigate. Do this before broader research so a useful artifact survives response exhaustion.
3. Perform bounded local research using only the evidence packet's `research_allowlist`. Use targeted `rg`, small line
   ranges, and small CSV selections; do not recurse through run directories or dump large artifacts. Allowlist entries
   are relative to the packet's `research_root` (`/Users/et/Documents/BayesFolio`).

## Required design

- Propose 4-8 new configurations: at least one exploitation candidate, one frontier-bridging candidate, and one
  genuinely exploratory candidate.
- Exploit promising neighborhoods of the measured frontier, bridge complementary strengths of existing nondominated
  configurations, and explore coherent cached feature/kernel/task-covariance ideas not tested in the four-config
  confirmation.
- Keep `SPY,MGK,VTV,IWM`, `t_index`, seeds `27,127,227`, the same 24 scored dates, rank 2, maxiter 15, 256 scenarios,
  target, optimizer settings, and cached-data-only rule fixed. Never include IJR, turnover, a time ablation, external
  data refresh, or the production-budget reference.
- Change one coherent feature or covariance hypothesis per candidate when possible. Specify exact parent
  configuration, feature additions/removals, model changes, and runner arguments. Mark unsupported ideas as deferred
  data/code gaps rather than executable candidates.
- IC, Gaussian NLL, and conventional Sharpe are the only objectives. Return and RMSE remain diagnostic-only.
- Do not claim numerical expected hypervolume improvement from four observations. Rank candidates by evidence,
  novelty, frontier-gap coverage, and falsifiability; actual hypervolume improvement is measured only after execution.
- Include deterministic execution order, controls, early stopping rules, run counts, and validation requirements.

## Required outputs

Write both files and do not launch GP fits or modify existing code/configuration:

1. `/Users/et/Documents/BayesFolio/experiments/2026-10-etf-family-feature-confirmation/planning/NEXT_EXPERIMENT_PLAN.md`
2. `/Users/et/Documents/BayesFolio/experiments/2026-10-etf-family-feature-confirmation/planning/next_experiment_candidates.json`

The JSON must conform to `/Users/et/Documents/BayesFolio/experiments/2026-10-etf-family-feature-confirmation/planning/next_experiment_candidates.schema.json`.
Finish by running:

```bash
/Users/et/Documents/BayesFolio/.venv/bin/python   /Users/et/Documents/BayesFolio/experiments/2026-10-etf-family-feature-confirmation/validate_next_experiment_plan.py
```

If validation fails, repair only the two planning outputs and rerun it. Stop after validation passes.
