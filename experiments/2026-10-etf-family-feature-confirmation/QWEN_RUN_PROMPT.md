# Qwen confirmation workflow

Run the complete workflow with one command:

```bash
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/run_qwen_confirmation_workflow.py \
  --execute
```

The workflow performs two strictly separated stages:

1. Deterministic stage 3 executes or resumes the four configurations over seeds `27`, `127`, and `227`, validates
   every run, calculates the Pareto frontier, and calculates dominated hypervolume from the fixed raw-objective
   reference `(IC=0, Gaussian NLL=0, Sharpe=0)`.
2. After—and only after—all 12 runs pass, one Qwen turn runs with `xhigh` reasoning. It researches a bounded local
   allowlist and preregisters the next exploit/bridge/explore batch intended to improve the frontier and hypervolume.

The Qwen turn cannot launch GP fits. It may write only its planning artifacts under this experiment's `planning/`
directory, and its machine-readable candidate batch must pass `validate_next_experiment_plan.py`. The resulting batch
still requires human review before execution.
