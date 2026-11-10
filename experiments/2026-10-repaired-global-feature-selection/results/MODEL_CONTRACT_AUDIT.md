# Actual model and causal-preprocessing audit

`audit_model_contract.py` constructed the largest completed registered mask in
each of the four family universes and the actual fifteen-task universe. It used
the same wrapper, command builder, parser and model builder as the experiments.
No optimizer, posterior sampling or new backtest was invoked. All five checks
passed; each first window has 84 training months (336 rows for four tasks, 252 for
three tasks, 1,260 for fifteen tasks). The JSON report records actual model reprs,
resolved covariance configs, plan/data/source hashes and installed runtime versions.

## What is actually fitted

- Data covariance: additive time, ETF-local and macro blocks plus all three
  configured pairwise block interactions. Time and ETF blocks use ARD Matern
  nu=0.5 kernels. The macro block adds ARD Matern nu=0.5, ARD rational quadratic
  and linear components. Direct components and interaction terms are scaled.
- Task covariance: signed `IndexKernel`, rank two, with
  `LKJCovariancePrior(eta=2, sd_prior=LogNormal(0,0.5))`.
- Mean: task-specific constants (`MultitaskMean` of `ConstantMean`).
- Likelihood: `HadamardGaussianLikelihood` with learned task-specific noise,
  `LogNormal(-4,1)` noise prior and variance floor 0.005 in each task's standardized
  target units. Recency/fixed-observation-noise experiments are not active.
- Predictive spreads include observation noise (`posterior(..., observation_noise=True)`);
  they are predictive-return uncertainty, not only latent-function uncertainty.
- No time-varying outputscale/lengthscale modulation or higher-order interaction
  tuning. t_index still participates as an ordinary feature/block.
- The fixed adaptive lengthscale rule uses LogNormal location
  sqrt(2)+0.5*log(d), scale sqrt(3), and lower bound 0.025*sqrt(d) for a component's
  ARD dimension d. Removing a dimension therefore changes geometry/prior bounds
  through the same documented rule; the test does not isolate information removal
  from all finite-budget GP effects.

The signed-kernel replacement and likelihood were checked as actual constructed
objects, not inferred solely from a CLI experiment name. Saved completed seed-27
first-window likelihood diagnostics agree with the construction and noise floor.

## Training-only preprocessing

The selected CLI mode is `botorch_normalize`, with non-task feature normalization
learned from each training window. Outcome scaling is per-task
`StratifiedStandardize` built from training X/Y. The alternate legacy
`june_manual_minmax` path includes evaluation inputs in its bounds, but that path
is not selected by this study.

The audit replaced evaluation predictors and targets with extreme values and
confirmed that training tensors did not change. Constructed input bounds exactly
matched training minima/maxima. After switching the model to evaluation mode,
extreme prediction inputs could not update learned input bounds or outcome
means/standard deviations. No full-panel learned preprocessing is used.

## Legacy metadata caveats, not model changes

1. The June runner writes `modeling.task_covar_prior = BetaPrior(2.5,1.5)` for
   signed-LKJ variants. This is a misleading reporting branch. The actual
   model-builder objects use LKJ eta 2 with the LogNormal scale prior above.
   Original manifests and execution-critical source files remain unchanged;
   use this source-hashed overlay to interpret that field.
2. The reusable July wrapper assigns every ticker `us_equity` in `ASSET_GROUPS`.
   That mapping is consumed only by the legacy within/between-group correlation
   summary, not model construction. Thus its between-group correlation is empty
   and its within-group correlation is not a four-family statistic. Neither is
   used for selection. The study's family IC/NLL/coverage/allocation reports use
   the approved four-family ticker map independently, including the joint-objective audit.

These findings do not justify rerunning or changing the GP. They do require
clear interpretation of raw legacy fields. Any future upstream metadata cleanup
must be separately versioned so it does not invalidate this frozen study's
source fingerprints or rewrite original results.

## Reproduce

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_model_contract.py
.venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_model_contract.py
```

This checks construction and preprocessing, not predictive economic validity or
untouched generalization. Those claims still require the actual joint results,
uncertainty reports and the final research limitations.
