# Worker-queue repair — October 25, 2026

The first detached bulk invocation stalled exactly after 14 workers completed
four jobs each. There were 84 complete new configuration/month/seed checkpoints
(28 pilots plus56 bulk fits), and four successful baseline replay audits.
The parent was idle, all fit workers had exited, and no new task had started.
No GP fit reported a model or numerical failure.

This matches the Python ProcessPoolExecutor worker-recycling deadlock documented
in [CPython issue115634](https://github.com/python/cpython/issues/115634).

Repair: remove max_tasks_per_child=4. Retain the same bounded spawn executor,
14 single-thread workers, per-fit explicit garbage collection, model builder,
GP parameters, seeds, constraints and static blending recipes. Workers now live
for the fit phase, then the entire pool closes before any allocator pool starts.
Every fit already records its peak RSS for memory diagnostics.

Only the idle campaign parent PID67989 was terminated. No output was deleted or
modified. The restarted workflow validates and reuses all completed checkpoints.
The remaining planned compute is2,268 new fits, not a rerun of completed fits.

Verification: targeted ProcessPool source guard test, all campaign tests, Ruff,
protected-input/checkpoint hash verification, and actual progress past the old
56-job recycling boundary after restart. The old launch record is retained in
outputs/repairs/first_launch.json before writing the new launch record.
