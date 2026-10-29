# Phase-scoped coverage study v1 (2026-10-29)

This is a POST-RESULT development revision, not independent confirmation. Parent
pilot: ad66df18e3c2e64f1df7202d790eacfa23bddd63; execution cfdf7c30216585b3920f23723236708859435ce7.
The discovery gate is NOT passed: protection improves cold discovery but degrades
paired warm Rosenbrock objectives in 13 of 14 runs relative to same-run RA-v2.
Consequently representation/baselines full pilots are blocked in this revision.
Their smoke and historical code remain; do not rewrite old plans to make them load.

## Mechanism and seven controls

New CLI stage: `phase`, schema `ra_lsbo_phase_discovery_v1`.
- random: unchanged common-pool Random.
- ra_v2: unchanged original RA implementation.
- ra_coverage: old all-phase coverage + near guard; local augmentation in both phases.
- ra_discovery: old all-phase coverage + near guard; local augmentation only after first feasible.
- ra_guard_only: near guard always; no additional scheduled/singleton coverage; local in both phases.
- ra_phase_coverage: near guard always; scheduled and singleton coverage ONLY until first raw usable feasible; local in both phases.
- ra_phase_discovery: same phase-scoped coverage and guard; local only after first raw usable feasible.

Raw feasibility defines the phase, never model-predicted feasibility or existence
of a constraint value. A completed evaluation with missing objective is NOT a
usable feasible incumbent. After first usable feasible observation, preserve it
and disable BOTH added coverage mechanisms in cold-scoped arms; do not disable
the near guard. Original boundary slots and original acquisition, posterior, EP,
encoder, kernel grids and radius adaptation remain unchanged. No guarantee of
restoring development: warm near-filtering can itself change search, hence the
near-only control. No hypothesis is declared correct before this experiment.

The all-phase defaults keep the prior decisions; extra metadata records scope
and activity. No forced short kernels, parameter tuning by task, larger initial
sets, feasibility-based seed replacement, task formula change or oracle screening.
The singleton fallback is not called information gain. Coverage is pool/branch
sampling, not physical safety, chance-constraint or CVaR optimization.

## Budget and execution

Pilot unchanged tasks branin_sparse/rosenbrock_sparse and seeds 9701--9720; 12
shared initial labels, 30 online per arm, pool128, panel16 and two anchors per task.
Online 2*20*7*30=8400, shared initial480, panels32, anchors4: ceiling **8916**.
Smoke four ordinary/sparse tasks, seed9601, initial6, horizon2, pool32, panel8:
ceiling **120**. Local smoke + full phase pilot ceiling **9036**. Old pilot costs,
regression-test evaluators and repeated validation smokes remain separate costs.
All MDA ceilings zero. No automatic next stage, resume, free retries or deletion
of an execution lock. Subsequent version changes require NEW plans/directories.

`prepare --stage phase --profile pilot --prerequisite OLD_DISCOVERY_FOLDER`
requires the exact reviewed parent evidence. `reviewed_evidence.py` validates
published plan/audit/results/registration hashes, ALL sealed evidence and source
snapshots and ledger linkage without executing snapshots or modifying Windows
paths. The receipt enters the new plan. This is a portable IMPORT check, NOT a
rerun of the original environment-bound audit and NOT optimizer replay. Current
run/audit/source/runtime checks remain strict. Unreviewed parents and changed
artifacts are rejected. Do not normalize old CRLF bytes or overwrite old hashes.

Explicit run approval: `--approved-call-limit 8916` (120 for smoke).
Prepare/run/audit/report all use the same new fixed commit and environment.

## Endpoints and promotion

Per task: retain all cold denominators, discovery within30, first-feasible online
cost with censored runs; compare paired warm final objective, NOT merely whether
each algorithm improves its own initial value. Report all pairwise warm win/tie/
loss and median signed difference (left-right, ties rtol1e-10/atol1e-12). No objective
penalty imputation, ratios or confirmatory p-values. Record first-discovery source
and cold/warm coverage, model, boundary and feasible counts from raw prefixes.
Channel attribution describes recorded decisions, not counterfactual causation.

Evaluate both cold-scoped variants against their respective all-phase control,
ra_guard_only, RA-v2 and Random. No aggregate cross-task win can hide a task-level
regression. Review unresolved cold failure and warm degradation explicitly; do not
pick a version only by conditional median or pooled success. The current seeds are
development seeds already inspected. Independent confirmation requires frozen rules,
unseen instances/seeds, primary estimand, precision/power and multiplicity planning.

No CBOB official integration or general finite-interval/branch-tradeoff benchmark
is claimed here. Those research tasks remain after the discovery/development gate.
The existing CBOB-inspired model and factors are not newly invented contributions.
