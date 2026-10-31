# Five-arm aircraft corpus closed-loop development protocol

Date: 2026-10-31. New implementation based on `725bd0e8131f977c912b55e2fe95e87760864d3c`.
This is a candidate engineering revision, not a claim that VAE or corpus training is novel.
Historical source files, objective/constraint equations and outcome records are not changed.

## Scientific question and frozen candidate

Does adequate structural learning with the user-reported 2,000 simulation-free,
possibly repeated configurations and persistent incremental parameter updates
improve actual optimization, beyond both adequately trained reset features and
hierarchical one-hot? Prior offline reconstructions are not closed-loop results.
No new expert ensemble, scaling heuristic, acquisition formula, mean-function
change, constraint relaxation or task redefinition is introduced here.

Arms: `random`, `old30`, `old480`, `corpus_persistent`, `onehot`.
All learned arms use identical phase-discovery policy, interval evidence, kernel
grid, global pool stream, local search settings and near-repeat protection.
Random retains its frozen candidate-pool selector. Old30/old480 rebuild the
existing encoder at each posterior fit for 30/480 updates on the 12-configuration
training grammar. They are not the original LSBO algorithm. One-hot uses no VAE.
Old480 is an adequately trained control, NOT cumulative FLOPs/wall-time matched.

The corpus arm uses the accepted `CorpusRepresentation` implementation: uniform
canonical with-replacement draws, exactly 2,000, preserving draw order and counts;
current observed structures are appended, but only trusted exact objective
observations enter performance supervision. No synthetic physics/feasibility
labels are generated. The distribution is a declared adaptation, not a claim to
exactly replicate an unavailable original X-57 sampler.

Initial joint training uses 30 epochs, batch 128, learning rate 0.001. Persistent
updates before proposals 8,16,24 use 20 epochs and 0.0004. Initial training occurs
before proposal 0, even when that proposal uses scheduled cold coverage. Later
updates are likewise independent of outcomes/coverage channels. With 12 initial
rows and the registered horizon this gives 480+320+320+320=1,440 updates per
corpus trial. Initial smoke uses six rows and still 480 updates. Neural weights
and private RNG are retained; Adam is recreated per fit. No extrapolated state
or previous trial/mission is reused. Each trial has its own corpus/session seed.

All current GP training rows are re-encoded by an immutable feature snapshot;
encoder training prefix and current GP prefix are logged separately. Numerical
NPZ+JSON checkpoints and complete deterministic/stochastic codebook diagnostics
are saved at every scheduled fit, before its first real evaluation. Structural
quality does NOT choose another method, change hyperparameters, stop a trial,
remove a seed or authorize new calls. A failed fit stops the campaign, preserving
evidence; no retries, silent fallbacks or relaxed quality thresholds.

## Tasks, seeds and complete budget

Keep 100 km and 140 km missions, all 12 configurations, all constraints unchanged.
`aircraft_sparse` remains a legacy identifier, not proof of <10% feasibility.
Optimization step is 20 s. Post-optimization 5 s verification evaluates the
same chronological nominal incumbent once; no feedback or replacement selection.
No nominal incumbent means no invented check; spare calls are not reassigned.

Smoke: seed 9930, both tasks, six shared initial rows per case, two online calls
per arm; global pool 32 and inherited reduced GP/local-search settings. Unlike
old smoke, encoder training remains the full registered 30/480/corpus budget.
Maximum =12 initial+20 online+10 verification=42 calls, 10 trials.

Pilot: seeds 9931–9938, both tasks, 12 common initial rows and 30 sequential calls
per arm, global pool 128; 80 trials. New seeds were not used in the inspected
tracked protocols; do not reroll or screen them with outcome labels.
Maximum=192 initial+2,400 online+80 verification=2,672 calls.
Total local smoke+pilot <=2,714 reduced-aircraft evaluations, RCAIDE=0.
Historical 352/3,718 calls and offline fit costs remain separate.
Regression tests use artificial observations; report their cost separately.

Arm execution order is a predeclared cyclic rotation by seed index and case,
not outcome-dependent. Eight seeds / five methods do not give perfect timing
balance. Algorithm RNG and pool streams do not depend on execution order.
Keep every trial and failed candidate. Only infrastructure/unknown/numerical
terminal evidence or an actual software error stops the full campaign after
journaling. Known physical violation and lawful partial feedback are not errors.
There is no resume mode and no deletion of locks.

## Provenance and numerical gates

Prepare a NEW write-once result directory in a clean, committed worktree.
Plan binds raw source hashes, source snapshots, exact source commit, environment,
registered seeds/pools/specifications, qualification receipt and full call cap.
No-argument invocation displays CLI usage; prepare never evaluates designs.
Explicit `--approved-call-limit` must equal the registered complete cap.

Pilot registration requires the complete fixed previous qualification directory
and a newly audited smoke from the SAME commit and runtime. No old qualification
is rerun or fed to training. Require at least one held-out check per task and
all evaluated smoke checks usable feasible and `passed_screen`; no incumbent
outcomes remain explicit. Gate failure means stop for review, not choose a new
smoke seed. Preserve the smoke files; changing them invalidates the pilot plan.

Audit reconstructs budget, initial sharing, pool generation, proposals, statuses,
trajectory metrics, final fixed incumbent and verification association. It also
checks the persistent checkpoint chain, native/raw prefixes, corpus draw identity,
updates, rates, training times, diagnostics and feature identities using forward
passes only. It does NOT refit SGD/GP or replay acquisition optimization.
NPZ, JSON and JSONL are sealed. Training time is already inside proposal/total
optimization time: do not add it twice. Report stage training cumulatively once
per trial, and all control proposal/optimization times separately.

## Endpoints, interpretation and stopping research

Primary comparisons: corpus_persistent vs onehot and vs old480, by paired seed
within each task. Primary numeric outcome: final nominal feasible energy in Wh
at the fixed online horizon. Always show feasible/all and verification/all; never
impute a large number for a missing objective. Report feasibility-first paired
win/tie/loss and conditional paired objective differences, all denominators and
raw seed results. Secondary: first feasible cost, warm/cold strata, same-design
5 s outcomes, objective trajectory, known violations, runtime and training work.

Keep ordinary/constrained tasks separate; shared seeds are not 16 independent
replicates. This is eight-seed prospective development evidence, not an
independent confirmation or a statistical safety guarantee. Do not select per-task
winners or post-hoc hyperparameters. A corpus advantage over old30 alone is
insufficient. If it remains inferior to adequate-training/one-hot controls,
stop adding representation machinery and reassess the latent-space claim; partial
feedback/CBOB comparisons remain separate research questions. No automatic
promotion, further campaign, scenario optimization or RCAIDE authorization.
