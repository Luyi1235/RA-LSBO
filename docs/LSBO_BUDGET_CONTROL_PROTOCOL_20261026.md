# Secondary native LSBO full-budget control

Registered after observing native early termination in the ongoing development
study, and before any new continuation truth calls. This is a secondary fairness
check, not a contrast preregistered with the primary experiment.

Apply the same rule to all three seeds in every qualified primary task. Preserve
the native LSBO proposal, model fitting, acquisition, local search, seed, horizon
12 and shared initial history. Change only max_consecutive_solver_failures from
5 to 12, so missing objectives cannot terminate the run before its online budget.
Physical constraint aborts remain distinct from actual numerical solver failures
in the common endpoint records, even though native LSBO counts missing objectives
toward its own historical failure-streak rule.

Before any continuation, replay every cached native prefix and require exact
candidate and metadata equality with the primary trajectory. Reuse those already
charged deterministic truth outcomes. Any prefix divergence stops the secondary
control before a new call. Native trajectories already at 12 need no new truth.
For early-stopped trajectories, evaluate only the unexecuted suffix. An evaluation
at the final budget call may trip the larger threshold; treat this as budget
completion, not premature termination. Do not change the original primary result.

Each native trajectory starts with streak zero and stops no earlier than five
online calls. Thus the complete additional upper bound is 2 tasks*3 seeds*(12-5)
=42. With the 430-call primary upper bound, the complete new plan is <=472 cheap
model calls, zero RCAIDE calls. All actual suffix calls have separate started/
settled records; reused prefixes are counted as reuse, never charged again or
represented as new evaluations. No RA method receives these extra control calls.

Freeze the secondary tool and this document before its execution. Bind manifests
to the original primary proposal/observation files. Replay all full trajectories
using saved outcomes only, verify linkage and report every qualified seed,
including cases with no feasible result. Compare RA against this 12-call variant
as a secondary engineering control alongside the original stopping-policy arm.
Three development seeds still do not establish stable population superiority.
