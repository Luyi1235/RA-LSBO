# SOC revisit diagnosis and independent revision pilot

Date: 2026-10-26. Freeze this protocol and all execution sources before new calls.
This is one complete new study, at most 492 cheap evaluations and zero RCAIDE MDA.
It follows the completed 430-call refinement campaign; no prior unused calls are
transferred, no existing study is reopened, and no physical workers are launched.

## Motivation and fixed revision

Historical aircraft seed8202 has eight SOC reserve aborts. Five have predicted
SOC feasibility approximately .5 while the EICB balance factor is approximately
.975. The latter is an exploration weight, not a feasibility probability. Short
continuous length selection can also return unobserved areas to the zero-mean
prior. Some other aborts have high predicted feasibility, so neither observation
alone establishes a complete causal explanation. A same-pool counterfactual is
prediction-only and must not be represented as an evaluated better design.

Retain the frozen LSBO-style global/branch-local search, exact geometry screen,
active variables, structured representation and radius rules. Keep physical
models, integration step, observations, kernel grid, noise and normalization.
The fixed candidate revision has two independently switchable changes:

1. Average all valid grouped Matérn kernel posteriors using normalized current-
   prefix approximate EP evidence and a uniform discrete kernel prior. Report
   component means, variances and weights. Total variance includes between-model
   variation. Compute CDF, probit completion and EI componentwise before averaging;
   do not substitute a moment-matched Gaussian for the mixture's probabilities.
2. Once a feasible incumbent exists, use EI times the completion probability and
   response feasibility probabilities on exploitation rounds. Keep balanced
   feasibility discovery before an incumbent and the preallocated boundary slots
   (iterations5 and10, one-based). Keep the optimistic support/restoration rules.

Both are literature-based adaptations, not new standalone mathematical formulas.
Independent output probabilities remain a joint proxy; approximate evidence and
EP do not produce a calibrated flight-safety certificate. Representation and
cross-output uncertainty are not integrated out. No aircraft-specific SOC
penalty, exclusion radius, physical oracle or manually supplied feasible start is
introduced. New code is isolated in ra_lsbo/soc_revision; frozen sources stay fixed.

## Independent data and comparisons

Use the same declared hierarchical Branin extension and the already qualified
140km reduced electric aircraft, at the unchanged20s integration step. These are
fresh optimization seeds and labels on development-selected tasks, not independent
task selection or final journal confirmation. Three paired seeds:8401,8402,8403.
Each task/seed has12 newly sampled initial designs shared across five arms, and
each arm has12 online evaluations. No method may stop early or receive more calls.
The independent30-point panel per task is generated before execution, never used
for training, calibration, tuning, acquisition or method selection.

Five fixed arms:

1. revised_full: evidence averaging and probability-weighted exploitation.
2. average_only: averaging with the frozen balanced exploitation rule.
3. probability_only: frozen single-kernel posterior with revised exploitation.
4. frozen_full: the actual frozen previous grouped/local implementation.
5. revised_no_intervals: full revision with interval outputs removed from model
   training only; preserve actual observations for accounting and endpoint scoring.

All arms use the same192-point global stream rule, four local branches, two starts,
twelve local random points and25 SLSQP iterations. Seen-design exclusion can differ
after trajectories diverge. Truth duplicates across arms are charged separately.
Cold and warm initial prefixes are both retained. No optimizer-outcome-dependent
qualification gate, mission retuning, seed replacement or automatic retry.

Exact complete-study maximum:

    2 tasks * 3 seeds * (12 initial + 5 arms * 12 online) + 2 * 30 panels = 492.

No new original LSBO or one-hot comparison is included in this mechanism pilot;
old cohorts cannot substitute for such comparisons on these new seeds. Full paper
design must include them independently.

## Outcomes, analysis and stopping

Primary contrasts are fixed revised_full versus frozen_full on both tasks. The
other arms diagnose the two switches and partial feedback; do not select the best
ablation after seeing results and relabel it as the prespecified full method.
Report every seed's observed feasible endpoint, no-feasible cases, online usable
feasible count, first-feasible call, SOC reserve aborts and physical/model/numerical
status. Raw endpoint order is supplemented by prespecified practical ties:40Wh
for aircraft and .01 for Branin. These are reporting tolerances, not certificates.
Report common-panel Brier/log loss, positive and negative class scores and the
smoothed training-rate constant reference; repeated forecasts of one panel do not
increase its independent design count. Report local queries and elapsed time.

Development continuation gate for the fixed full revision: on aircraft, fewer
aggregate SOC aborts, no fewer runs with feasible endpoints, and more practical
endpoint wins than losses against frozen_full; on Branin, no fewer feasible runs
and no more practical losses than wins. Passing is only a three-seed engineering
screen, not a significance test or proof of stable superiority. A failed gate is
reported without post-hoc changes or extra experiments in this study.

Bind all source files, protocol and historical diagnostic inputs; save snapshots.
Audit all492 started/settled calls, candidate/contract identities, initial and
panel separation, all360 online proposals including continuous-search metadata,
all30 initial-prefix panel model fits, and final summaries using saved outcomes
only. An exception leaves the call charged and study stopped; no automatic retry.
Historical MDA's unknown call stays charged/unknown and is untouched.
