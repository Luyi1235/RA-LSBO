# Constraint-evidence representation transfer: development protocol (2026-10-30)

This protocol is written after inspecting phase pilot 41042dcd62b696bbaf2bcbd9f365310fbe2f4607.
It is NOT independent confirmation, original pre-registration, or a claim of superiority.
The existing phase policy is retained as a frozen control, not replaced by an untested ensemble.
No historical task, seed, plan, observation, archive or cost is rewritten.

## Decision supported by the phase pilot

Cold-scoped coverage retains 9/9 and 5/6 cold discoveries for phase_coverage, and
8/9 and 6/6 for phase_discovery. Each achieves 39/40 final usable-feasible runs.
Random remains 40/40; no task-dependent winner is selected. Both phase variants
match RA-v2 warm final objectives in 24/25 pairs; the remaining Rosenbrock pair
(seed9706) is worse by about 5.4522e-6 under the original strict numerical tie rule.
Warm equivalence is largely imposed by policy, not a statistical noninferiority result.
A post-first-feasible switch cannot change the earlier first-feasible prefix.
These experiments do NOT identify an independent latent-representation advantage.

The previous declared phase_discovery reference is therefore retained for mechanism
controls. Do not keep tuning coverage on the same 20 seeds or claim it is a new GP.

## Research hypothesis and literature boundary

Hypothesis: early constraint evidence can expose harmful sharing by a performance-
supervised latent representation. Constraint-specific trust in latent vs hierarchical
one-hot surrogates may help discovery without modifying successful warm development.
This is optional experimental code, not a proven innovation or default replacement.

Mixed exact/censored likelihoods and EP follow the CBOB-related existing implementation:
Wang & Li (AAAI2024), https://ojs.aaai.org/index.php/AAAI/article/view/29488 .
Adaptive GP ensembles already exist: Lu, Polyzos, Li & Giannakis (2023),
Surrogate Modeling for Bayesian Optimization beyond a Single Gaussian Process,
https://experts.umn.edu/en/publications/surrogate-modeling-for-bayesian-optimization-beyond-a-single-gaus/ .
Prequential model selection/aggregation is established; see
https://pmc.ncbi.nlm.nih.gov/articles/PMC2898163/ .
Censored active learning is also not new: Karlova et al., COBALT (UAI2026),
https://proceedings.mlr.press/v337/karlova26a.html .

No originality is claimed for likelihood weighting, mixture moments, interval CDFs,
GP ensembles or phase switching separately. The candidate contribution is the
controlled coupling of output-specific partial evidence, representation transfer
and discovery/development separation in hierarchical mixed design. Only matched
experiments on valid, nontrivial tasks can establish an incremental contribution.
Not official CBOB integration, risk-supervised encoder training, chance/CVaR
optimization, calibrated joint feasibility or physical safety certification.

## Six registered methods and isolated change

| ID | Constraint representation / discovery |
|---|---|
| random | Existing common-pool Random control |
| ev_latent | Existing phase_discovery policy, latent posterior |
| ev_onehot | Same phase policy with hierarchical one-hot representation |
| ev_random_then_latent | Uniform novel-pool discovery then identical latent local development |
| ev_equal | Cold-only equal mixture of latent and one-hot constraint heads |
| ev_prequential | Same two experts and search effort; cold-only evidence-updated constraint weights |

The last two retain the LATENT objective and completion heads unchanged. They do
not mix those heads. Each expensive constraint has its own pair of weights.
The equal/prequential comparison isolates weighting; single-expert comparisons
include additional computation, which must be reported rather than called free.
Both portfolio arms fit both experts on every cold step, including coverage slots,
so pre-evaluation scores exist on all actually evaluated cold points. No extra
physical evaluations or labels of unselected candidates are acquired. On entering
the warm phase, both stop updating/fitting the second expert and use the original
latent phase policy. The ev_onehot control uses one-hot in both phases as declared.

All guarded methods use original 1e-8 same-branch active-coordinate resolution,
cold-only coverage at iterations0,5,10,..., and cold singleton fallback. Original
boundary slots remain. Do not interpret this as a total20% exploration/safety bound.

## Causal weighting and observation semantics

For constraint j, expert m predicts a Gaussian before the call, with physical mean
mu[j,m] and variance v[j,m]. Initial weights are(1/2,1/2); initial samples train
experts but are NOT scored retrospectively to initialize weights.

After settlement, likelihood L[j,m] is the Gaussian density of an exact observation,
or the Gaussian probability of its actual reported interval [l,u]. Missing and
untrusted unknown/infrastructure outputs produce NO update. Endpoints are not
midpoint-imputed; open/closed endpoints have identical continuous Gaussian mass.
Exact observation SD is1e-3 times the head scale, matching LearnedMixedHead;
a fixed1e-12*scale^2 predictive variance floor handles degeneracy. Nonfinite scores
stop explicitly, never silently floor probabilities or fabricate measurements.

log w_new[j,m] = log w_old[j,m] + log L[j,m] - logsumexp_m(log w_old + log L).
Equal control keeps weights fixed. Updated weights affect only future proposals.
Prediction packets, candidate identity and prefix are saved and hashed before
calling the evaluator. Auditing reconstructs score arithmetic and phase transitions,
not GP fits. Exact scores are densities and may be positive. Compare experts for
the SAME output/event/units, never sum unnormalized scores across different heads.
The event/coarsening mechanism must be shared by experts and the interval must be
valid evidence; this is not a correction for arbitrary informative missingness.

Constraint probability/factor uses componentwise weighted CDFs, not a Gaussian CDF
of mixture moments. Mixture variance includes between-expert disagreement. The
existing optimistic gate uses moments, and joint feasibility remains an independence
proxy. Weights are predictive trust scores, NOT posterior encoder correctness or
calibration guarantees, especially since experts/encoders are refitted over time.
All-negative evidence can mislead BOTH experts; this design cannot ensure discovery.

## Frozen prospective development scale

New schema ra_lsbo_constraint_evidence_transfer_v1, CLI stage transfer.
Pilot: existing two sparse tasks,8 newly allocated development seeds9801-9808,
12 shared initial samples,30 online calls per method,pool128,panel16/task,two anchors/task.
No feasibility-based resampling. Do not replace warm starts to manufacture cold ones.
These seeds differ from reviewed9701-9720, but the algorithm was selected after
prior pilots; eight seeds and a possibly tiny cold stratum are not confirmation.

Pilot cap = 2*8*6*30 + 2*8*12 + 2*16 + 2*2 = 3108.
Smoke retains four ordinary/sparse cases,seed9601,initial6,online2,pool32,panel8,anchors2:
cap = 112. Workstation smoke+pilot cap3220. Tests' artificial/cheap internal
calls and historical expenses are additional. CI authorizes only112 smoke calls,
zero full pilots and zero RCAIDE MDA. No runtime/quality advantage is promised.

Prepare pilot requires the FULL immutable phase directory and a pinned read-only
receipt (1126 audit evidence files,166 source bindings,8916settled calls).
The importer does not execute old source or rewrite old Windows paths/hashes.
The NEW plan binds current code,dependencies,platform,commit and its own snapshots.
Run requires an explicit full cap; no default run/resume/automatic retry, overwrite,
seed replacement, downstream chaining or MDA. Representation/baselines full-pilot
entrypoints remain withheld. Source changes after prepare invalidate a new plan.

## Acceptance and stop conditions

Primary descriptive endpoint: per-task all-seed and cold-stratum usable-feasibility
at horizon. Report censored first-feasible costs (including no-solution), paired
warm final objective, all-feasible conditional denominators, near/coverage channels,
expert weights and prequential scores, runtimes and all failures. Warm sameness is
an invariant sanity check, not novel empirical benefit. Cold post-discovery quality
has differing residual budgets and selection; report it without naive causal claims.

Compare adaptive vs equal, latent,one-hot and simple random-then-latent. Do not call
adaptive effective solely because it beats latent while failing to beat the simple
hybrid. No multiple-comparison-adjusted or confirmatory claim. If mixtures add cost
without discovery improvement, retain the simpler control; do not silently deploy
or keep tuning until the same seeds win. Review before another full experiment.

General finite-interval value is NOT tested by current mainly sign-censored math
tasks. Nontrivial branch-tradeoff tasks, output-validity guarantees, reference CBOB,
matched representation studies, independent power/precision planning and qualified
aircraft/MDA remain separate future work. They are not completed by this module.
