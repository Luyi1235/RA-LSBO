# Discovery stages v1 — prospective development protocol (2026-10-29)

Base: `9360da8e4d9777228a873257b93992234f81be2b`; branch
`research/discovery-stages-20261029`. Schema: `ra_lsbo_discovery_stages_v1`.
This protocol is written AFTER inspection of the 20-seed mathematical pilot.
It registers development hypotheses, not independent confirmation or superiority.
Historical algorithms, source snapshots, ledgers, seeds and task formulas remain unchanged.

## Implemented and deliberately gated stages

The new CLI is `python -m ra_lsbo.experiments.discovery_stages.compare`.
No arguments run no experiment. Commands are prepare, run, audit, report.
Each plan is a NEW directory with its own source snapshot, commit, dependency and
platform binding. A full explicit call limit is required by run. There is no resume,
free retry, auto-update, stage chaining, confirm profile, aircraft/MDA entry or
performance-based seed replacement. A leftover lock or unsettled call is preserved.

| Stage | Arms | Mathematical pilot ceiling | Smoke ceiling |
|---|---:|---:|---:|
| discovery | 5 | 6,516 | 104 |
| representation | 8 | 10,116 | 128 |
| baselines | 8 | 10,116 | 128 |

Pilot: branin_sparse, rosenbrock_sparse; seeds 9701–9720; 12 shared initial calls,
30 online calls/arm, pool 128, 16 independent panel points/task and two anchors/task.
Discovery budget: 6,000 online + 480 shared initial + 32 panel + 4 anchor.
Eight-arm budget: 9,600 online + the same 516 non-online calls.
Smoke retains the four mathematical ordinary/sparse cases, seed9601, six initial,
two online, pool32, eight panel, two anchors. All MDA ceilings are zero. Native LSBO
may stop earlier under its original missing-objective rule; unused calls are never
reassigned. These are upper bounds, not accomplished experiments. Historical calls,
unit-test evaluators and new smoke calls are additional costs. CI authorizes ONLY
three smokes (combined ceiling360), not any full pilot.

Later pilot plans require `--prerequisite <preceding-pilot-directory>` and a matching
read-only audit. This is a provenance gate, NOT automatic scientific approval.
Review the preceding results before separately authorizing the next budget. All
stages must use the same frozen checkout/environment for that prerequisite audit;
changing code requires a new version and migration review, not rewriting old plans.

## P0/P1: discovery policy and factorial controls

1. random: unchanged historical common-pool Random adapter.
2. ra_v2: unchanged RA-v2 implementation and settings.
3. ra_cold_global: same posterior/acquisition; disable ALL local augmentation
   (both local random candidates and SLSQP) until a raw usable feasible incumbent
   exists, then restore original local augmentation. No new coverage/near guard.
4. ra_coverage: retain cold/warm local search; add the protection package below.
5. ra_discovery: combine cold-global and the protection package.

Protection package: (a) no send-to-model candidate within L-infinity 1e-8 of history
on active normalized continuous dimensions in the SAME canonical discrete branch;
also deduplicate the candidate list at that resolution; (b) coverage at iterations
0,5,10,..., independent of learned constraints; (c) after model selection, if only
zero/one eligible candidate remains, use the same global coverage channel instead.
Coverage selects the least-visited available branch (ties randomly, sorted branch
keys), then samples uniformly from that branch's novel common-pool candidates.
It applies legal exact geometry screening and the near guard only. No hidden true
constraint, qualifier/panel label or oracle is consulted. It is not uniform sampling
of the entire design domain, a true information-gain acquisition, a physical-risk
budget or a chance/safety guarantee. The 20% schedule is fixed, not tuned to new results.

The 1e-8 tolerance is an explicit numerical decision resolution chosen during this
post-pilot revision, NOT physical equivalence, simulator accuracy or learned truth.
It never changes design_id, observations or historical costs. Do not apply this
policy unchanged to noisy tasks with intentional replication. Exhausted novel pools
raise BEFORE any model call; no tolerance relaxation or silent repeat. Guarded
coverage slots skip posterior fitting; cost in model evaluations stays one per step.

We intentionally do not force short kernels, change EP, add risk supervision to the
encoder or enable kernel averaging. Those would confound this first diagnosis.
Metadata records raw/model history, phase, model eligibility, fallback, source of
selection, exact near-rejection distances and the original local solver reports.
A singleton entropy slot is now explicitly named a coverage fallback.

## P2: matched representation / information views

Eight arms: Random, random_then_local, and {latent, hierarchical-onehot} crossed
with {exact, sign, interval}. The six matched arms use the combined discovery policy,
identical kernels grids, local effort, initialization, completion evidence and budget.
Only the representation and model input view vary. Exact constraint values remain
exact in all three views; exact removes interval targets; sign reduces resolved
constraint intervals to g<=0 or g>0 and omits overlapping intervals; interval retains
all valid bounds. Missing/infrastructure evidence is never fabricated. Objective
intervals are omitted in exact/sign because there is no universal objective sign
threshold. Original evaluator records remain byte-exact; only model views change.
Raw usable-feasibility still defines incumbents and phase transitions for every arm.

random_then_local uses uniform novel common-pool discovery until first feasible,
then the SAME hierarchical-onehot, exact+sign RA local development as onehot_sign.
It is a transparent simple hybrid, not a claim to be the frozen original LSBO.

IMPORTANT: current mathematical tasks mostly return (0,+infinity) and exact values.
Sign and interval variants may therefore be identical. This stage does not establish
the value of general finite intervals. Nontrivial branch-tradeoff tasks, valid
multistage finite bounds and a faithful CBOB integration require separate protocols;
this implementation does not invent them or call its own adapters CBOB.

## Reference checks

Eight arms: Random, combined RA, frozen original LSBO through the existing settled
outcome adapter, unchanged LogCEI/SCBO/NSGA adapters, historical conservative SEGO
under explicit new label sego_conservative_v2, and sego_utb_v3. The latter changes
only the inequality-bound sign: g<=0 uses mu-beta*sigma<=0, beta1.96. It retains
fixed kernels, completion/sign weighting and other repository adaptations; not an
official SEGO reproduction. Historical results keep their original records/labels.

Source: Priem et al., Upper Trust Bound Feasibility Criterion for Mixed Constrained
Bayesian Optimization with Application to Aircraft Design (2020), equation(3):
https://arxiv.org/html/2005.05067 . Original convention g>=0 becomes the above rule
under g->-g. Regression tests check sign equivalence, beta0 and expanding uncertainty.

The native LSBO retains its search settings and consecutive-missing-objective stop
(default5); it is NOT search-effort matched. No synthetic objective is added to make
it run. Its internal evaluate adapter consumes an ALREADY settled label and cannot
invoke an unaccounted physical evaluator. Early-stop trajectories are carried to
nominal horizon, costs and stop reasons reported, never replaced by successful seeds.

## Metrics, evidence and decision gates

Primary development endpoint: cold-start discovery at nominal horizon per task;
secondary: censored first-feasible cost, warm improvement, full paired success table,
conditional paired objective with denominators, unknown/failed/stopped counts and
time. Cold/warm is assigned after evaluating shared initialization, not resampled.
Reports never replace missing objectives with penalty numbers. Current bootstrap
intervals condition on both arms feasible and are exploratory, not a confirmatory
family of tests. No multi-comparison-adjusted superiority claim is made.

Software gate: full ledger pairing and expected tags; all non-native trials at full
horizon; native stops substantiated by five consecutive online missing objectives;
shared initialization, source bytes, canonical candidates, deterministic common
pools and summary reconstruction; protected policy invariants; fresh evidence seal
before reporting. An audit is not posterior/optimizer replay or model truth validation.

Scientific review before promotion: on each task, compare combined RA against ra_v2
AND Random on cold discovery with all cold denominators; inspect every regression;
verify warm-start improvements have not been sacrificed for discovery. Do not select
a strategy merely by the smallest conditional median. If combined RA does not improve
its original discovery failure, stop promotion and revise the mechanism hypothesis.
This review gate deliberately requires a recorded human decision, not an automatic
claim based on low-powered, already-inspected development seeds.

Independent confirmation must use a later frozen algorithm, unseen task instances/
seeds, explicit primary endpoint, power/precision analysis and multiplicity plan.
Only after mathematical and simplified-aircraft qualification should an RCAIDE MDA
protocol be authorized. Those future stages are NOT implemented/validated by these
smokes and cannot be enabled through this CLI. Scenario validation remains outside
nominal optimization; there is no chance-constraint or CVaR optimization claim.
