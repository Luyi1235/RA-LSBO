# 转入简化飞机：接口 smoke 与模型资格检查

Date: 2026-10-30. Schema `ra_lsbo_aircraft_transition_v1`.
This is an entry/qualification protocol, not a final algorithm comparison or a new
algorithm claim. It follows the mathematical development results already examined.
The optional 33-test transfer diagnostic is NOT a prerequisite. No old result
folder, old pilot audit or copied Windows path is required by this new entry.

## Frozen scope

Reuse `AircraftCase` and its frozen cell model without changing equations,
constraints, mission distances, integration logic, or historical source snapshots.
Tasks retain 100 km / 140 km nominal cruise and SOC reserve 0.2. There are 12
canonical discrete branches (layout/count combinations x three battery choices).
These are reduced point-mass/polar/ideal-propulsor assumptions, not calibrated
flight data or an RCAIDE MDA. `aircraft_sparse` is only a historical task label;
actual sparsity must be reported under the specified sampling distribution.

New entry:
`python -m ra_lsbo.experiments.simplified_aircraft.transition`
Commands: prepare/run/audit/report. Stages: smoke/qualification. No arguments
start nothing. No full pilot, confirm, resume, automatic retry, scenario validation,
CVaR optimization or MDA is exposed. Each directory is write-once. Source bytes,
commit, runtime, tasks and budget are bound before the first evaluation.
A declared complete call limit is mandatory. Locks and unsettled calls are retained.

## 1. Interface smoke: ceiling 36 model calls

Two tasks, one seed (9901), six common initial evaluations per task, six methods,
two online evaluations each: 12 + 2*6*2 = 36. No performance ranking is inferred.
The small model-fit/search settings come from the historical smoke profile.

- random: historical random adapter;
- ev_latent: frozen single-latent phase-discovery policy, cold-only coverage,
  near-repeat guard, no cold local augmentation, original warm development;
- ev_onehot: same phase policy and fitting/search settings with hierarchical one-hot;
- phase_exact: same latent phase policy, exact-only model view, raw observations
  and known feasible incumbent retained; no interval midpoint/penalty imputation;
- ev_random_then_latent: random until first usable feasible, then same latent development;
- original_lsbo: frozen settled-label adapter; original search and stop semantics.

No equal/adaptive experts or kernel averaging is enabled. This is not an official
CBOB implementation. Strict causal model views and evidence/policy auditing are
retained. The legacy native algorithm is not search-effort matched. Smoke only
has two online steps so its five-missing-output native stop cannot yet trigger.

## 2. Model qualification: ceiling 316 model calls

| Role | Calls |
|---|---:|
| Independent random panel, 128 per mission | 256 |
| Two fixed anchors per mission at 20/10/5 s | 12 |
| One default-coordinate probe per 12 canonical branches per mission at 20/5 s | 48 |
| Total | 316 |

The two missions use the same prospective random designs (seed 43101001). Nominal
constraints/labels never select, reject or resample a panel point. The common exact
geometry screen is allowed. Sampling is uniform over configurations and active
coordinates BEFORE that screen; accepted branch proportions need not be equal.
Report total and per-branch counts. All 128 accepted labels remain in denominators.
The anchors and deliberately selected probes are separate, never pooled into an
estimated random feasibility fraction and never fed to an optimizer.

`--size ci` uses two panel points per mission plus the same 60 fixed checks: 64
calls. This only validates execution. It does not replace the 316-call qualification.

Physical states are reported as known-infeasible / usable-feasible / unresolved.
Numerical and infrastructure errors are not used as physical-infeasibility evidence.
The raw evaluator record is unchanged, including any legacy hidden-failure status.
Output kinds (exact/interval/missing) and termination reasons are counted separately.
The unknown-conservative binomial envelope uses the lower endpoint for feasible/n
and upper endpoint for (feasible+unresolved)/n; it is not a physical safety guarantee.
A historical "sparse" label is supported only when that upper endpoint is below
0.10. Inconclusive results are retained; do not silently enlarge samples or tune
mission length to force the label. A zero-success panel does not establish existence.

Energy integration screening compares a fixed design at the registered steps.
Complete outputs require unchanged termination, unchanged constraint satisfaction
labels and an energy range <=1% of the 5 s value. Early/censored outputs are marked
inconclusive, not converged. A failure of this screen is a review item, not a reason
to discard the point or rerun for a favorable answer. Fixed branch probes do not
prove global convergence, non-domination, benefit of latent compression or fidelity.

Manual scientific review must check actual feasible existence, numerical-error
incidence, uncertainty, model response to branch choices, legally derived energy/SOC
bounds, and whether the model supports the intended claims. No automatic pilot
approval is granted even if all software audits pass.

## Observation and risk boundaries

The frozen model already emits nonnegative accumulated-energy lower bounds and
SOC lower bounds under its positive-discharge model. Untested general finite
intervals are not claimed. Nothing here certifies the same bounds for a future
regenerative/nonmonotone model. Numerical failures in the original evaluator
can carry a hidden_failure process label; qualification reports these separately,
and any such occurrence requires review before a performance campaign.

No post-optimization disturbance scenarios are run in this handoff; nominal
optimization remains different from physical-risk or chance-constrained optimization.
Do not claim current CBOB-inspired components as original or as an official CBOB
replication. CBOB integration and final comparator/statistical design are separate.

## Execution and audit

Local handoff: affected regression tests -> 36-call interface smoke -> 316-call
qualification -> fresh audit/report -> stop and review. Research ceiling 352,
0 mathematical benchmark evaluations, 0 RCAIDE MDA; unit-test synthetic/cheap
fixtures are separate and must not be presented as physical research evaluations.
CI authorizes only 36 smoke + 64 small qualification = 100 reduced-model calls.
Full qualification and pilot are not run by CI. No model parameters are selected
using CI rankings.

Auditing reconstructs calls, common initials, candidate streams, raw metrics and
phase/near-guard invariants. It checks hashes again immediately before reporting.
It is not independent posterior refitting, optimizer replay, empirical validation
or proof of algorithm superiority. Old mathematical ledgers and fees remain intact.

## After qualification

Freeze a separate aircraft comparison protocol using the qualified nominal model,
main outcomes, matched representation and feedback controls, and sample-size/budget
reasoning. Keep ordinary and harder conditions; do not tune thresholds by ranking.
Use stage-latent as a candidate, not an already proven winner. Formal CBOB, paired
statistics, disturbance screening and RCAIDE require distinct scope/budget checks.
No additional algorithm module is needed to begin this qualification.
