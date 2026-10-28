# External comparison reliability revision (2026-10-28)

Schema: `ra_lsbo_external_comparison_v2`. Base commit:
`8e141c821702b052df4b916c0e4d86a275432497`. Development branch:
`fix/comparison-reliability-20261028`.

This is a correctness/numerical revision, not evidence of optimization superiority.
It supersedes the execution/configuration descriptions in the dated v1 comparison
protocol for NEW plans only. Old plans, source snapshots, observations and costs
must not be rewritten or resumed under this implementation. Do not recompute a
registration hash to make an old plan pass. The core seven-arm historical study
and frozen LSBO/aircraft baseline are not replaced.

## Fixed experimental scope and budget

The roster remains RA-LSBO, Random, logCEI, SCBO-style adapted, SEGO-UTB-style
adapted, and NSGA-II-style adapted. The four baselines remain repository-specific
adapters, NOT official implementations or faithful reproductions of their papers.
The GP baseline kernels still have fixed hyperparameters; the SCBO adapter uses
exact categorical equality, a common candidate pool and Bernoulli feasibility
sampling, not full reference SCBO constraint-function sampling.

Pilot: seeds 9701--9720, 12 shared random initial labels, 30 sequential evaluations
per method; pool 128. Cases remain branin_sparse, rosenbrock_sparse, and the
separate aircraft_sparse task. No seed, domain, objective, constraint threshold,
anchor, panel, or flight scenario is selected using observed method rankings.

| Role | Mathematical pilot | Aircraft pilot |
|---|---:|---:|
| Anchors | 4 | 2 |
| Independent panels | 32 | 16 |
| Shared initial labels | 480 | 240 |
| Sequential evaluations | 7,200 | 3,600 |
| Integration checks | 0 | 2 |
| Post-optimization scenarios | 0 | 960 |
| Complete ceiling | 7,716 | 4,820 |

Combined ceiling 12,536, zero RCAIDE MDA. Aborted historical calls and new smoke
calls are separate incurred costs and are not erased by starting a fresh pilot.
Smoke ceilings are 112 mathematical calls and 108 aircraft calls. Aircraft calls
may be fewer if no nominal feasible incumbent exists; unused scenario allowance
is not reassigned. Neither smoke nor pilot is publication confirmation.

## Correctness changes

1. Local acquisition SLSQP search checks finite coordinates, acquisition and
   support values, projects only round-off excursions (1e-10 in normalized
   coordinates), and rejects material out-of-bounds/nonfinite endpoints. Valid
   cached candidates and the global pool remain available when an optional local
   solve fails. Solver `success`, `status`, `message`, local error and endpoint
   diagnostics remain separate. An iteration-limit result may contribute valid
   points but is never relabelled as solver convergence.
2. Only the exact known SciPy RuntimeWarning about clipping an out-of-bounds
   SLSQP trial is captured within one serial solve. Its occurrence count is saved
   as `slsqp_clipping_warnings`; other warnings retain their normal handling.
   `bound_projections` and `max_callback_bound_excess` describe callbacks observed
   by this code, NOT excursions already clipped inside SciPy. A warning count is
   not a severity measure or evidence of physical infeasibility. No site-packages
   files are edited; no global warning suppression or constraint relaxation.
3. Stable analytic Gaussian log-EI replaces log(floored ordinary EI) in logCEI
   and the SEGO-style adapter. Negative-tail values use erfcx and a Mills-ratio
   expansion below z=-20; exact zero variance uses deterministic improvement.
4. Evidence-first baseline ordering distinguishes usable feasible, completed
   known-infeasible, unresolved, hidden-failure and infrastructure/unknown states.
   Within known-infeasible states it compares known violation count, unresolved
   count and observed numeric lower bound. A strict interval (0,+infinity) remains
   a known violation although its infimum is zero. This is a declared decision
   policy, not a claim that unknown physical designs are worse or that an interval
   lower bound equals the true violation. Missing objectives are not assigned a
   large penalty. The policy is applied to NSGA parent ranking and SCBO centers.
5. SCBO best-so-far is initialized from the actual shared/history prefix and
   cannot be overwritten by a worse feasible observation. Success/failure counts
   therefore refer to improvement over the incumbent, not the last observation.
6. Baseline objective GPs use all trusted completed exact objectives, including
   explicitly constraint-infeasible designs. Unknown/infrastructure outputs do not
   train physical or completion heads. Partially observed bounds are still not
   fabricated as exact targets. SEGO uses completion/sign evidence weighting,
   an explicit eligible mask and lexicographic restoration when no candidate is
   eligible; an arbitrary numeric penalty can no longer beat a very small log-EI.
7. Audit reconstructs all registered trials, every expected call/role, common
   initialization, candidate pool generation, qualification statistics,
   integration evidence and aircraft scenario risks. It checks exact horizon
   completeness and all summary identities. A structured-evidence hash manifest
   seals the audit. Reporting re-audits and rejects changed evidence. Repeated
   auditing is idempotent only for unchanged evidence. This is NOT independent
   optimizer replay or validation of the physical model's truth.
8. Execution always requires an explicit complete-call cap. No-argument compare
   entrypoints prepare a plan only. The external `confirm` profile is rejected
   because its anchored policy is not implemented by this runner; core study
   profiles remain untouched. sklearn and matplotlib versions and new tests are
   included in source/runtime registration.

## Restart and interpretation

Stop only the identified old mathematical process; preserve its lock, ledger,
plan, snapshots and console log. Account separately for settled and started-but-
unsettled calls. Do not automatically retry an unsettled call. Use a separate
checkout/worktree for v2 so the old code paths remain available for historical
verification. Run regression tests and a fresh smoke, then prepare a NEW pilot.
Do not modify any registered source between prepare, run, audit and report.

No-argument execution now only prints the explicit run command. Recommended
commands (use a new output directory every time):

```text
python -m ra_lsbo.experiments.mathematical_benchmarks.compare prepare --profile pilot --output runs/math_v2_NEW
python -m ra_lsbo.experiments.mathematical_benchmarks.compare run --output runs/math_v2_NEW --approved-call-limit 7716
python -m ra_lsbo.experiments.mathematical_benchmarks.compare audit --output runs/math_v2_NEW
python -m ra_lsbo.experiments.mathematical_benchmarks.compare report --output runs/math_v2_NEW
```

The task remains a small declared hierarchical extension, not a published
unmodified benchmark. Under its registered uniform branch/active-coordinate
sampling the two sparse mathematical tasks have theoretical feasibility
pi*0.16*0.20*(1+0.93+0.81+0.69)/4, about 8.62%. A 12-point initial set is cold
with probability about 33.90%; do not replace warm seeds to increase this number.
The basic branch has a non-worse counterpart for every feasible augmented design
in these sparse tasks. These limitations are retained, not tuned away in a bug
fix. New branch-tradeoff benchmarks and confirmatory statistics need separate
prospective designs. Conditional objectives retain feasible denominators. RA's
marginal-product feasibility remains an independence proxy with no safety
probability guarantee; aircraft scenarios remain post-optimization validation.

## Validation and references

`tests/test_numerical_safeguards.py` exercises high-precision log-EI values,
bounds, finite checks and scoped warning handling without GP/model dependencies.
`tests/test_comparison_reliability.py` exercises the real adapter interfaces and
complete-campaign audit with synthetic labelled fixtures, plus solver endpoint
fault injection. The CI workflow additionally runs existing affected tests and
two bounded smoke campaigns (combined ceiling 220, zero RCAIDE MDA). See the
workflow attached to the final code commit for actual pass/fail status and counts;
workflow presence alone is not a passing test result.

SciPy 1.13.1 explicitly wraps SLSQP objective/gradient evaluations to mitigate
small out-of-bound solver moves; constraints may also invoke its clipping guard:
https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_slsqp_py.py
https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_optimize.py
The stable log-EI numerical principle is also used in BoTorch:
https://botorch.readthedocs.io/en/latest/_modules/botorch/acquisition/analytic.html
Our NumPy/SciPy implementation is independently tested, not claimed to be the
BoTorch reference implementation or its multivariate/qLogEI extension.
