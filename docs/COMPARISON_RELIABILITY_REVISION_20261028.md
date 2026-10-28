# External comparison reliability revision (2026-10-28)

Schema: `ra_lsbo_external_comparison_v2`. Base commit:
`8e141c821702b052df4b916c0e4d86a275432497`. Development branch:
`fix/comparison-reliability-20261028`.

This is a correctness/numerical revision, not optimization-performance evidence.
It supersedes the execution/configuration descriptions of the dated v1 external
comparison protocol for NEW plans only. Never rewrite old registration hashes,
source snapshots, observations or costs. Do not resume an old plan under v2.
The core seven-arm study and frozen LSBO/aircraft baseline are not replaced.

## Scope and complete budget

Roster remains RA-LSBO, Random, logCEI, SCBO-style adapted, SEGO-UTB-style
adapted and NSGA-II-style adapted. The four adapters are NOT official software
or faithful paper reproductions. Baseline GP kernel hyperparameters remain
fixed. SCBO uses exact categorical equality, common-pool trust regions and
Bernoulli feasibility sampling, not reference SCBO constraint-function sampling.

Pilot seeds 9701--9720, 12 common random initial labels and 30 sequential calls
per method; pool 128. Mathematical cases remain branin_sparse and
rosenbrock_sparse; the separate aircraft pilot remains aircraft_sparse. No seed,
objective, threshold, domain, anchor, panel or scenario is chosen using observed
method rankings. Smoke and pilot remain exploratory.

| Role | Mathematical pilot | Aircraft pilot |
|---|---:|---:|
| Anchors | 4 | 2 |
| Independent panels | 32 | 16 |
| Shared initial labels | 480 | 240 |
| Online | 7,200 | 3,600 |
| Integration checks | 0 | 2 |
| Held-out scenarios | 0 | 960 |
| Complete ceiling | 7,716 | 4,820 |

Combined pilot ceiling 12,536, zero RCAIDE MDA. Interrupted historical calls and
smoke calls are separate incurred costs; restarting does not erase them. Smoke
ceilings: mathematical 112 and aircraft 108. Aircraft validation may use fewer
calls if there is no nominal feasible incumbent; unused calls are not reassigned.

## Changes

1. Local SLSQP candidate generation validates finite coordinates, scores and
   supports. Only round-off excursions up to 1e-10 in normalized coordinates are
   projected; material/nonfinite endpoints are rejected. Valid cached candidates
   and the global pool survive a local failure. Solver status/success/message
   remain distinct from endpoint validity. A nonconverged solve can supply a
   valid candidate but is never relabelled as converged.
2. Only the exact known SciPy SLSQP clipping warning is intercepted within one
   serial solve. Its count is retained in `slsqp_clipping_warnings`; unrelated
   warnings keep their normal handling. `bound_projections` and
   `max_callback_bound_excess` describe points seen by our callbacks, not moves
   already clipped inside SciPy. Counts are not severity measures or evidence of
   physical feasibility. No global warning suppression, site-packages changes,
   solver replacement or physical-constraint relaxation.
3. Stable Gaussian log-EI replaces log(floored ordinary EI). Use erfcx and a
   Mills-ratio expansion below z=-20; zero variance has the deterministic limit.
   Finite representable log-EIs preserve tail rankings. Values beyond floating
   range are diagnosed without serializing infinity in proposal metadata.
4. Evidence ordering distinguishes usable feasible, completed known-infeasible,
   unresolved, hidden failure, and infrastructure/unknown. Within known-infeasible
   records compare known violation count, unresolved count and observed numeric
   lower bound. Thus (0,+infinity) remains a known violation despite infimum zero.
   This is a declared adapter decision policy, NOT an assertion that unknown
   physical designs are worse. Missing objectives no longer receive a large
   fabricated numeric penalty. NSGA parent ranking and SCBO centers use the key.
5. SCBO best-so-far includes the shared initial/history prefix and never regresses
   when a worse feasible point arrives. Trust-region counters compare against the
   incumbent, not the last observation.
6. Baseline objective GPs use all trusted completed exact objectives, including
   explicitly infeasible designs. Untrusted infrastructure/unknown outputs do
   not train heads. SEGO also uses completion/sign evidence, an explicit eligible
   mask and lexicographic restoration; a fixed numeric penalty can no longer
   beat an eligible candidate with a very negative log-EI.
7. Audit reconstructs full registered horizons, expected call identities/roles,
   shared labels, candidate-pool generation, qualification, integration and
   scenario-risk summaries. It seals structured evidence with hashes and
   reporting re-audits against the seal. Repeated audit is idempotent only for
   unchanged evidence. This is NOT optimizer replay or physical-model validation.
8. Execution requires an explicit full-call cap. No-argument compare commands
   prepare only. External confirm is rejected until its anchored policy is
   implemented; core study profiles are untouched. New tests, this revision
   protocol, sklearn and matplotlib versions enter registration.

## Restart rules

Stop only the identified old mathematical process; preserve its lock, ledger,
plan, snapshots and console log. Count settled and started-but-unsettled calls
separately; no automatic retry. Use a separate checkout/worktree for v2, leaving
old source paths available. Run regression tests and a new smoke, then prepare a
NEW pilot. Do not change registered files or the environment between prepare,
run, audit and report. Do not reuse or overwrite any earlier result directory.

```text
python -m ra_lsbo.experiments.mathematical_benchmarks.compare prepare --profile pilot --output runs/math_v2_NEW
python -m ra_lsbo.experiments.mathematical_benchmarks.compare run --output runs/math_v2_NEW --approved-call-limit 7716
python -m ra_lsbo.experiments.mathematical_benchmarks.compare audit --output runs/math_v2_NEW
python -m ra_lsbo.experiments.mathematical_benchmarks.compare report --output runs/math_v2_NEW
```

This revision does not fix all scientific limitations. Current tasks are small
hierarchical extensions, not unmodified published benchmarks. Under uniform
sampling over their four branches and active coordinates, each sparse math task
has theoretical feasibility pi*0.16*0.20*(1+0.93+0.81+0.69)/4, about 8.62%; a
12-point initial set is cold with probability about 33.90%. Do not replace warm
seeds. Each feasible augmented design has a non-worse basic counterpart in these
sparse tasks. Preserve this limitation rather than tune the task during a fix.
New branch-tradeoff tasks and confirmatory statistics need separate prospective
protocols. Conditional objective summaries retain feasible denominators. RA's
marginal-product feasibility is still an independence proxy with no safety
probability guarantee. Aircraft scenarios remain post-optimization validation.

## Validation and sources

`test_numerical_safeguards.py`: high-precision log-EI, bounds, finite checks,
scoped warnings. `test_comparison_reliability.py`: adapter regressions, solver
fault injection and synthetic campaign corruption tests. The CI workflow runs
these plus existing affected tests and bounded smoke campaigns (combined ceiling
220; no RCAIDE MDA). Unit tests also construct synthetic evidence and may invoke
cheap evaluators; these are not the smoke/pilot research campaign ledger. Read
actual CI logs for completion and counts; a committed workflow is not a pass.

SciPy explicitly clips SLSQP trial points, including objective/gradient and
finite-difference constraint checks:
https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_slsqp_py.py
https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_optimize.py
Stable log-EI numerical reference:
https://botorch.readthedocs.io/en/latest/_modules/botorch/acquisition/analytic.html
Our NumPy/SciPy implementation is independently tested, not BoTorch reference
code or a qLogEI implementation.
