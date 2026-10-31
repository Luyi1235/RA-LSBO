# Validation receipt — candidate patch, NOT an uploaded/CI-accepted release

Date: 2026-10-31. Intended application base: `725bd0e8131f977c912b55e2fe95e87760864d3c`.
This receipt deliberately distinguishes completed local tests from remaining gates.

## Completed in this conversation

- New protocol and persistent-optimizer tests: 29 passed.
- Those 29 plus 38 existing structure-corpus tests and 24 mixed-observation/GP tests:
  **91 main tests passed, zero failures/errors/skips; five unittest subtests also passed**.
  Pytest 9 writes 96 JUnit testcases because it includes those subtests. Do not
  mistake 96 for 96 separate main tests or add the 29 to the 91 again.
- Actual VAE training, custom mixed-observation GP fitting, injected-feature
  posterior equivalence, scheduled incremental training, numeric checkpoint
  loading, codebook forward validation, history/cost identities and tampering
  checks were exercised on synthetic data. No real evaluators were called.
- In these core tests only the frozen schema/record loader was narrowed to the
  two genuine modules; no VAE or GP implementation was mocked. This isolates
  unused native BoTorch dependencies and is NOT full-runner acceptance.
- Seven new Python sources parse/compile successfully.
- A full-runner test collection was attempted and stopped with missing `botorch`.
  `gpytorch` is also absent in this environment. No integration test or research
  smoke was run. The failure log is delivered; no test was skipped or suppressed.
- Current validation runtime: Linux/Python 3.13.5, torch 2.10.0+cpu, NumPy 2.3.5,
  SciPy 1.17.0, pytest 9.0.2. This is NOT the pinned Windows/Rcaide_vsp environment.
  The user's environment was not changed. Dependency acquisition was unavailable.

Research aircraft/math evaluator calls=0; RCAIDE=0; BO campaign runs=0.
Synthetic VAE/GP work and unit-test compute are nonzero. No result ranking exists.

## GitHub status

Repository metadata still reports push permission for Luyi1235. However this
conversation exposes GitHub reads, not create/update/commit actions. CLI access
also failed to resolve github.com and had no authenticated bridge. The connected
browser exposes read-only tab tools. Therefore no new remote commit, branch, PR
or CI run was created. Previous PRs and history remain untouched.

The delivery is an additive Git patch against the stated base. It contains code,
tests, protocol, local guide and a manual-only CI workflow. Apply it without
reimplementing the patch, create a real local commit, then run all prescribed
regressions. The full runner must not be called with skipped/failed tests.

## Required local acceptance

Run all seven named regression modules with the original working-machine
packages and validate the JUnit with tools/verify_aircraft_corpus_tests.py.
Then run and audit the new maximum-42-call smoke. Pilot registration requires
that same clean commit/runtime and unchanged successful smoke evidence, plus the
complete previous qualification. Full maximum-2,672-call pilot follows only after
all gates pass. Any failure stops and is returned, not fixed ad hoc by changing
seeds, budgets, epochs, thresholds or evidence. Do not launch RCAIDE.

The optional GitHub Actions workflow is manual-only. It has not run here; running
it later would consume up to an additional 42 reduced-aircraft calls, separate
from the local 2,714-call cap. It does not run full pilot or MDA.
