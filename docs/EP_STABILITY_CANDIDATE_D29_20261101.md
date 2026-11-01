# EP numerical-stability candidate: bounded D29 SOC verification

**Scope:** A candidate numerical change, NOT a fixed pilot, not a new airplane
evaluation, and not permission to resume the 2026-10-31 interrupted campaign.

## Frozen facts

Original interrupted pilot: aircraft_sparse, seed 9936, old30, online
iteration 29. Original code fc1ac773f5bd9e60abd2ca4b56bd971a707735e2.
Its 2,212 settled simplified-model evaluations remain charged.
Original release:
https://github.com/Luyi1235/RA-LSBO/releases/tag/aircraft-corpus-pilot-interrupted-20261031
Immutable archive SHA256:
aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9

Sealed Windows diagnosis (eight D29 SOC kernels):
https://github.com/Luyi1235/RA-LSBO/releases/tag/aircraft-soc-diagnostic-20261031
7/8 failed within 60 EP sweeps, 1/8 failed truncated-normal tail moments;
none reached log_evidence. This is an observed SOC GP failure, not proof that
SOC was the original optimizer's first failing constraint.

Read-only inspection of the 41 D29 observations (no evaluator):
17 exact SOC margins range roughly [-.18965, -.02780];
24 lower-censored SOC margins have lower bounds roughly
[1.56e-5, .005863]; all 41 design IDs are distinct. The 29th
new online observation is exact, not a new censored interval.
The data can be numerically difficult without containing a duplicate design
or an invalid fabricated target.

## Opt-in candidate only

- New risk_core/ep_numerics.py: 64-node Gauss-Legendre quadrature for
  extreme one-sided standardized Gaussian tails (threshold 8 sigma),
  computed relative to the truncation boundary. Reflected left tails are
  handled by Gaussian symmetry. Unlike direct variance subtraction at
  hundreds of sigma, this calculates the small conditional variance in
  local coordinates. Normal intervals retain the legacy moment routine.
- MixedGPSettings has two new booleans with false defaults:
  stable_tail_moments and record_ep_trace.
- When both are false, the original EP numerical iteration remains unchanged.
  The new flags *do* appear in serialized model settings, so numerical legacy
  equivalence is not a byte-identical historical model identity claim.
- With record_ep_trace=true, each sweep records its maximum relative site
  change and site precision/linear-term diagnostics. On failure, exception
  attributes carry trace, last change, and the offending site (if applicable).
- This does **not** change damping=.7, max_sweeps=60,
  tolerance=1e-8, kernel grids, exact/interval noise, or any output labels.
  It never silently accepts an unconverged EP posterior.

## Windows original-environment test — one bounded run, no simulator

Use an independent NEW worktree at the exact *tested* candidate commit
provided with the handoff; original Windows Rcaide_vsp environment, unchanged
dependencies, thread environment OMP/MKL/OPENBLAS=1, PYTHONHASHSEED=0.
Do NOT run on the original pilot worktree or modify its locks.

The registered script is tools/probe_ep_stability_D29.py.
It is not a candidate generator or evaluator. It performs two 30-epoch
old30 feature recoveries (D28 and D29), requires both historical Windows
feature identities to match, then tries exactly eight D29 SOC kernels with
stable_tail_moments=true, record_ep_trace=true. It **does not** redo the
already-sealed baseline failures.

Plan-only, no fits:
python tools/probe_ep_stability_D29.py plan --archive FULL_ORIGINAL_ZIP --output BRAND_NEW_OUTPUT_DIRECTORY

Execute only after plan and source/contract match:
python tools/probe_ep_stability_D29.py run --archive FULL_ORIGINAL_ZIP --output BRAND_NEW_OUTPUT_DIRECTORY

No --cross-platform-check on Windows. That flag is for a Linux CI-only
NON-AUTHORIZING plan preview. If D28 or D29 features do not reproduce exact
historical Windows identities, the tool stops before any GP attempt.

The probe writes plan.json, features.json, D29_soc_gp_features.npz (numeric,
no pickle), D29_soc_normalized_evidence.json, candidate_attempts.jsonl,
result.json, and a *new* diagnostic_execution.lock. Each attempt is written
once, with the precise exception/EP trajectory, finite-prediction checks
for successful models and elapsed time.

Maximum incremental work: two VAE refits and eight D29 SOC GP candidates.
New aircraft/math evaluator calls=0; BO or RCAIDE calls=0. No original lock
deletion, parameter tuning, retry, rescue fallback, or pilot restart.

## Interpretation and stop

If seven EP candidates still fail, examine final EP trace and site-specific
failures; the remedy likely requires a separately verified convergence
strategy, not more VAE data. If the eighth previously tail-failing
candidate now succeeds, that specifically supports the tail fix; it does
not establish complete EP stability or BO superiority. If the same
candidate still fails, record the new lowest-level cause.

**Do not promote this mode into the original five-arm pilot or full
algorithm before independent numerical accuracy tests and the fixed
Windows D29 GP-only result.** After this one bounded probe, stop and
return the outputs and SHA256 manifest. Any new optimization protocol is
a separate decision.
