# RA-LSBO-H2 P0 code-readiness review — 2026-11-02

**Scope:** small opt-in stability and chronology hardening, proposed separately from Draft PR #21.
Base source: `7874e72da264be0c117193471db68b05286ea1e4`.
No main/legacy/H1 branch, old source snapshot, simulator equations, experiment seed, budget, or archived Release is modified.

## Review finding A — high-offset Gaussian-mixture variance

The old H2-only `component_mixture` computed `E[var + mean^2] - E[mean]^2`. In standard
float64, two GP heads with common mean `1e9` and variance `1e-6` produced **zero**
mixture variance, although the law of total variance requires `1e-6`.
A 2-element local NumPy reproduction also showed zero for slightly different large means.

The isolated fix uses the algebraically equivalent **centered** identity:

`Var(Y) = sum_j w_j [ Var(Y | view_j) + (mu_j - mu_mix)^2 ]`.

This reduces catastrophic cancellation without changing the Gaussian component CDF used
by the risk acquisition or the deliberately documented moment approximation used by
optimistic-support margins. It is *numerical robustness*, not an improvement in GP
calibration or proof of BO benefit. Regression exercises both large-offset and usual-scale
two-view data; inputs and observation labels remain identical.

## Review finding B — registered smoke does not exercise adaptive weights

H2-M has `warmup_scored=4`, but the registered 48-call Windows smoke has only
`horizon=2` online proposals per arm. That smoke can validate the evaluation
pipeline, source hashes, first two prior commitments, GP fit, and ledger audit,
**but cannot demonstrate a learned H2 weight in the actual airplane engine**.
Do **not** modify the locked smoke roster or increase calls to conceal this.

A new synthetic-only P0 integration regression executes the real
`CausalH2CorpusBO.propose/accept` adapter on five chronologically settled
`RunObservation` fixtures. Only the upstream GP/VAE proposer is mocked:
all four early proposal weights must equal the 0.5 prior, the fifth proposal
must use the weight updated from four earlier outcomes, and
`replay_h2_controller` must reconstruct the entire five-event ledger.
Tampering the fifth-step weight must fail the independent replay.
This does not replace a Windows full GP/VAE smoke.

## Next gates and evidence status

1. Run the new isolated Linux CI, along with H2 numerical and frozen EP/dual-view
   tests, on the exact proposed commit. Record run IDs and any failed attempts.
2. If green, review the branch difference against base and retain as Draft until
   the maintainer decides whether to adopt this H2-only fix.
3. Re-pin the full proposed H2 source SHA *after* any code merge. A prior H2 or
   Linux CI result cannot qualify a modified source for Windows execution.
4. In the original Windows/Rcaide_vsp environment, run source- and environment-
   bound regressions first. Only an independently approved **48-call** smoke may
   then run. A fully audited same-source smoke and a separate authorization are
   prerequisites for the **1,008-call** four-seed development pilot.
5. Only after independent new-seed on-policy data should risk-view adaptation,
   cost/performance and feasibility be assessed against fixed controls.
   No promotional/airworthiness/confirmatory claim follows from these tests.

All historical H1 856-call, original interrupted 2,212-call, and 384-GP SOC panel
evidence must stay immutable. The new CI explicitly performs **zero aircraft model
evaluations and zero RCAIDE MDA**. A synthetic `RunObservation` is not a physical
model evaluation.
