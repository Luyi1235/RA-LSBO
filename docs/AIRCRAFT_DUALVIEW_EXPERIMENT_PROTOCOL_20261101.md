# RA-LSBO heterogeneous risk-view optimization: prospective aircraft study

## Context and scientific question

The original 2026-10-31 five-arm aircraft corpus pilot is **interrupted**, not a
complete study. Its 2,212 settled simulator calls, 65 completed trials,
one partial trial and all locks, logs and snapshots are immutable. A new
read-only matched-cohort audit confirms 13 complete five-arm (case,seed) sets:
eight 100km seeds (9931..9938) and five 140km seeds (9931..9935). These
descriptive histories suggest persistent corpus latent and onehot
representations are both competitive for energy, while the original
old30 uncertainty can be severely miscalibrated for SOC. None of these
adaptive, incomplete pilot results establish a winner.

The new **risk-decoupled representation** asks: should the response model for
energy and the response models for feasibility use the same learned feature
space? The default paper-style RA-LSBO shares a latent view for all outputs.
The candidate instead transfers learned structural similarity to the energy
response while retaining explicit hierarchical categories in constraint GPs.

This is a **new opt-in research algorithm** inspired by the prior study; do
not claim the general notion of heterogeneous GP representations is novel in
the wider BO literature. The novel contribution to this implementation is
the specific trust-aware, risk-decoupled hierarchical mixed-domain
acquisition and registered degradation behavior.

## Locked arm roster and exact ablations

All five arms share the exact same prospective **initial design labels per
(case,seed)**, global candidate RNG stream, acquisition phase schedule,
near-duplicate guard, integration-step contract (20s) and final-horizon
budget. No legacy labels are inserted into training.

| Arm | Objective GP view | Completion/constraint GP views | Additional rule |
|---|---|---|---|
| random | none | none | matched random control |
| onehot_stable | hierarchical onehot | hierarchical onehot | EP stable-tail |
| corpus_stable | 2,000-corpus persistent latent | same latent | EP stable-tail |
| hybrid_onehot | persistent corpus latent | hierarchical onehot | risk-head decoupling |
| hybrid_guard | persistent corpus latent | hierarchical onehot | also fit latent SOC and choose per-candidate less optimistic P(g<=0) |

For hybrid_guard, the SOC Gaussian moment pair is selected from the view
with **smaller** predicted feasibility probability per queried design. This
is *not* a statistically calibrated joint posterior and not a physical
safety guarantee; it is a deterministic, prefix-only conservative decision
rule. Its effect must be isolated against hybrid_onehot. No learned
ensemble weights, post-hoc arm selection, adaptive parameter tuning,
scenario tuning or performance claim.

The objective response remains an exact+interval mixed-likelihood
surrogate, with exact-only objective supervision in the corpus encoder.
The 2,000 structures are canonical uniform-with-replacement draws with
duplicates retained; training sessions at online indices 0,8,16 use the
unchanged corpus settings (30 epochs initially, 20 update epochs). Surrogate
GPs refit on all settled historical responses. All model arms use the same
grouped Matérn 4x2 grid, stable-tail interval moments, EP damping .7, max
sweeps 60, tolerance 1e-8, fixed exact/interval observation semantics.

### Numerical failure behavior

A candidate kernel failing is a recorded excluded candidate, **not** a
fabricated constraint or a crash. If all eight kernels fail for any required
response head (or the supported model prediction fails numerically), the
entire proposal uses deterministic least-visited-branch coverage from the
already exact-screened, near-novel global pool; the failed head and all
candidate errors are recorded in proposal metadata. This can change the
selection policy, so report fallback counts separately and do NOT call it
a normal GP-guided suggestion. Failure of the actual model evaluator,
infrastructure error, invalid evidence, corrupted code/environment, or
unsettled model call still stops the campaign after journaling. No silent
physical labels, retries or rewriting history.

## Prospective budgets and seeds

| Stage | Seeds | Shared initial / case / seed | Online / arm | Total new simplified calls | RCAIDE calls |
|---|---|---:|---:|---:|---:|
| smoke | 9960 | 6 | 2 | **42 maximum** | 0 |
| pilot | 9961,9962,9963,9964 | 12 | 18 | **856 maximum** | 0 |

Pilot budget calculation: 2 cases x 4 seeds x 12 shared initial =96,
plus 2 x 4 x 5 arms x 18 online =720,
plus up to 2 x 4 x 5 same-incumbent 5s step checks =40.
Unused budget is not reassigned. Smoke: 12+20+10=42. Combined maximum
of any completely new smoke + pilot is **898** calls. Smoke does not
qualify for the pilot merely by being attempted: it must be completed,
audited and numerically reviewed in the **original Windows/Rcaide_vsp**
environment with the same source commit and dependencies.

Fixed task order: aircraft_ordinary 100km, aircraft_sparse 140km; cyclic
arm order from seed and task, predetermined without performance information.
The 20s/5s nominal/refinement step comparison is purely a deterministic
numerical sensitivity check, not aircraft certification or robust flight
validation. Training/evaluation uses 20s only; 5s checks never feedback.

## Research endpoints

Primary exploratory comparison (case/seed paired): final **nominal feasible
energy Wh at exactly 18 online calls**, with feasible/not-feasible and
missing outcomes reported separately (do not impute a numeric penalty).
Primary contrasts, in this order:

1. hybrid_onehot vs corpus_stable: change constraint feature view.
2. hybrid_guard vs hybrid_onehot: enable disagreement-aware SOC conservatism.
3. hybrid_onehot vs onehot_stable: change objective feature view.

Secondary: first nominal feasible call, feasible-return rates, GP
failure/coverage counts and named heads, SOC prequential exact NLL and
interval log probability on each arm's **saved adaptively selected**
candidate, same-incumbent 5s verification, conditional objective differences,
number of VAE gradient updates, encoder/GP runtime and total wall time.

This is a **4-seed development pilot**, not an independently powered
confirmation. No single best arm promotion or physical-risk guarantee based
on these 4 seeds.

## Reproducibility and stop rules

The 2026-10-31 original pilot must never be resumed or have its locks
deleted. This study has its own new branch, commit, independent Windows
worktree, two new write-once directories (smoke and pilot), registered
source snapshot, manifest, SHA256 evidence map and complete call ledger.
Run full relevant regression and dependency check before preparing smoke.
Only run new pilot after same-source smoke and complete qualification receipt
are accepted by the standalone runner. Every generated artifact must be
audited before exploratory reporting. No PR is merged into main unless
subsequent evidence warrants it.

Entrypoint:
  python -m ra_lsbo.experiments.simplified_aircraft.dualview_pilot

commands: prepare, run, audit, report. Explicit --approved-call-limit equals
the FULL registered maximum (42 smoke or 856 pilot), not a soft allowance.
Do not initiate an unapproved 856-call local run; demonstrate source/CI/Windows
smoke and review budget first.

The original historical 2,212 calls remain fully charged in the overall
research ledger, separate from the 898 maximum new calls.
