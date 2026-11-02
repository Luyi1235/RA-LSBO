# RA-LSBO-H2 P3: fixed-mixture causal ablation protocol (prospective, zero data evaluated)

**Registration date:** 2026-11-02. **Status:** proposal; no simulator execution or source promotion.
**Implementation base:** `7ca420613ccb50e138dc3c9d4292ac8e863c4ecf`.
**Historical evidence:** independent P2 raw Release audit, PR #26 / run 41442847625; read-only.

## Scientific hypothesis and important confound

The previous H2 P2 development panel (seeds 11062–11065) showed H2-M beating
`hybrid_onehot` on final feasible energy by 4/4 paired seeds at 140 km
(mean left-minus-right -1402.28 Wh), but H2 changes *two* things:
(1) uses both one-hot and latent SOC posteriors through an exact two-Gaussian
mixture, and (2) adjusts the mixture weight from settled past likelihood scores.
That six-arm P2 design had no frozen 0.5/0.5 mixture comparator.
**Its success does not identify the incremental adaptive-weight effect.**
In particular, two cold-start feasible discoveries can happen before the
4-scored-observation warmup, so they cannot be attributed to online adaptation.

This P3 compares **only** the additional effect of the causal weight adaptation.
No confirmatory, calibration, physical validation or airworthiness claim follows.

## Exactly two preregistered arms

- `h2_dynamic`: current audited H2-M, including full 2k synthetic structure corpus,
  latent objective, one-hot completion and non-SOC risk heads, stable-tail SOC
  one-hot and latent Gaussian components, exact mixture CDF for feasibility,
  moment-approximate optimistic margin, selected GP kernel grid and all original
  model-failure coverage rules. Controller prior one-hot 0.5, warmup 4 scored
  outcomes, eta 0.35, clipped difference ±4, bounds [0.1,0.9].
- `h2_fixed50`: **the identical algorithm and two GP SOC heads**, same fixed
  training and search schedule, selected forecasts and exact mixture CDF,
  but **effective one-hot weight always 0.5** at proposals. It must still
  save component forecasts, score settled exact/interval responses
  prequentially, and produce auditable commitment and settlement evidence.
  No surrogate-specific tuning, altered budgets or different GP parameters.
  Prefer an explicit opt-in frozen-weight controller setting with a tested
  replay contract over abusing an unrealistically large warmup.

The only intended algorithmic difference is whether prior settled SOC scores
alter the mixture weight used for subsequent proposals. Candidate trajectories
may diverge after the first adaptive update. Do not make an ungrounded claim
that the two policies share all future posterior training prefixes: they
share **initial data, same deterministic pools, same seed and fitting rules**,
and then operate on their respective genuine selected observations.
Both arms must receive exactly the same number of online calls.

## Fixed prospective designs and maximum calls

| Stage | Seeds | Cases | Shared initial/case/seed | Online/arm/case/seed | 5s checks max | Total truth cap |
|---|---|---|---:|---:|---:|---:|
| `smoke` | 12070 | ordinary 100km; sparse 140km | 6 | 2 | 4 | **24** |
| `pilot` | 12071–12078 (8 seeds) | same two cases | 12 | 18 | 32 | **800** |

Smoke calculation: 12 shared initial + 8 online + up to 4 numerical checks = 24.
Pilot calculation: 192 shared initial + 576 online + up to 32 checks = 800.
All tasks use the unchanged simplified-aircraft equations and physical
contracts; **RCAIDE MDA = 0**. The online pool is 32 for smoke and 128 for
pilot; encoder initial epochs 30, periodic updates every 8 iterations,
and original native GP fitting grid/steps (smoke 2, pilot 15).
Use the fixed original task settings and same-incumbent held-out 5s checks;
no weather/thermal scenario extension, early stopping, seed reroll or
post-hoc per-task method selection.

## Outcome and scientific analysis

Primary **task-stratified, seed-paired** outcome:
1. all-trial nominal feasibility and known-infeasible/missing status, retaining
   failures without replacement or invented infinite/zero objective;
2. final nominal feasible energy difference **conditional on both arms feasible**,
   clearly stating the number of such pairs, mean, median and individual deltas.
Report feasibility and energy jointly; a smaller conditional mean cannot erase
a loss in discovery rate.

Prespecified contrasts: `h2_dynamic - h2_fixed50` in ordinary and sparse tasks
**separately**, including warm/cold initial status, first feasible online call,
per-step incumbent energy trajectories, 5s same-design status and actual call
counts. Secondary: exact SOC NLL vs interval SOC probability-mass NLL (never
pooled without type), resolved-label Brier, recorded causal weight trajectories,
GP numerical failures and deterministic coverage, VAE/GP training/optimization
time and per-trial overhead. Compare adaptive and fixed mixture *at matched
outcome types or common prefixes only* for predictive reliability; policies
are on-policy and adaptive, not i.i.d. samples.

Eight seeds are still a **developmental mechanism ablation**. Pre-registration
does not transform them into independent publication confirmation.

## Mandatory preregistration gates and immutable provenance

1. Implement P3 in a **new** code branch only; do not edit frozen H2 P0/P1/P2
   sources, historical Releases, locks or records, and do not update `main`.
2. Implement an opt-in method registry and independent `prepare/run/audit/report`
   support with content-addressed snapshot/ledger and explicit startup caps.
   Freeze the source commit before preparing any physical experiment.
3. Add zero-evaluator tests: fixed weight remains 0.5 despite opposite scores;
   dynamic changes after fourth scored event; prequential selected component
   density/interval scoring; commitment and prefix ordering; no future leakage;
   fallback registration; independent replay and tamper detection;
   24/800 caps and exact seed roster. Existing 44 P0 and other H2/EP/H1
   regressions must continue passing.
4. Linux CI and Windows Rcaide_vsp source/environment regressions first.
   Qualify the original 316-call reviewed evidence **read-only**. The original
   P1/P2 Release results may inform the design but must not be used to choose
   seeds/methods post hoc or injected into prospective training.
5. Only after code-review and fixed SHA: run a new **24-call** smoke in a
   clean worktree with a new write-once directory. Independently seal/audit it
   and check all its observed 5s numerical verifications. A smoke with only
   two online evaluations cannot establish adaptive effectiveness.
6. Only after independently auditing that smoke: prepare the separate
   8-seed **800-call** pilot bound to the same fixed source/environment.
   Fail closed on unknown/numerical/infrastructure outcomes; no retry/resume,
   no reopening a consumed result directory. Do not automatically promote
   a method, start MDA or run a confirmation campaign.

No physical model calls are authorized by **drafting or implementing this
protocol alone**. The coordinating research agent will review the code-level
CI and new smoke qualifications before issuing next local execution steps.

## Implementation and prerequisite binding

The independent entry point is `ra_lsbo.experiments.simplified_aircraft.h2_p3_pilot`.
Both profiles require the complete original 316-call qualification directory
through `--qualification`; the pilot additionally requires a successfully
audited, numerically qualified P3 smoke through `--smoke`. Smoke and pilot
each require their own new write-once output directory. `prepare` seals source,
runtime, physical contracts, the qualification receipt and the applicable
budget; `run` additionally requires the exact approved call limit. The source
manifest uses repository-relative paths, so the same fixed commit and runtime
can be checked from another clean worktree without changing historical files.

`h2_fixed50` still registers each forecast and scores each settled SOC outcome;
only the effective one-hot weight remains 0.5. P3 audit independently replays
each arm's committed predictions and subsequent outcomes. The mathematical
runner tests use synthetic observations and explicitly prohibit aircraft
evaluators; they do not qualify or replace a real future smoke.
