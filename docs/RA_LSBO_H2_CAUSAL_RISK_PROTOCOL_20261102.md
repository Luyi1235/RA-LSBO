# RA-LSBO-H2 — prefix-causal, censor-aware risk-view control

**Scope:** opt-in research prototype only. Frozen H1 Windows smoke, H1 856-call pilot, 384-GP SOC cross-policy panel, and interrupted 2212-call original corpus pilot remain completely unchanged. This protocol is written after inspecting those completed development experiments, hence it is **not confirmatory**.

## Research question

H1's 140km experiments favored the onehot-risk/latent-objective architecture on average, whereas adding a worst-view per-candidate SOC feasibility guard harmed the optimization outcome on four paired seeds. A separate 24-prefix/120-target matched GP panel found task- and censor-dependent SOC reliability, including a highly influential 140km censored point. A fixed risk view and a pointwise minimum probability are therefore both imperfect.

H2 tests an alternative: **an online SOC risk controller whose next model-weight decision is a function only of *previously committed, subsequently settled* paired predictions**, with no access to the next observation label, censoring type, or evaluator result.

This is a useful new *implementation-level research hypothesis* for RA-LSBO; it is not a claim that expert aggregation, hedge, likelihood ratio weighting, or adaptive mixtures are new in the general BO literature.

## Two distinct preregistered controller mechanisms

The current isolated branch contains complementary **research variants**, intentionally not silently conflated.

### H2-M: bounded paired proper-log-score Gaussian mixture (GP-integrated candidate)

Sources:
- ra_lsbo/experiments/simplified_aircraft/h2_risk_controller.py
- ra_lsbo/experiments/simplified_aircraft/h2_optimizer.py

Controller settings fixed:
- Prior onehot component weight 0.50, minimum 0.10, maximum 0.90;
- Four past *scored* observations before weight adaptation starts;
- Learning rate eta=0.35, clipped per-settlement paired log-score difference ±4.0;
- Weight update based only on past prequential latent NLL minus onehot NLL;
- For exact SOC use the continuous Gaussian **density** NLL;
- For censored SOC use the appropriate Gaussian **interval probability mass** NLL;
- Within each settlement, both models are scored against the **same evidence of the same measure**, so the paired log-score difference is a log-likelihood ratio; scores of different measurement types are never reported as a combined proper NLL metric. A bounded log-ratio controller is a heuristic, not a theorem of risk calibration.

For prediction at the next BO decision, two Gaussian SOC components retain the current causal onehot weight w. The existing RA-LSBO acquisition uses the **componentwise Gaussian mixture CDF** for SOC feasibility. Its optimistic support margin is based on the **mixture's moment-matched mean/variance**, explicitly only an approximation to Gaussian-mixture geometry. No per-candidate worst-P switching occurs. Completion/other risk heads still use onehot; energy objective uses the same frozen persistent corpus latent representation as H1. Stable-tail EP and fixed 4x2 kernel grid are inherited. When the model is skipped due to scheduled coverage or all-head failure, no paired log score is updated.

### H2-B: resolved-feasibility Brier head-gating ablation

Source:
- ra_lsbo/experiments/simplified_aircraft/h2_controller.py

Fixed settings:
- Prior onehot preference 0.75;
- Minimum three resolved, paired, previously settled feasibility labels;
- Bounded Brier loss in [0,1] for any exact or sign-resolvable censored SOC observation;
- Learning rate 1.0 and log-odds cap 4.0;
- One GP head selected for the **whole iteration**, avoiding candidate-specific arbitrary switches.

A censored interval whose SOC sign is ambiguous and a missing/untrusted SOC outcome produce no Brier update. Brier has one consistent Bernoulli event across exact and resolvable censored observations, but ignores the extra numeric magnitude of exact SOC; that is the methodological trade-off. Exact and censored proper NLLs are tracked in separate diagnostics and are not used to change H2-B weights.

H2-B is currently a self-contained controller and synthetic search mechanism, *not a registered end-to-end aircraft GP optimizer*. Never claim it is a validated interchangeable H2-M flight arm.

## Correct temporal order

For every online step t:

1. Read only the current, already-settled training prefix and current H2 controller state based on steps < t.
2. Fit the registered GP heads using the current prefix only (or skip GP under the unmodified coverage policy).
3. At the chosen design x_t, **commit two frozen SOC Gaussian forecasts** and the current view weight/head before running the evaluator; bind by candidate ID and prefix digest.
4. Evaluate x_t. The actual SOC may be exact, one-sided censored, unresolved, or missing. Keep its evidence type without inventing a label.
5. Only after the physical record is settled and tied to x_t, compute paired Brier or clipped log-likelihood ratio and update the controller.
6. Preserve current-step forecast, chosen view/weight, observation type, update amount, and next state for auditing.

Failure cases include mismatched candidate/prefix, duplicate settlement, future label access in selection, missing output evidence and an unfit GP head. They must not silently change a registered experiment's budget or emit invented GP feasibility.

### Fixed synthetic benchmarks (no aircraft, RCAIDE, VAE or GP data generation)

**A. Hard-head H2-B policy-search emulator:**
- Three synthetic hierarchical scenarios: smooth transfer, discrete category change, delayed regime shift.
- Eight fixed seeds 7301–7308 per scenario; four strategies: static onehot, static latent, old pessimistic minimum, H2-B.
- Twenty-four online mathematical evaluations per strategy; total 96 synthetic policy trajectories and **2304 synthetic oracle calls**.
- Nominal variables include material family, ordinal level, optional conditionally active z and continuous x. Positive SOC violations are one-sided censored; every 11th step is missing; fixed synthetic GP-head outages occur at steps 5/14. H2 uses only previous resolved actual evidence.

**B. H2-M predictive-density mixture emulator:**
- Four synthetic scenarios: onehot-valid, latent-valid, temporal shift, conditional hierarchy.
- Eight fixed seeds 16001–16008, 96 chronological observations per case/seed; total **3072** synthetic observations (2031 exact, 1041 censored).
- Same deterministic candidate/true-outcome sequence across fixed-onehot, fixed-latent, worst-view guard and H2-M. Forecasts are analytic *surrogate emulators*, not fitted GPs.
- Compare exact-density NLL and interval-mass NLL **separately**, plus resolved feasibility Brier, trajectories and cap/weight adaptation.

These two studies are development-only *mechanism checks*. Different scenarios, policies, horizons and sampled trajectories mean neither one is an independent confirmation and their objective energies cannot be merged or used for p-values. All synthetic evaluation charges are disclosed separately; new aircraft/RCAIDE evaluation calls = zero.

**C. GP integration gate (mathematical data only):**
- Unit tests build a synthetic hierarchical observation dataset with exact and censored SOC, fit existing grouped 4x2 stable-tail EP GP heads under the opt-in H2-M component mixture, and verify the existing RA-LSBO acquisition receives the exact mixture probability CDF rather than a moment-matched Gaussian CDF.
- Tests cover prior-forecast commitment, post-settlement-only controller update, deterministic coverage without fabricated predictions, and immutable source regression. No true aircraft model evaluator runs.

## Promotion and next-stage safeguards

Passing these checks authorizes **no automatic new airplane pilot**. In particular:
- Numerical EP convergence is not a guarantee of predictive calibration.
- Synthetic expert emulators are not evidence of real GP predictive superiority.
- H2-M is a mixture model, H2-B a hard-head selector, and H1's hybrid_guard a per-candidate worst-P rule; these have different uncertainty semantics and costs.
- Next confirmatory or exploratory airplane evaluation requires **new seeds**, a new write-once registration, a preregistered comparison roster and explicit maximum true-model charges, with original 856 and 2212 historical calls left unchanged.
- If a Windows mathematical/GP-only smoke is useful, it must be separately registered and bounded; it must not be described as a flight evaluation.

All work remains on an isolated Draft PR; do not merge or change the frozen H1 implementation.
