# Posterior uncertainty and compact-wing endurance MDA

Date: 2026-10-21. Development protocol; not a confirmatory efficacy claim.

## Scientific question

The previous X57 study measured 59% feasibility after analytical screening.
Its optimization histories already contained a feasible incumbent. It cannot
establish the value of risk learning for discovering a small feasible set.
Native LSBO's prior development on X57 is a possible source of task familiarity,
not an established causal explanation for all negative results. The controlled
sparse task also exposed limitations of the revised acquisition.

The new hypothesis is that a completion posterior, combined with continuous
constraint posterior samples and explicit feasibility restoration, uses failed
evaluations more efficiently than a terminal-class entropy bonus. Structured
latent representations and observable failure mechanisms remain research
questions, not assumed improvements.

## Method decisions

1. Fit a Bernoulli/probit variational GP to observed completion labels only.
   Infrastructure errors and unobserved completion labels are excluded. Fit
   mechanism-specific diagnostic GPs only when both labels are observed.
   Unsupported stages remain unsupported; they are not multiplied into a
   fictitious five-stage survival probability.
2. Use the existing structured encoder and objective/constraint response GPs.
   Analytical constraints are checked exactly, and are excluded from learned
   joint feasibility. Sample each remaining constraint's latent posterior
   jointly across the candidate pool. Different heads remain independent:
   this is a modeling assumption, not an identified physical dependence model.
3. Sample the completion latent GP jointly across candidates, then a probit
   completion outcome. Select the best objective sample among sampled feasible
   candidates. If none are sampled feasible, minimize summed positive
   constraint violations in training-standard-deviation units plus a
   dimensionless completion-failure penalty. Ties are broken by objective
   samples only. Do not normalize EI by the best EI in the candidate pool.
4. Candidate pools include local proposals and global coverage. Before the
   first feasible observation, center local proposals on the completed design
   with least observed normalized violation. Failed hidden evaluations have
   no invented constraint values.
5. Log uncalibrated posterior predictions and train-only model identities.
   Sparse new-task pilot data are insufficient for calibration. Do not reuse
   a temperature learned at 40 nmi for another mission. Calibration and risk
   budgets remain separate: every physical call is charged one.

This is an SCBO-inspired mixed-variable extension, not the reference SCBO
implementation. Combining known methods is not itself evidence of novelty.
The candidate contribution to test is censored completion learning in a
structured representation with discovery-to-improvement optimization.

## New real MDA task

Use the unchanged isolated X57 MDA backend, its geometry/propulsion/battery
models, reserve segments, and existing electrical/structural/stall limits.
The new scenario is compact-wing endurance design: wing area 9--17 m2,
all original conditional discrete choices and other continuous coordinates.
The backend already supports this domain and audits realized wing area.
The 9 m2 bound is conceptual sensitivity, not a NASA-validated aircraft bound.
Actual cruise distance is changed through the existing versioned mission
configuration. No constraint threshold is adjusted.

Predeclare distances 40 nmi (control), 55 and 65 nmi (candidate challenges).
An engineering pilot evaluates two explicitly selected anchors at each distance:
the best feasible old physical design and a compact derivative at 9.5 m2.
These six calls are matched engineering checks, not independent draws and
not evidence of feasible-region prevalence. Old labels may identify the
anchor but must never become new-mission response training labels.

After the pilot, qualification uses 30 independent conditional-uniform designs
at each challenge distance, with a new random seed; no optimizer is run during
task selection. Choose the shorter distance with 1--6 feasible results out
of 30, provided every outcome is resolved. Zero feasible is not proof of
emptiness; no eligible challenge means the proposed task is rejected.
An independent 60-design conditional-uniform confirmation must contain at
least one feasible point and have a 95% Wilson upper bound <= 0.20.
This operational definition of sparse feasibility is conditional on passing
the exact analytical precheck. Also report the unconditional precheck rate.
Do not report either challenge as sparse before this gate passes.

## Entire new research budget

This is a distinct physical task/uncertainty study, not an extension of the
completed v1--v5 cumulative 500-call study. Standing user authorization permits
complete plans of at most 500 evaluations. Freeze this entire new study at
300 real calls, counted cumulatively across phases, errors, and failed designs:

- Six matched engineering pilot calls.
- Sixty task qualification calls (30 per challenge).
- Sixty independent confirmation calls for the selected task.
- Twelve common new-task initial calls.
- Three methods x three seeds x 18 online calls = 162 calls.

Total: 300. Single physical worker, no retries. Unknown or infrastructure
outcome stops the phase. No automatic progression past a failed scientific
gate. Unused calls do not justify retuning the task against algorithm rankings.
Pilot completion does not mean the full study has been executed.

The physical comparison methods are posterior risk-aware selection, matched
structured local constrained Thompson sampling without the completion head,
and unchanged original LSBO proposals. Report discovery probability, calls
to first feasibility, total failures, best feasible objective, and anytime
curves. Do not select a feasible starting subset retrospectively. The common
12 initial outcomes are visible to every method, including when none is feasible.
The small number of seeds limits statistical conclusions.

## Primary literature checked

- Eriksson and Poloczek (2021), Scalable Constrained Bayesian Optimization,
  AISTATS, PMLR 130, 730--738.
  https://proceedings.mlr.press/v130/eriksson21a.html
  Use constrained posterior sampling and feasibility restoration; do not
  label the mixed-variable implementation as their unmodified algorithm.
- Hensman, Matthews and Ghahramani (2015), Scalable Variational Gaussian
  Process Classification, AISTATS, PMLR 38, 351--360.
  https://proceedings.mlr.press/v38/hensman15.html
  Use an established GPyTorch variational Bernoulli GP implementation.
  Variational variance is approximate epistemic uncertainty, not coverage.
- Osband, Aslanides and Cassirer (2018), Randomized Prior Functions for
  Deep Reinforcement Learning, arXiv:1806.03335.
  https://arxiv.org/abs/1806.03335
  Supports investigating missing prior uncertainty in bootstrap ensembles.
  Randomized prior networks are not implemented in this revision.

The links and titles above were fetched from the primary publication pages.
An initially guessed unrelated arXiv identifier was discarded after checking
the title; it is not a source for this method.
