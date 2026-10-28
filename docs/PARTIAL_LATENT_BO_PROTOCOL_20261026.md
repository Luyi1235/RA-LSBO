# RA-LSBO: partial observations in a structured latent space

Date: 2026-10-26. Status: prospective development protocol, not superiority evidence.
This protocol is frozen with the code before the first new problem evaluation.
The complete low-cost budget is 476; this runner cannot launch RCAIDE MDA.

## Scientific question and proposed contribution

Can a structured representation of hierarchical aircraft choices make mixed
numeric/inequality observations more useful for finding and improving a sparse
feasible design, without excluding unexplored architectures too early?

RA-LSBO combines the structured LSBO representation with an output-specific
mixed-likelihood posterior, prefix-only kernel evidence selection, and an
acquisition rule distinguishing response feasibility from analysis completion.
The proposed contribution is this coherent method for hierarchical partially
observed design and its empirical characterization. Component assembly and
software tests alone do not establish a publishable performance contribution.

## Primary literature

1. Wang, S.; Li, K. (2024). Constrained Bayesian Optimization under Partial
   Observations: Balanced Improvements and Provable Convergence. AAAI 38(14),
   15607-15615. DOI: 10.1609/aaai.v38i14.29488.
   https://ojs.aaai.org/index.php/AAAI/article/view/29488
   Numeric/sign likelihoods and balanced feasible improvement motivate the core.
   Our representation, objective intervals, kernel grid and support gates adapt
   these ideas; this is not an unchanged CBOB reproduction or theorem.
2. Rasmussen, C. E.; Williams, C. K. I. (2006). Gaussian Processes for Machine
   Learning, Chapters 3 and 5. https://gaussianprocess.org/gpml/chapters/RW3.pdf
   https://gaussianprocess.org/gpml/chapters/RW5.pdf
   EP site normalizers and approximate marginal likelihood motivate inference
   and kernel selection. The small grid does not integrate hyperparameters.
3. Tfaily, A.; Diouane, Y.; Bartoli, N.; Kokkolaras, M. (2024). Bayesian
   optimization with hidden constraints for aircraft design. Structural and
   Multidisciplinary Optimization 67, 123. DOI: 10.1007/s00158-024-03833-8.
   Published 2024-07-10. https://doi.org/10.1007/s00158-024-03833-8
   Examples involve landing gear sizing and aircraft performance; uncertainty
   protects exploration from an overconfident failure classifier. Their industrial
   simulation code is not public. Our electric aircraft is a new reduced case,
   not a reproduction of those industrial aircraft problems.
4. Priem, R.; Bartoli, N.; Diouane, Y.; Sgueglia, A. (2020). Upper trust bound
   feasibility criterion for mixed constrained Bayesian optimization with
   application to aircraft design. Aerospace Science and Technology 105, 105980.
   DOI: 10.1016/j.ast.2020.105980. https://arxiv.org/abs/2005.05067
   Our posterior-support rule adapts uncertainty-aware feasibility exploration;
   it is not the original UTB refinement algorithm or a safety certificate.
5. Doff-Sotta, M.; Cannon, M.; Bacic, M. (2020). Optimal energy management for
   hybrid electric aircraft. IFAC World Congress. https://arxiv.org/abs/2004.02582
   Point-mass mission/propulsion/battery coupling motivates the model structure.
   Their 100-seat hybrid aircraft parameters are not used.
6. Chin, J. C.; Schnulo, S. L.; Miller, T. B.; Prokopius, K.; Gray, J. (2019).
   Battery Performance Modeling on Maxwell X-57.
   https://openmdao.org/pubs/chin_battery_performance_x57_2019.pdf
   We reuse previously frozen workspace cell maps at fixed 20 deg C, using
   OCV and R0+RTh, balanced 128S40P and an energy-conserving discharge update.
   Transient RC/thermal dynamics are omitted; this is not the complete validated
   NASA equivalent-circuit model.

## Data and posterior

Retain each output independently: exact y, interval [a,b], one-sided sign, or
missing. Run status is separate from output validity. Numeric observations use
a Gaussian likelihood; interval likelihood is Phi((b-h)/s)-Phi((a-h)/s), with
hard limits for s=0. An objective bound requires physical justification, e.g.
cumulative nonnegative mission energy before termination. Missing outputs add
no likelihood. Solver failure never creates an invented positive constraint.

Inputs are [normalized active continuous values, mask, discrete embedding].
Refit the structured encoder on the current history prefix; all legal unlabeled
configurations may define its grammar. Future response labels and qualification
labels cannot train it. The one-hot arm replaces only the discrete embedding.

Each response/completion head selects length scale sqrt(input_dimension) times
{0.25,0.5,1,2}, maximizing current-prefix approximate EP marginal likelihood.
Condition exact values by Gaussian regression and assimilate interval evidence
by damped EP. Completion uses probit noise 1. Independent output heads are a
modeling assumption, not proven independence of aircraft failure mechanisms.

## Acquisition and audit

Apply exact geometry screening uniformly. On the remaining pool require
posterior completion support Phi(-mu_c+kappa*sigma_c)>=0.5 and optimistic response
support mu_j-kappa*sigma_j<=0. Kappa decreases from 2 to 1 over 14 calls. If no
completion support exists, use maximal posterior upper probability; if no joint
optimistic domain exists, minimize normalized positive optimistic violation.

Main score: log(EI)+log(p_completion)+sum(log(EICB_factor_j)); before an incumbent,
omit EI. Allocation floor((t+1)*0.2)-floor(t*0.2) reserves two of 14 calls for
joint feasibility entropy in the same support domain. This is a boundary
heuristic, not mutual information, a regret proof or calibrated safety guarantee.

Save pools, training-prefix digest, encoder identity, per-kernel evidence,
posteriors, support and fallback for every proposal. Every call has an append-only
started/settled pair; pending calls stop further evaluation. No automatic retry
or hidden budget reset. Freeze source snapshots and replay proposals without
new truth calls. Primitive tests use mock labels, not model oracle evaluations.

## Development cases

### A. Hierarchical analytic extension

Extend Branin with basic/augmented architecture, active module and conditional
trim. A declared elliptical feasible region with radii 0.16,0.2 in unit
coordinates and a resource constraint induce sparse feasibility. Outside the
ellipse return its violated sign and skip the objective. This is an artificial
feedback experiment, not a physical solver failure or an unchanged public
hierarchical benchmark. Future confirmation should include public benchmarks.

### B. Reduced electric fixed-wing aircraft

Wingtip/distributed propulsion activates count and diameter/spacing variables.
Wing area and aspect ratio are continuous. Ideal actuator-disk power, mass,
span and packing create architecture tradeoffs; there is no layout efficiency
bonus or artificial failure island. A parabolic polar and point-mass force
balance drive DC power through a fixed electrical efficiency.

Mission: 500 m climb, 100 km cruise at 50 m/s, descent, 600 s reserve. Constraints:
span, packing, stall speed, root bending, minimum SOC, current, voltage and
instantaneous battery power. Mass/polar/limits are research assumptions, not
validated X57/certification parameters. Minimize full mission DC energy in Wh;
aborted missions supply prefix lower bounds. Root loading is approximate;
thermal, wake-wing interaction, stability, flutter and detailed stress are omitted.

Geometry is an exact precheck. Known SOC/power thresholds provide certified
inequality evidence at controlled termination. Generic battery update domain
exceptions provide a model-domain failure label and already supported bounds,
not a fabricated physical margin. Numerical exceptions are recorded separately.

Both fixed anchors are checked at 20,10,5 s battery integration steps. Only the
20 s contract enters task training. Coarse/fine energy error must be <=1% and
terminal status/physical labels must agree. Partial anchor objectives fail this
convergence test conservatively. Finer checks never supply training labels.

## Fixed design and decision rules

Upper bound: 476 cheap full-model calls, zero RCAIDE calls. Per case:
2 anchors + 30 independent screened random qualification points +
3 paired seeds * (12 shared random initial + 4 methods * 14 online).
Aircraft additionally uses 4 finer-integration calls. Charge shared labels once;
charge duplicate online designs again. Anchors/panels are excluded from training.

Methods: full RA latent; same core with hierarchical one-hot; latent core omitting
interval evidence; unmodified frozen original LSBO through native proposal/update.
Preserve its search effort, missing-value handling and early-stop policy; unused
calls are not reassigned. This is a practical native baseline; publication still
needs matched-pool/surrogate controls to attribute gains precisely.

Proceed only with at least one usable feasible prespecified anchor; aircraft
also requires integration convergence. No mission/domain/algorithm retuning
after results. Failed qualification remains in the record, with no transfer of
unused calls. Report screened-random feasibility with Wilson 95% intervals and
unknown-physical counts; 30 points cannot precisely certify a sparse rate.

Endpoints: best observed feasible objective including initial calls; online
feasible count; calls to first online feasible result; model-domain/numerical
failures; partial objective count; wall time. Keep no-feasible outcomes explicit,
without invented penalty objectives. Three seeds are development evidence,
insufficient for strong significance or stable-superiority claims.

## Publication and RCAIDE progression

Independently freeze confirmation with more paired seeds, public hierarchical
benchmarks, prequential probability scoring/calibration, kernel/support/boundary
ablations, paired uncertainty intervals and architecture-stratified results.
Include compute time and failure discovery cost; retain failed development.

One real mainline: X57-derived electric fixed-wing design in RCAIDE, with physically
motivated range/reserve/loading/geometry requirements. Fix airframe/propulsion/
battery versions before independent random qualification. Record module-stage
outputs and termination evidence. Do not use reduced labels as RCAIDE truth.
Compare reduced/high-fidelity predictions on a small common panel before online
optimization, without choosing the task because a favored method wins.

An old compact-aircraft ledger contains an unsettled started call. Reconcile it
before real continuation; this runner neither resets it nor launches parallel
MDA. Original LSBO sources and old frozen studies remain unchanged.
