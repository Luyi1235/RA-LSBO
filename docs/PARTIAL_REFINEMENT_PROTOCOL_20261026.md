# RA-LSBO grouped posterior and local continuous refinement

Date: 2026-10-26. Prospective development protocol; freeze before any new calls.
This is a distinct follow-up to the completed 491-call development study, prompted
by its diagnosed search-resolution and kernel-scale limitations. The complete
new plan has at most 430 cheap evaluations, zero RCAIDE MDA, no parallel physical
workers and no transfers from or resets of previous budgets.

## Retained LSBO mechanisms and literature basis

Retain the structured encoder and useful LSBO engineering: elite-centered branch
bounds, only active continuous coordinates, multistart continuous optimization,
adaptive branch radii and final rescoring in a shared candidate union. Local
optimization operates on the new mixed posterior, not on objective/constraint
pseudo-values. Old LSBO and all previous source bindings remain unchanged.

Eriksson and Poloczek (2021), Scalable Constrained Bayesian Optimization, AISTATS,
PMLR 130:730-738, https://proceedings.mlr.press/v130/eriksson21a.html supports the
use of local constrained searches. Its trust region, posterior sampling, restart
algorithm and convergence result are not reproduced here. Our branch radius
updates instead retain the frozen LSBO engineering convention: initial .2,
improvement times 1.1 up to .5, three failed improvements times .7 down to .025.

Rasmussen and Williams (2006), GPML Chapters 3 and 5,
https://gaussianprocess.org/gpml/chapters/RW3.pdf and
https://gaussianprocess.org/gpml/chapters/RW5.pdf motivate the likelihood/EP and
evidence learning. Wang and Li (2024), CBOB, AAAI 38(14):15607-15615,
https://ojs.aaai.org/index.php/AAAI/article/view/29488 motivate mixed exact/sign
observations and balanced feasibility. This method retains its own documented
acquisition adaptations and does not inherit either paper's convergence proof.

## Grouped mixed-observation model

Keep features x=[u*m, m, z] with the prefix-fitted encoder, or hierarchical
one-hot in place of z. For p continuous variables and q discrete feature
coordinates, the kernel is Matern-5/2 with squared distance

    ||du_masked||^2/(p*l_cont^2) + ||dm||^2/p + ||dz||^2/(q*l_disc^2).

This is a positive definite kernel implemented by explicit diagonal input
scaling and the unchanged mixed GP. Use l_cont in {.05,.15,.4,1} and l_disc
in {.5,1.5}, selecting one of eight pairs separately per output by current-prefix
EP evidence. This separates physical-coordinate resolution from representation
dimension, without claiming full coordinate-wise ARD or hyperparameter inference.

Likelihoods, output normalization, numeric noise, interval/probit observation
semantics and missing-output handling remain the frozen implementation. The
legacy-kernel control uses its four old scalar lengths, including its dimension
factor; both kernel arms use identical continuous-search settings.

## Acquisition optimization

Each round first fits a single sealed posterior and generates 192 exact-screened
global candidates with no truth queries. Use the existing balanced-improvement /
feasible-discovery / budgeted-boundary rule. Preserve completion uncertainty and
optimistic response support, including the documented empty-support restoration.

Rank branches by eligible global candidates first. Select at most four branches,
add 12 local random points per branch, and solve at most two SLSQP subproblems
per branch with at most 25 iterations each. For a branch with a feasible elite,
bounds intersect its adaptive unit-coordinate neighborhood with [0,1]. For a
branch without an elite, use the whole continuous box for discovery. Only active
coordinates move; inactive fields remain canonical.

The local objective is batch-independent: log balanced improvement, discovery or
boundary entropy, with completion/optimistic support and exact geometry as
SLSQP inequalities. If the reference pool has no support, minimize normalized
optimistic violation with exact geometry constraints. The round's local mode is
fixed from the reference pool. After adding returned endpoints and the best
valid cached visits, apply the existing acquisition rule once to the full union.
This final union can change the zero-EI or empty-support fallback. A local
solver's success is diagnostic, not a guarantee of global acquisition optimality.

Never invoke problem.evaluate while fitting or searching. Exact geometry checks
are cheap and common. Charge only selected truth calls, while reporting posterior
query counts and wall time. Protect pending proposals, sequential iteration,
feedback identity, training-prefix mutation and encoder state. Save initial and
augmented pools, heads, forecasts, branch state, bounds, local solver reports
and the selected source. Refit/replay without new truth calls in the audit.

## Tasks and independent qualification

Task A reuses the hierarchical Branin extension and its three 12-point initial
prefixes from the completed study. New random seeds are 8201/8202/8203 mapped to
old initial sets 8101/8102/8103. These are development data, not independent
confirmation. Source hashes bind every reused label. Old random panel labels
are used only for common-query prediction scoring, never training.

Task B fixes the same reduced electric-aircraft physics at 140 km cruise. This
is selected from the previous optimizer-free range sensitivity: its two random
feasible designs become inherited qualification anchors. The fact that the
original engineering anchors failed does not make this task empty. These new
anchors remain excluded from optimizer training. The choice uses feasibility,
not which optimization method performed best; no optimizer has been run at this
mission setting in the previous study.

Before online aircraft optimization: evaluate 30 newly fixed independent
exact-screened random designs and both inherited anchors at 10/5 s time steps
(their existing 20 s results are reused). Proceed only if both anchors are
usable feasible, random feasible count <=6/30, physical-unknown count zero,
and the integration check passes. This is a development gate based on a point
estimate, not a certified population sparse rate. Report Wilson intervals.

Integration gate compares 20/10/5 s terminal status and every constraint label;
20/5 s full mission energy relative error <=.01; SOC/current/voltage margin
absolute errors <=.001; power margin <=.01. Stage power is fixed, so energy alone
cannot establish convergence. The reduced model remains unvalidated against
RCAIDE; no thermal, wake-wing, stability or detailed structural claim is made.

If the aircraft gate fails, retain all results and skip its online comparisons,
without changing mission parameters or transferring calls to another task.

## Fixed comparisons, cost and interpretation

Five arms, three paired seeds and twelve online calls per task:

1. Structured latent + grouped kernel + branch-local search (full new method).
2. Hierarchical one-hot + grouped kernel + identical branch-local search.
3. Structured latent + grouped kernel + global pool only (local-search ablation).
4. Structured latent + old scalar kernel + identical branch-local search.
5. Unmodified frozen LSBO native proposal/update and native early-stop policy.

New complete-plan budget: 2*3*5*12 online + 3*12 aircraft initial + 30 aircraft
qualification + 4 integration = 430. Reuse 36 historical Branin initial labels
and two aircraft anchor labels, not newly charged. Qualification panels and
anchors never train the optimization models. Duplicates between online methods
are charged again. Native LSBO unused calls stay unused.

The same local effort caps apply to arms 1/2/4, though solver convergence can
produce different actual query counts; save these and wall time. Native LSBO's
search still differs, so 1/2/4 are the primary matched implementation controls.
Arm 3 deliberately has less search effort: it tests the practical value/cost of
local refinement, not an equal-CPU theoretical comparison.

Before online calls for each seed, forecast the same 30-point independent panel
from the shared initial prefix using grouped latent, grouped one-hot and legacy
latent models. Save response/completion posteriors and joint probability proxy
Brier. No calibration or kernel retuning on panel outcomes. The panel is for
diagnosis, not a hidden method selection loop. On-policy forecasts alone cannot
demonstrate common-distribution predictive superiority.

Endpoints: observed feasible best including initial calls; no-feasible retained
explicitly; online feasibility count; first online feasible cost; partial output,
model-domain/numerical failure counts; local contribution, surrogate queries and
wall time. Compare paired endpoints including no-feasible outcomes, never just
the best seed or conditional means over successes. Three seeds support development
diagnosis only. Physical aborts are not numerical solver failures.

No physical RCAIDE job is launched. The single-aircraft final MDA progression
and the old unsettled-event reconciliation remain separate prerequisites.
