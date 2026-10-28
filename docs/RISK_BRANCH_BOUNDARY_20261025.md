# Boundary ensemble branch, 2026-10-25

## Scope and evidence contract

This branch owns only `ra_lsbo/risk_core/boundary_ensemble.py`,
`tests/test_risk_core_boundary_ensemble.py`, and this document. Import directly
from the new module; no package exports or frozen models are modified.
No truth labels, MDA calls, installations, fabricated failures, or inserted
feasible starts are used. The main runner owns discovery, the objective GP,
and numeric constraint GPs.

`BoundaryEnsemble(x, labels, seed=0, settings=None)` fits one named binary
boundary. `x` is a finite real `(n, d)` matrix with `d > 0`; NumPy and CPU/GPU
PyTorch inputs are copied to CPU float64. All rows must have finite features,
even masked rows. Labels are scalar numeric 0/1 (including Python booleans),
or `None`. A label of 1 means the head's event is satisfied. NaN is rejected,
not interpreted as missing. Tensor labels and NumPy numeric scalar labels work.
Unknown labels are removed before training; their features never influence the
model, normalization, initialization, loss, or class counts.

`constraint_labels(outputs)` converts explicit `OutputObservation` objects
using the existing `constraint_satisfaction()` contract:

| Evidence | Label |
| --- | --- |
| Exact finite `g <= 0`, including zero | 1 |
| Exact finite `g > 0` | 0 |
| Interval entirely at/below zero (`upper <= 0`) | 1 |
| Interval entirely above zero, including open `lower = 0` | 0 |
| Crossing interval, closed lower zero without upper proof, missing output | None |

The caller must first filter run status and validate evidence provenance
consistently with `observations.output_rows`. Do not feed unvalidated
infrastructure/unknown runs into output classifiers. A known completion
failure can supply label 0 to a *completion* head through explicit
`completion_label()` evidence; it does not supply label 0 to every physical
constraint. Missing objective values do not become objective targets.
Do not infer constraint labels from aggregate feasibility, simulator status,
held-out outcomes, or a planted benchmark boundary.

## Primary sources and adaptation

Tian et al., *Boundary Exploration for Bayesian Optimization With Unknown
Physical Constraints*, ICML 2024, PMLR 235:48295-48320 (paper identifier
`pmlr-v235-tian24g`, arXiv `2402.07692v2`, sections 4.1-4.2 and appendix A.1).
The paper proposes learned boundary exploration with deep ensembles.
The checked-out author reference is pinned in the source manifest to commit
`76c12b049c8748d0616039608323d78ab0c51f43`:

```text
runs/risk_redesign_20261025/primary_sources/repositories/becbo/
    algorithms/becbo/becbo.py
    model/classifier/de.py
```

`becbo.py` maximizes objective EI under an upper feasibility-band threshold
of 0.5. Its multiple-head implementation multiplies upper probabilities;
this branch deliberately uses the requested *per-head* threshold instead.
The reference asserts that an observed feasible incumbent exists. Its
boundary-seeking initialization, acquisition optimization, and random
fallback after initialization search failure are not reproduced by this
classifier. Its acquisition-optimization exception handler re-raises errors.

The checked-out `de.py` combines neural outputs into a latent Gaussian,
uses a softplus-transformed variance, `ApproximateGP`, a variational strategy,
`VariationalELBO`, `BernoulliLikelihood`, and latent mean +/- standard deviation
before likelihood transformation. It trains up to 1,000 iterations. This
branch instead trains independent networks with regularized probit Bernoulli
likelihood and uses their probability spread directly. There is no
variational posterior, KL term, softplus variance floor, or integrated
Gaussian probability. This is **BE-CBO-inspired, not reference-faithful
BE-CBO**, and reference performance claims do not transfer.

In section 4.1, the paper specifically reports poorer uncertainty calibration
and BO performance with ordinary maximum likelihood training than with its
variational approach. Consequently this small-budget likelihood adaptation
requires its own empirical uncertainty and optimization evaluation. The paper's
description of ensemble variance uses denominator K-1; this branch explicitly
uses population spread with denominator K, another documented departure.

The local environment was inspected: PyTorch 2.8.0+cpu, GPyTorch 1.11,
and `gpytorch.lazy.DiagLazyTensor` is present. Therefore this adaptation must
not be justified by claiming that that legacy API is unavailable.
Compatibility of the complete historical optimizer was not established;
the reference README pins GPyTorch 1.7.0 and BoTorch 0.6.5, rather than the
installed GPyTorch 1.11 and BoTorch 0.10.0. The new module needs only existing
NumPy and PyTorch.

## Fixed model and uncertainty definitions

Defaults are fixed before experiments and are not tuned on their outcomes:

| Setting | Value |
| --- | --- |
| `n_networks` | 5 (minimum 2) |
| Hidden architecture | two ReLU layers, `hidden_width = 16` |
| `steps` | 80 full-batch updates per member |
| `learning_rate` | 0.01 |
| `weight_decay` | 0.0001 |
| Optimizer | separate Adam per member, `foreach=False` |
| `band_scale` | 1.0 |
| Arithmetic | CPU float64 |

`settings` accepts a `BoundarySettings` instance or a mapping of these
named fields; unknown fields and nonfinite/invalid values raise `ValueError`.
`.diagnostics` is a dictionary containing `status`, `observed`, `positive`,
`negative`, `missing`, `calibrated`, `seed`, `likelihood`, and a nested
`settings` dictionary with the full effective configuration.
Features should use the main runner's fixed normalized representation;
the classifier neither fits a scaler nor clamps candidates to a domain.
Extremely large finite inputs can overflow a network: nonfinite losses,
fitted weights, or prediction latents raise `ValueError`.

For member latent `z_k`, let `p_k = Phi(z_k)`. Each member minimizes:

```text
mean_over_observed_rows(-log Phi((2*y - 1) * z_k))
```

The stable `torch.special.log_ndtr` expression is the Bernoulli/probit
negative log likelihood, without clipping tail gradients. Adam additionally
applies its coupled L2 penalty with the declared `weight_decay` to all weights
and biases. Members have
separate parameters and optimizers. They use the same observed rows, with
no bootstrap, class balancing, shuffled minibatches, dropout, or pseudo-data.
For predictions:

```text
probability  = mean_k(p_k)
disagreement = sqrt(mean_k((p_k - probability)**2))
lower        = max(0, probability - band_scale * disagreement)
upper        = min(1, probability + band_scale * disagreement)
```

`predict(x)` returns exactly those four keys, each a NumPy float64 array
of shape `(n,)`, with empty batches supported. Disagreement measures spread
across fitted members, an epistemic proxy. It excludes Bernoulli outcome
variance `p*(1-p)`. Identical member functions give zero spread and zero-width
bands. Random initialization does not guarantee reliable uncertainty away
from observed data. Bands are heuristic, uncalibrated, and have no coverage
or physical safety guarantee.

Single-class data are valid: all members fit only that class, and diagnostics
report `fitted_single_class`. No opposite-class observations or spatial
boundary are invented. A single row can already produce a confident raw
prediction; a low spread is not evidence that an unobserved boundary is
known. Two-class data report `fitted_two_class`, even with one row per class;
no minimum class-count shortcut is imposed.

With zero observed labels, there are no trained members. Diagnostics report
`prior_only_no_labels`, and every candidate receives probability 0.5,
lower 0, upper 1, disagreement 0. The wide band expresses absent evidence;
zero disagreement in this state does not mean certainty.

## Runner acquisition interface and discovery

```python
from ra_lsbo.risk_core.boundary_ensemble import (
    BoundaryEnsemble, boundary_acquisition, constraint_labels,
)

# outputs must already have validated physical evidence provenance.
labels = constraint_labels(outputs)
head = BoundaryEnsemble(history_x, labels, seed=trial_seed)
components = boundary_acquisition({"stress": head}, candidate_x)
eligible = components["eligible"]                     # (n,) bool
mean_p = components["heads"]["stress"]["probability"] # (n,) float64
upper = components["heads"]["stress"]["upper"]
```

The helper accepts a nonempty mapping of head names to fitted heads.
Eligibility is the conjunction of `upper >= 0.5` for **each head**; equality
is included. Per-head means, bands, and disagreement remain available in
`components["heads"]`. Upper bands are not multiplied, and neither joint
feasibility nor an objective score is returned. If the runner multiplies
per-head means as a heuristic feasibility weight, it must disclose its
constraint-independence approximation. An empty mapping raises `ValueError`
so a missing model list cannot silently approve every candidate.

An unknown head's broad prior admits all candidates optimistically. This
means only model eligibility; it does not establish an actual feasible
incumbent or authorize objective EI around a fabricated best value.

The main runner should implement an explicit discovery policy outside
this class:

1. Before any evidence-supported feasible objective observation exists, use
   a seeded space-filling/Sobol candidate pool with a discovery policy.
   Per-head means and disagreement can rank candidates, but no feasible
   incumbent, failure margins, or objective pseudo-values may be invented.
2. If every candidate fails the optimistic gate, record that condition and
   use a declared discovery evaluation, for example maximal minimum
   per-head upper probability with distance-to-history as a deterministic
   tie-break. This is an exploration fallback, not a feasible recommendation.
3. Even with a nonempty eligible pool, single-class heads need an explicit
   exploration budget in the runner because their disagreement can collapse.
   Record status, actual evidence, and fallback reason in experiment traces.

Cold-start discovery (no evidence-supported feasible incumbent) and
feasibility restoration (no eligible candidate at a later iteration) are
different runner states; preserve that distinction in records and reports.
The same classifier API can model overall completion or individual stages
when their labels come from verified traces. A missing/unverified stage
remains `None`, and a downstream failure does not imply earlier-stage
failure. Stagewise combination and dependence assumptions belong to the
runner; no stage labels are inferred here.

These are integration recommendations, not silently executed behavior.
The runner must fix its acquisition/discovery policy before comparing runs.
The classifier does not optimize candidates, compute objective EI, construct
a numeric GP, evaluate a simulator, or select a feasible start.

## Reproducibility and validation

Seeds are integers in `[0, 2**63)`. Member `k` uses a private CPU
`torch.Generator` with seed `(seed + 104729*k) % 2**63`. Layer parameters
are initialized directly from that generator, avoiding `nn.Linear`'s global
random initialization. Training has a fixed update count. Python, NumPy,
and PyTorch global RNG states, default dtype, thread count, and global
deterministic-algorithm settings are left untouched. Identical ordered
evidence, settings, seed, and local CPU environment reproduce fitted weights
and predictions. Cross-platform/BLAS/PyTorch-version bitwise identity is
not promised.

`state_dict()` returns independent tensor copies plus version, dimensions,
settings, seed, observed mask, and class counts. It omits raw training data,
optimizer state, and GP state. Use:

```python
torch.save(head.state_dict(), checkpoint_path)
restored = BoundaryEnsemble.from_state_dict(
    torch.load(checkpoint_path, weights_only=True, map_location="cpu"))
```

Restore loads weights without training. Refit by reconstructing
`BoundaryEnsemble` from the same original inputs and settings; retain the
validated evidence/history separately. No warm start or online update is
implied by checkpoint restoration.

Focused command:

```powershell
& D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe -m pytest -q -W error tests/test_risk_core_boundary_ensemble.py
```

Tests cover explicit sign evidence including interval endpoints, unknown-row
mask equivalence, orientation learned from observations, independent members,
spread collapse when member functions are identical, one-class and no-label
behavior, finite checks, inclusive per-head gating, seed isolation,
repeat prediction, exact refit, and weights-only checkpoint restoration.
These checks establish implementation behavior, not empirical calibration,
optimization performance, or equivalence to the reference variational model.

Validated in the specified environment on 2026-10-25: **43 passed**, with
warnings treated as errors (6.51 seconds for the focused suite).
After adding effective settings to diagnostics, the boundary and existing
observation suites together passed **50 tests in 6.05 seconds**, also with
warnings treated as errors.

A local synthetic timing smoke check with 488 rows, 8 normalized features,
and the default configuration took 2.256 seconds to fit one head (including
first optimizer initialization) and 0.001 seconds to predict all 488 rows.
It ran alongside the test suite; this illustrative timing is not a fixed
runtime guarantee or an optimization-quality result.
