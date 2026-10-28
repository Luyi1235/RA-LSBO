# Mixed-observation GP branch, 2026-10-25

Scope: independent scalar surrogate and per-constraint EICB factor. No runner,
MDA, controlled evaluations, dependency installation, original LSBO changes, or
changes to the observation contract are part of this branch.

## API

Import directly from `ra_lsbo.risk_core.mixed_gp` (no package exports changed):

```python
model = MixedObservationGP(x, observations, seed=0, settings=MixedGPSettings())
prediction = model.predict(query_x)
# prediction["mean"], prediction["variance"]: shape (n_query,), physical units
state = model.to_dict()
restored = MixedObservationGP.from_dict(state)
factor = eicb_factor(prediction["mean"], prediction["variance"], beta=1.96)
```

`x` and query inputs have shape `(n, d)`. Observations are aligned
`OutputObservation` instances. Exact values enter Gaussian likelihoods regardless
of their sign. Bounds enter interval likelihoods, never numeric pseudo-targets.
Missing rows are excluded entirely. An empty `(0, d)` or all-missing batch
returns the configured prior (`supports_empty_observations = True` and
`diagnostics["prior_only"] = True`). The input dimension must still be explicit.
Nonfinite inputs are rejected, including missing-row coordinates. The caller
must select evidence with the observation contract before fitting; this model
does not infer which physical output failed from a completion label.

## Scientific specification

The read-only sources reviewed are the archived CBOB AAAI 2024 full PDF,
pages 3-5, equations (6), (10)-(14), and `ep_function.py` / `gp_model.py` inside
the archived public `cbob_gpflow.zip`. The source locations are recorded in
`RISK_CORE_REDESIGN_20261025.md`. This is a CBOB-inspired extension, not a
paper/theory reproduction or evidence of optimization performance.

The prior is constant-mean Matern 5/2. Numeric likelihoods are
`y | f ~ N(f, exact_noise**2)`. The interval likelihood is
`P(lower < f + epsilon < upper)`, `epsilon ~ N(0, interval_noise**2)`.
Zero interval noise gives a hard indicator. Single-sided nonzero-noise
likelihoods are probits, corresponding to the paper's construction. Bounded
intervals extend the paper's violated-sign observation. Strict and inclusive
endpoints have equal probability under these continuous likelihoods; original
endpoint metadata is retained in serialization.

Numeric evidence is conditioned analytically first. Sequential damped EP then
matches each interval tilted distribution's analytic moments against its cavity.
For cavity `(m, v)`, set `s**2 = v + interval_noise**2`; if `(t, w)` are the
standard normal moments truncated to `((lower-m)/s, (upper-m)/s)`, the tilted
latent moments are `m + v*t/s` and
`v*(interval_noise**2/s**2 + v*w/s**2)`.
SciPy's truncated-normal implementation uses log-tail CDF calculations. Narrow
finite intervals use centered, log-rescaled 32-node Gaussian quadrature to avoid
catastrophic variance cancellation in analytic CDF-ratio formulas. This is
moment integration, not interval-midpoint imputation. Other unresolved numerical
failures raise explicitly. EP stores natural site parameters, with rank-one
updates and a whitened Cholesky refactor each sweep. Concentrated sites use
direct cavity refactoring to avoid precision subtraction cancellation.
No third-party source was
copied. One interval's moments are exact up to numerical tolerance; multiple
correlated interval sites have an approximate Gaussian posterior.

Defaults are prior mean 0, signal variance 1, length scale 0.5, numeric noise
SD 1e-6, interval noise SD 0, diagonal jitter 1e-10 (output variance units),
damping 0.7, relative site tolerance 1e-8, at most 60 sweeps.
No hyperparameters are optimized or inferred from signs/bounds. Input and output
units are not silently standardized. A scalar length scale is isotropic; a
tuple provides fixed ARD scales. The runner should use predeclared input
normalization and physically meaningful prior/noise scales equally across
compared methods. These fixed settings intentionally differ from the paper's
hyperparameter update strategy. Predictions report latent variance, without
adding measurement noise or a query nugget.

One-class sign evidence is supported, remains finite under the fixed prior, and
reverts to that prior far from observations. Signs alone do not identify the
physical response amplitude; neither predictive calibration nor finite-budget
safety is guaranteed. Jitter regularizes conditioning and interval sites; it
is explicit numerical tolerance, not evidence noise. Hard intervals at identical
inputs are merged by intersection before inference, so duplicates are
idempotent. Original evidence remains in serialization; diagnostics distinguish
raw interval counts and the merged site count. Contradictory hard intervals
raise rather than being explained by jitter. Noisy interval observations
remain separate factors under an independent-noise assumption.
Extremely concentrated tails beyond floating-point moment resolution can fail.
Failure to converge raises RuntimeError; numerical inference failures raise
FloatingPointError. Diagnostics record counts, sweeps, and final site change
on successfully constructed models, as well as fixed length scales, settings,
identity input/output normalization, kernel and inference type, and prior-only
status. Dense EP is intended for small BO batches.

The same API supports a completion latent response: a completed label is
`OutputObservation("interval", source, upper=0.)`; a known failure is
`OutputObservation("interval", source, lower=0., lower_inclusive=False)`.
An unknown label remains missing. Feasibility/completion probability is
`Phi(-mean/sqrt(variance))` under the approximate latent predictive Gaussian;
the EICB exploration factor is a separate acquisition choice. No classifier
threshold, hyperparameter, or calibration parameter is optimized on labels.

`eicb_factor` implements the requested `g <= 0` factor with clipping and NumPy
broadcasting. At exactly zero variance it uses deterministic `mean <= 0`,
including the threshold. The runner owns EI, constraint-factor multiplication,
completion risk, and no-feasible-incumbent policy. This factor is not a safety
guarantee. `to_dict`/`from_dict` preserve inputs/settings/evidence for deterministic
refit rather than serializing a trained opaque object; seed is recorded but
inference uses no random draws. Exact refit across different SciPy/BLAS versions
is not promised.

## Focused verification

`tests/test_risk_core_mixed_gp.py` covers an independent exact-GP closed form in
physical units, analytic half-normal moments, independent quadrature for finite
and censored intervals, soft probit likelihoods, conditional numeric-plus-bound
moments, spatial mixed-site evidence, one-class behavior, missing exclusion,
tail/narrow-interval stability, evidence-preserving deterministic refit,
endpoint semantics, duplicate-bound idempotence/intersection, unobserved-location
moment propagation, explicit invalid/nonconverged failures, and the EICB formula.
These are mathematical/unit checks, not controlled optimization evaluations.

Verified with `D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe`:
`-m pytest tests/test_risk_core_mixed_gp.py tests/test_risk_core_observations.py
-q -p no:cacheprovider`: **24 passed** (17 new model/acquisition tests and 7
existing observation tests). Additional inference-only shape checks with 12
completion labels and 192 prediction candidates converged in 33 sweeps
(one class) and 29 sweeps (opposing classes), with finite predictions and
nonnegative variances. These checks make no evaluator calls.
