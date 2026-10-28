# Scalar online residual calibration branch

Date: 2026-10-25. This branch adds a small, standalone calibration component.
It does not integrate the runner, train a GP, evaluate an objective or constraint,
call MDA, or change existing modules.

## Literature and scope

Deshpande, Marx and Kuleshov, *Online Calibrated and Conformal Prediction Improves
Bayesian Optimization*, AISTATS 2024, PMLR 238:1450-1458: Sections 4-5 discuss
online recalibration and historical forecast/outcome pairs.
The local `online_calibration_aistats_2024.source.gz` contains the proceedings
landing page, not the full paper. The official full PDF was additionally read
online for this branch:
`https://proceedings.mlr.press/v238/deshpande24a/deshpande24a.pdf`.
Their quantile-function recalibration and cross-validation heuristic are not
reproduced here. Signed normalized residuals, a rolling window, and scalar
adaptive-alpha feedback are this branch's explicit implementation choices.

Zhang, Park and Simeone, *Bayesian Optimization with Formal Safety Guarantees
via Online Conformal Prediction*, arXiv:2306.17815v3, 2024-07-04:
Sections II and V-VI specify safety feedback, an initial safe set, and feedback
control of violation rates. The archived `safe_bocp_full_v3.source.gz` was read.
That method's safety error, safe-set fallback, scaling rule, horizon adjustment,
and noise assumptions are not reproduced. Our error is upper-prediction
miscoverage, not its physical safety error.

Archives are under
`runs/risk_redesign_20261025/primary_sources/papers/`.
The observation semantics and independent core follow
`docs/RISK_CORE_REDESIGN_20261025.md`.

## API

Import directly; package exports and runner files are intentionally untouched:

```python
from ra_lsbo.risk_core.online_calibration import OnlineResidualCalibrator

cal = OnlineResidualCalibrator(target_alpha=.2, learning_rate=.05, window=64)
pool_calibration = cal.current_quantile()  # No keys or prediction tokens reserved.
forecast = cal.predict(mean=-.3, sd=.1, key="contract/output/evaluation-1")
# Persist the forecast before revealing this evaluation's constraint.
feedback = cal.observe("contract/output/evaluation-1", -.2)
checkpoint = cal.state_dict()
restored = OnlineResidualCalibrator.from_state_dict(checkpoint)
```

Use one instance per scalar constraint and compatible physical/model contract.
Keys must be unique nonempty strings, including after missing outcomes or
residual-window eviction. `observe(key, None)` represents unknown missingness.
An inequality, censored value, failure label, imputed margin, or unconverged
intermediate value must never be supplied as an exact number. The caller owns
evidence validation; this numeric API cannot authenticate the source of a scalar.

`current_quantile()` returns a detached dictionary with `quantile`, `alpha`,
`quantile_index`, `quantile_status`, `n_residuals`, and `n_observed`. It neither
reserves keys nor creates pending tokens. Read it once to compute candidate-pool
uppers, then call `predict` only for the selected point. Pool scoring and the
selected prediction must use the same calibration state: serialize them against
feedback updates. For infinite quantiles, assign that signed infinity directly
instead of multiplying by a zero candidate standard deviation.

The planned runner uses one instance per constraint, persists the selected
forecast before the true outcome, and passes exact margins or `None` for
censored/missing evidence. After an incumbent exists, it can screen with the
adaptive upper. An unavailable `+infinity` bound requires the runner to record
an explicit uncalibrated fallback. Pool eligibility, fallback logging, and runner
integration are outside these three files; the fallback has no safety guarantee.

`predict` returns `upper`, `quantile`, `alpha`, `mean`, original `sd`, stabilized
residual `scale`, one-based `quantile_index`, `quantile_status`, `n_residuals`,
`n_observed`, `key`, and a frozen `PredictionToken` under `token`.
The token freezes alpha, quantile and prediction history. Returned dictionaries
are detached; editing one cannot change an internal pending prediction.
`alpha` on the calibrator is read-only.

`observe` returns whether an exact value was observed, residual/error (both
`None` for missing), alpha before/after, exact/missing counts, and the original
token. Unknown keys raise `KeyError`; duplicate predictions/observations raise
`ValueError`. Missing consumes the key once, but changes neither numeric
residuals, alpha, nor the exact-observation/error counts.

## Update and boundaries

With the last at most `window` previously observed finite signed scores:

```text
scale = max(sd, epsilon)                 epsilon defaults to 1e-12
s = (observed_g - frozen_mean) / frozen_scale
k = ceil((n + 1) * (1 - frozen_alpha))
q = kth smallest past score             for 1 <= k <= n
q = +infinity                           if that rank is unavailable
upper = mean + q * sd
error = int(observed_g > frozen_upper)   equality is covered
alpha_next = current_alpha + learning_rate * (target_alpha - error)
```

Only after computing the error against the frozen bound does `observe` append
the score and advance alpha. The current observation cannot calibrate its own
prediction. Predict calls without observations change neither residuals nor
alpha. Pending outcomes may arrive out of order: their means, scales, quantiles
and errors remain frozen; the controller advances from the current alpha in
feedback arrival order. This is not a delayed-feedback theorem.

Alpha is never clipped. For `alpha <= 0`, `q = upper = +infinity` (full upper
prediction set; every exact finite outcome is covered). For `alpha >= 1`,
`q = upper = -infinity` (empty set; every exact finite outcome misses).
These extensions allow the feedback to move alpha back toward the interior.
They also apply when `sd == 0`, avoiding `infinity * 0` producing NaN.
Inside `(0, 1)`, cold start or an unavailable rank yields `+infinity`;
there is no Gaussian fallback or interpolated quantile.

Finite quantiles remain signed; they are not forced nonnegative. A finite
upper uses the original `sd` exactly. Only the residual denominator is floored.
Consequently, for `sd < epsilon`, score exceedance and actual upper-bound
exceedance need not agree; error always measures the actual returned upper.
This stabilizing choice is explicit and carries no conformal-coverage claim.
Finite inputs with nonfinite computed residuals, upper bounds, or alpha updates
raise without mutating state; no overflow score is silently inserted or dropped.

Feasibility uses `g <= 0`, so an upper-bound feasibility screen would test
`upper <= 0`. Positive observed `g` violates the constraint. A feasible negative
`g` can still miss a more negative upper, and an infeasible positive `g` can be
covered by a positive upper. Coverage error is therefore not physical violation.
Infinite upper at startup does not establish any feasible starting design.

## Checkpoint and resource behavior

`state_dict` returns detached Python data including parameters, alpha, ordered
residual window, all pending frozen tokens, closed keys, and counts.
`from_state_dict` validates structural/numeric consistency and restores pending
predictions without recalibrating them from newer scores. Restore only trusted
checkpoints; provenance and model identity must be recorded by the runner.
Extended float infinities are preserved; strict JSON requires an external
encoding for them. Python's default JSON round trip is covered by a unit test.

Residual memory is bounded by `window`; sorting costs `O(window log window)`.
Lifetime duplicate protection retains `O(number of closed keys)` string keys;
pending tokens require `O(number of outstanding predictions)` memory. No
unbounded numerical residual history or general event framework is introduced.
There is no thread synchronization; serialize calls and checkpoint writes.

## Limits and validation

Adaptive BO sampling is generally not exchangeable. A finite-sample rank formula
alone does not establish exchangeable conformal coverage for these adaptive,
rolling-window GP residuals. There is no formal safety guarantee, no joint
constraint guarantee, no finite-horizon risk-budget enforcement, and no assurance
of feasible-start safety. Missingness may select a biased observed subset; no
statement here extends calibration to missing outputs or latent noiseless g.
The GP posterior must be computed before seeing the outcome; this component
cannot detect leakage already present in caller-supplied means or scales.
Reset or explicitly version a stream when its constraint/model contract changes.

Focused tests cover prequential ordering, frozen tokens and future data,
finite-sample ranks, missing/nonexact outcomes, finite residuals and atomic
overflow rejection, unclipped excursions and recovery, checkpoint continuation,
duplicate prevention after eviction, and the signs of feasibility/coverage.
Run only the focused file:

```powershell
& D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe -B -m pytest -q -p no:cacheprovider tests/test_risk_core_online_calibration.py
```

No controlled evaluations, MDA calls, or package installations are part of
this branch. Unit checks establish implementation behavior only; optimization
benefit and safety remain untested.

The focused run uses `-B` and disables pytest's cache provider to avoid producing
bytecode or pytest-cache files in the shared checkout. Candidate-pool snapshot
checks also verify that reading the quantile creates no pending prediction.
Actual result on 2026-10-25 with the specified interpreter:
**51 passed in 0.12s**.
