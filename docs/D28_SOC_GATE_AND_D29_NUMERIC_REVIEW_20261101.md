# EP numerical geometry and D28 prospective SOC gate — 2026-11-01

Status: **static D29 audit completed; D28 prospective-fit gate implemented and plan-only checked; Windows D28 GP fits have NOT run.**

## Immutable evidence and code

- Original interrupted pilot code: `fc1ac773f5bd9e60abd2ca4b56bd971a707735e2`; remains interrupted after 2,212 settled simplified-aircraft calls.
- Original interrupted ZIP SHA256: `aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9`.
- Windows stable-tail D29 ZIP SHA256: `0787fed8ce9ce59fa568c761e1bfe01337c672ebd74965ec970235cf30e1279f`.
- New analysis code **tested** commit: `a07fe72410aaf219456f57ed979e10cc5a209ffa`. Documentation-only later commits do not change the tested executable.
- Static D29 CI: https://github.com/Luyi1235/RA-LSBO/actions/runs/41299427599 — successful (4 new static unit tests and fixed Release audit, no GP/EP fitting). Previous independent artifact from the same audited logic: https://github.com/Luyi1235/RA-LSBO/actions/runs/41299316353.
- D28 chronological plan-only CI: https://github.com/Luyi1235/RA-LSBO/actions/runs/41299427578 — successful (dependency imports, original 116 MB archive hash, fixed proposal/pool/label association, and denial of model fitting from Linux preview).
- Fixed geometry raw JSON: GitHub Actions artifact `d29-static-audit-output`, generated from the 14-file Windows Release. The archive verifies member hashes, 41x16 feature/label correspondence and 8 original EP traces.

## New verified D29 numeric findings — NO model fitting

The 17 exact SOC margins and 24 one-sided lower-censored SOC observations are independently conditioned using the original grouped Matérn-5/2 kernel, **before EP**. The kernel implementation is re-expressed in a separate NumPy/SciPy script, not by calling the optimizer or MixedObservationGP.

| Grouped length pair | Condition number: exact 17x17 covariance | Condition number: 24x24 conditional interval covariance | Maximum standardized lower-bound distance | Intervals above 8 sigma |
|---|---:|---:|---:|---:|
| 0.05 / 0.5 | 5,245 | 1,294 | 59.55 | 4 |
| 0.05 / 1.5 | 12,570 | 3,085 | 100.72 | 4 |
| 0.15 / 0.5 | 176,654 | 80,630 | 111.41 | 5 |
| 0.15 / 1.5 | 274,039 | 205,500 | 242.18 | 5 |
| 0.40 / 0.5 | 2,861,537 | 983,906 | 175.25 | 6 |
| 0.40 / 1.5 | 6,510,800 | 2,622,600 | 461.71 | 5 |
| 1.00 / 0.5 | 10,176,238 | 2,544,897 | 230.88 | 8 |
| 1.00 / 1.5 | 14,210,750 | 5,058,087 | 556.13 | 9 |

All 8 **pre-EP** conditional covariance matrices had positive minimum eigenvalues (about 8.96e-4 down to 7.85e-8). Their conditioning deteriorates strongly with increasing kernel lengths. The standardized positive lower-bound distances confirm extremely rare inequalities in the exact-data-conditioned baseline. These are *pre-EP conditional tail distances*, NOT GP posterior calibration z-scores, and do not prove the likelihood approximation is accurate.

Under the saved approximate log evidences, the normalized eight-kernel weighting is numerically `[1,0,0,0,0,0,0,0]`, with effective component count `1.0`. All 8 stable-tail D29 fits converged in the original Windows run; convergence is not proof of reliable uncertainty. The selected pair is 0.05/0.5.

## Historic genuine one-step-ahead control

The last successful proposal before the original interruption used 40 observations (D28) to select the 41st future candidate. This is a real prior-to-evaluation old30 forecast, not an in-sample D29 prediction.

- Candidate index in saved forecast: 170; candidate and actual settled design identities match.
- Saved SOC predictive mean: -0.019670086090909394.
- Saved SOC predictive variance: 0.00005385913491917027.
- Subsequently observed exact SOC constraint margin: -0.09529698726991598.
- Standardized residual: approximately -10.304967 sigma.
- Gaussian exact negative log density in physical margin units: approximately 49.100545.

This is a poor old30 uncertainty forecast at ONE adaptively selected candidate. It does not show how stable-tail or onehot GP would predict that candidate.

## Implemented D28 one-holdout comparison, not executed

Tool: `tools/prequential_soc_D28_gate.py`. Source at the code SHA above; it is not an airplane run.

Protocol: build exactly the original D28 prefix of **40** already-settled observations, use the 41st only for future scoring; read original old30 saved forecast without refitting. Fit **one** D28 old30 encoder (30 epochs), create two SOC feature matrices: D28 latent and matched hierarchical one-hot; for each fit the fixed eight grouped kernels with opt-in stable-tail moments, same EP 60 sweeps/damping .7/tolerance 1e-8. Each uses only the 40 preceding SOC labels. Selection by training-prefix approximate evidence, not by future outcome. Report exact SOC predictive mean, variance, NLL, z-score, valid kernel count and elapsed training/fitting time. Maximum new work: 1 encoder fit and 16 GP kernel attempts, zero aircraft/RCAIDE/BO evaluations.

Original Windows/Rcaide_vsp only. Register a new worktree at the code SHA above; create a new output folder and run:

```
python tools/prequential_soc_D28_gate.py plan --archive ORIGINAL_INTERRUPT_ZIP --output NEW_OUTPUT_DIR
python tools/prequential_soc_D28_gate.py run --archive ORIGINAL_INTERRUPT_ZIP --output NEW_OUTPUT_DIR
```

DO NOT pass `--cross-platform-check` in Windows. Linux CI uses that option *only* to preview the plan; attempting a model run from preview is explicitly rejected before creating its execution lock. Original pilot locks/history remain untouched. Stop after this registered single-point check, return its raw output and SHA256 manifest; do not promote either representation or extend to other seeds automatically.

**No new Windows D28 encoder, GP, optimizer, aircraft, math truth or RCAIDE execution has occurred in this analysis turn.**
