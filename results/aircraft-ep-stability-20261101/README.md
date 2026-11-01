# D29 SOC EP Stability Candidate Evidence

This directory is an evidence-only archive for the opt-in stable-tail EP
numerical candidate. It does not authorize a pilot restart, evaluator call,
optimizer run, or RCAIDE run.

## Fixed Identity

- Result branch: `results/ep-stability-candidate-20261101`
- Diagnostic code commit: `b5744736c942eadf0a56f79d811f7e7b8ca3d52b`
- Original execution commit: `fc1ac773f5bd9e60abd2ca4b56bd971a707735e2`
- Target: `aircraft_sparse`, seed `9936`, method `old30`, failed online iteration `29`
- Input archive SHA-256: `aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9`

## Environment And Bound

- Windows / `Rcaide_vsp`
- Python `3.9.25`, `Windows-10-10.0.26200-SP0`
- Thread environment retained: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `PYTHONHASHSEED=0`, `MPLBACKEND=Agg`
- `environment.txt`, `packages.txt`, and `dependencies-freeze.txt` were
  captured during evidence sealing, not during the original execution.
- `execution.log` is a post-run reconstruction from recorded console output,
  not a contemporaneously captured console transcript.
- Encoder refits: `2`, one for D28 and one for D29
- D29 SOC candidates: `8`, each attempted once
- Aircraft/math evaluator calls: `0`
- BO, optimizer, or RCAIDE calls: `0`
- EP damping `0.7`, maximum sweeps `60`, tolerance `1e-8`, kernel grid,
  exact/interval noise, jitter, and labels remain unchanged.

## Prefix And Evidence

Both historical Windows feature identities matched exactly:

- D28: `75aa389992a576bacc3af2c0abe77526e7338bac01c1c12ea2b1525cc6c9ebfa`
- D29: `e458ae755ff5abbdbba4fa5069ad54f4cb1a7d004377831e69d409b48b88f687`
- D29 SOC observations: `17` exact, `24` interval, `0` missing

The original D29 baseline had all eight SOC GP candidates fail. With the
opt-in stable tail moments enabled, all eight candidates converged within
25-36 sweeps and produced finite marginal likelihoods and finite predictions:

| continuous | discrete | sweeps | last site change | log evidence |
|---:|---:|---:|---:|---:|
| 0.05 | 0.5 | 25 | `4.622e-09` | `-2099.498226675747` |
| 0.05 | 1.5 | 25 | `6.968e-09` | `-8670.103876866804` |
| 0.15 | 0.5 | 34 | `7.832e-09` | `-7032.163712887721` |
| 0.15 | 1.5 | 34 | `3.923e-09` | `-33354.165973741576` |
| 0.40 | 0.5 | 33 | `4.514e-09` | `-20165.681940431597` |
| 0.40 | 1.5 | 36 | `3.082e-09` | `-116591.08913501259` |
| 1.00 | 0.5 | 34 | `5.483e-09` | `-33763.119175418804` |
| 1.00 | 1.5 | 36 | `4.709e-09` | `-225647.48777219746` |

No candidate reached the 60-sweep limit, so there is no unconverged
"60th-sweep" change to report. The final changes above are the terminal
changes of the successful convergence traces. The previous truncated-normal
tail-moment failure did not reappear.

## Claim Boundary

This run establishes only GP numerical fault recovery on the fixed D29 SOC
input. It does not establish predictive calibration, closed-loop optimization
benefit, or cross-task robustness. Those require separate acceptance tests
and are not claimed here.
