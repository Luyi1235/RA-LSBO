# D29 SOC Interruption Diagnosis Evidence

This directory is an evidence-only archive. It does not authorize a pilot
restart, parameter change, evaluator call, optimizer run, or RCAIDE run.

## Source Identity

- Original execution commit: `fc1ac773f5bd9e60abd2ca4b56bd971a707735e2`
- Diagnostic code commit: `0a43300646a6968be908b4a7d296089f4f5518de`
- Analysis branch: `analysis/corpus-interruption-diagnosis-20261031`
- Target: `aircraft_sparse`, seed `9936`, method `old30`, failed online iteration `29`
- Input archive SHA-256: `aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9`

## Environment

- Windows / `Rcaide_vsp`
- Python `3.9.25`
- Platform `Windows-10-10.0.26200-SP0`
- `numpy==2.0.2`, `scipy==1.13.1`, `torch==2.8.0+cpu`,
  `botorch==0.10.0`, `gpytorch==1.11`, `scikit-learn==1.6.1`
- Original thread settings were retained: `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `PYTHONHASHSEED=0`,
  `MPLBACKEND=Agg`.

## Prefix Verification

D28 and D29 each used one original 30-epoch encoder refit. No prior D28
candidate was rerun.

- D28 feature identity matched the historical successful proposal exactly:
  `75aa389992a576bacc3af2c0abe77526e7338bac01c1c12ea2b1525cc6c9ebfa`
- D29 feature identity:
  `e458ae755ff5abbdbba4fa5069ad54f4cb1a7d004377831e69d409b48b88f687`
- D29 SOC observations: `17` exact, `24` interval, `0` missing.

## SOC Candidate Result

All eight original D29 SOC kernel candidates failed. The outer wrapper error
was `RuntimeError: No kernel candidate produced a valid EP posterior`.
The lowest-level failures were recorded inside posterior construction, not
inside `log_evidence()`:

| continuous | discrete | result | lowest-level failure |
|---:|---:|---|---|
| 0.05 | 0.5 | fail | `RuntimeError: MixedObservationGP EP did not converge within max_sweeps` |
| 0.05 | 1.5 | fail | `RuntimeError: MixedObservationGP EP did not converge within max_sweeps` |
| 0.15 | 0.5 | fail | `RuntimeError: MixedObservationGP EP did not converge within max_sweeps` |
| 0.15 | 1.5 | fail | `RuntimeError: MixedObservationGP EP did not converge within max_sweeps` |
| 0.40 | 0.5 | fail | `RuntimeError: MixedObservationGP EP did not converge within max_sweeps` |
| 0.40 | 1.5 | fail | `FloatingPointError: Truncated tail moments exceeded numerical precision` |
| 1.00 | 0.5 | fail | `RuntimeError: MixedObservationGP EP did not converge within max_sweeps` |
| 1.00 | 1.5 | fail | `RuntimeError: MixedObservationGP EP did not converge within max_sweeps` |

Seven candidates reached the EP sweep limit of `60` and failed convergence.
One candidate failed earlier in interval truncated-moment evaluation. The
evidence therefore distinguishes EP inference failure from
`log_evidence()` failure: the observed failures occurred before evidence
evaluation.

## Bound and Preservation

- Encoder refits: `2` (D28 and D29)
- D29 SOC GP attempts: `8`
- New aircraft or mathematical truth evaluations: `0`
- New optimizer or BO proposal calls: `0`
- New RCAIDE calls: `0`
- Original interruption archive hash after diagnosis: unchanged
- Original pilot lock, ledger, plan, snapshots, and source: untouched

The raw plan, feature identities, per-candidate attempts, result summary, and
the new diagnostic execution lock are preserved in this directory.
