# PR150 finite-roster energy audit

Audit code SHA: `df181f3b53bdf7af66b7b194f71353bf5a840c1f`
Input archive: PR149 confirmation commit `f3dd32eba5d97f8523b72fc850ae283171790f96`
Input `SHA256SUMS.txt`: `72951311edc3fedd39242932de50dae875d6db4e7b846141d25b03370c377c0f`

| scenario | status | supplemental calls | RCAIDE/MDA |
|---|---|---:|---:|
| payload | `VERIFIED_FINITE_ROSTER_ENERGY_AUDIT payload 145` | 145 | 0 |
| headwind | `VERIFIED_FINITE_ROSTER_ENERGY_AUDIT headwind 118` | 118 | 0 |

The supplemental labels complete the archived PR149 finite candidate roster at
`D:\LSBO\RA-LSBO-pr149-confirm-results\results\aircraft-external-full-budget-confirmation-20261120-2393ff2`.

Summary (per initial size 8/16):

| scenario | n | pooled at finite optimum | ArchSBO below finite optimum | ArchSBO advantage eliminated |
|---|---:|---:|---:|---:|
| payload 8 | 24 | 24 | 23 | 0 |
| payload 16 | 24 | 24 | 22 | 0 |
| headwind 8 | 24 | 23 | 22 | 0 |
| headwind 16 | 24 | 22 | 23 | 0 |

`archsbo_below_finite_optimum` counts seeds where ArchSBO's selected terminal energy
is below the complete finite-roster optimum; it is retained as a diagnostic, not
treated as a valid bound. `pooled_minus_finite_optimum_Wh` is 0 for most seeds;
headwind has a small positive residual because the original pooled choice was not
always the finite-roster optimum.
