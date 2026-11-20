# PR148 ArchSBO screened-integration qualification

Fixed code SHA: `25eb360c407fe60870aa8f650f2b1a962acf9792`
Input PR147 result: `c80abe137f751051bc87df0c961e0a90f8bbc62d`
Input archive: `results/aircraft-external-baseline-pilot-20261120-f495032`

## Environment

Python 3.12.10 (embeddable distribution), CPU PyTorch 2.8.0, botorch 0.10.0,
sb-arch-opt 1.7.1, pymoo 0.6.2, smt 2.15.1, numba 0.68.1, numpy 2.5.5,
scipy 1.18.1, scikit-learn 1.9.1. See `environment.txt`.

## Acceptance

| profile | probe | online smoke |
|---|---|---|
| payload | `VERIFIED_ARCHSBO_SCREEN_INTEGRATION probe pilot-payload 8 0` | `... online-smoke pilot-payload 1 2` |
| headwind | `VERIFIED_ARCHSBO_SCREEN_INTEGRATION probe pilot-headwind 8 0` | `... online-smoke pilot-headwind 1 2` |

- Probe: 8 historical configurations each; `new_aircraft_calls=0`, `rcaide_mda_calls=0`.
- Online smoke: one cold history each, 2 charged reduced-aircraft calls; `rcaide_mda_calls=0`.
- Both smoke steps selected candidates with all exact structural margins <= 0; first step outcome `abort`, second step outcome `abort` for both profiles.

No code, protocol or input archive was modified. This is an integration qualification, not a performance comparison.
