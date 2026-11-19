# PR130 — Nested candidate coverage at a fixed online budget

Fixed code SHA: `3f4993b5ec4821acc40d917b17b3a421eacacef5`

Inputs: PR127 result `e23b151…`, PR128 supplement `9d59c38…`.

## Acceptance

| Profile | Status | New diagnostic calls | New replay calls | RCAIDE/MDA |
|---|---|---:|---:|---:|
| smoke | `VERIFIED_TIMING_NESTED_POOL smoke 24` | 24 | 0 | 0 |
| headwind-online | `VERIFIED_TIMING_NESTED_POOL headwind-online 699` | 699 | 0 | 0 |

The 699 calls are new reduced-aircraft diagnostic evaluations used to widen the
completed pool from 32 to 64 before the structural screen; they are not a replay
of previously evaluated points. RCAIDE/MDA remain zero.

## `coverage`

| Panel | has usable | no usable |
|---|---:|---:|
| small (32) | 24 | 9 |
| large (64) | **28** | **5** |

Recovered seeds: `28701100`, `28702900`, `28703200`, `28703300` — four cold
prefixes that had no usable candidate at 32 become coverable at 64. The other
five remain without a usable candidate even at 64.

## `methods` (cold, n=33)

| Method | small discovered | large discovered | small mean first | large mean first | large−small first [95%] | large−small discovered [95%] |
|---|---:|---:|---:|---:|---:|---:|
| `original_probit` | 16 | 22 | 3.545 | 2.970 | −0.576 [−1.182, 0] | +0.182 [0, +0.364] |
| `timing_interval` | 17 | **25** | 3.485 | **2.485** | **−1.000 [−1.606, −0.424]** | **+0.242 [+0.091, +0.424]** |
| `timing_sign` | 14 | 23 | 3.727 | 2.848 | **−0.879 [−1.455, −0.303]** | **+0.273 [+0.121, +0.424]** |

Replayed queries over the four-query budget:

| Method | small | large |
|---|---:|---:|
| `original_probit` | 100 | 87 |
| `timing_interval` | 99 | 74 |
| `timing_sign` | 104 | 84 |

## Interpretation

- Widening the completed candidate pool from 32 to 64 recovers four cold seeds
  and raises usable coverage from 24/33 to 28/33.
- Every timing method benefits from the larger pool, including the control:
  discovery increases by 0.18–0.27 and capped first call drops by 0.58–1.00.
- The interval and sign methods gain more than the control, but the effect is
  confounded with coverage: a larger pool offers more opportunities to every
  method. This design measures pool coverage, not a timing-feedback mechanism
  on a fixed pool.
- `timing_interval` is the only arm whose large−small first-call interval excludes
  zero in the strongest direction; `timing_sign` also excludes zero. The control
  interval touches zero.
- This is a reduced-aircraft, headwind-only diagnostic; no aircraft safety,
  default-policy, or confirmatory claim.

## Files

- `analyses/{smoke,headwind-online}-analysis.json`
- `logs/{smoke,headwind-online}-{run,verify}.log`
- `smoke-plan.json`, `headwind-online-plan.json`
- `fixed_sha.txt`, `environment.txt`, `SHA256SUMS.txt`
