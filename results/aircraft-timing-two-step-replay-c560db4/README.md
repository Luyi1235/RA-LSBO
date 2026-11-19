# PR129 — Greedy vs two-step acquisition under timing feedback

Fixed code SHA: `c560db4bc4074495116bbf8787c95733b26e6271`

Inputs:

- PR127 result SHA: `e23b1510e4638582b1766e51f57fbdd0b48137d5`
- PR128 supplement SHA: `9d59c38ea89d80b70d10d397c679335b04b6716e`

## Acceptance

| Profile | Cold seeds | Status | New calls |
|---|---:|---|---:|
| smoke | 1 | `COMPLETE_TIMING_TWO_STEP_REPLAY` | 0 |
| headwind-online | 33 | `COMPLETE_TIMING_TWO_STEP_REPLAY` | 0 |

New RCAIDE/MDA: **0**. Query budget: 4.

## Cold summary (headwind-online)

Greedy and two-step are paired on the same completed pool and the same
archived GP fits; only the acquisition rule is replayed.

| Method | Greedy discovered | Two-step discovered | Greedy mean capped-first | Two-step mean capped-first | two-step − greedy [95%] |
|---|---:|---:|---:|---:|---:|
| `timing_interval` | 17 | 17 | 3.4848 | 3.4848 | 0 [0, 0] |
| `timing_sign` | 14 | 14 | 3.7273 | 3.7576 | **+0.0303** [0, +0.0909] |

Feedback × acquisition interaction: **−0.0303** [−0.0909, 0]; one negative
pair, none positive.

For the 24 pools with a usable candidate:

- `timing_interval`: 17/17 discovered, mean 2.9167 both.
- `timing_sign`: 14/14 discovered, greedy 3.2500, two-step 3.2917,
  difference +0.0417 [0, +0.125].

Numerical greedy fallback: **0** for every method and both cohorts.

## Smoke

1 cold seed; all four method/acquisition combinations discover at call 1, with
zero difference and zero fallback.

## Interpretation

Two-step acquisition does not improve cold first discovery over greedy on the
same archived pools. For the interval model it is exactly identical; for the
sign model it is slightly slower (one extra call in one seed). The
feedback×acquisition interaction is negative but tiny and interval-bounded at
zero. This is a replay of already-evaluated pools, not a new online experiment.

## Files

- `analyses/{smoke,headwind-online}-analysis.json`
- `logs/{smoke,headwind-online}.log`
- `input-online-SHA256SUMS.txt`, `input-pool-audit-SHA256SUMS.txt`
- `fixed_sha.txt`, `SHA256SUMS.txt`
