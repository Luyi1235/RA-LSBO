# PR126 — timing-guided adaptive replay

Fixed code SHA: `073572c3f6b550888f6b9ff98da351eef38c3cd7`
Input result SHA: `4f1aad00e2570df5019d7e25212b1cbbaa0e95c6`
Subdir: `results/aircraft-initial-timing-validation-e8c0158`

Read-only replay from sealed PR125 panels. 0 new evaluator calls and 0
RCAIDE/MDA calls; the script replays archived four-query prefixes.

## Status

`COMPLETE_TIMING_ADAPTIVE_REPLAY` for both profiles.

| Profile | Cold seeds | Excluded | New aircraft calls | New RCAIDE |
|---|---:|---:|---:|---:|
| smoke | 2 | 0 | 0 | 0 |
| headwind-confirm | 30 | 0 | 0 | 0 |

## headwind-confirm cold summary

`timing_interval − control` (original probit):

| Metric | Mean | 95% paired interval | Negative / positive |
|---|---:|---|---:|
| capped first discovery call | **−0.6333** | [−1.2333, −0.0333] | 13 / 3 |
| number discovered | +0.1333 | [−0.0667, +0.3333] | 3 / 7 |

`timing_interval − timing_sign`:

| Metric | Mean | 95% paired interval | Negative / positive |
|---|---:|---|---:|
| capped first discovery call | **−0.9333** | [−1.4667, −0.4333] | 12 / 2 |
| number discovered | **+0.2667** | [+0.1000, +0.4333] | 0 / 8 |

Absolute methods:

| Method | Discovered / 30 | Mean capped first | Replayed queries |
|---|---:|---:|---:|
| original_probit | 19 | 3.133 | 83 |
| timing_interval | **23** | **2.500** | **68** |
| timing_sign | 15 | 3.433 | 88 |

## Interpretation

Timing interval is faster than control by 0.63 calls on the capped-first
metric and discovers four more seeds; the capped-first interval excludes zero.
Timing sign is slower and discovers fewer seeds than interval in this replay.
The discovered-count interval for interval vs control still crosses zero, so
the gain is primarily speed, not a guaranteed extra discovery.

## Files

- `pr126-timing-adaptive-replay.json` (SHA256 a0a91efa…)
- `pr126-timing-adaptive-replay.log`
- `SHA256SUMS.txt`
