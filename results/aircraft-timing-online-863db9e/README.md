# PR127 — timing feedback on independent online aircraft trajectories

Fixed code SHA: `863db9e2a6da2a53c5dda58a19b945c4b79b142a`

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| smoke | 19 | `VERIFIED_TIMING_ONLINE smoke 19` |
| headwind-online | 687 | `VERIFIED_TIMING_ONLINE headwind-online 687` |

- shared initial evaluations: 384
- total calls: 687 (cap 960)
- new RCAIDE/MDA: 0
- initially solved: 15/48
- cold panel: n = 33, excluded 0

## Primary

`primary` in the analysis is `timing_interval − timing_sign` on the cold panel:
**−0.2424**, 95% [−0.7879, +0.2727], 8 negative / 7 positive.

Additional interval−control:

| Comparison | Mean capped first | 95% | neg/pos |
|---|---:|---|---:|
| interval − control | −0.0606 | [−0.5152, +0.3939] | 7/8 |
| interval − sign | −0.2424 | [−0.7879, +0.2727] | 8/7 |

## Four-query first discovery

All methods use four online queries; all discoveries happen within those four.

| Method | first ≤4 | discovered | mean capped first | queries |
|---|---:|---:|---:|---:|
| original_probit | 16 | 16 | 3.545 | 100 |
| timing_interval | **17** | 17 | **3.485** | 99 |
| timing_sign | 14 | 14 | 3.727 | 104 |

## Fallbacks

`fallbacks` is empty for all three arms; every query used the greedy decision
path (`reason = greedy`), no coverage or model-failure fallback was triggered.

## Interpretation

The online replay is underpowered and produces no significant primary effect:
interval vs sign −0.24 calls with a wide interval. Interval discovers one more
cold seed than control and three more than sign, but this is a single
headwind condition with 33 cold seeds. The result is consistent with a small
speed signal, not a confirmed online gain.

## Files

- `smoke/`, `headwind-online/`
- `analyses/*-analysis.json`
- `logs/`
- `fixed_sha.txt`, `environment.txt`, `SHA256SUMS.txt`
