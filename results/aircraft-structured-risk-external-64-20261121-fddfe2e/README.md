# Structured-risk external comparison at 64 evaluations

Fixed code SHA: `fddfe2e476b09db2a38e8b957646bac4d7017055`, profile `structured-risk-external-64`.

## Run

- 24 seeds/scenario; 176 calls/seed cap (8 initial + 3 arms × 56 online); 8,448 paid calls; 0 RCAIDE/MDA
- Verify: `VERIFIED_STRUCTURED_RISK_EXTERNAL_64 structured-risk-external-64 8448`
- Arms: `ra_structured_greedy`, `archsbo_screened`, `logcei_screened_online`
- Archived per seed: `plan.json`, `calls.jsonl`, three arm `*.json`, `seal.json`

## Best feasible energy at 48 and 64

`ra_structured_greedy − archsbo_screened`, all 24 seeds:

| Scenario | 48 eval | 64 eval |
|---|---:|---:|
| headwind | **−496.59 Wh** [−940.46,−112.36], 19W/5L | **−293.08 Wh** [−617.30,−1.40], 17W/7L |
| payload | −123.98 Wh [−423.79,+41.56], 15W/9L | −109.53 Wh [−409.33,+54.32], 12W/12L |

Headwind energy gap attenuates from 48 to 64; payload gap is not distinguishable from zero.

## Time to energy target at 64

`ra_structured_greedy − archsbo_screened`, capped time to target:

| Scenario | mean Δcall | 95% | W/L/T | attainment Δ |
|---|---:|---|---:|---:|
| headwind | **−14.71** | [−22.67,−7.38] | 18/6/0 | +12.5 pp |
| payload | **−4.38** | [−9.33,−0.38] | 14/6/4 | +4.17 pp |

The target-attainment advantage persists at the 64-evaluation budget, though the cold-cohort headwind time difference is the same −6.5 calls as the 32-budget run; the long-budget all-seed endpoint shows the persistent cumulative benefit.

## Found / hit at 64

| Arm | headwind found/hit | payload found/hit |
|---|---:|---:|
| `ra_structured_greedy` | 24/24 | 24/24 |
| `archsbo_screened` | 24/21 | 24/23 |
| `logcei_screened_online` | 24/23 | 24/24 |

## Limits

Screened external implementations, not official ArchSBO/LogCEI; reduced aircraft; no RCAIDE. The 64-budget run was a fresh full trajectory, not a continuation of the 32-budget run. The 48/64 energy gap shrinking is evidence against an unbounded terminal-energy advantage.
