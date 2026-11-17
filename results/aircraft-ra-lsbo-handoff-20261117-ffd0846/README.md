# PR116 RA-LSBO cold start + original LSBO handoff

Reduced-aircraft test of whether RA-LSBO cold-start discovery can be kept while handing the evaluation history to original LSBO for the remaining budget.

## Identity

- Fixed code SHA: `ffd084627ab03bf2d4faecd9dd38ebae68b50f49`
- Entrypoint: `run_aircraft_ra_lsbo_comparison.py`
- Evidence root: `D:\LSBO\aircraft-ra-lsbo-handoff-20261117-ffd0846`
- RCAIDE/MDA calls: 0

## Protocol

- 24 seeds per condition; 4 shared initial + 20 nominal online calls per arm; no held-out panel.
- Arms: `ra_to_lsbo`, `lsbo_full_budget`, `original_lsbo`, `ra_separate`, `logcei_to_lsbo`, `logcei`.
- `ra_to_lsbo`: RA-LSBO from cold start until the first feasible design; then all evaluation history is handed to original LSBO for the remaining budget.
- `lsbo_full_budget`: original LSBO with early stopping disabled and the full nominal budget.
- Endpoint: per-seed terminal win/loss/tie: feasibility first, then lowest best-feasible energy.

## Calls

- `handoff-smoke`: `VERIFIED_RA_LSBO_COMPARISON`, total 79, roles {'initial': 8, 'online': 71}, max 80.
- `handoff-headwind`: `VERIFIED_RA_LSBO_COMPARISON`, total 2799, roles {'initial': 96, 'online': 2703}, max 2976.
- `handoff-payload`: `VERIFIED_RA_LSBO_COMPARISON`, total 2901, roles {'initial': 96, 'online': 2805}, max 2976.

Some handoff/original arms stop early after repeated failed objective returns; total online calls are below the 2,880 nominal cap. `lsbo_full_budget` does not use that early stop.

## Primary endpoint: terminal win/loss/tie (all 24 seeds)

`+1` = RA handoff wins (feasible vs infeasible, or lower best-feasible energy); `-1` = loss; `0` = tie/both infeasible.

| condition | comparator | W/L/T | win mean | first-call Δ | discovery Δ | energy Δ Wh (n) |
|---|---|---:|---:|---:|---:|---:|
| headwind 4 m/s | `lsbo_full_budget` | 15/7/2 | 0.333 | -2.542 [-6.375, 1.293] | 0.083 [-0.125, 0.292] | -679.9 [-1828.6, 435.0] (17) |
| headwind 4 m/s | `original_lsbo` | 18/3/3 | 0.625 | -4.792 [-8.708, -0.750] | 0.417 [0.167, 0.625] | -761.0 [-1693.5, -16.4] (10) |
| headwind 4 m/s | `ra_separate` | 16/5/3 | 0.458 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -1546.5 [-2635.1, -615.2] (21) |
| headwind 4 m/s | `logcei_to_lsbo` | 13/9/2 | 0.167 | -4.167 [-8.250, 0.083] | 0.250 [0.000, 0.500] | 562.7 [-1107.5, 2196.0] (13) |
| headwind 4 m/s | `logcei` | 17/6/1 | 0.458 | -4.167 [-8.250, 0.083] | 0.250 [0.000, 0.500] | -1460.7 [-2791.8, -305.9] (13) |
| payload +24 kg | `lsbo_full_budget` | 9/8/7 | 0.042 | -0.042 [-3.625, 3.793] | -0.042 [-0.208, 0.125] | 296.3 [-816.1, 1589.3] (19) |
| payload +24 kg | `original_lsbo` | 9/7/8 | 0.083 | -1.208 [-5.125, 2.875] | 0.125 [-0.083, 0.333] | 556.8 [-752.3, 2132.3] (16) |
| payload +24 kg | `ra_separate` | 15/5/4 | 0.417 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -1097.1 [-2273.0, 217.3] (21) |
| payload +24 kg | `logcei_to_lsbo` | 12/4/8 | 0.333 | -1.542 [-5.125, 2.125] | 0.167 [-0.042, 0.375] | -183.9 [-1434.8, 1404.8] (15) |
| payload +24 kg | `logcei` | 18/5/1 | 0.542 | -1.542 [-5.125, 2.125] | 0.167 [-0.042, 0.375] | -1596.2 [-3780.0, 853.4] (15) |

Positive energy Δ means RA handoff's best feasible energy is higher (worse); negative means lower (better). Negative first-call Δ means earlier discovery; positive discovery Δ means RA found more feasible designs.

## Interpretation

- **vs full-budget original LSBO (primary):** headwind 15W/7L/2T, energy −680 Wh (CI crosses 0); payload 9W/8L/7T, energy +296 Wh (CI crosses 0). No consistent terminal advantage.
- **vs full RA-LSBO:** handoff is better in headwind 16W/5L/3T, energy −1,546 Wh CI [−2,635,−615]; payload 15W/5L/4T, energy −1,097 Wh CI [−2,273,+217]. Handing over recovers part of the energy deficit of full RA.
- **vs LogCEI→LSBO handoff:** headwind 13W/9L/2T, energy +563 Wh (CI crosses 0); payload 12W/4L/8T, energy −184 Wh (CI crosses 0). RA handoff is not consistently better than a generic risk-aware handoff.
- **vs original LSBO with early stop:** headwind 18W/3L/3T, energy −761 Wh CI [−1,694,−16]; payload 9W/7L/8T, energy +557 Wh (CI crosses 0). The headwind comparison is confounded by the original arm's early stopping.
- No condition supports a universal handoff gain. The headwind condition suggests a direction, but the payload condition is a tie.
- Reduced-aircraft, 24-seed descriptive experiment; no RCAIDE/MDA, no held-out panel.

## Files

- `handoff-{smoke,headwind,payload}/`: full ledgers, proposals, roster, logs.
- `handoff-*-analysis.json`: paired terminal outcomes and calibration.
- `handoff-*-plan.json`, `*.log`, `python-version.log`, `environment.txt`.
- `SHA256SUMS.txt`.
