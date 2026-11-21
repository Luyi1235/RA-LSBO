# Aircraft censored-energy value pilot (energy-pilot)

Fixed code SHA: `e3ca2619a22c044e20db473ab13ef78b00803c5f`.

## Run

- Entry: `run_aircraft_refinement_confirmation.py --profile energy-pilot`
- Seeds: payload 29430100/200/300/400/500/600; headwind 29440100/200/300/400/500/600
- 168 paid calls/seed; 2,016 total; 0 RCAIDE/MDA
- Verify: `VERIFIED_COLD_ENERGY_VALUE_PILOT energy-pilot 2016`
- Three arms share interval-risk refinement; only cold acquisition differs:
  `..._cold_energy_greedy`, `..._cold_energy_lookahead`, `..._cold_active`
- Baseline `ra_refine_interval` is the unmodified interval policy.

## Cold cohort (initial not feasible)

Headwind: 4/6 seeds cold; payload: 1/6 cold.

| Scenario | Arm | best usable Wh | first target call | energy-bounds-only calls |
|---|---|---:|---:|---:|
| headwind | greedy | 38235.77 | 31.00 | 40 |
| headwind | lookahead | 38192.56 | 24.50 | 32 |
| headwind | cold active | 38182.22 | 18.75 | 0 |
| payload | greedy | 36321.60 | 15.00 | 1 |
| payload | lookahead | 36321.67 | 19.00 | 9 |
| payload | cold active | 36319.23 | 27.00 | 0 |

Paired lookahead − greedy, best usable energy: headwind −43.21 Wh (3 W/1 L, n=4); payload +0.06 Wh (0 W/1 L, n=1). First-target lookahead − greedy: headwind −6.5 calls; payload +4 calls.

All-seed (n=6) lookahead − greedy energy: headwind −28.81 Wh; payload +0.01 Wh. Energy-bounds-only calls total: headwind 32 vs 40; payload 9 vs 1. No energy-policy fallbacks.

## Limits

This is a 6-seed pilot, not a confirmation. Cold payload has one paired seed only; no interval estimate is meaningful. Lower energy bounds changed cold acquisition on 32/40 headwind and 9/1 payload calls; the pilot cannot separate a genuine value effect from exploratory selection. No RCAIDE/MDA and no safety claim.
