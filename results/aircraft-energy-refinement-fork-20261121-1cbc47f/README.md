# Aircraft censored-energy refinement fork (energy-refinement-fork)

Fixed code SHA: `1cbc47f5c16fab95ad0094baf36fb9b532c3260f`.

## Run

- Entry: `run_aircraft_refinement_confirmation.py --profile energy-refinement-fork`
- Seeds: payload 29460100..29460600; headwind 29470100..29470600
- 6 seeds per scenario, 2 arms: `ra_refine_energy_bounds` vs `ra_refine_interval`
- Planned cap 88 calls/seed; actual 1,002 paid calls; 0 RCAIDE/MDA
- Verify: `VERIFIED_CENSORED_ENERGY_REFINEMENT_FORK energy-refinement-fork 1002`
- `energy_update_fallbacks=0`; energy bounds updates 197 headwind / 187 payload

## Cold cohort (initial not feasible)

Headwind 3/6 cold; payload 2/6 cold.

| Scenario | Arm | mean best usable Wh | mean first target call |
|---|---|---:|---:|
| headwind | energy_bounds | 38,177.14 | 17.50 |
| headwind | interval | 38,178.44 | 16.50 |
| payload | energy_bounds | 36,344.17 | 18.33 |
| payload | interval | 36,373.82 | 19.33 |

Paired `energy_bounds − interval`, best usable energy:

- headwind cold n=3: **+0.39 Wh**, W/L/T 2/1/0.
- payload cold n=2: **+1.11 Wh**, W/L/T 0/2/0.
- all seeds: headwind −1.30 Wh (4/2/0, n=6); payload −29.64 Wh (3/3/0, n=6).

## Limits

Six seeds per scenario; cold pairs only 3 and 2. The energy-bound fork did not improve cold terminal energy in this pilot. It updated energy beliefs on 197/187 calls without fallback, but no stable terminal or threshold-time gain is established. Reduced aircraft only; no RCAIDE/MDA, no safety claim.
