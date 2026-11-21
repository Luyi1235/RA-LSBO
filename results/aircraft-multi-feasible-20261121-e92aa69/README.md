# Aircraft multi-feasible cold-start pilot

Fixed code SHA: `e92aa69fc8f23590c4084ac0b25487578e1a13d5`.

## Run

- Entry: `run_aircraft_refinement_confirmation.py --profile multi-feasible-pilot`
- 12 seeds/scenario: payload 29560100..29561200; headwind 29570100..29571200
- 104 calls/seed cap; actual 2,496 paid calls; 0 RCAIDE/MDA
- Verify: `VERIFIED_MULTI_FEASIBLE_COLD_START multi-feasible-pilot 2496`
- Arms: `ra_refine_interval_cold_active` vs `ra_refine_interval`
- Primary: headwind cold cohort, capped calls to **three usable designs**

## Cold cohort

headwind 5/12 cold; payload 2/12 cold.

`cold_active − interval`, capped target call to three usable designs:

- headwind n=5: **−0.40 calls**, 95% [−1.80,+0.60], W/L/T 1/1/3.
- payload n=2: **+7.00 calls**, 95% [6.00,8.00], W/L/T 0/2/0.

All 2,496 calls; target hits 12/12 in both scenarios and both arms. Distinct usable designs through target: headwind mean −0.20 [−0.60,0.00]; payload 0.

## Secondary energy / completion

All-seed best usable energy `cold_active − interval`:

- headwind: −4.16 Wh [−10.61,−0.07], W/L/T 7/4/1.
- payload: +11.26 Wh [−2.22,+24.10], W/L/T 3/8/1.

Cold held-out completion Brier delta `cold_active − interval`:

- headwind n=16 checkpoint? checkpoint 16: −0.0100 [−0.0196,−0.0005], 5 seeds.
- payload checkpoint 12: −0.0096 [−0.0143,−0.0050], 2 seeds; but configuration rank error worsened by +0.36 at 12 and +0.45 at 16.

## Limits

Cold pairs are 5 (headwind) and 2 (payload); payload primary is not interpretable as a population effect. The active cold rule helped headwind target time slightly and hurt payload; energy effects are mixed. Reduced aircraft only; no RCAIDE/MDA and no safety claim.
