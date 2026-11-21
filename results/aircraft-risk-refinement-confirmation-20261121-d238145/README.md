# PR154 risk-aware LSBO refinement confirmation

Fixed code SHA: `d238145afc345ea475910bcbf1f11244c65ae960`.
24 new seeds per scenario, five arms, 208 reduced-aircraft calls per seed;
9,984 paid calls total; 0 RCAIDE/MDA.

## Acceptance

`VERIFIED_RISK_AWARE_LSBO_CONFIRMATION confirmation 9984`; 48 sealed seed directories
(24 headwind, 24 payload); `rcaide_mda_calls=0`.

Seeds: headwind `29310100..29312400`, payload `29300100..29302400`.

## Target attainment (first call reaching <= 40,000 Wh)

| scenario | interval | sign | greedy | lsbo_native | archsbo |
|---|---:|---:|---:|---:|---:|
| headwind | **24/24** | 24/24 | 20/24 | 19/24 | 20/24 |
| payload | **24/24** | 24/24 | 21/24 | 20/24 | 22/24 |

## Primary: headwind, budget 24, `ra_refine_interval − ra_interval_greedy`

- feasible at 24: 23/24 both; terminal usable 24/24 both.
- paired terminal usable energy at 24 (23 pairs): mean **−601.85 Wh**
  [−1,379.91, +122.29], W/L/T 14/9/0.
- at 48 (24 pairs): mean **−415.05 Wh** [−816.75, −101.47], W/L/T 14/10/0.
- first energy-target call: mean **−6.58 calls** [−10.21, −3.29], W/L/T 15/4/5;
  target attainment +16.7 pp [4.2, 33.3].

## Payload secondary

- terminal usable energy at 48: **−430.10 Wh** [−876.03, −8.37], W/L/T 12/12/0.
- first energy-target call: **−5.08 calls** [−9.00, −1.50], W/L/T 16/6/2.
- at 24: −522.38 Wh [−1,525.76, +512.65], W/L/T 16/8/0 (n=24).

## Versus native LSBO

Headwind terminal −605.64 Wh [−1,203.49, −124.88] (23 pairs), energy target
−7.13 calls [−12.21,−2.00]; payload terminal −565.69 Wh [−1,135.96,−133.03],
target −5.67 calls [−10.29,−1.54]. All 24 arms were compared; no hidden evaluator calls.

## Notes

- `ra_interval_greedy` was byte-identical to `lsbo_native` in the PR153 pilot; in
  PR154 it differs from native LSBO in feasibility (21 vs 19 headwind at budget 24).
- `archsbo_screened` is an external screened baseline with no native refinement-risk calls.
- This is a reduced-aircraft confirmation, not RCAIDE validation; no safety or
  full-scale claim.
