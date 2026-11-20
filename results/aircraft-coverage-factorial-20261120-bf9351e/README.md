# PR143 — Coverage × risk-active factorial

Fixed code SHA: `bf9351e255ff9585a392229fdb40ef190196db0a`

No RCAIDE/MDA calls. All profiles `VERIFIED_AIRCRAFT_COVERAGE_FACTORIAL`.

## Design

2×2 factorial on the reduced aircraft:

- `standard` vs `covered`: enhanced candidate pool with additional coverage.
- `greedy` vs `active`: risk-active selection versus greedy, in the same pool.

Arms: `standard_greedy`, `standard_active`, `covered_greedy`,
`covered_active`.

## Acceptance and calls

| Profile | initial | online | audit | total |
|---|---:|---:|---:|---:|
| smoke | 16 | 19 | 58 | 93 |
| payload | 256 | 407 | 309 | 972 |
| headwind | 192 | 197 | 233 | 622 |

`rcaide_mda_calls=0` in all three.

## Audit: usable candidates in the enhanced pool

Cold seeds where the enhanced pool is measurable:

| Profile | cold n | covered has usable | standard has usable | usable count difference [95%] |
|---|---:|---:|---:|---|
| payload | 4 | 4 | 3 | **+0.75 [+0.25, +1.00]** |
| headwind (initial 8) | 2 | 1 | 1 | +0.50 [0.00, +1.00] |
| headwind (initial 16) | 1 | 0 | 0 | 0 |

The enhanced pool adds usable candidates in payload; headwind is too sparse
to support that claim.

## Same-pool active vs greedy

`active_under_covered` = capped-first-call difference for active minus greedy
inside the covered pool; negative favors active.

| Profile | cohort | n | Δ calls [95%] | hits active / greedy |
|---|---:|---:|---|---:|
| payload, n=8 | all | 16 | −0.50 [−1.63, +0.38] | 14 / 14 |
| payload, n=16 | all | 16 | **−0.94 [−1.88, −0.13]** | 14 / 14 |
| payload, n=16 | cold | 8 | **−1.88 [−3.38, −0.50]** | 6 / 6 |
| headwind, n=8 | all | 12 | −0.50 [−1.42, +0.08] | 11 / 11 |
| headwind, n=16 | all | 12 | −1.25 [−3.42, 0.00] | 11 / 11 |
| headwind, n=16 | cold | 3 | −5.00 [−13.0, 0.00] | 2 / 2 |

Active selection is slightly faster in both conditions, significantly so for
payload n=16. Top-one hits are identical, so this is discovery timing, not a
hit-rate gain.

## Interaction

`interaction` = (active − greedy under covered) − (active − greedy under
standard); negative means active's advantage is larger with the covered pool.

| Profile | initial | n | interaction [95%] |
|---|---:|---:|---|
| payload | 8 | 16 | +0.125 [−1.31, +1.63] |
| payload | 16 | 16 | −0.688 [−1.81, +0.25] |
| headwind | 8 | 12 | −1.083 [−3.58, +0.67] |
| headwind | 16 | 12 | −0.833 [−3.00, +0.50] |

All intervals cross 0. The point estimates lean toward a negative (synergistic)
interaction at n=16, but this is not separated from noise.

## Coverage main effect

Coverage under standard is large and negative for payload and headwind cold:

- payload n=16 all: coverage−greedy **−2.13 [−5.88, +0.75]**; cold −4.25.
- headwind n=8 all: coverage−greedy **−4.67 [−11.75, 0.00]**; cold −11.2.
- headwind n=16 all: coverage−greedy +0.08 [−1.25, +1.50]; the effect is
  gone once the standard pool already contains the usable designs.

## Interpretation

- The enhanced pool increases **usable candidate count** in payload
  (+0.75, CI excludes 0); headwind evidence is too sparse.
- Within the same covered pool, risk-active selection is modestly faster
  (payload n=16 significant; headwind borderline) without changing top-one
  hit counts.
- The interaction is directionally negative at n=16 but not significant;
  no additive or multiplicative synergy can be claimed.
- Coverage alone helps only while the standard pool lacks usable candidates;
  at n=16 it no longer separates. All of this is reduced-aircraft and
  descriptive.
