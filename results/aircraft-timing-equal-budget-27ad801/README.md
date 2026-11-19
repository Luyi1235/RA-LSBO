# PR131 — Equal-budget timing feedback on the reduced aircraft

Fixed code SHA: `27ad80110d14ed840df5a5f0bf5032a696392bf1`

## Design

- 24 paired seeds per condition.
- Nine policies: `lsbo`, `timing_sign`, and `timing_interval`, each at
  initial sizes 8, 16, and 32.
- Every trajectory has the same **total budget of 48 evaluations**, split
  into initial evaluation and online allocation.
- Primary: cold cohort, `timing_interval-8 − timing_sign-8`, capped
  first-feasible call. Negative means interval is faster.
- 0 RCAIDE/MDA calls.

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| `timing-budget-smoke` | 34 | `VERIFIED_INITIAL_BUDGET 34` |
| `timing-budget-headwind` | 8,064 | `VERIFIED_INITIAL_BUDGET 8064` |
| `timing-budget-payload` | 8,064 | `VERIFIED_INITIAL_BUDGET 8064` |

`rcaide_mda_calls = 0` in all three analyses. Each policy trajectory is
48 evaluations; 8,064 is the cross-seed/cross-policy/cross-heldout total.

## Primary: interval-8 vs sign-8

| Condition | cold n | initial-unsolved | paired mean | 95% paired | W/L/T | energy Δ Wh [n] |
|---|---:|---:|---:|---|---:|---:|
| headwind | 24 | 12 | **−2.125** | [−4.208, −0.458] | 9/2/13 | −287.9 [24] |
| payload | 24 | 10 | **−0.708** | [−1.708, +0.042] | 3/7/14 | +12.6 [24] |

Both arms discovered all 24 first-feasible solutions on both conditions.

Headwind: interval-8 is significantly faster. Payload: point estimate favours
interval-8 but the interval includes zero.

## Initially unsolved subset

| Condition | n | paired mean | 95% paired |
|---|---:|---:|---|
| headwind | 12 | −4.25 | [−8.08, −1.17] |
| payload | 10 | −1.70 | [−3.90, +0.10] |

The headwind advantage is concentrated in seeds that start without a feasible
incumbent; the payload subset is directionally similar but inconclusive.

## Other fixed-budget comparisons

Headwind, interval-8 vs:

| Comparator | capped-first mean | 95% | energy Δ Wh |
|---|---:|---|---:|
| `timing_sign-8` | −2.125 | [−4.208, −0.458] | −287.9 |
| `lsbo-8` | −3.667 | [−6.000, −1.667] | −447.7 |
| `coverage` | −1.750 | [−3.167, −0.583] | −3738.4 |

`timing_interval-16` and `timing_interval-32` lose most of the advantage:
−0.375 [−1.0, 0.0] and 0 [0.0, 0.0] vs their sign counterparts. At
initial=32 all policies are identical on the first-feasible metric.

## Interpretation

- At equal total budget, interval-8 improves cold first-feasible speed versus
  sign-8 on headwind and is directionally positive on payload.
- The effect is not a simple “more initial points” effect: the 16/32 variants
  do not show it, and the LSBO comparison is significant on headwind.
- Terminal energy and W/L/T are mixed; this is a discovery-timing result, not
  a final-design quality result.
- Reduced-aircraft only; no RCAIDE and no flight-safety claim.

## Evidence

`evidence/` contains gzip-split tar archives of the three run directories;
reassemble with `cat part-* | tar -xzf -`. The full per-call ledgers and seals
are inside.
