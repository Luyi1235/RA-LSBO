# PR132 — Completion-information acquisition at equal total budget

Fixed code SHA: `2962da5729cf9c9c4a21c9ab9831809e63c89881`

## Design

- 24 paired seeds per condition, 4 policies at initial size 8 or 16:
  `timing_interval-8`, `timing_interval-16`, `timing_information-8`,
  `timing_information-16`; 48 evaluations per trajectory and a common
  independent test panel, so total reduced-aircraft evaluations are equal.
- Primary: `timing_information-8 − timing_interval-8`, cold cohort,
  capped first-feasible call; negative means the information policy is faster.
- 0 RCAIDE/MDA calls.

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| `timing-information-smoke` | 28 | `VERIFIED_INITIAL_BUDGET 28` |
| `timing-information-headwind` | 5,184 | `VERIFIED_INITIAL_BUDGET 5184` |
| `timing-information-payload` | 5,184 | `VERIFIED_INITIAL_BUDGET 5184` |

The 5,184 totals are across seeds, policies and the independent test panel;
each policy trajectory still has 48 evaluations.

## Primary: information-8 vs interval-8

| Condition | cold n | capped-first mean | 95% paired | W/L/T | energy Wh (both feasible, n) |
|---|---:|---:|---|---:|---:|
| headwind | 24 | **0.000** | [−0.167, +0.208] | 6/3/15 | +199.2 (24) |
| payload | 24 | **+0.250** | [0.000, +0.583] | 2/3/19 | −90.5 (24) |

Initial-unsolved subgroup:

| Condition | n | capped-first mean | 95% paired | W/L/T | energy Wh (n) |
|---|---:|---:|---|---:|---:|
| headwind | 16 | 0.000 | [−0.250, +0.313] | 6/3/7 | +298.8 (16) |
| payload | 9 | +0.667 | [0.000, +1.444] | 2/3/4 | −241.2 (9) |

Both primary intervals include zero on the favourable side; payload’s point
estimate is positive (information policy slower). No discovery advantage.

## Reference: each policy vs coverage

| Condition | Policy | capped-first vs coverage | 95% paired |
|---|---|---:|---|
| headwind | `timing_information-8` | −2.083 | [−3.875, −0.333] |
| headwind | `timing_information-16` | −0.500 | [−1.208, +0.042] |
| payload | `timing_information-8` | −1.000 | [−2.168, 0.000] |
| payload | `timing_information-16` | −0.125 | [−0.333, 0.000] |

The information policies beat coverage on cold first discovery, but not the
matched interval policy.

## Initial-size effect

- `information-8` vs `information-16`: headwind −1.583 [−2.875, −0.125];
  payload −0.875 [−1.875, 0.000].
- `interval-8` vs `interval-16`: headwind −1.583 [−2.917, −0.125];
  payload −1.167 [−2.250, −0.250].
  Larger initial budget does not help first discovery under a fixed total
  budget.
- All arms discovered 24/24 cold seeds; 16 headwind and 9 payload seeds were
  initially unsolved.

## Calibration

Held-out completion/usable Brier and timing-completion Brier differences at
checkpoints 8/12/24/48 are stored in each analysis JSON. No checkpoint shows a
significant information gain over interval; the paired intervals include zero
or are dominated by tiny deltas.

## Interpretation

- Completion-information acquisition does not beat the interval policy at the
  same total evaluation budget on either condition.
- The point estimates are zero/slower; initial-unsolved payload is if anything
  slower for information.
- Both policies still beat coverage, so the timing mechanism is not useless;
  the extra completion-information routing adds no measured value.
- Reduced-aircraft, 24-seed descriptive study; no RCAIDE, no safety claim.

## Files

- `analyses/` — formal analysis JSON for headwind and payload
- `plans/`, `logs/`, `environment.txt`, `fixed_sha.txt`
- `evidence/` — gzip-split raw ledgers and seals (48 MB parts)
- `SHA256SUMS.txt`
