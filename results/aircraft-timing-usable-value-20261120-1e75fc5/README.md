# PR140 — Two-call usable-design discovery with timing feedback

Fixed code SHA: `1e75fc58edf380e96ac60ab2e57e3edcd3ee9c3b`

## Acceptance

| Profile | Calls | RCAIDE/MDA | Verify |
|---|---:|---:|---|
| smoke | 23 | 0 | `VERIFIED_TIMING_USABLE_VALUE smoke 23` |
| headwind-derated | 289 | 0 | `VERIFIED_TIMING_USABLE_VALUE headwind-derated 289` |
| headwind-nominal | 147 | 0 | `VERIFIED_TIMING_USABLE_VALUE headwind-nominal 147` |

Total **459** reduced-aircraft calls; experiment ceiling 672. No aircraft/RCAIDE
calls.

## Arms

- `interval_greedy`: interval timing information, immediate probability.
- `usable_greedy`: expected usable-design value, greedy first choice.
- `usable_lookahead`: two-call expected usable-design value.

Primary is `usable_lookahead − usable_greedy`, `any_usable_by_two`, initially
unsolved cohort.

## Primary

| Condition | obs pairs | changed first | wins | losses | paired Δ [95%] | shared first miss, 2nd usable (greedy / lookahead) |
|---|---:|---:|---:|---:|---|---:|
| headwind-derated | 17 | 2 | 0 | **1** | **−0.0588 [−0.1765, 0.0000]** | 1 / 1 |
| headwind-nominal | 9 | 1 | 0 | 0 | 0.0000 [0.0000, 0.0000] | 1 / 1 |

There is no win for lookahead in either condition. Under derated thrust it has
one paired loss (a case where greedy found a usable design within two calls but
lookahead did not); nominal is a wash.

## Control contrast

`usable_greedy − interval_greedy` is exactly zero in both conditions:

- changed first: **0/17** and **0/9**
- paired Δ: **0**
- shared first-miss second-usable: **1 vs 1**

So adding the usable-design value term does not change the greedy action at
all in this dataset; the only path change comes from the lookahead term.

## Coverage

headwind-derated:

| Arm | first usable | first observed | second observed | second usable |
|---|---:|---:|---:|---:|
| interval_greedy | 6 | 17 | 11 | 1 |
| usable_greedy | 6 | 17 | 11 | 1 |
| usable_lookahead | 5 | 17 | 12 | 1 |

headwind-nominal:

| Arm | first usable | first observed | second observed | second usable |
|---|---:|---:|---:|---:|
| interval_greedy | 1 | 9 | 8 | 1 |
| usable_greedy | 1 | 9 | 8 | 1 |
| usable_lookahead | 1 | 9 | 8 | 1 |

## Interpretation

- Lookahead does not increase two-call usable hits; it changes only two
  derated histories, losing one and tying the other.
- On histories where both arms miss the first point, both arms find exactly
  one usable second design (derated) or one (nominal). Lookahead does not add a
  rescue.
- The usable-value greedy arm is behaviorally identical to interval-greedy;
  the value term alone does not alter the first action.
- This supports stopping this branch of the timing-feedback design rather than
  spending another online budget.

## Files

- `runs/{smoke,headwind-derated,headwind-nominal}` sealed ledgers
- `analyses/{profile}.json`
- `logs/`
- `SHA256SUMS.txt`
