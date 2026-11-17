# PR120 — Sign-only stop evidence in aircraft cold starts

Fixed code SHA: `bc3f61348d3b5fede7584af974c94a9b1ede85c3`

## Design

- 40 paired seeds per condition; 3 arms: `ra_stop_sign_to_lsbo` (direction-only
  stop evidence), `ra_stop_status_to_lsbo` (masked stop evidence), and
  `ra_to_lsbo` (complete stop evidence).
- 4 shared initial evaluations + 20 online evaluations per arm.
- Primary: cold cohort, `ra_stop_sign_to_lsbo − ra_stop_status_to_lsbo`,
  capped first-feasible call. Negative means the direction arm is faster.
- Secondary: direction vs complete evidence, with the pre-registered practical
  tolerance of **1 call** on the paired mean.
- 0 RCAIDE/MDA calls; all evaluations are reduced-aircraft analytical calls.

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| `stop-signs-smoke` | 44 | `VERIFIED_RA_LSBO_COMPARISON` |
| `stop-signs-headwind` | 2,560 | `VERIFIED_RA_LSBO_COMPARISON` |
| `stop-signs-payload` | 2,560 | `VERIFIED_RA_LSBO_COMPARISON` |

Total 5,164 calls.

## Primary: direction vs masked

| Condition | cold n | paired mean | 95% paired interval | W/L/T (energy) | energy mean Wh [95%] |
|---|---:|---:|---|---:|---:|
| headwind | 31 | **+0.968** | [−1.387, +3.452] | 9/8/2 | −388.1 [−1794.6, +921.1] |
| payload | 26 | **+1.269** | [−0.962, +3.654] | 8/8/8 | +364.7 [−729.5, +1505.9] |

Both intervals cross zero. The point estimates are positive (direction arm
slightly slower), so the direction-only signal does **not** replicate the
PR119 result that stop evidence speeds cold discovery.

## Secondary: direction vs complete evidence

| Condition | cold n | paired mean | 95% paired interval | W/L/T | energy mean Wh [95%] |
|---|---:|---:|---|---:|---:|
| headwind | 31 | +1.935 | [−0.516, +4.452] | 8/6/7 | −246.8 [−1140.4, +680.4] |
| payload | 26 | +0.500 | [−2.077, +3.000] | 7/7/9 | +140.4 [−1686.3, +1797.5] |

The pre-registered tolerance is 1 call. Headwind is outside it in point
estimate (+1.94); payload is inside in point estimate (+0.50) but its interval
is too wide to establish practical equivalence. No equivalence claim is made.

## Replication check: complete vs masked (PR119)

| Condition | cold n | paired mean | 95% paired interval |
|---|---:|---:|---|
| headwind | 31 | −0.968 | [−3.355, +1.323] |
| payload | 26 | +0.769 | [−1.346, +3.077] |

This is the PR119 contrast on fresh seeds. PR119 estimated −2.64/−2.56 and
both intervals excluded zero; the fresh-seed intervals now cross zero. The
earlier stop-evidence advantage does not replicate.

## Stop-evidence fits

Cold-start fits that actually used each evidence mode:

| Condition | signed stop evidence | masked | complete |
|---|---:|---:|---:|
| headwind | 262 | 238 | 217 |
| payload | 140 | 117 | 132 |

Scheduled coverage was 74/69/65 (headwind) and 50/41/46 (payload).
No `fit_without_stop_evidence` occurred: every arm had at least one usable
stop-evidence or masked fit on every cold path.

The sign arm's direction observations therefore entered hundreds of cold fits;
the null result is not an instrumentation failure.

## Interpretation

- Direction-only evidence does not speed cold discovery relative to masking in
  either condition; both primary intervals cross zero.
- It is not practically equivalent to complete evidence under the 1-call
  tolerance: headwind is outside, payload is inconclusive.
- The PR119 “stop evidence speeds discovery” result does **not** replicate on
  fresh seeds; complete vs masked is now centred near zero in both conditions.
- Terminal energy and W/L/T remain mixed and are descriptive only.
- This is a reduced-aircraft mechanism study, not a flight-safety result.

## Files

- `stop-signs-*/calls.jsonl`, `seal.json`, `plan.json`, per-seed directories
- `analyses/*-analysis.json`
- `logs/` — tests, run, verify logs, Python version
- `fixed_sha.txt`, `environment.txt`, `SHA256SUMS.txt`
