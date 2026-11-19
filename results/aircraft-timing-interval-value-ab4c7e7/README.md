# PR134 — Interval-valued vs sign-valued first-feedback acquisition

Fixed code SHA: `ab4c7e712754175abd5e34d5219f0a91e7d0d838`

## Design

- 48 paired seeds per condition; 3 arms: `greedy` (completion probability),
  `sign_value` (sign-based information value), `interval_value`
  (termination-interval information value).
- A shared interval-risk GP, identical initial history and candidate pool.
- Both information rules use the same 90% band; each trajectory has one
  subsequent evaluation opportunity.
- Initial sizes 8 and 16; 0 RCAIDE/MDA.
- Primary: after 8 initial evaluations, does one real feedback from
  `interval_value` produce lower Brier on the same structurally screened
  independent test points than `sign_value`?

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| `timing-value-smoke` | 34 | `VERIFIED_TIMING_INTERVAL_VALUE timing-value-smoke 34` |
| `timing-value-headwind` | 3,123 | `VERIFIED_TIMING_INTERVAL_VALUE timing-value-headwind 3123` |
| `timing-value-payload` | 3,091 | `VERIFIED_TIMING_INTERVAL_VALUE timing-value-payload 3091` |

Calls are within the expected 3,072–3,360 formal window per condition and vary
with initially solved seeds and same-point reuse. RCAIDE/MDA: **0**.

## Primary at initial 8

`interval_value − sign_value`, Brier after one feedback:

| Condition | cold n | changed first | first usable Δ | Brier all [95%] | Brier screened [95%] |
|---|---:|---:|---:|---|---|
| headwind | 28 | 1 | 0.000 | +0.000056 [0.000000, +0.000168] | +0.000016 [0.000000, +0.000049] |
| payload | 13 | 0 | 0.000 | 0.000000 [0, 0] | 0.000000 [0, 0] |

The headwind difference is numerically positive (interval slightly worse) but
tiny; payload is exactly identical because both arms chose the same point.
Neither condition shows a practically meaningful advantage.

## Secondary at initial 8

`interval_value − greedy`:

| Condition | cold n | changed first | first usable Δ | Brier all [95%] | Brier screened [95%] |
|---|---:|---:|---:|---|---|
| headwind | 28 | 12 | +0.0357 | −0.001008 [−0.002739, +0.000769] | −0.001819 [−0.005665, +0.001174] |
| payload | 13 | 4 | 0.000 | −0.000769 [−0.004006, +0.001699] | −0.003233 [−0.009459, +0.000765] |

Both intervals cross zero; no significant Brier or first-usable gain.

## Initial 16

| Condition | cold n | changed vs sign | Brier vs sign | changed vs greedy | Brier vs greedy |
|---|---:|---:|---|---:|---|
| headwind | 9 | 0 | 0 | 1 | +0.001092 [0, +0.003275] |
| payload | 2 | 0 | 0 | 0 | 0 |

At initial 16 both information rules and greedy are nearly identical; the two
arms select the same candidate in almost every seed.

## Choice-space limitation

Single-candidate probability bands:

| Condition | initial 8 | initial 16 |
|---|---:|---:|
| headwind | 15/28 | 4/9 |
| payload | 7/13 | 2/2 |

In more than half of initial-8 cold prefixes there was only one candidate in
the band, so the “rule” had no effective choice. This is the main mechanism
behind the near-zero differences.

## Interpretation

- Interval-valued and sign-valued information do **not** separate at initial 8:
  headwind is a negligible positive Brier difference with an interval touching
  0; payload is exactly identical.
- Neither information rule significantly improves over completion-probability
  greedy; both intervals cross 0.
- Changing from sign to interval almost never changes the first choice at
  initial 8 (1/28 and 0/13); most bands are singleton.
- No claim of interval superiority, and no terminal-quality claim. This is a
  reduced-aircraft, one-feedback mechanism test.

## Files

- `analyses/timing-value-{headwind,payload}.json`
- `plans/`, `logs/`
- `evidence/*.tar.gz.part-*` gzip-split archives; reassemble with
  `cat part-* | tar -xzf -`
- `fixed_sha.txt`, `SHA256SUMS.txt`
