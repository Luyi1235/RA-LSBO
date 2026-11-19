# PR135 — Decision-valued cold-start feedback with two paired evaluations

Fixed code SHA: `9a8d841e8e3006070471ca9cad2dc82afb2328a8`

## Design

- 96 paired seeds per condition; 3 arms: `greedy`, `gated_decision`,
  `decision_value`.
- Initial sizes 8 and 16; two subsequent real feedback evaluations per arm.
- Shared interval-risk GP, initial history and candidate pool.
- Primary: initial 8, `decision_value − greedy`, `any_usable_by_two`.
- 0 RCAIDE/MDA.

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| `timing-decision-smoke` | 69 | `VERIFIED_TIMING_DECISION_VALUE timing-decision-smoke 69` |
| `timing-decision-headwind` | 6,339 | `VERIFIED_TIMING_DECISION_VALUE timing-decision-headwind 6339` |
| `timing-decision-payload` | 6,237 | `VERIFIED_TIMING_DECISION_VALUE timing-decision-payload 6237` |

Both formal profiles are inside the expected 6,144–7,296 window; variation comes
from initially solved seeds and reuse.

## Primary — initial 8, `decision_value − greedy`, any usable by two

| Condition | cold n | mean Δ | 95% paired | changed first / second |
|---|---:|---:|---|---:|
| headwind | 58 | −0.0172 | [−0.1034, +0.0690] | 18 / 8 |
| payload | 30 | 0.0000 | [−0.1333, +0.1333] | 13 / 7 |

No improvement in the probability of obtaining at least one usable result
within two feedback evaluations.

`decision_value − gated_decision`:

| Condition | cold n | mean Δ | 95% |
|---|---:|---:|---|
| headwind | 58 | −0.0172 | [−0.0517, 0.0000] |
| payload | 30 | +0.0667 | [0.0000, +0.1667] |

The two directions are opposite and sample sizes are small; no stable
difference.

## Brier after feedback, initial 8

| Comparison | headwind all [95%] | headwind screened [95%] | payload all [95%] | payload screened [95%] |
|---|---|---|---|---|
| decision − greedy | +0.00065 [−0.00177, +0.00347] | −0.00013 [−0.00246, +0.00232] | −0.00280 [−0.00858, +0.00216] | −0.00660 [−0.01556, +0.00096] |
| gated − greedy | −0.00044 [−0.00227, +0.00152] | +0.00006 [−0.00208, +0.00230] | −0.00348 [−0.00894, +0.00098] | **−0.00855 [−0.01727, −0.00108]** |
| decision − gated | +0.00109 [−0.00006, +0.00310] | −0.00018 [−0.00116, +0.00059] | +0.00068 [−0.00062, +0.00266] | +0.00195 [−0.00002, +0.00505] |

Only `gated_decision` on payload screened points has a Brier interval excluding
zero; it does not translate into any usable-by-two gain.

## Initial 16

| Condition | cold n | decision − greedy any2 | decision − greedy Brier all [95%] | gated − greedy Brier all [95%] |
|---|---:|---:|---|---|
| headwind | 23 | 0.000 | −0.00191 [−0.00441, −0.00013] | −0.00092 [−0.00261, 0.00000] |
| payload | 4 | 0.000 | +0.00431 [0.00000, +0.01294] | 0.00000 [0, 0] |

At initial 16, any-usable is already 1.0 in every arm; only Brier differs, and
the signs conflict between conditions.

## Choice space

Single-candidate bands:

| Condition | initial 8 | initial 16 |
|---|---:|---:|
| headwind | 28/58 | 15/23 |
| payload | 6/30 | 3/4 |

## Interpretation

- Decision-valued acquisition does not improve the primary two-query usable
  outcome; headwind is slightly negative and payload is exactly zero.
- The Brier effects are inconsistent: decision improves headwind initial 16 but
  worsens payload initial 16; gated improves payload screened Brier at initial 8
  without improving usable discovery.
- No stable ordering among `decision_value`, `gated_decision` and `greedy`.
- This is a reduced-aircraft, two-feedback mechanism test; no terminal-quality,
  flight-safety or default-method claim.

## Files

- `analyses/timing-decision-{headwind,payload}.json`
- `plans/`, `logs/`
- `evidence/*.tar.gz.part-*`; reassemble with
  `cat part-* | tar -xzf -`
- `fixed_sha.txt`, `SHA256SUMS.txt`
