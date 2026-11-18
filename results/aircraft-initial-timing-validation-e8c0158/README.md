# PR125 — timing validation on new seeds

Fixed code SHA: `e8c0158996e53e68bfaa1390a17c35f057dd76cf`

Two profiles: `smoke` (80 calls) and `headwind-confirm` (1,920 calls).
No RCAIDE/MDA calls. The script re-runs the timing-information validation on
new paired seeds with the structurally screened common panel.

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| smoke | 80 | `VERIFIED_TIMING_VALIDATION smoke 80` |
| headwind-confirm | 1,920 | `VERIFIED_TIMING_VALIDATION headwind-confirm 1920` |

## Panel

- retained prefixes: 1,050
- retained usable candidates: 92
- screened out: 486
- retained completed-with-violation: 0

## Primary (n0=4, cold)

`timing_interval − original_probit`, paired seeds.

| Metric | Mean | 95% paired interval |
|---|---:|---|
| completion Brier | **−0.012786** | [−0.022758, −0.003143] |
| top-one usable hits | −0.027027 | [−0.135135, +0.081081] |
| top-five usable fraction | +0.021622 | [−0.027027, +0.070270] |

Absolute methods:

| Method | Brier | top-one hits | top-five fraction |
|---|---:|---:|---:|
| original_probit | .145764 | 9 | .140541 |
| timing_interval | **.132977** | 8 | **.162162** |
| timing_sign | **.131196** | 7 | .145946 |

n = 37 paired cold seeds; 33 had at least one usable candidate.

At n0=8 (n=30): Brier −0.014252 [−0.028916, −0.001727]; top-one +0.166667
[0.0, 0.333333]; top-five +0.013333 [−0.046667, +0.086667].

## Interpretation

The Brier improvement replicates on new seeds at n0=4 and n0=8, but the
top-one usable hit rate does not improve: it moves −0.027 at n0=4 and
+0.167 at n0=8. `timing_sign` has similar Brier but fewer top-one hits.
The timing signal is a calibration gain, not yet a stable ranking gain.
No new online policy is claimed.

## Files

- `smoke/`, `headwind-confirm/` — raw ledgers and per-seed records
- `analyses/*-analysis.json`
- `logs/`
- `fixed_sha.txt`, `environment.txt`, `SHA256SUMS.txt`
