# PR128 — Timing discovery against completed candidate pools

Fixed code SHA: `d8135723f782be2e8fb7f51141f3e2d7ffe37f4b`

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| smoke | 18 | `VERIFIED_TIMING_POOL_AUDIT smoke 18` |
| headwind-online | 525 | `VERIFIED_TIMING_POOL_AUDIT headwind-online 525` |

- New RCAIDE/MDA calls: **0**
- Input result SHA: `e23b1510e4638582b1766e51f57fbdd0b48137d5`
- Input code SHA: `863db9e2a6da2a53c5dda58a19b945c4b79b142a`
- Archived unique evaluations: 172
- Eligible comparisons: 697
- Cold prefixes: 33; prefixes with a usable candidate: 24
- Prefixes without any usable candidate in the completed pool: 9

## `pools`

| Pool state | Count |
|---|---:|
| has usable candidate | **24** |
| no usable candidate | **9** |

The nine no-usable pools are not model failures: they are cold prefixes whose
completed evaluation pools genuinely contained no usable candidate.

## `previous_online_outcomes`

For the 24 prefixes with a usable candidate:

| Method | Discovered | Missed usable pool | No usable pool |
|---|---:|---:|---:|
| original_probit | 16 | 8 | 9 |
| timing_interval | **17** | **7** | 9 |
| timing_sign | 14 | 10 | 9 |

“Missed usable pool” means the arm ended its four online queries without
finding the usable candidate that was present in the completed pool.

## Initial ranking against the completed pool

Absolute metrics over the 33 cold prefixes:

| Method | Completion Brier | Top-one hits | Top-five usable fraction |
|---|---:|---:|---:|
| original_probit | 0.09998 | 8 | 0.1212 |
| timing_interval | **0.08565** | 6 | **0.1515** |
| timing_sign | 0.09380 | 7 | 0.1455 |

Paired `interval − control`:

| Metric | Mean | 95% paired |
|---|---:|---|
| completion Brier | **−0.01433** | [−0.02776, −0.00376] |
| top-one usable | −0.06061 | [−0.24242, +0.09091] |
| top-five usable fraction | +0.03030 | [−0.00606, +0.06682] |

Paired `interval − sign`:

| Metric | Mean | 95% paired |
|---|---:|---|
| completion Brier | −0.00815 | [−0.01992, +0.00223] |
| top-one usable | −0.03030 | [−0.18182, +0.12121] |
| top-five usable fraction | +0.00606 | [−0.02424, +0.03636] |

`changed_first`: interval changed the top-one choice in 25/33 prefixes vs
control and 22/33 vs sign.

## Interpretation

- Interval calibration improves Brier significantly against control, but the
  top-one usable hit difference is negative (more misses) and its interval
  crosses zero; top-five coverage is only slightly positive.
- The online discovery advantage of interval (17 vs 16 vs 14) is small and
  comes with one additional pool miss corrected vs control, but still seven
  usable pools were missed.
- Nine cold prefixes had no usable candidate at all; those are excluded from
  the “missed usable pool” denominator.
- This is a completed-pool audit of an already executed online study, not a new
  optimization result. It does not establish stable ranking superiority.

## Files

- `smoke/`, `headwind-online/` evidence
- `analyses/*-analysis.json`
- `logs/`
- `fixed_sha.txt`, `environment.txt`, `SHA256SUMS.txt`
