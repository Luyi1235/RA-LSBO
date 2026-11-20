# PR142 — Stopped-trace state transfer audit

Fixed code SHA: `24bbbf1df91224f345e3c5ee4dd795987308683a`

Input: PR135 sealed archive `3b6141131ec790a829f5267f285910a7b0a6f077`.
No new evaluator calls; all reports `new_evaluator_calls=0`,
`rcaide_mda_calls=0`.

Cross-state correlation:

- `prefix_fixed_full`: 1.0 (full sharing)
- `prefix_fixed_half`: 0.5 (reduced sharing)

## Acceptance

All four: `VERIFIED_CURRENT_PREFIX_AUDIT`.

| Profile | initial | cold ranked | model failures |
|---|---:|---:|---:|
| pr135-headwind | 8 | 73 | 0 |
| pr135-headwind | 16 | 40 | 0 |
| pr135-payload | 8 | 84 | 0 |
| pr135-payload | 16 | 63 | 0 |

## Usable Brier vs `completed_exact`

Negative favors the transfer model.

| Profile | n | full − exact [95%] | half − exact [95%] | half − full [95%] |
|---|---:|---|---|---|
| headwind | 8 | −.00071 [−.00157, −.00015] | −.00048 [−.00104, −.00008] | +.00023 [+.00005, +.00053] |
| headwind | 16 | −.00109 [−.00199, −.00036] | −.00085 [−.00162, −.00025] | +.00024 [+.00005, +.00049] |
| payload | 8 | −.00102 [−.00194, −.00026] | −.00051 [−.00110, −.00002] | +.00051 [+.00003, +.00124] |
| payload | 16 | +.00074 [−.00129, +.00307] | −.00068 [−.00168, +.00034] | −.00143 [−.00338, +.00010] |

## Top-one usable hits

Positive favors the first model.

| Profile | n | full − exact [95%] | half − exact [95%] | half − full [95%] |
|---|---:|---|---|---|
| headwind | 8 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| headwind | 16 | +.025 [0, +.075] | 0 [0, 0] | −.025 [−.075, 0] |
| payload | 8 | +.0119 [−.0357, +.0595] | 0 [−.0357, +.0357] | −.0119 [−.0357, 0] |
| payload | 16 | +.0317 [0, +.0794] | +.0159 [0, +.0476] | −.0159 [−.0476, 0] |

## Conclusions

1. **Full sharing lowers headwind usable Brier at both n8 and n16**; the
   paired interval excludes 0. At n16 it also raises top-one hits by .025
   [0, .075].
2. **Payload full sharing lowers Brier at n8, but loses it at n16**
   (+.00074, CI crosses 0). Half sharing is still slightly better than exact
   at n16 (−.00068) but its CI crosses 0; the half−full comparison
   (−.00143) also crosses 0.
3. **Reducing sharing to .5 does not improve headwind**: half−full is
   +.00023/+ .00024 (slightly worse) and top-one is 0/−.025. Full sharing is
   the better headwind setting.
4. **Reduced sharing loses payload top-one gain**: half improves top-one by
   .0159 at n16, full by .0317, and half−full is −.0159.
5. Overall: full sharing is the only setting with a clear headwind gain and
   the larger payload hit gain; reducing sharing trades a small n16 payload
   Brier improvement for a smaller hit gain and no headwind benefit. This is
   a descriptive, reduced-aircraft result with no new simulations.

## Files

- `analyses/{profile}-n{8,16}.json`
- `logs/{profile}-n{8,16}.log`
- `SHA256SUMS.txt`
