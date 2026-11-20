# PR141 — Current lower bounds from stopped traces

Fixed code SHA: `efac3e55b137843a5d66a66251ff2970324e43a8`

Read-only audit. Inputs:

- PR135: `3b6141131ec790a829f5267f285910a7b0a6f077`
- PR122: `aac24f5a96a39ef26908ac21414382783e89dc73`

No new evaluator calls (`new_evaluator_calls=0`, `rcaide_mda_calls=0` in all
8 reports).

## Acceptance

All eight runs: `VERIFIED_CURRENT_PREFIX_AUDIT`.

| Profile | initial | rows | cold ranked | prefix bounds | proved violation |
|---|---:|---:|---:|---:|---:|
| pr135-headwind | 8 | 96 | 73 | 565 | 562 |
| pr135-headwind | 16 | 96 | 40 | 618 | 609 |
| pr135-payload | 8 | 96 | 84 | 598 | 597 |
| pr135-payload | 16 | 96 | 63 | 881 | 879 |
| pr122-headwind | 8 | 8 | 5 | 34 | 34 |
| pr122-headwind | 16 | 8 | 4 | 58 | 56 |
| pr122-payload | 8 | 8 | 8 | 45 | 45 |
| pr122-payload | 16 | 8 | 6 | 69 | 69 |

“Rows” includes all archived seeds; “cold ranked” is the subset with a usable
incumbent at the checkpoint.

## Cold usable Brier

Lower is better. `exact_plus_prefix` vs `completed_exact`; paired Δ is
`exact_plus_prefix − completed_exact` (negative favors prefix).

| Profile | n | exact | sign | prefix | Δ Brier [95%] | top-one hits exact/prefix |
|---|---:|---:|---:|---:|---|---:|
| pr135-headwind | 8 | .04159 | .04157 | .04088 | −.0007 [−.0029, +.0017] | 8 / 9 |
| pr135-headwind | 16 | .03501 | .03503 | .03880 | **+.0038 [+.0002, +.0079]** | 9 / 12 |
| pr135-payload | 8 | .02682 | .02755 | .02184 | **−.0050 [−.0074, −.0023]** | 6 / 9 |
| pr135-payload | 16 | .01683 | .01725 | .01709 | +.0003 [−.0032, +.0042] | 9 / 9 |
| pr122-headwind | 8 | .04525 | .04366 | .04992 | +.0047 [−.0117, +.0304] | 1 / 2 |
| pr122-headwind | 16 | .04625 | .04599 | .06214 | +.0159 [−.0030, +.0426] | 2 / 2 |
| pr122-payload | 8 | .04189 | .03716 | .02672 | **−.0152 [−.0238, −.0073]** | 2 / 3 |
| pr122-payload | 16 | .04046 | .03318 | .02297 | **−.0175 [−.0349, −.0063]** | 2 / 2 |

## Initial 16: does the bound still add information?

- PR135 headwind: bound count rises to **618** but Brier is significantly
  worse at n16 (+.0038), although top-one hits rise from 9 to 12 (Δ +.075
  [−.025, +.175], not separated).
- PR135 payload: neutral at n16 (+.0003 [−.0032, +.0042]); prefix no longer
  improves Brier once more exact currents are observed.
- PR122 headwind: n16 point estimate worse (+.0159, CI crosses 0); no top-one
  gain.
- PR122 payload: the only condition with a clear n16 improvement
  (−.0175 [−.0349, −.0063]); top-one hits unchanged (2 vs 2).
- The n16 no-exact cohort shrinks (e.g. PR135 headwind 58 → 23; payload 30 →
  4). PR135 payload’s no-exact subset is only 4 states and shows worse Brier
  (+.0209) and fewer top-one hits (2 → 0), so its apparent gain is not a
  small-sample rescue.
- Conclusion: the lower bound remains abundant at n16, but its extra
  information is condition-dependent; there is no consistent n16 benefit over
  exact current observations.

## n8 vs local

n8 reproduces the archived profile row counts and identity checks 8/8. All
reports are deterministic JSON; logs are included. No prior local PR141 file
existed to diff, so “local check” here means the archived PR135/PR122 input
hashes and the recomputed n8 tables.

## Files

`analyses/` contains the 8 JSON reports; `logs/` contains stdout/stderr;
`SHA256SUMS.txt` covers all payload files.
