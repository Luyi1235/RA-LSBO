# PR124 — timing-information diagnostic

Fixed code SHA: `330489e51dc5c63214beb5e01aeb06903028bb26`
Input result SHA: `aac24f5a96a39ef26908ac21414382783e89dc73`
Input subdir: `results/aircraft-initial-budget-ae4b9be`

Read-only replay of the PR122 archive. No new evaluator calls and no
RCAIDE/MDA calls. The script re-fits the frozen risk GP on archived histories
and compares `timing_interval` and `timing_sign` with the original probit
control on the common structurally screened cold panel.

## Row counts

| Profile | Rows | Cold n at n0=4 / n0=8 |
|---|---:|---|
| smoke | 4 | 1 / — |
| pilot-headwind-48 | 32 | 5 / 3 |
| pilot-payload-48 | 32 | 4 / 2 |

## Selected n0=4 / n0=8 results

`interval − control` (original probit), cold cohort.

| Condition | n0 | n | Brier mean [95%] | top-one Δ | top-five Δ | changed first |
|---|---:|---:|---|---:|---:|---:|
| headwind | 4 | 5 | −0.0213 [−0.0321, −0.0095] | +0.40 | +0.08 | 3/5 |
| headwind | 8 | 3 | +0.0126 [+0.0102, +0.0141] | 0 | +0.067 | 2/3 |
| payload | 4 | 4 | −0.0275 [−0.0376, −0.0150] | +0.25 | 0 | 1/4 |
| payload | 8 | 2 | −0.0133 [−0.0152, −0.0115] | −0.50 | +0.10 | 2/2 |

`sign − control` is generally weaker; the headwind n0=4 Brier interval
crosses zero, while payload n0=4 and n0=8 Brier intervals exclude zero but
top-one moves by +0.25/−0.50.

## Interpretation

Timing information has a real calibration signal at n0=4 (both conditions),
but it is not a stable ranking gain: at n0=8 the headwind Brier becomes worse,
and payload top-one loses one hit. The interval representation is stronger
than the sign representation. No new online policy is implied.

## Files

- `pr124-timing-diagnostic.json` (SHA256 62ea4e72…)
- `pr124-timing-diagnostic.log` (SHA256 572dc4d0…)
- `SHA256SUMS.txt`
