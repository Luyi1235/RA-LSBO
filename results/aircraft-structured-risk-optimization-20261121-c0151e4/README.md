# Structured-risk end-to-end optimization confirmation

Fixed code SHA: `c0151e4238d5acf3f56e56b1f6343d5844ab6263`.

## Run

- Entry: `run_aircraft_refinement_confirmation.py --profile structured-risk-optimization`
- 24 seeds/scenario; 80 calls/seed cap; 3,840 paid calls; 0 RCAIDE/MDA
- Verify: `VERIFIED_STRUCTURED_RISK_OPTIMIZATION structured-risk-optimization 3840`
- Arms: `lsbo_native`, `ra_refine_interval`, `ra_structured_greedy`
- Archived per seed: `plan.json`, `calls.jsonl`, three arm selection JSONs, `seal.json`; model snapshots/original repeat dirs omitted

## Primary: cold, capped time to energy target

Targets: headwind 40,000 Wh; payload 38,000 Wh.

`ra_structured_greedy −` comparator:

| Scenario | cold n | vs `lsbo_native` | vs `ra_refine_interval` |
|---|---:|---|---|
| headwind | 9 | **−5.222** [−10.111,−1.333], 7/1/1 | **−3.889** [−6.222,−1.667], 6/0/3 |
| payload | 11 | **−8.000** [−11.545,−4.364], 9/1/1 | **−2.273** [−4.091,−0.455], 8/2/1 |

All seeds:

- headwind vs LSBO −1.792 [−4.083,−0.083]; vs interval −1.917 [−3.208,−0.792].
- payload vs LSBO −3.583 [−5.917,−1.375]; vs interval −1.250 [−2.250,−0.375].

## Found usable / energy-threshold hit

| Arm | headwind found/hit | payload found/hit |
|---|---:|---:|
| `lsbo_native` | 23/20 | 24/15 |
| `ra_refine_interval` | 24/21 | 24/23 |
| `ra_structured_greedy` | 24/24 | 24/23 |

## Limits

This is an end-to-end optimization check on 9/11 cold pairs, reduced aircraft only. It measures time to energy target and found-usable counts, not safety or RCAIDE performance. The structured arm reaches the energy target faster than both baselines in both scenarios; the payload advantage over interval is smaller than over LSBO.
