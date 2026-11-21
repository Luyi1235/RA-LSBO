# Structured-risk external benchmark comparison

Fixed code SHA: `62750a562a868bc208568b30b65c08b848d2c2fb`.

## Run

- Entry: `run_aircraft_refinement_confirmation.py --profile structured-risk-external`
- 24 seeds/scenario; 80 calls/seed cap; 3,840 paid calls; 0 RCAIDE/MDA
- Verify: `VERIFIED_STRUCTURED_RISK_EXTERNAL structured-risk-external 3840`
- Arms: `ra_structured_greedy`, `archsbo_screened`, `logcei_screened_online`
- Archived per seed: `plan.json`, `calls.jsonl`, three arm selection JSONs, `seal.json`
- `archsbo_screened` and `logcei_screened_online` are screened/adapted implementations, not the official packages

## Primary: cold, capped time to energy target

Targets: headwind 40,000 Wh; payload 38,000 Wh.

`ra_structured_greedy −` comparator, cold:

| Scenario | cold n | vs `archsbo_screened` | vs `logcei_screened_online` |
|---|---:|---|---|
| headwind | 10 | **−6.500** [−9.800,−3.597], 8/1/1 | **−11.200** [−15.300,−6.897], 9/1/0 |
| payload | 7 | **−2.571** [−4.143,−1.000], 6/1/0 | **−10.714** [−16.857,−3.000], 5/1/1 |

All seeds:

- headwind vs ArchSBO −6.542 [−8.917,−4.292]; vs LogCEI −4.750 [−7.708,−2.125].
- payload vs ArchSBO −1.708 [−2.583,−0.875]; vs LogCEI −3.792 [−6.708,−1.083].

## Found usable / energy-threshold hit

| Arm | headwind found/hit | payload found/hit |
|---|---:|---:|
| `ra_structured_greedy` | 24/22 | 24/22 |
| `archsbo_screened` | 24/14 | 24/21 |
| `logcei_screened_online` | 23/16 | 23/21 |

## Limits

Reduced-aircraft, 7–10 cold pairs, screened external arms; not a claim about official ArchSBO/LogCEI implementations or RCAIDE. The structured arm is faster to the energy target than both screened baselines in both scenarios; the payload margin over ArchSBO is smaller (≈2.6 calls) than over LogCEI.
