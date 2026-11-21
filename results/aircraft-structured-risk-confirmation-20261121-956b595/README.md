# Structured-risk independent confirmation

Fixed code SHA: `956b5956538ae17def1edd97125c3538d4a26a6e`.

## Run

- Entry: `run_aircraft_refinement_confirmation.py --profile structured-risk-confirmation`
- 24 seeds/scenario; 144 calls/seed cap; 6,912 paid calls; 0 RCAIDE/MDA
- Verify: `VERIFIED_STRUCTURED_RISK_CONFIRMATION structured-risk-confirmation 6912`
- Arms: `lsbo_native`, `ra_refine_interval`, `ra_structured_sign_greedy`, `ra_structured_greedy`, `ra_structured_decision`
- Archived per seed: `plan.json`, `calls.jsonl`, five arm selection JSONs, `seal.json`; model snapshots/original repeat dirs omitted

## Primary: cold, capped target call to three usable designs

`ra_structured_greedy − ra_refine_interval`:

| Scenario | cold n | mean Δcall | 95% | W/L/T |
|---|---:|---:|---|---:|
| headwind | 13 | **−4.538** | [−7.154,−2.231] | 10/1/2 |
| payload | 9 | **−2.556** | [−4.333,−1.000] | 6/0/3 |

## Secondary cold contrasts

`ra_structured_greedy` vs:

| Comparator | headwind Δcall [95%] | payload Δcall [95%] |
|---|---|---|
| `ra_structured_sign_greedy` | −2.692 [−5.231,−0.615] | −3.556 [−5.778,−1.333] |
| `lsbo_native` | −7.231 [−10.923,−4.000] | −7.556 [−12.889,−3.222] |

`ra_structured_decision − ra_structured_greedy`: headwind −0.077 [−1.154,+1.154]; payload +0.667 [0.000,+1.222] (worse).

## Found usable / energy-threshold hit

| Arm | headwind found/hit | payload found/hit |
|---|---:|---:|
| `lsbo_native` | 23/13 | 23/17 |
| `ra_refine_interval` | 24/19 | 24/23 |
| `ra_structured_sign_greedy` | 24/21 | 24/20 |
| `ra_structured_greedy` | 24/22 | 24/21 |
| `ra_structured_decision` | 24/22 | 24/21 |

Energy-threshold hit is a secondary count, not the primary three-usable endpoint.

## Limits

Independent seeds; 13/9 cold pairs; reduced aircraft only; no RCAIDE/MDA. The structured-greedy effect replicates the earlier direction, but sign-aware scoring adds beyond plain structured greedy and the decision layer does not add on top.
