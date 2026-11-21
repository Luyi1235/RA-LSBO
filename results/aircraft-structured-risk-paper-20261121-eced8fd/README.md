# Structured cold-start risk paper study

Fixed code SHA: `eced8fdf928ef5a24b1a07a830b20255067a52a3`.

## Run

- Entry: `run_aircraft_refinement_confirmation.py --profile structured-risk-paper`
- 24 seeds/scenario; 144 calls/seed cap; 6,912 paid calls; 0 RCAIDE/MDA
- Verify: `VERIFIED_STRUCTURED_RISK_PAPER structured-risk-paper 6912`
- Five arms: `ra_refine_interval`, `ra_refine_interval_cold_active`, `ra_interval_decision`, `ra_structured_greedy`, `ra_structured_decision`
- Archived per seed: `plan.json`, `calls.jsonl`, five arm selection JSONs, `seal.json`
- Omitted: `representation.pt`, `original/`, repeat logs

## Primary: cold, capped target call to three usable designs

`ra_structured_decision − ra_refine_interval`:

| Scenario | cold n | mean Δcall | 95% | W/L/T |
|---|---:|---:|---|---:|
| headwind | 16 | **−2.875** | [−5.0625,−0.9375] | 8/3/5 |
| payload | 8 | **−2.875** | [−5.250,−0.125] | 6/2/0 |

Held-out completion at checkpoint 12:

- headwind: balanced Brier −0.03343 [−0.04859,−0.01836], 14W/2L; configuration rank error −0.31262 [−0.44351,−0.18513].
- payload: balanced Brier −0.03458 [−0.05673,−0.01913], 8W/0L; rank error −0.20031 [−0.41925,−0.02511].

## Other cold contrasts

`ra_structured_decision` vs:

| Comparator | headwind Δcall [95%] | payload Δcall [95%] |
|---|---|---|
| `ra_interval_decision` | −3.4375 [−5.8125,−1.25] | −6.500 [−10.5,−2.5] |
| `ra_refine_interval_cold_active` | −4.750 [−6.875,−2.811] | −3.000 [−5.375,−0.75] |
| `ra_structured_greedy` | +0.3125 [−0.3125,+1.0] | −0.500 [−1.25,0.0] |

All five arms found at least one usable design in 24/24 seeds per scenario. Energy-threshold hits (headwind 40,000 Wh; payload 38,000 Wh) were 22/21/21/21/18 headwind for refine/active/interval-decision/structured-decision/structured-greedy, and 22/21/21/23/21 payload; these are not the primary endpoint.

## Limits

The primary endpoint is the capped time to three usable designs, not terminal energy or safety. Cold pairs are 16 headwind and 8 payload. Results are reduced-aircraft only, with no RCAIDE/MDA; model snapshots and repeat logs remain on the work machine.
