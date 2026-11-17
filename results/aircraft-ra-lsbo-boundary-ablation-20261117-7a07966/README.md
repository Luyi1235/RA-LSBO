# PR118 — RA-LSBO boundary-scoring ablation (reduced aircraft)

## Identity
- Fixed code SHA: `7a0796673938fbba82e04f42fd13b37755b08cb4`
- Source worktree: `D:\LSBO\ra-lsbo-pr118-code` (detached, clean)
- Evidence root: `D:\LSBO\aircraft-ra-lsbo-boundary-ablation-20261117-7a07966`
- Profiles: `boundary-smoke`, `boundary-headwind`, `boundary-payload`
- Tasks: tradeoff_s0–s3; 32 cold seeds per formal profile (40 seeds, 8 initial-solved excluded from the primary cold cohort)
- 4 shared initial evaluations + 20 online decisions per arm; no held-out panel
- Arms: `ra_to_lsbo` (budgeted boundary-information score, information fraction 0.20) and `ra_feasibility_to_lsbo` (feasibility score at every slot)
- RCAIDE/MDA: **0**; VAE/aircraft truth calls: 0
- Evidence: all calls are recorded in `calls.jsonl`; no synthetic or replayed evaluations

## Protocol
The only experimental change is which score is used at scheduled boundary slots in cold start.
`ra_to_lsbo` spends 20% of cold-start slots on the information score; `ra_feasibility_to_lsbo` spends all slots on the feasibility score. Both arms share the 4 initial evaluations and the same candidate pool. The verifier checks that both arms agree until the first boundary intervention, that scheduled coverage counts are unchanged, and that the feasibility arm registers every replaced information slot.

## Acceptance
| Profile | Calls | Roles | Verify |
|---|---:|---|---|
| boundary-smoke | 32 | 16 initial +16 online | `VERIFIED_RA_LSBO_COMPARISON` |
| boundary-headwind | 1,760 | 160 initial +1,600 online | `VERIFIED_RA_LSBO_COMPARISON` |
| boundary-payload | 1,760 | 160 initial +1,600 online | `VERIFIED_RA_LSBO_COMPARISON` |
RCAIDE/MDA = 0; no model, kernel, or settlement failures in the completed ledgers.

## Primary (cold, paired, feasibility then energy)
`ra_to_lsbo` minus `ra_feasibility_to_lsbo`, capped first-feasible call:

| Condition | n | mean | 95% paired CI |
|---|---:|---:|---|
| headwind | 32 | +0.1875 | [-0.9375, +1.15625] |
| payload | 32 | -0.40625 | [-1.09375, +0.25000] |

Negative means the information-scored arm found its first usable feasible design earlier. Neither interval excludes zero.

## Boundary scoring actually used
Counts over all cold steps (summed over 32 seeds per condition):

| Condition | Arm | information | feasibility | replaced information | coverage | degenerate coverage |
|---|---|---|---:|---:|---:|---:|---:|
| headwind | ra_to_lsbo | 31 | 132 | 0 | 69 | 61 |
| headwind | ra_feasibility_to_lsbo | 0 | 126 | 32 | 68 | 60 |
| payload | ra_to_lsbo | 33 | 128 | 0 | 59 | 23 |
| payload | ra_feasibility_to_lsbo | 0 | 133 | 34 | 63 | 26 |

`information_replaced_by_feasibility` is the number of information slots converted to the feasibility score; it is zero for the information arm by construction. `budgeted_boundary_information` is the only mode that actually used the information score.

## Terminal outcomes
Headwind: win/loss/tie (feasibility then energy) **8/5/27**; energy delta +132.7 Wh [-335.2, +720.4] with 33 comparable seeds. Payload: **7/10/23**; energy delta -76.9 Wh [-499.4, +394.1] with 36 comparable seeds. Positive energy means the information-scored arm ended at a higher (worse) best feasible energy; the paired energy sample excludes seeds where either arm had no feasible design.

## Failed attempt (preserved)
The first `boundary-headwind` attempt stopped after four seeds without a `seal.json`; its partial ledger and complete log are kept in `failed-headwind-first/` and `logs/boundary-headwind-first.log`. It was not overwritten. The fresh rerun is `headwind/`.

## Limitations
- 32 cold seeds, reduced-aircraft task, two arms only.
- Headwind and payload are different scenarios, not repeated measures of one population.
- No aircraft/RCAIDE validation and no claim of a new algorithm.
- The boundary score fraction is a protocol parameter, not tuned here.
