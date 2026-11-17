# PR117 — controlled training ablation for RA-LSBO aircraft handoff

Fixed code SHA: `5fcb4e358a592dd13e93c12ae6624705c37cd4d2`

This experiment isolates independent risk-head training (`ra_to_lsbo`) from a
shared/joint kernel (`ra_joint_to_lsbo`) and compares both with a LogCEI
cold-start handoff (`logcei_to_lsbo`). It is a reduced UAM aircraft experiment;
RCAIDE/MDA calls are zero. It has no held-out panel; the terminal endpoint is
the best feasible energy after the 20 online calls.

Profiles: smoke 44 calls, training-headwind 2,560 calls, training-payload
2,560 calls. Each formal profile uses 40 seeds (`28480000`–`28483900`), four
shared initial designs and 20 online calls per arm. Every arm has the same
initial panel per seed.

Initial-unsolved seeds: **31/40** (headwind), **27/40** (payload); identical
for all three arms because the initial panels are shared.

## Primary comparison: RA vs joint kernel

The frozen `primary` field is the paired capped first-feasible-call delta
(reference `ra_to_lsbo`, comparator `ra_joint_to_lsbo`), cold cohort.

| condition | primary mean | 95% paired interval | n |
|---|---:|---:|---:|
| headwind | +0.129 | [−1.742, +2.162] | 31 |
| payload | +0.370 | [−1.223, +2.111] | 27 |

Positive means RA is slower. Both intervals cross zero, so independent risk
training does not produce a distinguishable discovery-speed gain here.

Terminal W/L/T (feasibility then energy), 40 seeds:

| condition | W/L/T | energy Δ (Wh) | 95% interval | energy n |
|---|---:|---:|---:|---:|
| headwind | 13/15/12 | +411.5 | [−556.8, +1328.0] | 34 |
| payload | 12/13/15 | +269.9 | [−485.5, +1046.6] | 39 |

Signed win rates: −0.05 and −0.025. No terminal advantage for the separate
training.

## Comparison: RA vs LogCEI handoff

| condition | W/L/T | energy Δ (Wh) | 95% interval | energy n | first-call Δ | 95% interval | discovery Δ | 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| headwind | 19/8/13 | −240.9 | [−1389.1, +912.0] | 25 | −3.250 | [−5.425, −1.349] | +0.275 | [+0.150, +0.425] |
| payload | 15/12/13 | +120.1 | [−559.5, +810.3] | 36 | −1.425 | [−3.451, +0.451] | +0.100 | [+0.025, +0.200] |

RA finds the first feasible design earlier and discovers more feasible seeds;
the headwind interval excludes zero, the payload first-call interval crosses
zero. Terminal energy is not significantly better on either condition and is
slightly worse on payload.

## Interpretation

- The independent-vs-joint training ablation is effectively a tie: all first-
  call and terminal intervals include zero, with mixed W/L/T.
- RA’s earlier discovery versus LogCEI is consistent on headwind and
  directionally present on payload, but the terminal energy effect is not
  significant.
- These are handoff-only arms; there is no full-budget original-LSBO arm and
  no held-out calibration panel. Do not read this as a comparison with
  full-budget LSBO or as evidence of a new default method.
- All calls are reduced-aircraft calls. No RCAIDE, no MDA, no full-fidelity
  validation.

## Artifacts

`training-{smoke,headwind,payload}/` hold the plan, ledger, roster, traces and
`report.json`; `training-{headwind,payload}-analysis.json` hold the frozen
paired analysis; `.log` files hold stdout; `environment.txt` and
`python-version.log` record the runtime. `SHA256SUMS.txt` covers every file
except itself.
