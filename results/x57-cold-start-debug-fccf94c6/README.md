# X-57/RCAIDE cold-start diagnostic

The frozen code is `fccf94c61d87b170824476046c8f3d2a4d503229`. [`campaign_plan.json`](campaign_plan.json) fixes the calibration and pilot rosters, the two mission configurations, the physical worker hashes, the distance selection rule, and the `46,000 Wh` objective target. The calibration roster uses seed `20261126` and has six designs per propulsion layout. The independent pilot seeds are `20261127` and `20261129`. Each uses eight shared initial designs followed by 12 sequential calls per arm.

## Zero-call verification

| Check | Result |
| --- | --- |
| 45 nmi engineering-stop replay | 5 of 44 completed trajectories crossed the `47,000 Wh` terminal-energy cap before the final segment |
| First proposal from the archived shared initialization | `7d106a8` uses completion and engineering feasibility. `5830ffd` enters `risk_weighted_native_search` with one usable design and eight exact objectives |
| Reserved local search slot | It changes the assignment at online step 9 of the five paired prefixes. The historical low-energy branch enters local search and wins the final joint ranking |
| Physical outcomes for the new candidates | Neither the revised first proposal nor the step-9 reserved-slot proposal has an exact candidate-hash match in the 45 nmi archive |
| Matched engineering feedback | The sign and interval paths use the same completion, target, and charge records in the 50 and 55 nmi replays |
| Mock | 112 isolated worker starts, with both RA arms incorporating partial observations at both pilot seeds |

[`proposal_replay.json`](verification/proposal_replay.json) contains candidate hashes, acquisition components, observed outcomes where available, local search allocations, and raw-pool order. The slot comparison fixes both the raw pool and the number of local optimizations. It uses the same final joint score rule with and without the slot. The live RA configuration uses that rule and activates the dual-head search when a feasible incumbent and three exact objectives are available. Native LSBO follows its own proposal procedure.

[`energy_stop_replay.json`](verification/energy_stop_replay.json) pairs the counterfactual observations with sealed historical truth. At the stop, the natural completion state is unknown and the mission objective is missing. All arms receive the established energy constraint violation. The sign arm uses `g > 0`; the interval arm uses `g >= E_prefix / 47000 - 1`. The prefix requires contiguous converged discharge segments with finite energy, physically consistent terminal power, and continuous SOC. The real worker checks it after postprocessing each segment, while retaining the frozen RCAIDE and X-57 sources.

## Calibration

The 24 paired calls settled at peak concurrency 12. Each row covers six new designs.

| Distance | Layout | Usable | Completed with violations | Engineering stops with verified bounds | Numerical nonconvergence |
| --- | --- | ---: | ---: | ---: | ---: |
| 45 nmi | Wingtip | 0 | 1 | 5 | 0 |
| 45 nmi | Leading-edge distributed | 2 | 0 | 4 | 0 |
| 47.5 nmi | Wingtip | 0 | 0 | 6 | 0 |
| 47.5 nmi | Leading-edge distributed | 0 | 2 | 4 | 0 |

The two usable 45 nmi designs had mission objectives of 48,373.69 and 48,464.22 Wh. The predeclared rule selected 45 nmi for the pilot. The per-design report includes solver times, constraint violations, failure diagnostics and the verified stopping boundary. [Calibration report](calibration/calibration_report.json)

## Paired pilot

The campaign exited with code `0`. Its 112 worker starts cover 24 calibration calls, 16 shared initial calls and 72 online calls. The independently reconstructed peak overlap was 12 overall and 6 online. The audit verified 34 engineering stops, comprising 19 in calibration and 15 in the pilot. [Physical audit](physical_audit.json)

| Seed | Arm | Initial usable | First usable, cumulative | First usable, online | First `<= 46,000 Wh`, cumulative | Best at 20, Wh | Fitted partials |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261127 | Original LSBO | 1 | 7 | 1 | 10 | 44,480.67 | 0 |
| 20261127 | RA-LSBO sign | 1 | 7 | 2 | 15 | 45,328.34 | 7 |
| 20261127 | RA-LSBO interval | 1 | 7 | 2 | 15 | 45,521.14 | 7 |
| 20261129 | Original LSBO | 3 | 2 | 1 | 20 | 45,886.15 | 0 |
| 20261129 | RA-LSBO sign | 3 | 2 | 2 | 10 | 44,686.80 | 5 |
| 20261129 | RA-LSBO interval | 3 | 2 | 2 | 10 | 44,635.76 | 5 |

All three arms reached the quality target in both seeds. The zero-initial-usable subgroup has no sampled seed. The interval arm finished 192.79 Wh above the sign arm on `20261127` and 51.04 Wh below it on `20261129`. None of the trajectories recorded an RA decision fallback or a native consecutive-missing-objective stop. [Campaign report](campaign_report.json) · [Complete trajectory CSV](trajectory.csv) · [Figure](best_feasible_trajectory.pdf) · [Worker ledger](calls.jsonl)

## Formal comparison

The protocol fixes a 45 nmi mission, six independent seeds and five arms, including screened ArchSBO and adapted LogCEI. Each arm receives a cumulative budget of 320 calls, with 32 shared initial designs in the main study and a nested prefix of eight in the sensitivity study. The two external X-57 adapters require qualification ahead of the formal run. [Frozen protocol](formal_protocol.json) · [Frozen design rosters](formal_rosters.json)

## Reproduce

Run from the fixed development commit in the qualified workstation environment. `-Mode verify` runs the archived replay. `-Mode mock` runs the full simulation. The real run accepts the completed mock and replay reports for that same commit and records their applicable hashes.

```powershell
$script = '.\scripts\reproduce_x57_cold_start_debug.ps1'
$verify = 'D:\RA-LSBO\x57-cold-debug-repro-verify'
$mock = 'D:\RA-LSBO\x57-cold-debug-repro-mock'
& $script -Mode verify -Output $verify
& $script -Mode mock -Output $mock
& $script -Mode real -Output 'D:\RA-LSBO\x57-cold-debug-repro-real' `
  -Verification "$verify\verification.json" -MockReport "$mock\campaign_report.json"
```
