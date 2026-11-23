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
