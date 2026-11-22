# X-57 terminal energy constraint pilot

The pilot uses code commit `07a95dec9787c05601a64728d0b18cd073090db4` on top of `9cb135a238bd4de2b2df7945f8c2432d231af6c0`.

## 50 and 55 nmi replay

`offline_report.json` contains a screened pool of 32 unevaluated candidates, the full sign and interval rankings for each distance, and the completion, target, and charge alignment for each historical call. There are 12 historical calls at 50 nmi and 20 at 55 nmi. The replay launches zero workers.

| Distance | Exact constraint observations | Partial observations | Unique interval sites | Changed ranks of 32 | Maximum change in energy feasibility probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 nmi | 10 | 2 | 2 | 4 | 1.098 percentage points |
| 55 nmi | 12 | 8 | 6 | 4 | 1.611 percentage points |

The top three candidates retain their order at each distance. `p_usable_energy` is the marginal probability of satisfying the energy constraint. `p_engineering_feasible` multiplies the modeled constraint probabilities conditional on passing analytical prechecks. `log_acquisition` also includes the native objective acquisition and numerical solver factors. Rank 1 is the highest acquisition score. These model probabilities have no calibration assessment in this replay.

Both arms condition on the same exact values of `usable_energy` and use the same native representation, other constraint models, objective model, and acquisition. The constraint is `g = terminal_energy_Wh / 47000 - 1`. A verified discharge prefix above 47,000 Wh establishes `g > 0`. The sign fit uses lower endpoint 0 and the interval fit uses the positive numerical lower bound. The endpoint closure records the physical evidence semantics. The continuous GP likelihood depends on the endpoint value. The report contains the observations consumed by EP and the count of distinct interval sites after coalescing identical design inputs.

## 45 nmi pilot protocol

The frozen seed is `20261123`. The eight initial designs alternate the two layouts with four of each. Both arms replay those eight settled results in the frozen order. Archived outcomes are used for offline replay and the historical candidate IDs are used to exclude duplicates from the pilot initialization. Each arm uses its own online evaluations sequentially, with a cap of 12 worker starts. The original LSBO arm preserves the native missing-objective stop. The RA-LSBO arm uses the existing two-stage completion model and the new engineering energy constraint. During the completion stage it ranks the screened pool by the product of completion and energy feasibility probabilities. During performance search the energy probability replaces the native exact-only factor in the acquisition, including local refinement. A partial row is counted once per arm if it was consumed by at least one energy fit used for a recorded proposal.

`mock_report.json` validates the complete 8 + 12 + 12 schedule with a synthetic verified prefix. The RA-LSBO arm consumes one unique partial row. All 32 mock worker starts are recorded and there are zero RCAIDE starts in the mock.

## 45 nmi results

All 32 RCAIDE calls settled with exact objectives and confirmed completion. The two shared usable designs were at positions 4 and 8, with a shared best of 48,215.51 Wh. The wall time was 55 min 52 s and the peak was eight workers. The online trajectories each made 12 calls in order.

| Arm | First usable evaluation including shared initialization | First usable online evaluation | Best usable objective after 20 evaluations | Usable designs including shared initialization | Partial observations fitted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original LSBO | 4 | 1 | 46,486.61 Wh | 11 | 0 |
| RA-LSBO with energy constraint | 4 | 2 | 46,914.66 Wh | 11 | 0 |

Both arms found their best design at evaluation 20. RA-LSBO fit the energy model in 11 proposals, all from exact energy constraint observations. There were zero partial energy observations available at this distance for this seed, so evidence for the effect of the lower endpoint comes from the 50/55 nmi replay. The pilot also started online search with two usable designs already in hand.

`pilot_report.json` records the outcome and fit counts. `calls.jsonl` is the merged ledger, and `shared_initial.json` is the ordered batch that both arms replayed. The numbered folders under `initial/` and the two arm directories contain the per-worker journals and diagnostic artifacts. The pilot exited with status 0. The run stderr records native acquisition-optimizer line-search warnings and retries.

## Reproduction

Run `reproduce.ps1` from the named Windows workstation. With no switch it runs the offline and mock checks. `-ExecuteRcaide` additionally runs the paid pilot, with a fresh output directory and a maximum of 32 calls. The pinned code, local archive paths, and proof hash check are recorded in the script. Each real call has a dispatch journal with its worker PID, start time and settlement.
