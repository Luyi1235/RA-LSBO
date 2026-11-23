# X-57 five-arm cold-start comparison at 47.5 nmi

The campaign uses code `282647a90655714b251b47d3163b8b68c7210fbb`, based on `508398032f98b389be0701b8004aa5279c00b485`. The physical configuration and exclusions use result commit `9c5ac77c5ecc3987787292981dc679045441972c`.

`run_plan.json` (SHA-256 `1160101863a748cc0eda4283cd8c8cf1b269bcf432f441493114f9ada5b8531a`) fixes six independent seeds in advance. Each seed has four wingtip and four leading-edge distributed designs evaluated once for all five arms. All four 47.5 nmi confirmation candidates are excluded. Optimization starts with a 320-evaluation horizon per arm. The first reporting boundary is eight shared and eight online evaluations per arm, for at most 288 new physical starts across six seeds.

We froze the quality target at 47,500 Wh by rounding up the best prior 47.5 nmi confirmed feasible objective to a 500 Wh multiple. The 47,000 Wh physical terminal energy cap and the associated engineering stop remain in effect. The 45 nmi quality target remains 46,000 Wh. Its separate archive contains 48 verified and credited initialization records.

`mock_resume_report.json` and `mock_calls.jsonl` capture a zero-RCAIDE integration test. The runner dispatched one shared prefix of eight, advanced each arm to cumulative 10, then reconstructed each arm and checked the candidate hash and phase of proposal 11 against the saved preview. All five reproduced. Cumulative 12 accounted for 28 simulated worker starts and zero RCAIDE calls.

`reproduce.ps1` runs the mock check by default. `-ExecuteRcaide` starts the physical campaign. `-Resume` replays the ordered settled ledgers in the specified output directory and advances under the same plan. Raw artifacts are retained for any started call that cannot be replayed from a complete settlement.
