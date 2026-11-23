# X-57 five-arm cold-start comparison at 47.5 nmi

The campaign uses code `282647a90655714b251b47d3163b8b68c7210fbb`, based on `508398032f98b389be0701b8004aa5279c00b485`. The physical configuration and exclusions use result commit `9c5ac77c5ecc3987787292981dc679045441972c`.

`run_plan.json` (SHA-256 `1160101863a748cc0eda4283cd8c8cf1b269bcf432f441493114f9ada5b8531a`) fixes six independent seeds in advance. Each seed has four wingtip and four leading-edge distributed designs evaluated once for all five arms. All four 47.5 nmi confirmation candidates are excluded. Optimization starts with a 320-evaluation horizon per arm. The first reporting boundary is eight shared and eight online evaluations per arm, for at most 288 new physical starts across six seeds.

We froze the quality target at 47,500 Wh by rounding up the best prior 47.5 nmi confirmed feasible objective to a 500 Wh multiple. The 47,000 Wh physical terminal energy cap and the associated engineering stop remain in effect. The 45 nmi quality target remains 46,000 Wh. Its separate archive contains 48 verified and credited initialization records.

`mock_resume_report.json` and `mock_calls.jsonl` capture a zero-RCAIDE integration test. The runner dispatched one shared prefix of eight, advanced each arm to cumulative 10, then reconstructed each arm and checked the candidate hash and phase of proposal 11 against the saved preview. All five reproduced. Cumulative 12 accounted for 28 simulated worker starts and zero RCAIDE calls.

`reproduce.ps1` runs the mock check by default. `-ExecuteRcaide` starts the physical campaign. `-Resume` replays the ordered settled ledgers in the specified output directory and advances under the same plan. Raw artifacts are retained for any started call that cannot be replayed from a complete settlement.

## Physical shared initialization

| Seed | Exact objective | Feasible | Verified engineering stops | Best feasible (Wh) |
| --- | ---: | ---: | ---: | ---: |
| 307261 | 2 | 2 | 6 | 46,734.86 |
| 356821 | 2 | 1 | 6 | 49,175.68 |
| 412849 | 3 | 0 | 5 | |
| 470299 | 3 | 2 | 5 | 48,339.59 |
| 528899 | 2 | 1 | 6 | 47,753.34 |
| 590089 | 4 | 2 | 4 | 48,441.92 |

The 48 physical shared starts produced 8 feasible designs, 8 completed constraint violations, and 32 verified engineering stops. Seed `412849` is the zero-feasible cohort. The other five seeds had one or two feasible designs. Each `seed-*/shared/` folder contains the physical ledger and worker artifacts.

## Zero-feasible cohort at cumulative 16

All five trajectories for seed `412849` found a feasible design by evaluation 16.

| Arm | First feasible evaluation | First at or below 47,500 Wh | Best feasible (Wh) | Partial rows in RA energy fit |
| --- | ---: | ---: | ---: | ---: |
| LSBO | 15 | | 48,291.62 | |
| RA-LSBO sign | 12 | | 47,795.28 | 8 |
| RA-LSBO interval | 12 | | 47,795.28 | 8 |
| ArchSBO | 15 | 15 | 47,322.07 | |
| LogCEI | 11 | | 48,943.41 | |

The partial-fit column counts unique evidence rows across successful RA engineering-energy fits up to the checkpoint. For LogCEI, nine verified early-stop signs entered its `usable_energy` classifier in this seed. ArchSBO handled missing objectives through its native validity model. Compare the two RA arms using their per-evaluation ledgers in `seed-412849/`. The two arms have the same first hit and final best, with different feasible outcomes at evaluations 13 and 14.

## Full checkpoint

All 30 trajectories reached evaluation 16. The independent ledger audit verified 48 shared starts and 240 online starts, with an observed peak of 12 physical workers overall and online. All workers settled. The elapsed time from the first start through the last finish was 3 h 12 min 42 s. Across the 288 distinct charges there were 124 feasible outcomes, 67 completed violations, and 97 verified engineering stops.

| Seed | LSBO (Wh) | RA sign (Wh) | RA interval (Wh) | ArchSBO (Wh) | LogCEI (Wh) | RA sign / interval fitted rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 307261 | 46,633.62 | 46,540.08 | 46,601.08 | 45,271.87 | 46,734.86 | 8 / 8 |
| 356821 | 49,068.17 | 46,487.13 | 47,037.33 | 46,848.91 | 48,132.39 | 6 / 6 |
| 412849 | 48,291.62 | 47,795.28 | 47,795.28 | 47,322.07 | 48,943.41 | 8 / 8 |
| 470299 | 46,306.64 | 45,148.39 | 46,226.72 | 47,707.08 | 48,339.59 | 7 / 7 |
| 528899 | 47,753.34 | 46,418.28 | 47,321.17 | 46,262.53 | 45,882.37 | 7 / 7 |
| 590089 | 48,111.17 | 48,137.28 | 47,878.21 | 47,279.95 | 47,828.66 | 5 / 6 |

All five arms found a feasible design in all six seeds. The counts reaching 47,500 Wh or lower were 2, 4, 4, 5 and 2 in table order. The zero-feasible cohort had one seed. The two RA variants had identical first hit times for feasibility and the quality target in all six seeds. At the checkpoint, interval feedback produced a lower best energy in one paired seed, tied one, and had higher energy in four.

`report.json` contains the grouped endpoints and every evaluation's best-so-far trajectory. `calls.jsonl` contains the physical starts and settlements in frozen seed, arm, and evaluation order. Each arm's `checkpoint.json` stores the replayable settled observations. `ledger_audit.json` records the independent audit. `zero_feasible_first_hits.pdf` and `.png` show the zero-feasible cohort's attainment.

| File | SHA-256 |
| --- | --- |
| `report.json` | `1aa73c95a947e66b19c8eb12b0d675f2d652e9e94ad94d34393d04e68e6ae5d2` |
| `calls.jsonl` | `da57219a60b112165fce52ed2a048f2710872ee060541be24a084e5967022f1b` |
