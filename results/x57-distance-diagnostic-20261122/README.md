# Parallel X-57/RCAIDE distance diagnostic

The diagnostic pairs four designs from the 55 nmi qualification with new
evaluations at 40, 45, and 50 nmi. The derived configurations change only
`mission.cruise_distance_nmi`. Candidate hashes and analytical prechecks
were verified against the frozen qualification roster before dispatch.

The run exited with code `0`. All 14 RCAIDE calls completed and settled,
including two concurrent repeats. The initial wave reached 12 overlapping
worker processes. The total wall time was 733.207 seconds.

| Distance (nmi) | Wingtip usable | Leading-edge distributed usable |
| ---: | ---: | ---: |
| 40 | 2 / 2 | 2 / 2 |
| 45 | 0 / 2 | 1 / 2 |
| 50 | 0 / 2 | 0 / 2 |

All seven initial points that failed feasibility violated the SOC
constraint. Six also violated the terminal-output energy limit, and two
violated the loaded battery voltage limit.

The repeat pair was selected at 40 nmi. Wingtip task `13` repeated task
`02` at 48,737.89 Wh, and distributed task `14` repeated task `04` at
45,100.46 Wh. Both designs remained usable and each objective energy
difference was 0 Wh at the recorded precision.

The 55 nmi reference call `000001` did not converge at `Reserve_Cruise`.
Its eight verified prefix segments produced 48,442.56 Wh of terminal
output. All new calls completed, so their terminal output is reported
from the full mission audits.

- [Diagnostic report](diagnostic_report.json) contains every task and its
  corresponding 55 nmi observation, plus the repeat comparisons when applicable.
- [Summary table](summary.csv) contains the outcome, failed segment, constraint
  violations, SOC, electrical and objective energy, minimum loaded pack
  voltage, stall speed, elapsed time, PID, and UTC worker start and finish.
- [Run plan](run_plan.json) records the code and input identities, config hashes,
  task numbers, repeat selection rule, and 1,800-second worker timeout.
  See the [derived configs](configs/) and the [40 nmi repeat plan](repeat_plan.json).
- [Merged event ledger](calls.jsonl) is in task-number order.
  [Task directories](tasks/) preserve the original ledgers and worker artifacts.
  `worker_started` events determine the charge.
- [Standard output](stdout.log) and [standard error](stderr.log)
  preserve the process logs.
- [Mock report](mock_diagnostic_report.json) records the pre-commit check:
  12 overlapping subprocesses, 14 mock settlements including the
  conditional repeat pair, and zero RCAIDE calls.

`objective_Wh` measures audited stored-energy depletion. `terminal_energy_Wh`
measures audited mission-wide terminal output. For incomplete missions,
`terminal_energy_lower_Wh` sums the terminal output of the continuous
converged prefix. Each accepted segment must be in the expected mission
order, show finite and nonnegative electrical histories in the worker
summary, and preserve SOC continuity. The diagnostic report lists those
segments and the last prefix SOC. The objective stays missing.

When both layouts have a usable point at the same distance, the scheduler
repeats one point per layout at the longest shared distance. The first-run
objective determines the point within each layout, with task number
breaking ties.
