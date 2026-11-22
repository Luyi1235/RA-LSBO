# X-57/RCAIDE qualification, 2026-11-22

This 55 nmi qualification run used code commit
`94abf5a455e7913f81eb9ec39da01006e77bb162` and the frozen physical worker
described in the plan. The process exited with code `0`. All 20 calls were
started and settled in 4,215.705 seconds.

| Layout | Calls | Completed | Usable | Completed with violations | Numerical nonconvergence | Verified physical stops | Verified intervals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wingtip | 10 | 4 | 0 | 4 | 6 | 0 | 0 |
| Leading-edge distributed | 10 | 8 | 0 | 8 | 2 | 0 | 0 |

All four repeat pairs had the same outcome as their corresponding first
calls. The distributed reference and flutter-boundary pairs each had an
energy difference of 0 Wh. Neither wingtip pair returned objective energy.
The rejected battery-domain trial values in solver diagnostics do not
establish verified physical endpoints.

## Files

- [Qualification report](qualification_report.json) contains outcome counts,
  elapsed times, and repeat comparisons.
- [Call ledger](calls.jsonl) records dispatch, worker start, and settlement.
  Use the `settled` records for per-call feedback, candidate and constraint
  data, objective energy, and the original worker result.
- [Frozen qualification plan](qualification_plan.json) has SHA256
  `22ee810579fb83d35bf32869f388911d21cea51b2e53a73321c4d29df8bb34c8`.
  [Reviewed plan](reviewed_plan.json) is the runtime snapshot.
- [Worker diagnostics](workers/) contains `response.json`, `solver.log`,
  and `solver_error.json` for each of the eight numerical nonconvergences.
  No `energy_audit.json` was generated for these calls. The same 24 files
  are available in the [worker ZIP](anomalous_worker_artifacts.zip).
- [Standard output](stdout.log) and [standard error](stderr.log) are the
  original process logs. Standard error was empty.
