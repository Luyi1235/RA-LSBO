# X-57 shared initialization diagnostic at 40 nmi

The run uses the archived 40 nmi configuration and seed `0` with the formal
`freeze_roster` sampler. The plan fixes 16 wingtip and 16 leading-edge
distributed propulsion designs in an alternating order. It excludes the
16 unique design hashes found in the 55 nmi qualification and the distance
diagnostic worker-start events.

The process exited with code `0`. All 32 RCAIDE calls completed in
1,267.456 seconds with a peak of 12 concurrent worker processes.

| Layout | Usable | Completed with violations | Numerical nonconvergence | First three usable positions | Best usable objective (Wh) | Verified terminal-energy lower bounds |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Wingtip | 11 | 5 | 0 | 1, 3, 5 | 46,044.78 | 0 |
| Leading-edge distributed | 11 | 5 | 0 | 2, 6, 8 | 45,867.01 | 0 |

The third usable design appeared at position 3. The 22 usable designs place
this seed in the `3+` initialization group, which starts in the performance
stage. The worker produced zero verified completion-risk intervals. This
batch provides a feasibility and performance reference. Testing cold start
requires a seed with zero usable initial designs, and testing the incremental
value of interval feedback requires verified intervals for the common risk
metric.

- [Frozen plan](initialization_plan.json) records input commits, byte
  hashes, exclusions, design hashes, order, prechecks, and coverage.
- [Initialization report](initialization_report.json) records the overall
  initialization group, per-layout outcomes, usable positions, best usable
  objective, and verified partial terminal-energy observations.
- [Merged ledger](calls.jsonl) follows the frozen roster order.
  [Task directories](tasks/) contain the individual ledgers, worker
  responses, and original diagnostics. Each `worker_started` event counts
  as one call.
- [Worker configuration](x57_qualification_40_nmi.json),
  [stdout](stdout.log), and [stderr](stderr.log) are archived verbatim.

Positions are one-based indices in the 32-point shared roster. Each
terminal-energy lower bound comes from a continuous converged mission
prefix with verified electrical summary and SOC continuity. The
completion-risk interval field reflects the original worker normalization.
