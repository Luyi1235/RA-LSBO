# X-57 paired distance diagnostic and energy-feedback replay

The run uses the first 12 designs in the fixed 40 nmi initialization
roster, with six designs from each propulsion layout. Every design is
paired with its archived 40 nmi observation and evaluated at both 45 and
50 nmi. The two derived configurations change only
`mission.cruise_distance_nmi`. The ledgers and report sort by original
roster position, then by distance.

All 24 new calls settled with a peak of 12 concurrent worker processes.
The process exited with code `0` after 959.170 seconds.

| Distance (nmi) | Layout | Usable | Completed with violations | Numerical nonconvergence | Verified terminal-energy lower bounds |
| ---: | --- | ---: | ---: | ---: | ---: |
| 45 | Wingtip | 0 | 6 | 0 | 0 |
| 45 | Leading-edge distributed | 1 | 5 | 0 | 0 |
| 50 | Wingtip | 0 | 5 | 1 | 1 |
| 50 | Leading-edge distributed | 0 | 5 | 1 | 1 |

Nine of the paired 40 nmi observations were usable. At 45 nmi, position
6 was usable with an objective of 48,418.26 Wh. At 50 nmi, positions 11
and 12 failed to converge at `Reserve_Cruise`. Their terminal-output
energy lower bounds were 48,194.60 and 48,297.86 Wh, respectively.

- [Run plan](run_plan.json) records the source commits, SHA256 hashes,
  roster positions, candidate identities, prechecks, and call limits.
- [Report](diagnostic_report.json) and [summary table](summary.csv)
  contain the paired outcomes, terminal-output energy, converged prefix
  bounds, and worker timings.
- [Merged ledger](calls.jsonl) and [task directories](tasks/) retain
  the individual ledgers and original worker artifacts.
- [Offline feedback replay](feedback_replay.json) includes the eight
  55 nmi numerical failures, their prefix evidence, and the exact
  engineering-constraint observations supplied to each arm.

Both arm records retain the original completion state and missing
objective. A verified terminal-output lower bound above 47,000 Wh
proves `usable_energy > 0` on the normalized constraint scale. The
sign arm receives the strict inequality `g > 0`. The interval arm
receives `g >= E_lower / 47000 - 1`. The replay contains eight such
observations across six unique designs.
