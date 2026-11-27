# X-57 47.5 nmi feasible-neighborhood pilot

This paired RCAIDE pilot reached checkpoint 32 for six fresh seeds
(2026112801 through 2026112806) under code commit 1ecea8d694381d5091fcf4bd1205fc7c92c055bd. Each seed
shared eight initial evaluations across the three arms. The final ledger
records 48 shared and 432 online evaluations. Observed concurrent workers
peaked at 4 during initialization and 12
during online search.

- [Paired metrics for checkpoints 16 and 32](paired_metrics.csv)
- [Final report and per-seed trajectories](report.json)
- [Frozen run plan](run_plan.json)
- [Per-evaluation ledger](calls.jsonl)
- [Artifact checksums](SHA256SUMS)

The six historical proposal replays passed before this pilot started.
They consumed zero RCAIDE calls. The seeds in this result are the fresh
pilot seeds.
