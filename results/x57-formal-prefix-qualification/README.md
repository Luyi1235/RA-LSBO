# X-57 formal prefix qualification

The run imports the first eight members of each of the six sealed 45 nmi rosters from `de1d060b`. All six seeds are retained. The 47.5 nmi candidates come from the new 45 nmi rows, ordered within each layout by usable objective and deduplicated by physical candidate ID. The distance confirmation changes the cruise distance alone. The confirmation rows are excluded from optimizer initialization and training.

The batch settled all 52 RCAIDE starts. The six 45 nmi prefixes supplied 19 exact objectives, 12 usable designs and 29 verified energy lower bounds. Every seed supplied a usable design. At 47.5 nmi, one of two wingtip designs and both leading-edge distributed designs were usable. The other wingtip design violated the minimum final SOC by 0.008616. These observations qualify 47.5 nmi for the [independent cold-start condition](preregistered_47_5.json).

The physical ledger shows a peak of 12 RCAIDE workers. An offline mock worker overlapped with the physical batch, bringing the combined peak to 13. The reproduction script waits for the offline processes before launching the physical batch.

| Artifact | Contents |
| --- | --- |
| [`physical/report.json`](physical/report.json) | Six prefix summaries, paired distance outcomes, actual worker counts and worker times |
| [`physical/prefix_outcomes.json`](physical/prefix_outcomes.json) | All 48 outcomes in frozen seed and roster order |
| [`physical/confirmation_plan.json`](physical/confirmation_plan.json) | Confirmation candidates and their 45 nmi references |
| [`physical/calls.jsonl`](physical/calls.jsonl) | Merged ledger in task order. Unique charge key is `task_id` plus `call_id` |
| [`physical/formal_initial/credit.json`](physical/formal_initial/credit.json) | Credit and hashes for the six reusable initialization files |
| [`adapters/acceptance.json`](adapters/acceptance.json) | ArchSBO and LogCEI ask/tell acceptance and field-level evidence |
| [`ra-replay/ra_replay.json`](ra-replay/ra_replay.json) | Matched `20261127` RA proposals and the incumbent-neighborhood diagnostic |
| [`native-checkpoints/native_checkpoints.json`](native-checkpoints/native_checkpoints.json) | Next-proposal agreement for the three native policies after restoring the 16-call checkpoint |
| [`reproduce.ps1`](reproduce.ps1) | Separate physical and optimizer checkouts, with a mock prefix run by default |

The physical run used `8a40729177d3473b17f05655e3819167c943fe93` and the qualified Python 3.9 RCAIDE environment. The offline acceptance used `508398032f98b389be0701b8004aa5279c00b485`. ArchSBO ran in a separate Python 3.11 environment with SBArchOpt 1.7.1, SMT 2.14.1 and pymoo 0.6.1.5. The exact optimizer environment is recorded in [`environment/baseline_requirements.txt`](environment/baseline_requirements.txt).

Both external baselines received the archived exact constraints and the same verified energy violation sign. LogCEI fits the sign in its energy classifier and leaves the completion label unknown. ArchSBO receives the sign in `tell` and applies its native missing-output handling to the row. A numerical nonconvergence without physical evidence contributes no completion label or numerical target to the common record. The raw ledgers retain the original outcomes for both baselines.

The RA replay uses one feasible design and two exact objectives. The frozen policy reproduced both archived first proposals. Both of those proposed designs reached the engineering energy stop. The incumbent-neighborhood proposals are predictions within the same discrete branch and a radius of 0.03 in active normalized coordinates. The archive contains no observations for those candidates. Treat this as an ablation hypothesis and keep the frozen policy as the reference.

Each formal trajectory starts with a total optimizer budget of 320. Save settled observations at the checkpoints in the frozen protocol and resume within the same budget and physical contract. The accepted checkpoints reconstruct optimizer state by replaying the recorded proposals and settlements without new worker starts. The 48 reusable 45 nmi rows count toward the formal cap of 18,192. The distance confirmations belong to qualification.
