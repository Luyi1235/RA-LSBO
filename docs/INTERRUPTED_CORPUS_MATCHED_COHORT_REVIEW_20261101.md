# Read-only interrupted corpus pilot: complete matched seed cohorts

**Evidence source:** immutable 2026-10-31 116MB interrupted pilot Release
archive SHA256 aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9.

**Audit code:** tools/audit_interrupted_pilot_cohorts.py. GitHub Actions
run 41312433411, artifact partial-pilot-complete-cohort-description
(artifact id 13085902049; ZIP digest
6c821ae740cef548499574f11498575d62f56ca0089392acfb7b2caedacff5f0).
No evaluator, VAE, GP or BO call used.

This is **not** the completed five-arm, two-task, eight-seed experiment.
There are exactly 65 completed summaries of 80 registered, one partial old30
trial (aircraft_sparse/9936, 29 online calls), and 14 entirely unstarted
trials. Use ONLY 13 complete matched five-arm case-seed sets: eight 100km
seeds 9931–9938 and five 140km seeds 9931–9935. Historical adaptively
selected candidates and optional same-incumbent 5s checks retain their own
limits. Missing trials are not imputed. No formal full-pilot method ranking.

| Method | 100 km 8-seed mean final feasible energy Wh | 140 km 5-seed mean Wh | 100 km usable online total (out of 240) | 140 km usable online total (out of 150) |
|---|---:|---:|---:|---:|
| random | 29,488.77 | 41,620.26 | 114 | 17 |
| old30 | 27,919.42 | 35,866.80 | 186 | 65 |
| old480 | 26,275.53 | 34,243.11 | 144 | 73 |
| onehot | 26,255.36 | 33,843.85 | 163 | 84 |
| corpus_persistent | 26,519.71 | 33,802.96 | 150 | 71 |

All complete matched trial summaries have feasible nominal incumbents; all
their same-design 5-second checks report usable feasibility (8 per arm at
100km and 5 per arm at 140km). This does not imply validated physical
safety or that detailed MDA was performed. RCAIDE calls were zero.

On a seed-matched basis, the 100km onehot minus corpus final feasible
energy differences average −264.35 Wh (median −56.94), with onehot lower
for 7/8 matched seeds. In the 140km 5-seed completed cohort the same
difference averages +40.89 Wh (median −3.11); onehot lower on 3/5,
corpus lower on 2/5. At 140km the mean nearly cancels large opposing
per-seed differences: approximately [−14.5,−2,229.9,−3.1,+2,184.4,+267.6]
Wh. Therefore do not present the +40.89Wh mean as robust evidence for
corpus superiority.

The matched original structure-corps VAE and onehot energy results plus
old30's severe SOC predictive miscalibration support a *new hypothesis*:
an objective latent representation may help transfer structure, while
constraint-specific explicit hierarchical features may improve risk
inference. The new 2026-11-01 prospective risk-decoupled study (seeds
9961–9964, five arms, 856-call ceiling) tests that **separate question**.
It does not reuse, resume or overwrite these 2,212 already-settled calls.
