# Windows full structural-corpus offline validation

Execution commit: `08eed076f54fc6ed155f285a9bb695ff6d857bf8`.
Branch providing the pinned implementation: `research/structure-corpus-2000-20261031`.
An independent detached worktree was used; tracked execution sources remained clean.

The ZIP path in the request referred to the preceding representation handoff.
The desktop also contained `RA_LSBO_structure_corpus_2000_handoff_20261031.zip`;
its named guide, commit and scope matched this request exactly. That handoff was
used without reimplementing patches. Its SHA-256 is
`73e9d885a3602861724f0246fff4fb7f7765b55299d9c123e0a6c780b6c04b9e`.
The guide, protocol, validation note and delivery manifest are retained in `handoff/`.

## Environment and execution

Original Windows / Rcaide_vsp: Python 3.9.25, NumPy 2.0.2, SciPy 1.13.1,
PyTorch 2.8.0+cpu, GPyTorch 1.11, BoTorch 0.10.0, scikit-learn 1.6.1.
The full package snapshot matches the previous verified pilot environment.
No environment installation or modification was performed.

- 78 regression tests passed; no failures, errors or skips (JUnit retained).
- Exactly one full study; no additional smoke or retries.
- 160 encoder fit calls, 32 one-hot transforms, 320 GP head fit calls.
- All 2,560 registered GP kernel candidates were attempted: 2,541 `ok`,
  19 `RuntimeError`. The original candidate failure statuses are retained.
  All 320 selected GP heads have status `ok`; no failed candidate was retried.
- 160 case/seed/prefix/arm rows, five arms, two cases, eight fixed seeds.
- 80 existing historical designs, each scored by both heads for each arm.
- New aircraft/math evaluator calls: 0. RCAIDE calls: 0. Optimizer runs: 0.

The read-only output audit passed. The input ZIP SHA-256 was checked before and
after execution and is unchanged:
`35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd`.
No historical plans, raw records, source snapshots, locks or hashes were edited.
The new execution lock and complete fit-attempt ledger are retained.

## Identity and structure

Only old30 rows with a nonempty historical feature identity are compared:
29/29 match. Three original missing identities remain excluded, with no hash
substitution. All five arms used matched histories and prediction targets.
The 2,000 structural draws retained repetitions; no seeds or epochs were changed.

| Arm | Prefixes | Mean raw exact active reconstruction | Mean sampled reconstruction | Prefixes meeting mean grammar target |
| --- | ---: | ---: | ---: | ---: |
| old30 | 32 | 13.28125% | 12.109375% | 0/32 |
| old480 | 32 | 99.739583% | 99.739583% | 31/32 |
| corpus_reset | 32 | 100% | 100% | 32/32 |
| corpus_persistent | 32 | 100% | 100% | 32/32 |
| onehot | 32 | N/A | N/A | N/A |

These are training-grammar reconstruction results, not held-out generalization.
The strong old480 result means additional training is a material explanation;
reconstruction alone does not establish a distinct corpus mechanism benefit.

## Matched-history SOC forecasts

Below are per-design descriptive mean NLL values, grouped by case and observation
kind. Exact NLL is a predictive density score; interval NLL is a probability-mass
score. The columns must not be pooled into one ranking. Lower is better within a
column. Full objective/SOC scores, RMSE and Brier measurements are retained in
`local_summary.json`, `full/scores.json`, `full/rows.jsonl` and per-arm JSON files.

| Arm | Ordinary exact (n=37) | Ordinary interval (n=3) | Sparse exact (n=21) | Sparse interval (n=19) |
| --- | ---: | ---: | ---: | ---: |
| old30 | -0.136506 | 1.070015 | 3.248069 | 1.103572 |
| old480 | -0.552402 | 0.899059 | -0.571193 | 0.716148 |
| corpus_reset | -0.532065 | 0.966132 | -0.490176 | 0.946673 |
| corpus_persistent | -0.562613 | 0.965651 | -0.470895 | 0.918171 |
| onehot | -0.890915 | 0.483629 | -0.622092 | 0.349087 |

Corpus training improves structure recovery over old30, while these SOC NLL
aggregates still favor onehot. Prediction benefits depend on the metric and
observation type; a uniform advantage of the new representation is not established.
These historical adaptively selected targets are not independent confirmation or
counterfactual BO trajectories. No closed-loop improvement was tested.

## Training accounting

Reset training: 32 distinct checkpoint events, 40.1269045 seconds, 15,360 gradient
updates. Persistent training: 64 distinct events, 64.6796728 seconds, 23,040 updates.
Only the complete p29 event sequence is counted for each persistent trial; its
overlapping p0 checkpoint is not counted again. Every training event and duration
is included in `local_summary.json` and original checkpoint metadata.

old30/old480 feature preparation totals are 2.167/29.485 seconds, including fit
and setup; their gradient update totals are 960/15,360. These are not separate
pure-gradient timings. Persistent snapshot preparation (0.410 seconds) excludes
the preceding training and must not be substituted for its training cost.
Equal gradient steps do not imply matched batch work, MMD cost or wall time.
The persistent suffix has more cumulative training and does not isolate the
effect of incremental learning.

## Delivery and stop

The complete new output root is in the Release archive, including numeric NPZ
checkpoints, metadata, plan, audit, all rows, all prediction scores, fit attempts,
execution lock, environment, platform/build details, JUnit and console logs.
The original 176 MB pilot ZIP is not duplicated. The results branch contains
the report and reviewable JSON evidence; the Release retains every new file.
Only the named frozen study and its read-only audit were run. No new optimizer,
old pilot, parameter search or RCAIDE run was started. Execution stopped here.
