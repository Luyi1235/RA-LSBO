# Six-arm RA-LSBO-H2-M prospective development experiment
100/140 km are frozen mission settings. The old sparse identifier is not a below-10% claim.
Registered seeds [11061]; exploratory, not independent confirmation. No MDA or scenario risk optimization.
| Case | Method | Nominal feasible/all | Cold success/cold | 5 s verified/all | Conditional median Wh (n) |
|---|---|---:|---:|---:|---:|
| aircraft_ordinary | corpus_stable | 1/1 | 0/0 | 1/1 | 29073.326581 (1) |
| aircraft_ordinary | h2_causal | 1/1 | 0/0 | 1/1 | 29073.326581 (1) |
| aircraft_ordinary | hybrid_guard | 1/1 | 0/0 | 1/1 | 29073.326581 (1) |
| aircraft_ordinary | hybrid_onehot | 1/1 | 0/0 | 1/1 | 29073.326581 (1) |
| aircraft_ordinary | onehot_stable | 1/1 | 0/0 | 1/1 | 27822.689136 (1) |
| aircraft_ordinary | random | 1/1 | 0/0 | 1/1 | 29073.326581 (1) |
| aircraft_sparse | corpus_stable | 1/1 | 1/1 | 1/1 | 41995.508922 (1) |
| aircraft_sparse | h2_causal | 0/1 | 0/1 | 0/1 | NA (0) |
| aircraft_sparse | hybrid_guard | 0/1 | 0/1 | 0/1 | NA (0) |
| aircraft_sparse | hybrid_onehot | 0/1 | 0/1 | 0/1 | NA (0) |
| aircraft_sparse | onehot_stable | 0/1 | 0/1 | 0/1 | NA (0) |
| aircraft_sparse | random | 0/1 | 0/1 | 0/1 | NA (0) |

0/0 is an empty stratum, not zero probability. Verification never changes the nominal incumbent.
Missing objectives and unused verification slots are retained. No arm has native early stop. Paired objective analyses condition on both arms feasible.
Full raw/phase diagnostics, timing and paired tables are in results.json.
Calls 43/48; new RCAIDE MDA 0.

## Paired warm final objective (left minus right)
| Case | Left | Right | n | Better/tie/worse | Median difference Wh |
|---|---|---|---:|---:|---:|
| aircraft_ordinary | corpus_stable | h2_causal | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | corpus_stable | hybrid_guard | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | corpus_stable | hybrid_onehot | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | corpus_stable | onehot_stable | 1 | 0/0/1 | 1250.637444688251 |
| aircraft_ordinary | corpus_stable | random | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | h2_causal | hybrid_guard | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | h2_causal | hybrid_onehot | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | h2_causal | onehot_stable | 1 | 0/0/1 | 1250.637444688251 |
| aircraft_ordinary | h2_causal | random | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | hybrid_guard | hybrid_onehot | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | hybrid_guard | onehot_stable | 1 | 0/0/1 | 1250.637444688251 |
| aircraft_ordinary | hybrid_guard | random | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | hybrid_onehot | onehot_stable | 1 | 0/0/1 | 1250.637444688251 |
| aircraft_ordinary | hybrid_onehot | random | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | onehot_stable | random | 1 | 1/0/0 | -1250.637444688251 |
| aircraft_sparse | corpus_stable | h2_causal | 0 | 0/0/0 | None |
| aircraft_sparse | corpus_stable | hybrid_guard | 0 | 0/0/0 | None |
| aircraft_sparse | corpus_stable | hybrid_onehot | 0 | 0/0/0 | None |
| aircraft_sparse | corpus_stable | onehot_stable | 0 | 0/0/0 | None |
| aircraft_sparse | corpus_stable | random | 0 | 0/0/0 | None |
| aircraft_sparse | h2_causal | hybrid_guard | 0 | 0/0/0 | None |
| aircraft_sparse | h2_causal | hybrid_onehot | 0 | 0/0/0 | None |
| aircraft_sparse | h2_causal | onehot_stable | 0 | 0/0/0 | None |
| aircraft_sparse | h2_causal | random | 0 | 0/0/0 | None |
| aircraft_sparse | hybrid_guard | hybrid_onehot | 0 | 0/0/0 | None |
| aircraft_sparse | hybrid_guard | onehot_stable | 0 | 0/0/0 | None |
| aircraft_sparse | hybrid_guard | random | 0 | 0/0/0 | None |
| aircraft_sparse | hybrid_onehot | onehot_stable | 0 | 0/0/0 | None |
| aircraft_sparse | hybrid_onehot | random | 0 | 0/0/0 | None |
| aircraft_sparse | onehot_stable | random | 0 | 0/0/0 | None |

## Persistent training accounting
Recorded cumulative training is summed once per trial; it is already included in proposal and total optimization time.
H2 causal SOC-mixture risk with fixed prior/weight clipping, no calibration guarantee.
| Case | Seed | Fits | Gradient updates | Fit seconds | Full mean reconstruction checkpoints |
|---|---:|---:|---:|---:|---:|
| aircraft_ordinary | 11061 | 1 | 480 | 2.173510 | 1/1 |
| aircraft_ordinary | 11061 | 1 | 480 | 1.262193 | 1/1 |
| aircraft_ordinary | 11061 | 1 | 480 | 1.229074 | 1/1 |
| aircraft_ordinary | 11061 | 1 | 480 | 1.221609 | 1/1 |
| aircraft_sparse | 11061 | 1 | 480 | 1.060661 | 1/1 |
| aircraft_sparse | 11061 | 1 | 480 | 1.065986 | 1/1 |
| aircraft_sparse | 11061 | 1 | 480 | 1.075022 | 1/1 |
| aircraft_sparse | 11061 | 1 | 480 | 1.117069 | 1/1 |
