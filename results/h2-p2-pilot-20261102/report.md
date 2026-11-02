# Six-arm RA-LSBO-H2-M prospective development experiment
100/140 km are frozen mission settings. The old sparse identifier is not a below-10% claim.
Registered seeds [11062, 11063, 11064, 11065]; exploratory, not independent confirmation. No MDA or scenario risk optimization.
| Case | Method | Nominal feasible/all | Cold success/cold | 5 s verified/all | Conditional median Wh (n) |
|---|---|---:|---:|---:|---:|
| aircraft_ordinary | corpus_stable | 4/4 | 0/0 | 4/4 | 26328.735239 (4) |
| aircraft_ordinary | h2_causal | 4/4 | 0/0 | 4/4 | 26303.723201 (4) |
| aircraft_ordinary | hybrid_guard | 4/4 | 0/0 | 4/4 | 26391.563849 (4) |
| aircraft_ordinary | hybrid_onehot | 4/4 | 0/0 | 4/4 | 26325.405111 (4) |
| aircraft_ordinary | onehot_stable | 4/4 | 0/0 | 4/4 | 26326.318114 (4) |
| aircraft_ordinary | random | 4/4 | 0/0 | 4/4 | 29097.513352 (4) |
| aircraft_sparse | corpus_stable | 4/4 | 2/2 | 4/4 | 35810.309032 (4) |
| aircraft_sparse | h2_causal | 4/4 | 2/2 | 4/4 | 34618.354857 (4) |
| aircraft_sparse | hybrid_guard | 4/4 | 2/2 | 4/4 | 37185.864639 (4) |
| aircraft_sparse | hybrid_onehot | 4/4 | 2/2 | 4/4 | 35763.233842 (4) |
| aircraft_sparse | onehot_stable | 4/4 | 2/2 | 4/4 | 35800.398200 (4) |
| aircraft_sparse | random | 3/4 | 1/2 | 3/4 | 45640.068268 (3) |

0/0 is an empty stratum, not zero probability. Verification never changes the nominal incumbent.
Missing objectives and unused verification slots are retained. No arm has native early stop. Paired objective analyses condition on both arms feasible.
Full raw/phase diagnostics, timing and paired tables are in results.json.
Calls 1007/1008; new RCAIDE MDA 0.

## Paired warm final objective (left minus right)
| Case | Left | Right | n | Better/tie/worse | Median difference Wh |
|---|---|---|---:|---:|---:|
| aircraft_ordinary | corpus_stable | h2_causal | 4 | 2/0/2 | 25.01203873920167 |
| aircraft_ordinary | corpus_stable | hybrid_guard | 4 | 3/0/1 | -42.590897861449776 |
| aircraft_ordinary | corpus_stable | hybrid_onehot | 4 | 2/0/2 | 2.174257952186963 |
| aircraft_ordinary | corpus_stable | onehot_stable | 4 | 2/0/2 | -34.396136829767784 |
| aircraft_ordinary | corpus_stable | random | 4 | 4/0/0 | -2768.7781125306337 |
| aircraft_ordinary | h2_causal | hybrid_guard | 4 | 4/0/0 | -71.46259054067195 |
| aircraft_ordinary | h2_causal | hybrid_onehot | 4 | 2/0/2 | -32.219044049244985 |
| aircraft_ordinary | h2_causal | onehot_stable | 4 | 3/0/1 | -42.9923759726953 |
| aircraft_ordinary | h2_causal | random | 4 | 4/0/0 | -2793.7901512698354 |
| aircraft_ordinary | hybrid_guard | hybrid_onehot | 4 | 2/0/2 | -7.295540374212578 |
| aircraft_ordinary | hybrid_guard | onehot_stable | 4 | 2/0/2 | 35.72627404132072 |
| aircraft_ordinary | hybrid_guard | random | 4 | 4/0/0 | -2705.9495026830464 |
| aircraft_ordinary | hybrid_onehot | onehot_stable | 4 | 2/0/2 | -10.773331923450314 |
| aircraft_ordinary | hybrid_onehot | random | 4 | 4/0/0 | -2801.8849037439886 |
| aircraft_ordinary | onehot_stable | random | 4 | 4/0/0 | -2775.6585154230997 |
| aircraft_sparse | corpus_stable | h2_causal | 2 | 1/0/1 | -246.47020798272933 |
| aircraft_sparse | corpus_stable | hybrid_guard | 2 | 1/0/1 | 664.6528197865955 |
| aircraft_sparse | corpus_stable | hybrid_onehot | 2 | 1/0/1 | -799.3804242239239 |
| aircraft_sparse | corpus_stable | onehot_stable | 2 | 1/0/1 | -198.54486660700422 |
| aircraft_sparse | corpus_stable | random | 2 | 2/0/0 | -9589.672657649084 |
| aircraft_sparse | h2_causal | hybrid_guard | 2 | 1/0/1 | 911.1230277693248 |
| aircraft_sparse | h2_causal | hybrid_onehot | 2 | 2/0/0 | -552.9102162411946 |
| aircraft_sparse | h2_causal | onehot_stable | 2 | 1/0/1 | 47.92534137572511 |
| aircraft_sparse | h2_causal | random | 2 | 2/0/0 | -9343.202449666354 |
| aircraft_sparse | hybrid_guard | hybrid_onehot | 2 | 2/0/0 | -1464.0332440105194 |
| aircraft_sparse | hybrid_guard | onehot_stable | 2 | 1/0/1 | -863.1976863935997 |
| aircraft_sparse | hybrid_guard | random | 2 | 2/0/0 | -10254.32547743568 |
| aircraft_sparse | hybrid_onehot | onehot_stable | 2 | 1/0/1 | 600.8355576169197 |
| aircraft_sparse | hybrid_onehot | random | 2 | 2/0/0 | -8790.29223342516 |
| aircraft_sparse | onehot_stable | random | 2 | 2/0/0 | -9391.12779104208 |

## Persistent training accounting
Recorded cumulative training is summed once per trial; it is already included in proposal and total optimization time.
H2 causal SOC-mixture risk with fixed prior/weight clipping, no calibration guarantee.
| Case | Seed | Fits | Gradient updates | Fit seconds | Full mean reconstruction checkpoints |
|---|---:|---:|---:|---:|---:|
| aircraft_ordinary | 11062 | 3 | 1120 | 4.012295 | 3/3 |
| aircraft_ordinary | 11062 | 3 | 1120 | 3.108630 | 3/3 |
| aircraft_ordinary | 11062 | 3 | 1120 | 2.986242 | 3/3 |
| aircraft_ordinary | 11062 | 3 | 1120 | 3.001943 | 3/3 |
| aircraft_ordinary | 11063 | 3 | 1120 | 3.029104 | 3/3 |
| aircraft_ordinary | 11063 | 3 | 1120 | 3.031808 | 3/3 |
| aircraft_ordinary | 11063 | 3 | 1120 | 3.051219 | 3/3 |
| aircraft_ordinary | 11063 | 3 | 1120 | 3.051595 | 3/3 |
| aircraft_ordinary | 11064 | 3 | 1120 | 3.083154 | 3/3 |
| aircraft_ordinary | 11064 | 3 | 1120 | 3.134449 | 3/3 |
| aircraft_ordinary | 11064 | 3 | 1120 | 3.102287 | 3/3 |
| aircraft_ordinary | 11064 | 3 | 1120 | 3.128334 | 3/3 |
| aircraft_ordinary | 11065 | 3 | 1120 | 3.102600 | 3/3 |
| aircraft_ordinary | 11065 | 3 | 1120 | 3.078031 | 3/3 |
| aircraft_ordinary | 11065 | 3 | 1120 | 3.086308 | 3/3 |
| aircraft_ordinary | 11065 | 3 | 1120 | 3.109159 | 3/3 |
| aircraft_sparse | 11062 | 3 | 1120 | 2.819379 | 3/3 |
| aircraft_sparse | 11062 | 3 | 1120 | 2.835810 | 3/3 |
| aircraft_sparse | 11062 | 3 | 1120 | 2.828964 | 3/3 |
| aircraft_sparse | 11062 | 3 | 1120 | 2.800434 | 3/3 |
| aircraft_sparse | 11063 | 3 | 1120 | 2.836718 | 3/3 |
| aircraft_sparse | 11063 | 3 | 1120 | 2.838651 | 3/3 |
| aircraft_sparse | 11063 | 3 | 1120 | 2.844858 | 3/3 |
| aircraft_sparse | 11063 | 3 | 1120 | 2.857422 | 3/3 |
| aircraft_sparse | 11064 | 3 | 1120 | 3.103655 | 3/3 |
| aircraft_sparse | 11064 | 3 | 1120 | 3.097482 | 3/3 |
| aircraft_sparse | 11064 | 3 | 1120 | 3.092530 | 3/3 |
| aircraft_sparse | 11064 | 3 | 1120 | 3.107148 | 3/3 |
| aircraft_sparse | 11065 | 3 | 1120 | 2.907675 | 3/3 |
| aircraft_sparse | 11065 | 3 | 1120 | 2.893690 | 3/3 |
| aircraft_sparse | 11065 | 3 | 1120 | 2.915574 | 3/3 |
| aircraft_sparse | 11065 | 3 | 1120 | 2.910335 | 3/3 |
