# Two-arm RA-LSBO-H2 P3 mixture ablation
100/140 km are frozen mission settings. The old sparse identifier is not a below-10% claim.
Registered seeds [12071, 12072, 12073, 12074, 12075, 12076, 12077, 12078]; exploratory, not independent confirmation. No MDA or scenario risk optimization.
| Case | Method | Nominal feasible/all | Cold success/cold | 5 s verified/all | Conditional median Wh (n) |
|---|---|---:|---:|---:|---:|
| aircraft_ordinary | h2_dynamic | 8/8 | 0/0 | 8/8 | 26315.132118 (8) |
| aircraft_ordinary | h2_fixed50 | 8/8 | 0/0 | 8/8 | 26372.032217 (8) |
| aircraft_sparse | h2_dynamic | 8/8 | 3/3 | 8/8 | 33463.130040 (8) |
| aircraft_sparse | h2_fixed50 | 8/8 | 3/3 | 8/8 | 33583.012176 (8) |

0/0 is an empty stratum, not zero probability. Verification never changes the nominal incumbent.
Missing objectives and unused verification slots are retained. No arm has native early stop. Paired objective analyses condition on both arms feasible.
Full raw/phase diagnostics, timing and paired tables are in results.json.
Calls 800/800; new RCAIDE MDA 0.

## Paired warm final objective (left minus right)
| Case | Left | Right | n | Better/tie/worse | Median difference Wh |
|---|---|---|---:|---:|---:|
| aircraft_ordinary | h2_dynamic | h2_fixed50 | 8 | 5/0/3 | -0.5581698839268938 |
| aircraft_sparse | h2_dynamic | h2_fixed50 | 5 | 2/0/3 | 0.8840660482092062 |

## All seed-paired outcomes: dynamic minus fixed
| Case | Seed | Initial | Dynamic feasible | Fixed feasible | Dynamic Wh | Fixed Wh | Difference Wh |
|---|---:|---|---|---|---:|---:|---:|
| aircraft_ordinary | 12071 | warm | True | True | 26436.092143342394 | 26435.998862182158 | 0.09328116023607436 |
| aircraft_ordinary | 12072 | warm | True | True | 26284.60770780751 | 26551.729716405564 | -267.12200859805307 |
| aircraft_ordinary | 12073 | warm | True | True | 26333.85134423244 | 26389.727272949454 | -55.875928717014176 |
| aircraft_ordinary | 12074 | warm | True | True | 26273.15944915296 | 26256.81665737393 | 16.342791779032268 |
| aircraft_ordinary | 12075 | warm | True | True | 26296.412890965472 | 26346.480691539884 | -50.067800574412104 |
| aircraft_ordinary | 12076 | warm | True | True | 26280.348608583226 | 26262.28406057424 | 18.064548008987913 |
| aircraft_ordinary | 12077 | warm | True | True | 26353.342547805107 | 26354.337160916428 | -0.9946131113210868 |
| aircraft_ordinary | 12078 | warm | True | True | 26654.600655465852 | 26654.722382122385 | -0.12172665653270087 |
| aircraft_sparse | 12071 | warm | True | True | 33365.36452369213 | 33375.28637812981 | -9.921854437678121 |
| aircraft_sparse | 12072 | cold | True | True | 33468.480672941165 | 33437.49132501342 | 30.989347927745257 |
| aircraft_sparse | 12073 | warm | True | True | 36176.6779879553 | 36176.80351406891 | -0.125526113610249 |
| aircraft_sparse | 12074 | warm | True | True | 33457.77940624218 | 33378.54889439709 | 79.2305118450895 |
| aircraft_sparse | 12075 | cold | True | True | 33351.05797399589 | 35518.37098649095 | -2167.313012495055 |
| aircraft_sparse | 12076 | warm | True | True | 33729.41709207494 | 33728.53302602673 | 0.8840660482092062 |
| aircraft_sparse | 12077 | cold | True | True | 35871.538936663244 | 35911.901050982444 | -40.36211431919946 |
| aircraft_sparse | 12078 | warm | True | True | 33420.41939377561 | 33417.066612391245 | 3.352781384368427 |
aircraft_ordinary: both feasible n=8, mean=-42.46018208863461, median=-0.5581698839268938.
aircraft_sparse: both feasible n=8, mean=-262.9082250200163, median=0.3792699672994786.

## Persistent training accounting
Recorded cumulative training is summed once per trial; it is already included in proposal and total optimization time.
H2 causal SOC-mixture risk with fixed prior/weight clipping, no calibration guarantee.
| Case | Seed | Fits | Gradient updates | Fit seconds | Full mean reconstruction checkpoints |
|---|---:|---:|---:|---:|---:|
| aircraft_ordinary | 12071 | 3 | 1120 | 3.858770 | 3/3 |
| aircraft_ordinary | 12071 | 3 | 1120 | 3.063203 | 3/3 |
| aircraft_ordinary | 12072 | 3 | 1120 | 2.964906 | 3/3 |
| aircraft_ordinary | 12072 | 3 | 1120 | 2.959412 | 3/3 |
| aircraft_ordinary | 12073 | 3 | 1120 | 2.990428 | 3/3 |
| aircraft_ordinary | 12073 | 3 | 1120 | 2.978710 | 3/3 |
| aircraft_ordinary | 12074 | 3 | 1120 | 3.002982 | 3/3 |
| aircraft_ordinary | 12074 | 3 | 1120 | 2.983323 | 3/3 |
| aircraft_ordinary | 12075 | 3 | 1120 | 3.008567 | 3/3 |
| aircraft_ordinary | 12075 | 3 | 1120 | 3.014301 | 3/3 |
| aircraft_ordinary | 12076 | 3 | 1120 | 2.991295 | 3/3 |
| aircraft_ordinary | 12076 | 3 | 1120 | 3.008841 | 3/3 |
| aircraft_ordinary | 12077 | 3 | 1120 | 3.064129 | 3/3 |
| aircraft_ordinary | 12077 | 3 | 1120 | 3.050725 | 3/3 |
| aircraft_ordinary | 12078 | 3 | 1120 | 3.118827 | 3/3 |
| aircraft_ordinary | 12078 | 3 | 1120 | 3.112893 | 3/3 |
| aircraft_sparse | 12071 | 3 | 1120 | 3.061463 | 3/3 |
| aircraft_sparse | 12071 | 3 | 1120 | 3.060769 | 3/3 |
| aircraft_sparse | 12072 | 3 | 1120 | 2.815779 | 3/3 |
| aircraft_sparse | 12072 | 3 | 1120 | 2.802978 | 3/3 |
| aircraft_sparse | 12073 | 3 | 1120 | 2.870582 | 3/3 |
| aircraft_sparse | 12073 | 3 | 1120 | 2.853815 | 3/3 |
| aircraft_sparse | 12074 | 3 | 1120 | 2.971874 | 3/3 |
| aircraft_sparse | 12074 | 3 | 1120 | 2.971353 | 3/3 |
| aircraft_sparse | 12075 | 3 | 1120 | 2.909673 | 3/3 |
| aircraft_sparse | 12075 | 3 | 1120 | 2.920778 | 3/3 |
| aircraft_sparse | 12076 | 3 | 1120 | 2.989272 | 3/3 |
| aircraft_sparse | 12076 | 3 | 1120 | 3.000994 | 3/3 |
| aircraft_sparse | 12077 | 3 | 1120 | 3.090271 | 3/3 |
| aircraft_sparse | 12077 | 3 | 1120 | 3.069960 | 3/3 |
| aircraft_sparse | 12078 | 3 | 1120 | 3.088992 | 3/3 |
| aircraft_sparse | 12078 | 3 | 1120 | 3.093880 | 3/3 |
