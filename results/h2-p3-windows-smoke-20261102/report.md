# Two-arm RA-LSBO-H2 P3 mixture ablation
100/140 km are frozen mission settings. The old sparse identifier is not a below-10% claim.
Registered seeds [12070]; exploratory, not independent confirmation. No MDA or scenario risk optimization.
| Case | Method | Nominal feasible/all | Cold success/cold | 5 s verified/all | Conditional median Wh (n) |
|---|---|---:|---:|---:|---:|
| aircraft_ordinary | h2_dynamic | 1/1 | 0/0 | 1/1 | 30931.592280 (1) |
| aircraft_ordinary | h2_fixed50 | 1/1 | 0/0 | 1/1 | 30931.592280 (1) |
| aircraft_sparse | h2_dynamic | 1/1 | 0/0 | 1/1 | 39030.198851 (1) |
| aircraft_sparse | h2_fixed50 | 1/1 | 0/0 | 1/1 | 39030.198851 (1) |

0/0 is an empty stratum, not zero probability. Verification never changes the nominal incumbent.
Missing objectives and unused verification slots are retained. No arm has native early stop. Paired objective analyses condition on both arms feasible.
Full raw/phase diagnostics, timing and paired tables are in results.json.
Calls 24/24; new RCAIDE MDA 0.

## Paired warm final objective (left minus right)
| Case | Left | Right | n | Better/tie/worse | Median difference Wh |
|---|---|---|---:|---:|---:|
| aircraft_ordinary | h2_dynamic | h2_fixed50 | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | h2_dynamic | h2_fixed50 | 1 | 0/1/0 | 0.0 |

## All seed-paired outcomes: dynamic minus fixed
| Case | Seed | Initial | Dynamic feasible | Fixed feasible | Dynamic Wh | Fixed Wh | Difference Wh |
|---|---:|---|---|---|---:|---:|---:|
| aircraft_ordinary | 12070 | warm | True | True | 30931.592280470264 | 30931.592280470264 | 0.0 |
| aircraft_sparse | 12070 | warm | True | True | 39030.19885052945 | 39030.19885052945 | 0.0 |
aircraft_ordinary: both feasible n=1, mean=0.0, median=0.0.
aircraft_sparse: both feasible n=1, mean=0.0, median=0.0.

## Persistent training accounting
Recorded cumulative training is summed once per trial; it is already included in proposal and total optimization time.
H2 causal SOC-mixture risk with fixed prior/weight clipping, no calibration guarantee.
| Case | Seed | Fits | Gradient updates | Fit seconds | Full mean reconstruction checkpoints |
|---|---:|---:|---:|---:|---:|
| aircraft_ordinary | 12070 | 1 | 480 | 2.103906 | 1/1 |
| aircraft_ordinary | 12070 | 1 | 480 | 1.304673 | 1/1 |
| aircraft_sparse | 12070 | 1 | 480 | 1.148045 | 1/1 |
| aircraft_sparse | 12070 | 1 | 480 | 1.141322 | 1/1 |
