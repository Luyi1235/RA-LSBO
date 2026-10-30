# Nominal reduced-aircraft exploratory comparison
100/140 km are frozen mission settings. The old sparse identifier is not a below-10% claim.
Eight pilot seeds are development evidence, not independent confirmation. No MDA or scenario risk optimization.
| Case | Method | Nominal feasible/all | Cold success/cold | 5 s verified/all | Conditional median Wh (n) |
|---|---|---:|---:|---:|---:|
| aircraft_ordinary | ev_latent | 1/1 | 0/0 | 1/1 | 30593.197306 (1) |
| aircraft_ordinary | ev_onehot | 1/1 | 0/0 | 1/1 | 31835.851768 (1) |
| aircraft_ordinary | ev_random_then_latent | 1/1 | 0/0 | 1/1 | 30593.197306 (1) |
| aircraft_ordinary | original_lsbo | 1/1 | 0/0 | 1/1 | 26904.384545 (1) |
| aircraft_ordinary | phase_exact | 1/1 | 0/0 | 1/1 | 30593.197306 (1) |
| aircraft_ordinary | phase_sign | 1/1 | 0/0 | 1/1 | 30593.197306 (1) |
| aircraft_ordinary | random | 1/1 | 0/0 | 1/1 | 31686.318470 (1) |
| aircraft_sparse | ev_latent | 1/1 | 0/0 | 1/1 | 40611.570362 (1) |
| aircraft_sparse | ev_onehot | 1/1 | 0/0 | 1/1 | 40611.570362 (1) |
| aircraft_sparse | ev_random_then_latent | 1/1 | 0/0 | 1/1 | 40611.570362 (1) |
| aircraft_sparse | original_lsbo | 1/1 | 0/0 | 1/1 | 40463.897850 (1) |
| aircraft_sparse | phase_exact | 1/1 | 0/0 | 1/1 | 40611.570362 (1) |
| aircraft_sparse | phase_sign | 1/1 | 0/0 | 1/1 | 40611.570362 (1) |
| aircraft_sparse | random | 1/1 | 0/0 | 1/1 | 40611.570362 (1) |

0/0 is an empty stratum, not zero probability. Verification never changes the nominal incumbent.
All native stops, missing objectives and unused calls are retained. Paired objective analyses condition on both arms feasible.
Full raw/phase diagnostics, timing and paired tables are in results.json.
Calls 54/54; new RCAIDE MDA 0.

## Paired warm final objective (left minus right)
| Case | Left | Right | n | Better/tie/worse | Median difference Wh |
|---|---|---|---:|---:|---:|
| aircraft_ordinary | ev_latent | ev_onehot | 1 | 1/0/0 | -1242.6544616666324 |
| aircraft_ordinary | ev_latent | ev_random_then_latent | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | ev_latent | original_lsbo | 1 | 0/0/1 | 3688.812761438072 |
| aircraft_ordinary | ev_latent | phase_exact | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | ev_latent | phase_sign | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | ev_latent | random | 1 | 1/0/0 | -1093.1211640794209 |
| aircraft_ordinary | ev_onehot | ev_random_then_latent | 1 | 0/0/1 | 1242.6544616666324 |
| aircraft_ordinary | ev_onehot | original_lsbo | 1 | 0/0/1 | 4931.467223104704 |
| aircraft_ordinary | ev_onehot | phase_exact | 1 | 0/0/1 | 1242.6544616666324 |
| aircraft_ordinary | ev_onehot | phase_sign | 1 | 0/0/1 | 1242.6544616666324 |
| aircraft_ordinary | ev_onehot | random | 1 | 0/0/1 | 149.53329758721156 |
| aircraft_ordinary | ev_random_then_latent | original_lsbo | 1 | 0/0/1 | 3688.812761438072 |
| aircraft_ordinary | ev_random_then_latent | phase_exact | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | ev_random_then_latent | phase_sign | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | ev_random_then_latent | random | 1 | 1/0/0 | -1093.1211640794209 |
| aircraft_ordinary | original_lsbo | phase_exact | 1 | 1/0/0 | -3688.812761438072 |
| aircraft_ordinary | original_lsbo | phase_sign | 1 | 1/0/0 | -3688.812761438072 |
| aircraft_ordinary | original_lsbo | random | 1 | 1/0/0 | -4781.933925517493 |
| aircraft_ordinary | phase_exact | phase_sign | 1 | 0/1/0 | 0.0 |
| aircraft_ordinary | phase_exact | random | 1 | 1/0/0 | -1093.1211640794209 |
| aircraft_ordinary | phase_sign | random | 1 | 1/0/0 | -1093.1211640794209 |
| aircraft_sparse | ev_latent | ev_onehot | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_latent | ev_random_then_latent | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_latent | original_lsbo | 1 | 0/0/1 | 147.67251113199745 |
| aircraft_sparse | ev_latent | phase_exact | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_latent | phase_sign | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_latent | random | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_onehot | ev_random_then_latent | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_onehot | original_lsbo | 1 | 0/0/1 | 147.67251113199745 |
| aircraft_sparse | ev_onehot | phase_exact | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_onehot | phase_sign | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_onehot | random | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_random_then_latent | original_lsbo | 1 | 0/0/1 | 147.67251113199745 |
| aircraft_sparse | ev_random_then_latent | phase_exact | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_random_then_latent | phase_sign | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | ev_random_then_latent | random | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | original_lsbo | phase_exact | 1 | 1/0/0 | -147.67251113199745 |
| aircraft_sparse | original_lsbo | phase_sign | 1 | 1/0/0 | -147.67251113199745 |
| aircraft_sparse | original_lsbo | random | 1 | 1/0/0 | -147.67251113199745 |
| aircraft_sparse | phase_exact | phase_sign | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | phase_exact | random | 1 | 0/1/0 | 0.0 |
| aircraft_sparse | phase_sign | random | 1 | 0/1/0 | 0.0 |
