# Nominal reduced-aircraft exploratory comparison
100/140 km are frozen mission settings. The old sparse identifier is not a below-10% claim.
Eight pilot seeds are development evidence, not independent confirmation. No MDA or scenario risk optimization.
| Case | Method | Nominal feasible/all | Cold success/cold | 5 s verified/all | Conditional median Wh (n) |
|---|---|---:|---:|---:|---:|
| aircraft_ordinary | ev_latent | 8/8 | 1/1 | 8/8 | 28199.401685 (8) |
| aircraft_ordinary | ev_onehot | 8/8 | 1/1 | 8/8 | 26244.281053 (8) |
| aircraft_ordinary | ev_random_then_latent | 8/8 | 1/1 | 8/8 | 28199.401685 (8) |
| aircraft_ordinary | original_lsbo | 8/8 | 1/1 | 8/8 | 26260.605646 (8) |
| aircraft_ordinary | phase_exact | 8/8 | 1/1 | 8/8 | 28704.026084 (8) |
| aircraft_ordinary | phase_sign | 8/8 | 1/1 | 8/8 | 27090.337390 (8) |
| aircraft_ordinary | random | 8/8 | 1/1 | 8/8 | 29957.277009 (8) |
| aircraft_sparse | ev_latent | 8/8 | 2/2 | 8/8 | 37666.122316 (8) |
| aircraft_sparse | ev_onehot | 8/8 | 2/2 | 8/8 | 33471.303715 (8) |
| aircraft_sparse | ev_random_then_latent | 8/8 | 2/2 | 8/8 | 37138.024894 (8) |
| aircraft_sparse | original_lsbo | 8/8 | 2/2 | 8/8 | 35535.543737 (8) |
| aircraft_sparse | phase_exact | 8/8 | 2/2 | 8/8 | 40427.555557 (8) |
| aircraft_sparse | phase_sign | 8/8 | 2/2 | 8/8 | 39350.953411 (8) |
| aircraft_sparse | random | 8/8 | 2/2 | 8/8 | 39823.553099 (8) |

0/0 is an empty stratum, not zero probability. Verification never changes the nominal incumbent.
All native stops, missing objectives and unused calls are retained. Paired objective analyses condition on both arms feasible.
Full raw/phase diagnostics, timing and paired tables are in results.json.
Calls 3664/3664; new RCAIDE MDA 0.

## Paired warm final objective (left minus right)
| Case | Left | Right | n | Better/tie/worse | Median difference Wh |
|---|---|---|---:|---:|---:|
| aircraft_ordinary | ev_latent | ev_onehot | 7 | 0/0/7 | 1973.342321731263 |
| aircraft_ordinary | ev_latent | ev_random_then_latent | 7 | 0/7/0 | 0.0 |
| aircraft_ordinary | ev_latent | original_lsbo | 7 | 0/0/7 | 1973.1996468852994 |
| aircraft_ordinary | ev_latent | phase_exact | 7 | 4/0/3 | -151.70553260739325 |
| aircraft_ordinary | ev_latent | phase_sign | 7 | 3/0/4 | 697.4227590662122 |
| aircraft_ordinary | ev_latent | random | 7 | 6/0/1 | -1893.333729453887 |
| aircraft_ordinary | ev_onehot | ev_random_then_latent | 7 | 7/0/0 | -1973.342321731263 |
| aircraft_ordinary | ev_onehot | original_lsbo | 7 | 5/0/2 | -13.256130237412435 |
| aircraft_ordinary | ev_onehot | phase_exact | 7 | 7/0/0 | -3164.0212821222594 |
| aircraft_ordinary | ev_onehot | phase_sign | 7 | 7/0/0 | -1102.2265095918956 |
| aircraft_ordinary | ev_onehot | random | 7 | 7/0/0 | -3865.5399210819887 |
| aircraft_ordinary | ev_random_then_latent | original_lsbo | 7 | 0/0/7 | 1973.1996468852994 |
| aircraft_ordinary | ev_random_then_latent | phase_exact | 7 | 4/0/3 | -151.70553260739325 |
| aircraft_ordinary | ev_random_then_latent | phase_sign | 7 | 3/0/4 | 697.4227590662122 |
| aircraft_ordinary | ev_random_then_latent | random | 7 | 6/0/1 | -1893.333729453887 |
| aircraft_ordinary | original_lsbo | phase_exact | 7 | 6/0/1 | -3143.2294460218727 |
| aircraft_ordinary | original_lsbo | phase_sign | 7 | 7/0/0 | -1088.9703793544832 |
| aircraft_ordinary | original_lsbo | random | 7 | 7/0/0 | -3832.913853000475 |
| aircraft_ordinary | phase_exact | phase_sign | 7 | 4/0/3 | -349.04319488402325 |
| aircraft_ordinary | phase_exact | random | 7 | 4/0/3 | -180.13960502132613 |
| aircraft_ordinary | phase_sign | random | 7 | 4/0/3 | -974.548668510597 |
| aircraft_sparse | ev_latent | ev_onehot | 6 | 0/0/6 | 2878.87671859701 |
| aircraft_sparse | ev_latent | ev_random_then_latent | 6 | 0/6/0 | 0.0 |
| aircraft_sparse | ev_latent | original_lsbo | 6 | 1/0/5 | 1752.9086766345645 |
| aircraft_sparse | ev_latent | phase_exact | 6 | 6/0/0 | -2446.7024065302357 |
| aircraft_sparse | ev_latent | phase_sign | 6 | 5/0/1 | -2114.210165510747 |
| aircraft_sparse | ev_latent | random | 6 | 6/0/0 | -2760.674582486976 |
| aircraft_sparse | ev_onehot | ev_random_then_latent | 6 | 6/0/0 | -2878.87671859701 |
| aircraft_sparse | ev_onehot | original_lsbo | 6 | 4/0/2 | -1025.6267291376862 |
| aircraft_sparse | ev_onehot | phase_exact | 6 | 6/0/0 | -5974.190483530223 |
| aircraft_sparse | ev_onehot | phase_sign | 6 | 6/0/0 | -5862.635810554522 |
| aircraft_sparse | ev_onehot | random | 6 | 6/0/0 | -5518.573305165828 |
| aircraft_sparse | ev_random_then_latent | original_lsbo | 6 | 1/0/5 | 1752.9086766345645 |
| aircraft_sparse | ev_random_then_latent | phase_exact | 6 | 6/0/0 | -2446.7024065302357 |
| aircraft_sparse | ev_random_then_latent | phase_sign | 6 | 5/0/1 | -2114.210165510747 |
| aircraft_sparse | ev_random_then_latent | random | 6 | 6/0/0 | -2760.674582486976 |
| aircraft_sparse | original_lsbo | phase_exact | 6 | 6/0/0 | -5492.902859318263 |
| aircraft_sparse | original_lsbo | phase_sign | 6 | 6/0/0 | -4711.624245399424 |
| aircraft_sparse | original_lsbo | random | 6 | 6/0/0 | -4835.2992150289065 |
| aircraft_sparse | phase_exact | phase_sign | 6 | 1/0/5 | 725.3822244364965 |
| aircraft_sparse | phase_exact | random | 6 | 2/1/3 | 229.07033080409383 |
| aircraft_sparse | phase_sign | random | 6 | 3/0/3 | -750.5651003669227 |
