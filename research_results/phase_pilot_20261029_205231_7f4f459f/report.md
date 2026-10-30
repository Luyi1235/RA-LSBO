# Staged study: phase (pilot)

Exploratory only; failure, unknown and native early stops are retained. No MDA.
Calls: 8916/8916; unused calls are not reassigned.

| Case | Method | Feasible/all | Cold discovery/cold | Warm improved/warm |
|---|---|---:|---:|---:|
| branin_sparse | ra_coverage | 20/20 | 9/9 | 11/11 |
| branin_sparse | ra_discovery | 19/20 | 8/9 | 11/11 |
| branin_sparse | ra_guard_only | 12/20 | 1/9 | 11/11 |
| branin_sparse | ra_phase_coverage | 20/20 | 9/9 | 11/11 |
| branin_sparse | ra_phase_discovery | 19/20 | 8/9 | 11/11 |
| branin_sparse | ra_v2 | 12/20 | 1/9 | 11/11 |
| branin_sparse | random | 20/20 | 9/9 | 9/11 |
| rosenbrock_sparse | ra_coverage | 19/20 | 5/6 | 14/14 |
| rosenbrock_sparse | ra_discovery | 20/20 | 6/6 | 14/14 |
| rosenbrock_sparse | ra_guard_only | 17/20 | 3/6 | 14/14 |
| rosenbrock_sparse | ra_phase_coverage | 19/20 | 5/6 | 14/14 |
| rosenbrock_sparse | ra_phase_discovery | 20/20 | 6/6 | 14/14 |
| rosenbrock_sparse | ra_v2 | 17/20 | 3/6 | 14/14 |
| rosenbrock_sparse | random | 20/20 | 6/6 | 8/14 |

## Paired warm-start objective (left minus right; negative is better)

| Case | Left | Right | n | Left better/tie/worse | Median paired difference |
|---|---|---|---:|---:|---:|
| branin_sparse | ra_coverage | ra_discovery | 11 | 0/11/0 | 0.0 |
| branin_sparse | ra_coverage | ra_guard_only | 11 | 7/0/4 | -0.14242398452079175 |
| branin_sparse | ra_coverage | ra_phase_coverage | 11 | 7/0/4 | -0.14242398452079175 |
| branin_sparse | ra_coverage | ra_phase_discovery | 11 | 7/0/4 | -0.14242398452079175 |
| branin_sparse | ra_coverage | ra_v2 | 11 | 7/0/4 | -0.14242398452079175 |
| branin_sparse | ra_coverage | random | 11 | 10/0/1 | -5.755790892273198 |
| branin_sparse | ra_discovery | ra_guard_only | 11 | 7/0/4 | -0.14242398452079175 |
| branin_sparse | ra_discovery | ra_phase_coverage | 11 | 7/0/4 | -0.14242398452079175 |
| branin_sparse | ra_discovery | ra_phase_discovery | 11 | 7/0/4 | -0.14242398452079175 |
| branin_sparse | ra_discovery | ra_v2 | 11 | 7/0/4 | -0.14242398452079175 |
| branin_sparse | ra_discovery | random | 11 | 10/0/1 | -5.755790892273198 |
| branin_sparse | ra_guard_only | ra_phase_coverage | 11 | 0/11/0 | 0.0 |
| branin_sparse | ra_guard_only | ra_phase_discovery | 11 | 0/11/0 | 0.0 |
| branin_sparse | ra_guard_only | ra_v2 | 11 | 0/11/0 | 0.0 |
| branin_sparse | ra_guard_only | random | 11 | 11/0/0 | -5.477492456564492 |
| branin_sparse | ra_phase_coverage | ra_phase_discovery | 11 | 0/11/0 | 0.0 |
| branin_sparse | ra_phase_coverage | ra_v2 | 11 | 0/11/0 | 0.0 |
| branin_sparse | ra_phase_coverage | random | 11 | 11/0/0 | -5.477492456564492 |
| branin_sparse | ra_phase_discovery | ra_v2 | 11 | 0/11/0 | 0.0 |
| branin_sparse | ra_phase_discovery | random | 11 | 11/0/0 | -5.477492456564492 |
| branin_sparse | ra_v2 | random | 11 | 11/0/0 | -5.477492456564492 |
| rosenbrock_sparse | ra_coverage | ra_discovery | 14 | 0/14/0 | 0.0 |
| rosenbrock_sparse | ra_coverage | ra_guard_only | 14 | 1/0/13 | 0.7965864002033978 |
| rosenbrock_sparse | ra_coverage | ra_phase_coverage | 14 | 1/0/13 | 0.7965864002033978 |
| rosenbrock_sparse | ra_coverage | ra_phase_discovery | 14 | 1/0/13 | 0.7965864002033978 |
| rosenbrock_sparse | ra_coverage | ra_v2 | 14 | 1/0/13 | 0.7965891263065525 |
| rosenbrock_sparse | ra_coverage | random | 14 | 11/0/3 | -20.28597819768084 |
| rosenbrock_sparse | ra_discovery | ra_guard_only | 14 | 1/0/13 | 0.7965864002033978 |
| rosenbrock_sparse | ra_discovery | ra_phase_coverage | 14 | 1/0/13 | 0.7965864002033978 |
| rosenbrock_sparse | ra_discovery | ra_phase_discovery | 14 | 1/0/13 | 0.7965864002033978 |
| rosenbrock_sparse | ra_discovery | ra_v2 | 14 | 1/0/13 | 0.7965891263065525 |
| rosenbrock_sparse | ra_discovery | random | 14 | 11/0/3 | -20.28597819768084 |
| rosenbrock_sparse | ra_guard_only | ra_phase_coverage | 14 | 0/14/0 | 0.0 |
| rosenbrock_sparse | ra_guard_only | ra_phase_discovery | 14 | 0/14/0 | 0.0 |
| rosenbrock_sparse | ra_guard_only | ra_v2 | 14 | 0/13/1 | 0.0 |
| rosenbrock_sparse | ra_guard_only | random | 14 | 12/0/2 | -22.655490164999392 |
| rosenbrock_sparse | ra_phase_coverage | ra_phase_discovery | 14 | 0/14/0 | 0.0 |
| rosenbrock_sparse | ra_phase_coverage | ra_v2 | 14 | 0/13/1 | 0.0 |
| rosenbrock_sparse | ra_phase_coverage | random | 14 | 12/0/2 | -22.655490164999392 |
| rosenbrock_sparse | ra_phase_discovery | ra_v2 | 14 | 0/13/1 | 0.0 |
| rosenbrock_sparse | ra_phase_discovery | random | 14 | 12/0/2 | -22.655490164999392 |
| rosenbrock_sparse | ra_v2 | random | 14 | 12/0/2 | -22.655490164999392 |

Improvement over initialization is not preservation of baseline development.
Per-trial phase_diagnostics retain coverage/discovery channels; attribution is descriptive, not counterfactual.

0/0 means no runs in that stratum, not a zero success probability.
Paired objective summaries in results.json condition on both methods feasible and retain denominators.
Current math intervals mainly encode signs; sign and interval arms may be identical by construction.
Software audit is not optimizer replay, calibration certification or evidence of superiority.
