# Staged study: discovery (pilot)

Exploratory only; failure, unknown and native early stops are retained. No MDA.
Calls: 6516/6516; unused calls are not reassigned.

| Case | Method | Feasible/all | Cold discovery/cold | Warm improved/warm |
|---|---|---:|---:|---:|
| branin_sparse | ra_cold_global | 12/20 | 1/9 | 11/11 |
| branin_sparse | ra_coverage | 20/20 | 9/9 | 11/11 |
| branin_sparse | ra_discovery | 19/20 | 8/9 | 11/11 |
| branin_sparse | ra_v2 | 12/20 | 1/9 | 11/11 |
| branin_sparse | random | 20/20 | 9/9 | 9/11 |
| rosenbrock_sparse | ra_cold_global | 17/20 | 3/6 | 14/14 |
| rosenbrock_sparse | ra_coverage | 19/20 | 5/6 | 14/14 |
| rosenbrock_sparse | ra_discovery | 20/20 | 6/6 | 14/14 |
| rosenbrock_sparse | ra_v2 | 17/20 | 3/6 | 14/14 |
| rosenbrock_sparse | random | 20/20 | 6/6 | 8/14 |

0/0 means no runs in that stratum, not a zero success probability.
Paired objective summaries in results.json condition on both methods feasible and retain denominators.
Current math intervals mainly encode signs; sign and interval arms may be identical by construction.
Software audit is not optimizer replay, calibration certification or evidence of superiority.
