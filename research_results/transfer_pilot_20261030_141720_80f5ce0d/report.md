# Staged study: transfer (pilot)

Exploratory only; failure, unknown and native early stops are retained. No MDA.
Calls: 3108/3108; unused calls are not reassigned.

| Case | Method | Feasible/all | Cold discovery/cold | Warm improved/warm |
|---|---|---:|---:|---:|
| branin_sparse | ev_equal | 8/8 | 3/3 | 5/5 |
| branin_sparse | ev_latent | 8/8 | 3/3 | 5/5 |
| branin_sparse | ev_onehot | 8/8 | 3/3 | 5/5 |
| branin_sparse | ev_prequential | 8/8 | 3/3 | 5/5 |
| branin_sparse | ev_random_then_latent | 8/8 | 3/3 | 5/5 |
| branin_sparse | random | 8/8 | 3/3 | 2/5 |
| rosenbrock_sparse | ev_equal | 7/8 | 3/4 | 4/4 |
| rosenbrock_sparse | ev_latent | 7/8 | 3/4 | 4/4 |
| rosenbrock_sparse | ev_onehot | 7/8 | 3/4 | 4/4 |
| rosenbrock_sparse | ev_prequential | 7/8 | 3/4 | 4/4 |
| rosenbrock_sparse | ev_random_then_latent | 7/8 | 3/4 | 4/4 |
| rosenbrock_sparse | random | 6/8 | 2/4 | 3/4 |

## Paired warm-start objective (left minus right; negative is better)

| Case | Left | Right | n | Left better/tie/worse | Median paired difference |
|---|---|---|---:|---:|---:|
| branin_sparse | ev_equal | ev_latent | 5 | 0/5/0 | 0.0 |
| branin_sparse | ev_equal | ev_onehot | 5 | 2/0/3 | 0.15045934456781218 |
| branin_sparse | ev_equal | ev_prequential | 5 | 0/5/0 | 0.0 |
| branin_sparse | ev_equal | ev_random_then_latent | 5 | 0/5/0 | 0.0 |
| branin_sparse | ev_equal | random | 5 | 5/0/0 | -5.925172424314936 |
| branin_sparse | ev_latent | ev_onehot | 5 | 2/0/3 | 0.15045934456781218 |
| branin_sparse | ev_latent | ev_prequential | 5 | 0/5/0 | 0.0 |
| branin_sparse | ev_latent | ev_random_then_latent | 5 | 0/5/0 | 0.0 |
| branin_sparse | ev_latent | random | 5 | 5/0/0 | -5.925172424314936 |
| branin_sparse | ev_onehot | ev_prequential | 5 | 3/0/2 | -0.15045934456781218 |
| branin_sparse | ev_onehot | ev_random_then_latent | 5 | 3/0/2 | -0.15045934456781218 |
| branin_sparse | ev_onehot | random | 5 | 5/0/0 | -8.521740797454312 |
| branin_sparse | ev_prequential | ev_random_then_latent | 5 | 0/5/0 | 0.0 |
| branin_sparse | ev_prequential | random | 5 | 5/0/0 | -5.925172424314936 |
| branin_sparse | ev_random_then_latent | random | 5 | 5/0/0 | -5.925172424314936 |
| rosenbrock_sparse | ev_equal | ev_latent | 4 | 0/4/0 | 0.0 |
| rosenbrock_sparse | ev_equal | ev_onehot | 4 | 2/0/2 | 0.03082840325134444 |
| rosenbrock_sparse | ev_equal | ev_prequential | 4 | 0/4/0 | 0.0 |
| rosenbrock_sparse | ev_equal | ev_random_then_latent | 4 | 0/4/0 | 0.0 |
| rosenbrock_sparse | ev_equal | random | 4 | 4/0/0 | -1.272009543249632 |
| rosenbrock_sparse | ev_latent | ev_onehot | 4 | 2/0/2 | 0.03082840325134444 |
| rosenbrock_sparse | ev_latent | ev_prequential | 4 | 0/4/0 | 0.0 |
| rosenbrock_sparse | ev_latent | ev_random_then_latent | 4 | 0/4/0 | 0.0 |
| rosenbrock_sparse | ev_latent | random | 4 | 4/0/0 | -1.272009543249632 |
| rosenbrock_sparse | ev_onehot | ev_prequential | 4 | 2/0/2 | -0.03082840325134444 |
| rosenbrock_sparse | ev_onehot | ev_random_then_latent | 4 | 2/0/2 | -0.03082840325134444 |
| rosenbrock_sparse | ev_onehot | random | 4 | 4/0/0 | -1.8758496823855948 |
| rosenbrock_sparse | ev_prequential | ev_random_then_latent | 4 | 0/4/0 | 0.0 |
| rosenbrock_sparse | ev_prequential | random | 4 | 4/0/0 | -1.272009543249632 |
| rosenbrock_sparse | ev_random_then_latent | random | 4 | 4/0/0 | -1.272009543249632 |

Improvement over initialization is not preservation of baseline development.
Per-trial phase_diagnostics retain coverage/discovery channels; attribution is descriptive, not counterfactual.

0/0 means no runs in that stratum, not a zero success probability.
Paired objective summaries in results.json condition on both methods feasible and retain denominators.
Current math intervals mainly encode signs; sign and interval arms may be identical by construction.
Software audit is not optimizer replay, calibration certification or evidence of superiority.
