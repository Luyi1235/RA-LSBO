# PR #48 matched latent-vs-onehot GP mechanism diagnostic — 2026-11-04

Status: `COMPLETE_MATCHED_SURROGATE_DIAGNOSTIC`. Zero new aircraft, RCAIDE or mission-simulation calls; the PR #45 ledger was opened read-only.

## Provenance
- PR #48 head: `ebe395af9aba957af488d55685128d0276780f70`
- PR #45 runner source SHA: `386082beeacac8ae43d0dfb1ccda5c5ad3b356cf`
- PR #45 archived results commit: `d42ac115231372a39e0557aa8f61948b7783f61b`
- Source directory (read-only): `D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z`
- Entry point: `run_ra_lsbo_matched_gp_pycharm.py`, executed with the `Rcaide_vsp` interpreter (equivalent to PyCharm Run).

## Environment and commands
- Python 3.9.25 (`C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe`), SQLite 3.51.2
- `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- Smoke round: `--pairs-per-task 1` → 2 seed pairs, 384 prediction rows, exit 0.
- Full round: default 8 pairs per task → 16 seed pairs, 3,072 prediction rows, exit 0.
- Tests: `CONFIRM12_RUN_MATCHED_GP_SMOKE=1 python -m pytest -q tests/test_confirm12_matched_gp_mechanism.py` → **7 passed in 4.79 s** (without the opt-in flag: 6 passed, 1 skipped).

## Source evidence unchanged
| file | SHA256 |
|---|---|
| real_calls.sqlite | `a30b0dadb5d619be9b6c0d70c5aa29fe115f86b85d62a449e4acf05325d6f1b7` |
| plan.json | `ac5f1424f3c8d52eff9ffd835cb5004224be965afee8ec15069e1fea38c43012` |
| audit.json | `81e5cf82ae40e8bffd5ae085a9be5fca5849cd48ed50028f7913bdb1302adf17` |

Pre/post hashes and mtime are unchanged; `real_calls.sqlite` remained 856,600,576 bytes.

## Held-out results (train = 12 real shared-initial records; test = disjoint seed's initial designs; 8 pairs/task)
| task | objective RMSE latent (Wh) | objective RMSE onehot (Wh) | SOC Brier latent | SOC Brier onehot | constraint-mean Brier latent | constraint-mean Brier onehot | SOC false-feasible latent/onehot | all-constraint false-feasible latent/onehot |
|---|---|---|---|---|---|---|---|---|
| aircraft_ordinary | 3,665.13 | 3,657.20 | 0.19416 | 0.19433 | 0.04124 | 0.04317 | 13 / 13 | 22 / 22 |
| aircraft_sparse | 13,451.28 | 14,352.89 | 0.19227 | 0.19417 | 0.11779 | 0.11674 | 9 / 11 | 36 / 34 |

Completion Brier: ordinary 0.01077 (latent) vs 0.01145 (onehot); sparse 0.01192 vs 0.01239.

## Paired latent-minus-onehot differences (8 independent training/holdout seed clusters; negative favors latent)
- Objective squared error: ordinary +96,302.11 Wh² (95% CI −371,052.57 to 563,656.79); sparse −24,982,687.56 Wh² (95% CI −53,617,110.39 to 3,651,735.28). Both intervals include zero.
- SOC Brier: ordinary −0.000169 (95% CI −0.010155 to 0.009817); sparse −0.001904 (95% CI −0.008185 to 0.004378).
- Completion Brier: ordinary −0.000680 (95% CI −0.001049 to −0.000312); sparse −0.000471 (95% CI −0.000743 to −0.000199).
- Constraint heads are mixed (e.g. sparse current Brier −0.00384, power +0.00953, root_bending −0.00296, all CIs crossing zero).

All intervals and p-values are posthoc descriptive, clustered by the 8 independent seed pairs. The test designs are the held-out initial designs of another seed, not the original online trajectories.

## Files
| file | SHA256 |
|---|---|
| `matched_gp_report.json` | `9ffe949434b847044b2f15d9bdfec7b7a7a564fcec955dbedd23d5437d98886e` |
| `matched_gp_predictions.csv` | `8a987ec3a9d9af57a79e6bcdaf842ccf60f8e35f1ae9c0bd1feced5d58253806` |
| `console.log` | `664f86b09ffaa4144f491d2fe77386d88b51369470894f8bf35b3f894f5c4a38` |
| `smoke-console.log` | `afeb923ab21f52cf56668f6a58164c330bb2a66b2228f2d3c2ae439111a05d14` |
| `tests.log` | `6e6af768e1327ab5df8f12d494c5ae748736112bc524021b9fd083ff64667e47` |
| `pre_run_sha256.txt` | `92c65179aec3001ebaebb64c7cd5445fc3817888c51f7b89da4b1aad1623e93e` |

## Caveats
- Retrospective independent-seed initial holdout; not the original online trajectories.
- Numerical GP errors are preserved, not converted into feasible predictions.
- No significance or causal online-optimization claim.
- Shared initial samples do not establish 140 km sparse eligibility.
