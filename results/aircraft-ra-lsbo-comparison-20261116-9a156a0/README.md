# PR115 controlled RA-LSBO comparison

Reduced-aircraft, same-budget comparison (4 shared initial + 20 online calls per arm) of RA-LSBO against original LSBO, training/representation ablations, and two external risk-aware baselines.

## Identity

- Fixed code SHA: `9a156a0c0ab386dcfa474dada0e9bfd0d42c162d`
- Entrypoint: `run_aircraft_ra_lsbo_comparison.py`
- Worktree at execution: `D:\LSBO\ra-lsbo-pr115-code`
- Evidence root: `D:\LSBO\aircraft-ra-lsbo-comparison-20261116-9a156a0`
- RCAIDE/MDA calls: 0

## Protocol

- 24 seeds per condition, 4 shared initial calls per seed, 20 online calls per arm.
- Arms: `ra_separate`, `original_lsbo`, `ra_joint_kernel`, `ra_explicit`, `logcei`, `scbo_adapted`.
- Headwind: 4.0 m/s, 0 kg payload delta; payload: 0 m/s, 24 kg payload delta.
- Held-out 32 designs/seed; terminal energy is the best feasible held-out energy.
- `original_lsbo` may stop early on repeated missing/failed objective returns; each arm has the same 4+20 nominal budget.

## Calls

- `comparison-smoke`: status `VERIFIED_RA_LSBO_COMPARISON`, total calls 34, roles {'heldout': 12, 'initial': 4, 'online': 18}, max_calls 34.
- `comparison-headwind`: status `VERIFIED_RA_LSBO_COMPARISON`, total calls 3538, roles {'heldout': 768, 'initial': 96, 'online': 2674}, max_calls 3744.
- `comparison-payload`: status `VERIFIED_RA_LSBO_COMPARISON`, total calls 3635, roles {'heldout': 768, 'initial': 96, 'online': 2771}, max_calls 3744.

## Cold-start paired comparisons

`RA` is `ra_separate`; positive first-call delta means RA was slower; positive discovery delta means RA found more feasible designs; positive energy delta means RA's best feasible energy was higher (worse).

### Headwind 4.0 m/s

| comparator | n cold | first-call Δ | discovery Δ | energy Δ Wh | win rate |
|---|---:|---:|---:|---:|---:|
| `original_lsbo` | 19 | -9.263 [-12.684, -5.842] | 0.684 [0.474, 0.895] | 3191.6 [1189.237, 5870.618] | 0.474 |
| `ra_joint_kernel` | 19 | -0.632 [-2.421, 1.158] | -0.053 [-0.158, 0.000] | 288.5 [-957.805, 1631.263] | 0.000 |
| `ra_explicit` | 19 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.0 [0.000, 0.000] | 0.000 |
| `logcei` | 19 | -4.421 [-8.474, -0.211] | 0.158 [-0.053, 0.368] | 125.0 [-1390.716, 1911.010] | 0.368 |
| `scbo_adapted` | 19 | -6.368 [-9.686, -3.105] | 0.368 [0.158, 0.579] | -1001.5 [-3425.830, 1486.664] | 0.526 |

### Payload +24 kg

| comparator | n cold | first-call Δ | discovery Δ | energy Δ Wh | win rate |
|---|---:|---:|---:|---:|---:|
| `original_lsbo` | 20 | -3.100 [-6.450, 0.200] | 0.400 [0.200, 0.600] | 3564.2 [1850.816, 5266.528] | 0.050 |
| `ra_joint_kernel` | 20 | 0.750 [-1.400, 3.000] | -0.050 [-0.150, 0.000] | 1080.5 [129.629, 2145.015] | -0.350 |
| `ra_explicit` | 20 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -214.4 [-787.605, 122.126] | -0.050 |
| `logcei` | 20 | -0.350 [-3.850, 3.150] | 0.150 [-0.050, 0.350] | -589.7 [-2367.128, 1225.638] | 0.200 |
| `scbo_adapted` | 20 | -2.400 [-5.600, 0.600] | 0.250 [0.000, 0.500] | -2063.9 [-3985.819, 80.678] | 0.500 |

## All-seed paired comparisons

| condition | comparator | first-call Δ | discovery Δ | energy Δ Wh | win rate |
|---|---|---:|---:|---:|---:|
| headwind | `original_lsbo` | -7.333 [-10.417, -4.333] | 0.542 [0.333, 0.750] | 3807.2 [2536.923, 5094.316] | 0.167 |
| headwind | `ra_joint_kernel` | -0.500 [-2.000, 0.917] | -0.042 [-0.125, 0.000] | -52.5 [-1201.040, 1111.162] | 0.083 |
| headwind | `ra_explicit` | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -63.1 [-189.397, 0.000] | 0.042 |
| headwind | `logcei` | -3.500 [-6.750, 0.000] | 0.125 [-0.042, 0.292] | 13.1 [-1135.585, 1372.163] | 0.333 |
| headwind | `scbo_adapted` | -5.042 [-7.958, -2.333] | 0.292 [0.125, 0.500] | -1631.0 [-3710.195, 340.103] | 0.542 |
| payload | `original_lsbo` | -2.583 [-5.542, 0.208] | 0.333 [0.167, 0.542] | 4122.9 [2625.730, 5534.677] | -0.125 |
| payload | `ra_joint_kernel` | 0.625 [-1.208, 2.583] | -0.042 [-0.125, 0.000] | 792.4 [-21.632, 1723.646] | -0.250 |
| payload | `ra_explicit` | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -14.8 [-626.211, 496.709] | -0.083 |
| payload | `logcei` | -0.292 [-3.125, 2.583] | 0.125 [-0.042, 0.292] | -269.1 [-1704.979, 1156.083] | 0.125 |
| payload | `scbo_adapted` | -2.000 [-4.625, 0.500] | 0.208 [0.000, 0.417] | -1753.8 [-3353.489, 62.053] | 0.458 |

## Training/representation ablations

- `ra_explicit` is the explicit one-hot risk representation; headwind paths tie `ra_separate` exactly on first/discovery, payload energy −214 Wh with CI crossing zero. Representation alone does not explain an energy gain.
- `ra_joint_kernel` shares one grouped kernel; headwind first-call −0.63 with CI crossing zero, payload +0.75 with CI crossing zero; discovery −0.04/−0.05; energy +288/+1,080 Wh. Joint training does not reproduce the RA-vs-original discovery advantage.
- Calibration on matched prefixes (`ra_separate` vs `ra_joint_kernel`) changes completion Brier by −0.0019/−0.0007 and usable Brier by +0.0017/+0.0011; no path-level effect.

## Interpretation

- RA improves **time to first feasible** vs original LSBO on both conditions (headwind −9.26 cold calls; payload −3.10 cold calls; all-seed −7.33/−2.58) and discovers more feasible designs (cold +0.684/+0.400).
- RA does **not** improve terminal best-feasible energy vs original LSBO: +3,192/+3,564 Wh cold and +3,807/+4,123 Wh all-seed (positive CIs in all four); win rate is 0.474 headwind cold and 0.05 payload cold. The discovery gain is not an energy gain.
- Against `scbo_adapted`, RA is faster and lower-energy on average (cold energy −1,002/−2,064 Wh; win 0.526/0.500), though headwind energy CI crosses zero.
- Against `logcei`, RA is faster, with similar energy (cold +125/−590 Wh, CIs crossing zero).
- This is a reduced-aircraft, 24-seed descriptive comparison; original LSBO early stopping and the 4+20 budget limit mechanism attribution.

## Files

- `comparison-{smoke,headwind,payload}/`: raw unit ledgers, results, roster and proposal evidence.
- `comparison-{...}-analysis.json`: paired comparisons and calibration.
- `*-plan.json`: frozen plans.
- `*.log`, `tests.log`, `environment.txt`, `python-version.log`, `fixed_sha.txt`.
- `SHA256SUMS.txt`: artifact checksums.
