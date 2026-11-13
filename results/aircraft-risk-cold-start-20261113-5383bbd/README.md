# Aircraft risk cold-start pilot — PR #96

Fixed code SHA: `5383bbd9e414af8d50f2773b2260294254582878`  
Branch: `research/aircraft-risk-cold-start-20261113`  
Environment: `Rcaide_vsp` / Python 3.9.25.

This branch contains the complete smoke/pilot evidence for the aircraft risk cold-start experiment. It is a matched-pair risk-scoring study on the reduced X-57 model; no RCAIDE/MDA calls were made.

## Execution

| item | result |
|---|---|
| Unit tests | 5 passed |
| Smoke | 28 math calls; `COMPLETE_AIRCRAFT_RISK_COLD_START`; verify `VERIFIED_AIRCRAFT_RISK_COLD_START`; analysis `COMPLETE_AIRCRAFT_RISK_ANALYSIS` |
| Pilot | 2,136 calls = 72 shared initial + 1,296 online + 768 held-out; `COMPLETE_AIRCRAFT_RISK_COLD_START`; 1,915.4 s |
| Pilot verify | 2,136 calls verified |
| Seeds | 28412000–28412500 (6 seeds) |
| Arms | iid_explicit, coverage_explicit, boundary_explicit, brier_explicit, boundary_latent, brier_latent |
| Budget per arm | 12 shared initial + 36 online = 48 observed |
| Held-out tests | 128 per arm/seed |
| Aircraft / RCAIDE calls | 0 / 0 |

The five explicit arms start from identical 12 shared initial evaluations; each arm then uses only its own online history. All six arms share the same 128 held-out test designs per seed.

## Joint risk at n=48 (mean over 6 seeds)

| arm | Brier | Brier skill | AUROC | feasible recall | false feasible |
|---|---:|---:|---:|---:|---:|
| iid_explicit | .04857 | .4642 | .9696 | .7902 | .0401 |
| coverage_explicit | .04991 | .4531 | .9661 | .7687 | .0461 |
| boundary_explicit | .04711 | .4896 | .9769 | .8081 | .0489 |
| **brier_explicit** | **.04311** | **.5266** | **.9807** | **.8411** | **.0331** |
| boundary_latent | .09142 | .0279 | .8463 | .5146 | .0705 |
| brier_latent | .08875 | .0388 | .8702 | .5826 | .0721 |

`brier_explicit` has the best joint Brier, Brier skill, AUROC, feasible recall and false-feasible rate among the six arms at n=48.

## Paired Brier at n=48 (candidate − baseline; negative is better)

| comparison | mean | median | W/L |
|---|---:|---:|---:|
| brier_explicit − boundary_explicit | **−0.003999** | −0.001282 | 3/3 |
| brier_explicit − iid_explicit | −0.005458 | −0.003229 | 4/2 |
| brier_explicit − coverage_explicit | −0.006803 | −0.007651 | 4/2 |
| brier_latent − boundary_latent | −0.002666 | −0.009387 | 4/2 |
| brier_latent − brier_explicit | **+0.045645** | +0.054039 | 0/6 |

- `brier_explicit` improves Brier without a feasible-recall cost: recall **.8411 vs .8081** and false-feasible **.0331 vs .0489** versus `boundary_explicit`; the recall point estimate rises by 3.3 pp.
- The paired comparison is nevertheless 3W/3L and small in mean; one seed with a large gain (−0.0175) drives much of the mean.
- `brier_latent` improves on `boundary_latent` (4W/2L) but remains far worse than `brier_explicit` in absolute Brier/AUROC/recall (all 6 paired losses). The latent representation has no independent gain here.

## Learning curves (joint Brier, n=12/24/36/48)

| arm | n12 | n24 | n36 | n48 |
|---|---:|---:|---:|---:|
| iid_explicit | .1088 | .0996 | .0703 | .0486 |
| coverage_explicit | .1088 | .0885 | .0543 | .0499 |
| boundary_explicit | .1088 | .1077 | .0655 | .0471 |
| brier_explicit | .1088 | .1040 | .0635 | .0431 |
| boundary_latent | .1201 | .1043 | .0959 | .0914 |
| brier_latent | .1201 | .1082 | .0973 | .0888 |

- n12 explicit arms are identical by construction (same shared initial data and no online decisions yet).
- Latent arms are worse from n12 onward and do not catch up at n48.
- The explicit Brier head is best at n48 but has no separation at n24 (slightly worse than coverage/boundary there); the n48 advantage is a late-sample effect.

## Descriptive trajectories (diagnostic, not significance)

| arm | first feasible call mean | online feasible total | best median Wh |
|---|---:|---:|---:|
| iid_explicit | 9.7 | 24 | 39,336 |
| coverage_explicit | 10.2 | 16 | 40,059 |
| boundary_explicit | 8.5 | 49 | 39,387 |
| brier_explicit | 10.5 | 26 | 39,669 |
| boundary_latent | 7.7 | 43 | 39,561 |
| brier_latent | 9.7 | 27 | 40,256 |

The risk improvements do **not** translate into better best-observed energy: `brier_explicit` is slightly worse than `boundary_explicit` on median best (39,669 vs 39,387 Wh) and has far fewer online feasible evaluations. This experiment only scores risk predictions; it is not an optimization-benefit result.

## Interpretation boundaries

- Six seeds, one reduced-aircraft task, 36 online calls/arm; exploratory, not confirmatory.
- The Brier improvement is real in this panel but paired 3W/3L and one-seed sensitive; do not claim a stable aircraft-level gain.
- No RCAIDE or flight validation; no safety guarantee.
- Latent representation is decisively worse in absolute Brier/AUROC; no support for a latent advantage in this task.
- The next decision should be based on whether the Brier gain survives independent seeds and whether it persists without the recall/false-feasible tradeoff.

## Files

- `smoke/`, `pilot/`: sealed raw outputs (report, plan, source_receipt, evaluations, progress, units, manifest).
- `smoke-analysis/`, `pilot-analysis/`: metrics per seed/arm/head/n, joint pairs, trajectories, analysis.json.
- `logs/`: unit tests, plan, smoke run/verify/analyze, pilot run/verify/analyze.
- `fixed_sha.txt`, `environment.txt`, `environment_packages.txt`, `SHA256SUMS.txt`.
- `.gitattributes`: `* -text` for byte-exact evidence.
