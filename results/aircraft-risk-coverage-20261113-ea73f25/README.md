# Aircraft risk coverage pilot — PR #97

Fixed code SHA: `ea73f255ed7efe6f3898092ef79b97cbdc8ff094`  
Branch: `research/aircraft-risk-coverage-20261113`  
Environment: `Rcaide_vsp` / Python 3.9.25.

Complete smoke + phase-pilot evidence for the aircraft risk coverage experiment on the reduced X-57 model. No RCAIDE/MDA calls.

## Execution

| item | result |
|---|---|
| Unit tests | 6 passed |
| Smoke | 32 calls; `VERIFIED_AIRCRAFT_RISK_COLD_START`; `COMPLETE_AIRCRAFT_RISK_ANALYSIS` |
| Pilot | 3,984 calls = 144 initial + 2,304 online + 1,536 held-out; `COMPLETE_AIRCRAFT_RISK_COLD_START`; 3,446.6 s |
| Pilot verify | 3,984 calls verified |
| Pilot analysis | `COMPLETE_AIRCRAFT_RISK_ANALYSIS` |
| Seeds | 28414000–28415100 (12 seeds) |
| Arms | iid_explicit, coverage_explicit, boundary_explicit, brier_explicit, coverage_brier, coverage_boundary |
| Budget/arm | 12 shared initial + 32 online = 44 observed |
| Coverage switch | at n=24 |
| Aircraft / RCAIDE | 0 / 0 |

All six arms share the same 12 initial evaluations per seed; the three coverage arms share the coverage prefix and diverge only after the switch. Each arm uses only its own history afterward.

## Joint risk at n=48 (mean over 12 seeds)

| arm | Brier | Brier skill | AUROC | feasible recall | false feasible |
|---|---:|---:|---:|---:|---:|
| iid_explicit | .05956 | .4154 | .9392 | .5671 | .0349 |
| coverage_explicit | .05437 | .4646 | .9457 | .5792 | .0240 |
| boundary_explicit | .05336 | .4793 | .9478 | **.7190** | .0429 |
| brier_explicit | **.05318** | **.4809** | **.9522** | .6799 | .0393 |
| coverage_brier | **.05298** | .4798 | .9417 | .6342 | .0341 |
| coverage_boundary | .05366 | .4752 | .9510 | .6978 | .0423 |

At n=48 the six arms are tightly clustered on Brier (.0530–.0596); `brier_explicit` has the best AUROC, `coverage_brier` the lowest Brier by a negligible margin, and `boundary_explicit` the best feasible recall.

## Paired Brier at n=48 (candidate − baseline; negative is better)

| comparison | mean | median | W/L |
|---|---:|---:|---:|
| coverage_brier − coverage_explicit | **−0.001389** | −0.001440 | 7/5 |
| coverage_boundary − coverage_explicit | −0.000710 | −0.003523 | 8/4 |
| coverage_brier − brier_explicit | −0.000203 | −0.001909 | 8/4 |
| coverage_boundary − boundary_explicit | +0.000296 | −0.006182 | 9/3 |

- Coverage adds at most a ~0.0014 Brier change versus `coverage_explicit`, with 7/5 or 8/4 splits; this is not a robust coverage advantage.
- Adding coverage to `brier_explicit` is essentially neutral in mean (−0.0002) and lowers feasible recall (.634 vs .680); adding it to `boundary_explicit` is also neutral in mean but recall falls (.698 vs .719).
- n=24 coverage arms are identical to `coverage_explicit` by construction; n=36 effects are mixed and not sustained at n=48.

## Learning curves (joint Brier)

| arm | n12 | n24 | n36 | n48 |
|---|---:|---:|---:|---:|
| iid_explicit | .1085 | .0971 | .0716 | .0596 |
| coverage_explicit | .1085 | .0926 | .0712 | .0544 |
| boundary_explicit | .1085 | .1039 | .0741 | .0534 |
| brier_explicit | .1085 | .0946 | .0777 | .0532 |
| coverage_brier | .1085 | .0926 | .0707 | .0530 |
| coverage_boundary | .1085 | .0926 | **.0668** | .0537 |

The coverage arms share the same n=24 point; coverage_boundary is best at n=36 but the advantage shrinks to noise at n=48.

## Descriptive trajectories (diagnostic only)

| arm | first feasible mean | online feasible | best median Wh |
|---|---:|---:|---:|
| iid_explicit | 7.4 | 53 | 39,276 |
| coverage_explicit | 10.9 | 28 | 39,740 |
| boundary_explicit | 6.7 | 112 | 38,558 |
| brier_explicit | 7.7 | 58 | 40,990 |
| coverage_brier | 8.4 | 48 | 39,855 |
| coverage_boundary | 8.2 | 86 | **38,441** |

`coverage_boundary` gives the best median best-observed energy (38,441 Wh), but with 86 online feasible evaluations vs 112 for boundary and a 2.1 pp lower feasible recall. `brier_explicit` has the best risk ranking but the worst best-energy median (40,990 Wh). These are descriptive, not causal optimization results.

## Interpretation

- **Coverage did not add a robust risk-model gain.** The n48 Brier differences are tiny, paired splits are mixed, and the n36 advantage does not persist.
- The Brier change is not free: coverage_brier has a 4.6 pp lower feasible recall than brier_explicit; coverage_boundary has 2.1 pp lower recall than boundary_explicit. Do not describe this as a free improvement.
- `coverage_boundary` has the best best-energy median, but it also has fewer online feasible evaluations; the objective signal is not evidence of a superior optimizer.
- 12 seeds, one reduced-aircraft task, 32 online calls/arm; exploratory, not confirmatory.
- No RCAIDE/flight validation; no safety guarantee.

## Files

- `smoke/`, `pilot/`: sealed raw runs (report, plan, source_receipt, evaluations, progress, units, manifest).
- `smoke-analysis/`, `pilot-analysis/`: metrics by seed/arm/head/n, joint pairs, trajectories, analysis.json.
- `logs/`: tests, plan, smoke run/verify/analyze, pilot run/verify/analyze.
- `fixed_sha.txt`, `environment.txt`, `environment_packages.txt`, `SHA256SUMS.txt`.
- `.gitattributes`: `* -text`.
