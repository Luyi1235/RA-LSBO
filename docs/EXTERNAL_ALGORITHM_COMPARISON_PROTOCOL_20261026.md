# RA-LSBO External Algorithm Comparison Protocol

Date: 2026-10-26.

## Fixed Roster

The initial screening roster is `ra_lsbo`, `random`, `logcei`, `scbo_adapted`,
`sego_utb_adapted`, and `nsga2_adapted`. The original LSBO and component
ablations are excluded from this screening roster; their historical code,
plans, and results remain unchanged.

The constrained GP baselines use separate one-hot hierarchical features for
objective, completion, and constraint models. logCEI uses exact objective
observations from usable feasible runs and only sign-resolved exact/interval
constraint evidence. SCBO is a single-point candidate-pool adaptation using
Thompson sampling and a mixed categorical/continuous trust region. The
SEGO-UTB adaptation uses a fixed 1.96 upper-confidence feasible set. NSGA-II
uses nondominated sorting and crowding over observed constraint violation and
objective, with mixed-variable crossover/mutation. None of the adapted
implementations is described as official reference code or an exact paper
reproduction.

## Fairness and accounting

Each method receives the same evaluated initial designs and labels per case
and seed. The optimizer cannot access the independent qualification panel or
held-out aircraft scenarios. Candidate generation uses the exact geometry
screens; every objective or scenario evaluation is charged to the prospective
ledger. Aircraft validation uses the same scenarios for every method.

The sparse pilot uses two paired seeds and six online evaluations per method.
It is a screening study, not evidence of publication-level superiority.
Maximum complete budgets are 212 calls for the two mathematical sparse cases
and 204 calls for the simplified-aircraft sparse case, including anchors,
qualification panels, shared initial designs, integration checks, and
held-out aircraft scenarios. The model runner makes zero new RCAIDE MDA calls.

Primary report fields are feasible discovery, first-feasible cost, best
feasible objective versus calls, failed/unknown outcomes, and aircraft
scenario-risk bounds. Conditional objective summaries retain their feasible
run denominator. Results are paired by seed and no unsuccessful run is
dropped.

## Launch

From the repository root, using the configured `Rcaide_vsp` Python:

```powershell
python -m ra_lsbo.experiments.mathematical_benchmarks.compare prepare --profile pilot --output runs/comparison_math_pilot_20261026
python -m ra_lsbo.experiments.mathematical_benchmarks.compare run --output runs/comparison_math_pilot_20261026 --approved-call-limit 500
python -m ra_lsbo.experiments.mathematical_benchmarks.compare audit --output runs/comparison_math_pilot_20261026
python -m ra_lsbo.experiments.mathematical_benchmarks.compare report --output runs/comparison_math_pilot_20261026
```

```powershell
python -m ra_lsbo.experiments.simplified_aircraft.compare prepare --profile pilot --output runs/comparison_aircraft_pilot_20261026
python -m ra_lsbo.experiments.simplified_aircraft.compare run --output runs/comparison_aircraft_pilot_20261026 --approved-call-limit 500
python -m ra_lsbo.experiments.simplified_aircraft.compare audit --output runs/comparison_aircraft_pilot_20261026
python -m ra_lsbo.experiments.simplified_aircraft.compare report --output runs/comparison_aircraft_pilot_20261026
```

Each run directory is write-once. The complete call budget is checked before
the execution lock or any model call is started.
