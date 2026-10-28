# RA-LSBO Simplified Aircraft Package

Date: 2026-10-26. Package: `ra_lsbo/experiments/simplified_aircraft`.

## Scope and model boundary

The two declared tasks are `aircraft_ordinary` (100 km nominal cruise) and `aircraft_sparse` (140 km). The reduced electric-aircraft model combines the existing hierarchical aircraft design variables with cruise airspeed and battery parallel-string count. Parallel strings affect battery mass, capacity, and cell loading. The model includes nominal geometry, mission completion, energy, SOC reserve, electrical, and power constraints.

This is a reduced point-mass mission model with a frozen battery model, idealized propulsion, and limited wind/ageing assumptions. It is not a full RCAIDE MDA, validated X-57 digital twin, certification analysis, or empirical aircraft reliability model. The package makes zero new RCAIDE MDA calls.

## Risk-aware design and independent validation

The seven paired optimization arms and common-prefix protocol match the mathematical package. Optimization uses nominal conditions. After each arm selects its nominal feasible incumbent, the runner evaluates that fixed design on a separately generated shared set of screening scenarios. Scenario labels are never fed back into optimization or used to select another design.

Scenario variables are independently sampled from the declared supports: headwind Uniform[-4, 8] m/s, payload delta Uniform[-40, 60] kg, battery capacity retention Uniform[0.9, 1.0], and initial SOC Uniform[0.90, 0.98]. These are screening assumptions, not empirical distributions.

The report separates nominal posterior uncertainty from scenario outcomes. It reports SOC failure-fraction bounds and conservative binomial bounds. If a mission does not complete, terminal SOC is censored; the implementation reports bounded SOC shortfall and CVaR rather than inventing an exact terminal value. Completed-only energy is not an unconditional risk metric. The scenario study is not chance-constrained optimization or flight-safety certification.

Aircraft integration is checked at 20 s and 5 s for a registered anchor. Exact objectives, consistent termination/physical labels, and at most 1% energy discrepancy are required for `passed`; incomplete outputs remain `inconclusive`.

## Closed-loop commands

Run from the repository root:

```powershell
python -m ra_lsbo.experiments.simplified_aircraft prepare --profile smoke --output runs/aircraft_smoke_20261026
python -m ra_lsbo.experiments.simplified_aircraft run --output runs/aircraft_smoke_20261026 --approved-call-limit 500
python -m ra_lsbo.experiments.simplified_aircraft audit --output runs/aircraft_smoke_20261026
python -m ra_lsbo.experiments.simplified_aircraft report --output runs/aircraft_smoke_20261026
```

The `pilot` and `confirm` profiles use the registered complete budget. Each run directory is write-once. `prepare` freezes task contracts, candidates, scenarios, method settings, versions, source hashes, source snapshots, and costs. `run` writes a single-writer settled-call ledger. `audit` checks evidence identity, source snapshots, budgets, paired histories, scenario assignment, and reconstructed metrics; it does not independently replay optimizer decisions. `report` requires a passing audit.

The shared engine lives in `ra_lsbo/experiments/core`; aircraft definitions and CLI remain package-specific.
