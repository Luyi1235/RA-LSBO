# RA-LSBO Mathematical Benchmark Package

Date: 2026-10-26. Package: `ra_lsbo/experiments/mathematical_benchmarks`.

## Scope

This package evaluates the hierarchical, mixed-variable Branin and Rosenbrock extensions in ordinary and sparse-region variants. These are declared constrained extensions, not official reproductions of unconstrained published benchmarks. Each design has a basic/augmented architecture; augmented designs activate a discrete module count and continuous trim variable. The region constraint can censor the objective outside the declared feasible ellipse, while a resource constraint remains observable.

The sparse label denotes a narrower declared feasible region, not a measured feasible fraction. Each campaign reports the independently sampled qualification-panel fraction and its binomial interval. Anchor points are diagnostic and do not determine whether a task or seed is included.

## Methods and protocol

The seven paired arms are `ra_lsbo`, `original_lsbo`, `onehot_local`, `no_intervals`, `no_local`, `conventional_eic`, and `random`. All arms share the same initial designs and labels per case and seed. Candidate generation uses the same exact geometry screens. Panel labels are isolated from fitting, acquisition, stopping, and method selection.

The risk-aware arm uses the current mixed-observation posterior, interval constraints, structured latent representation, grouped-kernel selection, and local branch refinement. Ablations isolate one-hot representation, interval feedback, and local search. `conventional_eic` is a declared numeric-only hierarchical EIC adaptation, not an official CBOB/SCBO implementation.

Primary outcomes are feasible-run count, first feasible evaluation, best feasible objective versus actual calls, failure/unknown categories, and paired seed comparisons. Posterior diagnostics include Brier score, log loss, class-conditional scores, reliability bins, and high-confidence errors. A conditional objective statistic is only interpreted with its feasible-run denominator and paired count.

Smoke and pilot profiles are software/exploratory evidence, not confirmation of stable superiority. The confirm profile requires a separate explicit authorization because its complete prospective budget exceeds 500 calls. The runner rejects the complete campaign before execution if its registered maximum exceeds `--approved-call-limit`; budgets cannot be split to evade the cap.

## Closed-loop commands

Run from the repository root:

```powershell
python -m ra_lsbo.experiments.mathematical_benchmarks prepare --profile smoke --output runs/math_smoke_20261026
python -m ra_lsbo.experiments.mathematical_benchmarks run --output runs/math_smoke_20261026 --approved-call-limit 500
python -m ra_lsbo.experiments.mathematical_benchmarks audit --output runs/math_smoke_20261026
python -m ra_lsbo.experiments.mathematical_benchmarks report --output runs/math_smoke_20261026
```

The `pilot` and `confirm` profiles use the registered complete budget. Each run directory is write-once. `prepare` freezes task contracts, candidates, method settings, versions, source hashes, source snapshots, and costs. `run` writes a single-writer settled-call ledger. `audit` checks evidence identity, source snapshots, budgets, paired histories, and reconstructed metrics; it does not independently replay optimizer decisions. `report` requires a passing audit.

The shared engine lives in `ra_lsbo/experiments/core`; domain definitions and CLI remain package-specific. No RCAIDE MDA is called by this mathematical package.
