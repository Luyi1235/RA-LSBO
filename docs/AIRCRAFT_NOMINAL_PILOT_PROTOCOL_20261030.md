# Nominal aircraft pilot v1 (2026-10-30)

Prospective development protocol written AFTER inspection of the 352-call transition.
Not independent confirmation or a guarantee of publishability. Historical algorithms,
plans, byte-addressed snapshots, all 12 branches, physical equations and results remain.

## Entry and decision

`python -m ra_lsbo.experiments.simplified_aircraft.nominal_pilot`
Commands prepare/run/audit/report; explicit cap required. No automatic stage chaining,
resume, free retry, confirm, weather scenarios or RCAIDE entry. A run lock is retained.

Approve only a bounded exploratory comparison under these limitations: 100 km panel
65/128 feasible; 140 km 19/128 (95% interval 9.18%-22.21%) does NOT establish <10%.
The identifier `aircraft_sparse` remains for contract compatibility, reported as the
140 km constrained mission. Do not lengthen its mission to make a sparse claim true.
Completed fixed checks have stable labels; 16/28 checks remain partial/inconclusive.
No flight-data calibration, wake-wing model, detailed stress or thermal validation.
Complete mission energy is a sum of fixed stage loads, so its machine-precision
step agreement is NOT a general battery/SOC integration accuracy guarantee.

Pilot prepare requires the full qualification directory from result commit
59643c3d544aaca5d97144e83b95b5ec782b8c60, executed at b8a94c3. The importer verifies
fixed headers, 4 sealed JSON/JSONL evidence files, 156 snapshots and all 316 ledger
links/recipes without executing old snapshot code or rewriting Windows paths.
It is a read-only provenance check, not a repeat of the original runtime audit.
Only a receipt enters the new plan; no qualification labels enter training. Physical
source LF-normalized identity is checked for platform compatibility; each new plan
still binds its actual RAW source bytes, full runtime and commit separately.

## Frozen arms

Seven: random, ev_latent (phase single-latent reference), ev_onehot, phase_exact,
phase_sign, ev_random_then_latent, original_lsbo. No adaptive experts or new acquisition.

All common initial labels and raw outcomes are identical across arms within a seed.
Exact and sign views only coarsen MODEL inputs. Sign keeps resolved constraint signs,
not objective intervals; interval keeps legal lower/upper bounds. Exact constraints
stay exact in all three. Raw feasibility/incumbents survive view ablation. Interval
versus sign thus tests the JOINT additional information in constraint magnitudes
and objective bounds; it is not an isolated objective/constraint-factor experiment.
Current aircraft feedback includes one-sided numeric bounds, not general finite-width
intervals. Unknown checks stay unknown; no midpoint, penalty or synthetic safe label.

Phase RA uses the existing cold-only coverage, near guard and cold-local restriction;
first RAW usable feasible result restores the unchanged latent development policy.
The onehot control uses the same strategy; random-then-latent tests the simple hybrid.
Native LSBO retains its own pool/settings and five consecutive missing-objective stop;
not effort-matched. Unused calls are never reassigned, and stopped trajectories are
reported at the nominal horizon. None of these arms is an official CBOB reproduction.
Nearest-literature comparisons require a later separately validated protocol.

## Prospective scope and accounting

| Profile | Cases | Seeds | Initial | Online per arm | Verification per trial | Total ceiling |
|---|---:|---|---:|---:|---:|---:|
| smoke | 2 | 9910 | 6 | 2 | at most 1 | 54 |
| pilot | 2 | 9911-9918 | 12 | 30 | at most 1 | 3664 |

Pilot: 192 shared initial + 3360 online + at most 112 incumbent verifications;
112 trials. Smoke: 12 initial + 28 online + at most 14 verifications. New work-machine
research ceiling smoke+pilot=3718. Historical 352 calls and unit-test calls are extra.
CI authorizes ONE <=54-call smoke and prepare-only pilot; no complete performance pilot.
Pool128/encoder30/nativeGP15 in pilot; pool32/encoder4/nativeGP2 in smoke. Other model,
EP, kernel grid, local effort and acquisition settings remain the registered phase ones.

Initial candidate generation uses a fixed stream, paired across both missions without
any truth filter beyond common analytic geometry. Keep all seeds, warm/cold strata and
failures. At the observed 140 km panel rate the plug-in probability of a cold 12-point
initial set is about14.5%; at100 km about0.0202%. These are rough planning calculations,
not guarantees. Eight seeds may yield very few cold runs; don't resample to fix that.
Main purpose is partial-feedback/representation tradeoffs on a physical model, NOT a
powerful independent cold-start superiority claim. Seed count is not a power guarantee.

## Held-out numerical checks and faults

Optimization uses20 s. After each trial choose the FIRST chronological minimum nominal
usable feasible incumbent, then evaluate that EXACT design once at5 s. No best5 s
reselection, verification feedback, adaptive tolerance or scenario sampling. No nominal
feasible incumbent means no verification call, an explicit missing endpoint, and no
unused budget reuse. Store both contracts and labels. A changed label is a reported
nominal-versus-refined discrepancy, NOT silently corrected optimization performance.
Energy screen1% and exact constraint deltas are numerical sensitivity diagnostics,
not physical accuracy or safety certification. Initial incumbents are eligible.

Every evaluation is started/settled in the single-writer ledger. Numerical failure,
unknown or infrastructure status stops the entire campaign after recording the outcome,
before feeding it to an optimizer. A raised evaluator error leaves a charged pending
call. Do not delete locks, relabel errors as physical infeasibility, restart or exclude
the affected seed. Scientific/technical review is needed for a new version and budget.

## Metrics and gate

Per task report all-run nominal feasibility,5 s verified feasibility, censored first
feasible cost, conditional objective and denominator, paired warm final differences,
violations/abort causes, output kinds, actual/nominal calls and native stop reasons.
Save decision/phase diagnostics only where applicable; random/native channels are not
mislabelled as RA model decisions. Save proposal/acceptance wall-clock times separately
from shared initialization and final verification; fixed method order limits runtime
comparisons. Paired bootstrap remains conditional/exploratory, not multiplicity-adjusted
confirmation. Do not pool missions as independent identically distributed repeats.

Advance only after reading all cases and failures, checking refined feasibility and
whether interval feedback or representation brings an effect beyond the simple hybrid.
No requirement to beat Random on every metric. No case selection based on winners.
If a numerical issue appears, fix that before interpreting rank. If no core increment
appears, retain negative evidence and revise scientific scope rather than add modules.
RCAIDE engineering MDA and chance/CVaR/safety claims remain outside this protocol.
