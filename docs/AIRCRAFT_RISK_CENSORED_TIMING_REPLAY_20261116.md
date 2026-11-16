# Completion-risk learning from interval-censored reserve-crossing time

PR113 restored configuration coverage and retained much of the first-discovery benefit. The frozen primary Brier differences versus maximin remained −0.001034 in still air and +0.001466 in headwind, with both 95% intervals spanning zero. At 16 calls in headwind, the configuration-macro signed usable probability errors were +0.02922 for coverage, +0.04962 for unrestricted information acquisition and +0.04207 for balanced information acquisition. The PR113 formal ledgers contained 6,144 reduced calls, all settling as completed tasks or physical SOC-reserve stops. This makes them suitable for isolating additional supervision in the completion head.

## Paired archival experiment

Input is the sealed PR113 result commit 99e868125d2dde224b6203899382a3ddd77152e5, with explicit hashes for the three profile manifests. This is a retrospective comparison on fixed data selected after reviewing PR113. It adds zero real or simulated aircraft calls. It keeps both original training paths (balanced information and maximin), checkpoints, held-out labels and explicit features. It replaces only the completion probability P(C); it holds the original conditional feasibility forecast P(F|C) fixed and multiplies the two for usability. Acquisition never receives replay forecasts.

Let D be the planned duration of the climb, wind-adjusted cruise, descent and reserve. Let the latent margin be the fraction of mission time remaining at the first reserve-threshold crossing, with positive values on a stop. An observed stop is detected at time t, after an integration step of at most h. Its margin is observationally bracketed by (D−t)/D and (D−t+h)/D, intersected with the valid time support and positive margin. Completed missions provide only the bound margin ≤ 0. No SOC beyond termination, extrapolated endurance or completed-task time-to-failure is imputed. This experiment requires the sealed native reserve-stop contract. Competing termination reasons would require a separate model and cause accounting.

| Completion forecast | Feedback |
| --- | --- |
| original_probit | PR113 recorded completion probabilities |
| timing_sign | fixed-scale latent deficit GP with sign-only bounds |
| timing_interval | identical latent deficit GP with stop-time intervals instead of positive signs |

The two latent deficit GPs both use the existing grouped Matérn search grid, 0.25 prior scale in fraction-of-mission units, noiseless interval likelihood and the same inference code. Completed missions have identical supervision in both. Their paired difference isolates the interval information within this model family. Relative to the original probit model, there is also a likelihood change. Scale is fixed before this replay, with no search over the held-out results.

## Reporting

The primary contrast is timing_interval minus timing_sign on the balanced-information path. Score held-out configuration-macro usable Brier, averaged over calls 8, 12 and 16 per seed, separately for each of the two conditions. Also report completion Brier, other forecast comparisons, and the matched coverage path. Use the existing paired bootstrap protocol. Report every contrast regardless of direction. Treat any gain as retrospective evidence for confirmation on fresh seeds before testing downstream acquisition or energy performance.

Each formal replay produces 864 metric rows (24 seeds × 2 paths × 3 checkpoints × 3 forecasts × 2 endpoints). Smoke produces 12 rows. Original source ledgers are read-only. See run_aircraft_risk_timing_replay.py for the command with required input profile, output directory and full code SHA.
