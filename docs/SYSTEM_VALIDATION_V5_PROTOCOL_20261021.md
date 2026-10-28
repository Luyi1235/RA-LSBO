# Architecture-aware acquisition amendment before real optimizer outcomes

Date: 2026-10-21, Asia/Shanghai.

The human has authorized complete real MDA plans up to 500 calls. This amendment
counts every executed current-study call across v1 through v5, including aborted
calibration designs, within a single maximum of 500. Historical warm-start cost is
reported separately (420). Existing frozen studies, physical sources and outcomes
are immutable. No real optimizer outcome exists at amendment time.

The preceding explicit-constraint-only calibration design had no mission-failure
positive among its first 44 settled calls. It is stopped, with every in-flight
call allowed to settle and charged. Its entire settled calibration block becomes
training acquisition. Its original artifacts and labels remain unchanged.

The full budget is 140 cumulative training-acquisition calls, 72 completely new
calibration calls, and at most 288 optimization calls (8 methods x 3 seeds x 12).
All prior calls count in the 140, leaving only 140 minus prior actual cost for
new architecture stress samples. There is no reset or per-version exemption.

New training candidates are frozen before their outcomes. They change one legal
discrete factor around verified failure anchors 5 and 19, with priority to airfoil
and blade count. Up to four configurations receive deterministic continuous
jitter at unit radii 0.002, 0.01, 0.03, 0.06. All pass the unchanged cheap precheck;
duplicate designs are excluded. Correlated neighborhoods do not establish broad
generalization or independent failure mechanisms.

After all training acquisition settles, fit the unchanged planning GP and freeze
all 72 new calibration inputs before any of their outcomes: 24 fresh legal
neighbors of known mission-failure configurations, 12 GP explicit outer points,
12 near points, and 24 inner points. Represent every observed positive discrete
configuration in the hidden neighborhood stratum, then cycle fixed radii.
These are held out from initial RA training. Directed calibration prevalence
does not estimate population prevalence and is not a safety confidence bound.
There is no further quota adjustment based on calibration outcomes.

The eight algorithms, seeds 31/32/33, 12-call ceilings, encoder settings, risk
models, calibration policy, acquisition rules and stop conditions are unchanged
from v4. Common history is chronological initial data, training acquisition,
then the fresh calibration block. All methods receive identical shared history.
The original LSBO source remains unchanged. Source hashes, exact past-only
forecast reconstruction, native baseline replay and independent worker/ledger
audit accompany the final descriptive comparison.

First unknown or infrastructure error triggers global stop; all in-flight calls
settle and remain charged. Each optimization run retains its six-nonconfirmed-
feasible stop rule. All early stops, failures and zero improvements are retained.
No positive superiority conclusion is required for experiment completion.
Missing real failure classes remain explicitly unavailable; controlled labels
never enter physical training.
