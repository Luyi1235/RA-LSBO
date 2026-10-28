# Partial observation continuation, frozen before further MDA

The entire experiment remains capped at 480 real MDA calls across versions:
4 in v1, 48 in v2, 16 in v3, at most 412 in this continuation. No new
candidate sampling or optimizer tuning is introduced. All 120 probe inputs,
the 20-positive/20-negative training target, fresh 72-point calibration,
eight methods, three seeds and 12-call optimizer budgets remain unchanged.

Probe 66 completed all 11 mission segments and passed the frozen energy audit
with mission depletion 49431.89097596289 Wh. Its subsequent simplified lift
calculation failed because dry CLmax plus the finite native delta was nonpositive.
The original ValueError, requests, responses, solver diagnostics, ledgers and all
prior reports remain immutable.

The new source-bound normalization rule accepts only that exact postprocessing
failure, expected helper source hash, audit version, passed audit, positive finite
energy and converged mission segments. It retains the audited energy, masks all
engineering constraints and leaves final feasibility unknown. It labels none of
the five preceding failure mechanisms positive. Other errors retain their prior
semantics and stop policy. The physical model is not changed.

The new annotation is stored alongside the original record and its digest.
The independent auditor checks the ledger against the original annotation and
the new annotation against the unchanged raw response. New workers use the same
explicit observation rule. A separate label-contract hash identifies this change.
All changed source files are frozen before any additional MDA.

Real mechanism support and optimization outcomes must be reported separately
from controlled-process results. Neighbor counts are not independent failure
generalization evidence. Missing mechanisms cannot be trained by manufacturing
labels. Completion of comparisons is not a claim of RA-LSBO superiority.
