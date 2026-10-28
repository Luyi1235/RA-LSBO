# Sparse interior-failure controlled supplement

This supplementary case is designed after the first controlled case, before any
labels or optimizer outcomes of this case. It is not a retrospective tuning or
replacement of the original results. No aircraft or physical MDA data enter it.
Its additional real MDA cost is zero.

The serial five stages, hierarchy and objective stay as in the original
controlled process. The feasible envelope radii become .12 and .15 around
(.69, .63). A mission-failure disk of radius .055 around (.78, .69) is added.
It lies partly within the explicit feasible envelope and contains the
unconstrained objective minimizer (.81, .72). A failure censors later stages.
No source code of the original controlled case or aircraft optimizer is edited.

The case, code and method settings are frozen before sampling. Use the existing
eight methods and seeds 31, 32, 33, 900 cheap initial observations per seed:
600 chronological training, 300 calibration. Each method gets at most 16 new
controlled calls, at most 384 across all optimizers. Existing failure stop 6,
discovery/boundary limits 4, GP 45, z8/600 initial epochs and 20 incremental
epochs stay fixed. A separate 1500-point static train/calibration/test split
uses 600/300/600, as in the first controlled case. An independent 10000-point
cheap sample estimates the screened-domain feasible fraction descriptively.

No safety-bound or broad superiority claim is permitted. Include all early
stops, paired actual-call comparisons, independent source/label/ledger checks
and checkpoint-based forecast reconstruction. Report the original case too.
