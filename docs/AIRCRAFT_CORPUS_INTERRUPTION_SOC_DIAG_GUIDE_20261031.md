# Interrupted aircraft corpus pilot: bounded SOC diagnosis

This document is a new diagnostic task, NOT permission to restart, resume, or
modify the interrupted 2,672-call pilot. The original 2,212 charged calls are
preserved and there are no complete pilot results or final audit.

## Immutable source

Original code commit: fc1ac773f5bd9e60abd2ca4b56bd971a707735e2
Original Release: https://github.com/Luyi1235/RA-LSBO/releases/tag/aircraft-corpus-pilot-interrupted-20261031
Input archive: partial_output_records.zip
SHA256: aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9

Target: aircraft_sparse, seed 9936, method old30, failed online iteration=29.
The diagnostic must run against original Windows / Rcaide_vsp. Keep all
original lock and evidence files untouched.

## Why examine SOC first

In successful original iteration 27 the SOC GP had 2/8 valid kernel candidates;
in iteration 28 it had 1/8, surviving only (.05, .5). The other seven were
recorded as RuntimeError. All seven other response heads had 8/8 valid
candidates at iteration 28. The failing iteration 29 proposal was not saved.
SOC is therefore a strong candidate, not a confirmed failed head.

The 40-observation D28 SOC dataset has 16 exact and 24 interval outputs.
The 41-observation D29 SOC dataset has 17 exact and 24 interval outputs.
The newly added observation was a completed exact-value record. That does
not prove the new observation was erroneous, nor that EP failed to converge.

Original grouped GP wrappers catch errors and obscure the lowest-level
message. The separate local diagnostic failed writing a nonfinite numeric
diagnostic into JSON. That is a diagnostic serialization bug, not a proven
second model/optimizer bug.

## Registered bounded operation

Use tools/diagnose_failed_soc_prefix.py from the pinned diagnostic commit
in the handoff message, in a fresh independent Windows worktree.

From PowerShell at repository root, set unchanged original thread variables:
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
PYTHONHASHSEED=0, MPLBACKEND=Agg.

Command 1 (pure registration, zero fits):
python tools/diagnose_failed_soc_prefix.py plan --archive FULL_ZIP_PATH --output NEW_DIAGNOSTIC_DIRECTORY

Command 2 (only if the plan succeeds; original Windows environment):
python tools/diagnose_failed_soc_prefix.py run --archive FULL_ZIP_PATH --output NEW_DIAGNOSTIC_DIRECTORY

The plan registers two old30 feature fits and just eight D29 SOC kernels.
The run checks D28 original feature identity, stops BEFORE GP if it differs,
then attempts D29 SOC candidates once each using original parameters. The
original code and inference update rules are never changed. Root exceptions
from posterior construction vs log-evidence evaluation are recorded separately.

NaN and Inf in diagnostics are preserved as explicit nonfinite markers rather
than silently replaced by zero; the original numerical input is untouched.
The new diagnosis owns a NEW lock and must not be retried or resumed if
interrupted. It does not re-run the previously unlogged D28 candidate.

The CI cross-platform-check option is for Linux structure-only PREVIEW:
it never authorizes GP fitting. DO NOT pass it on Windows. The real
Windows run requires physical contract and D28 identity to match exactly.

Maximum: 2 encoder refits, 8 original D29 SOC kernel attempts, 0 new aircraft
evaluations, 0 BO proposals, 0 mathematical true-function evaluations,
0 RCAIDE calls.

## Return and stop

Return plan.json, features.json, candidate_attempts.jsonl, result.json,
environment/logs, and before/after original ZIP SHA256. Report the precise
candidate-level error types/messages and exception stages. If all eight
D29 SOC candidates work, report SOC failure NOT reproduced and STOP;
do not automatically expand to other constraints.

Do not change EP max_sweeps, damping, jitter, GP length grid, sample sets,
training epochs, any labels, or the historical experiment source/lock.
This is an isolated diagnostic, not a corrected experiment version.
