# Compact MDA Interruption Audit - 2026-10-26

This document records a read-only audit of one interrupted historical RCAIDE
qualification call. The generated JSON is independent audit evidence, not a
ledger `settle` event, a recovered physical evaluation, or completed qualification.
The original started call remains charged. No retry is authorized or performed.

## Scope and Identity

- Study: `D:/LSBO/RA-LSBO/runs/compact_mda_20261021T081006Z_20eb608d`.
- Phase: `qualification_recovered_20261025` under that study.
- Phase call: **37**; distance: **65 nmi**; seventh candidate at that distance.
- Ledger: `task_65/budget.jsonl`; token: `call_19`.
- Candidate key: `61f3bbb27e0cba1e1a2f4d126bd99766c7830b5f3217adc55841c15500451413`.
- Contract: `a16f28b756f61e8f35c9f7e45394c15b0c33b350a3c40791a7c3d45866d24378`.
- Worker: `workers_65/128c10d7e34d4a159aba376ab8062967` under the phase.
- Actual solver working directory: `solver_work` under that worker; initial
  subprocess working directory: `frozen_base` under the phase.

The candidate is identical in the frozen plan, submission, reservation, and
worker request. The tool checks all 37 worker bindings, both ledger hash chains,
and all 36 observation-to-settlement bindings without importing project modules.
It pins eight artifact SHA-256 values to the preceding read-only inspection and
checks six relevant executor/worker source hashes against the original plan.
Hash agreement establishes artifact identity, not physical validity.

## Accounting and Last Evidence

| Distance | Submissions / Charged Calls | Settled / Observations | Pending Started |
| --- | --- | --- | --- |
| 55 nmi | 30 | 30 | 0 |
| 65 nmi | 7 | 6 | 1 |

The pending ledger reservation is sequence 19; its start is sequence 20.
The latter remains the last ledger event, with hash
`209db409cceba4b21ba52ed81ef5a81580fc0c852d2bbc92d440beafa1f1752e`.
The audit validates the SHA-256 chain and state transitions independently using
the same standard-library JSON serialization used in the historical ledger.

The worker log records completion of `procedure.simple_sizing` and
`procedure.weights`, then entry into `procedure.missions.design_mission`.
Its last lines are:

```text
Solving DER segment.
[RCAIDE] DER fallback: evaluations=5, residual_inf=4.45397, rejected=0
```

This is intermediate progress in DER, not a terminal convergence/failure report.
The console ends with a 15-second waiting heartbeat and configured timeout of
1800 seconds. There is no observed timeout, exit code, final candidate status,
mission-end marker, or exception for this request. Earlier DomainSolveError
messages belong to already settled calls and must not be assigned to call 37.

The worker log last-write time is `2026-10-25T12:17:37.2393475Z`; the console
last-write time is `2026-10-25T12:17:42.2173476Z` (20:17:37 and 20:17:42 in
Asia/Shanghai). These timestamps do not establish the exact interruption time.
The JSON records current file metadata and the UTC audit interval.

The weights log prints `b=11.1m`, `RCAIDE 96.6`, `Torenbeek 170.0 kg`, and
`MTOW 1326.0 kg`. These numbers have only the printed log precision. They are
diagnostics, not precise training targets, objective values, or constraint
observations. The JSON does not extract those values into numeric outputs.

## No Recoverable Terminal Result

The pending worker contains only `request.json`, `solver.log`, an empty
`battery_domain_trials.jsonl`, and 83 copied Airfoils input files. Every copied
input is checked against its frozen source. There is no serialized mission
checkpoint or unexplained solver artifact. The tool records the full inventory.

`response.json`, `response.tmp`, `result.json`, `solver_error.json`,
`energy_audit.json`, and `simplified_powered_lift.json` are absent in this worker.
The phase has no `result.json`. Its 36 existing responses are all bound to
already settled calls. The worker source writes `response.tmp` only after
evaluation returns, then renames it to `response.json`; neither file is present.
An empty numerical-trial log and `rejected=0` prove no final physical outcome.

## Audit Contract

`outcome` is `unknown`. Physical success, solver success, feasibility, objective,
all constraints, all five hidden failure labels, and hidden success are `null`.
Observation masks are false. MDA entry is recorded separately as execution
evidence; completed-MDA status and final elapsed time remain unknown.

```json
{
  "original_call_remains_charged": true,
  "administrative_settlement_performed": false,
  "qualification_completed": false,
  "qualification_gate_satisfied": false
}
```

The record preserves the original `started` state, token, charge, reserve/start
events, source paths, hashes, and UTC check time. It makes no replacement ledger
entry, cancellation, refund, or duplicate charge. Both original qualification
rules retain their `no unknown outcomes` requirement. This report does not clear
that requirement and cannot supply the missing physical outcome.

The old `study_active.lock` is read and hashed but never removed or changed.
The tool performs no process query; process liveness and interruption cause at
audit time are unknown. It cannot establish whether an external actor killed
the process or whether the machine or parent process stopped unexpectedly.

## Reproduction and Verification

Use Python 3.12 or newer. All imports are from the standard library; no RCAIDE,
model runner, project evaluator, or project ledger class is imported or invoked.
From `D:/LSBO/RA-LSBO`, run:

```powershell
python -I -B tools/audit_interrupted_compact_mda.py --self-test
python -I -B tools/audit_interrupted_compact_mda.py
python -I -B tools/audit_interrupted_compact_mda.py --verify
```

The default command exclusively creates:
`D:/LSBO/RA-LSBO/runs/compact_mda_interruption_audit_20261026/audit.json`.
It refuses an existing output directory. Subsequent checks must use `--verify`,
which recomputes the evidence and compares all fields except the audit interval,
without writing any file. `--self-test` uses only in-memory fixtures.

Before publication, every file and directory in the old study is inventoried;
all file hashes, sizes, and modification times are compared again at the end.
The JSON includes that manifest, its aggregate SHA-256, key evidence records,
and the audit tool's own source hash. A changed pinned artifact, broken ledger
chain, ambiguous candidate, newly found result, or changing source causes the
audit to fail before output creation. The JSON may be large because it retains
the complete original-study file manifest for independent review.

Only the new audit directory is written by the default command. The two new
source/document files are separate from the old study and the running low-cost
study; no existing source, ledger, lock, or execution artifact is edited.

## Recorded Verification

The audit ran from `2026-10-26T04:24:23.424127+00:00` through
`2026-10-26T04:24:38.105051+00:00`.

- Eight in-memory self-tests passed, including tampered chains/candidate hashes,
  invalid state transitions, nonstandard JSON, and preservation of unknown labels.
- An independent AST check found only standard-library imports.
- Both historical phase ledger chains and all 37 candidate/worker bindings passed.
- The 4013-file original-study manifest and directory inventory matched before
  and after the audit. The report retains 257 explicit evidence-source records.
- Read-only `--verify` reproduced every report field except the UTC audit interval.
- A separate JSON check confirmed all requested null labels and false settlement/
  qualification flags. Re-running the default command refused the existing output.
- The output directory contains only `audit.json` (1,639,333 bytes).

Audit JSON SHA-256:
`58c875978ea9bf375a2d649f700be6efb1049546acfd1cc6b31438f0958fd5b7`.

Original-study manifest SHA-256:
`45cd81958de93a252349ea5ccdfdb33a067707c2aeaac2fbbf8c2a06cbfab563`.

These checks validate the audit and retained accounting evidence. They do not
recover a physical outcome or perform an administrative settlement.
