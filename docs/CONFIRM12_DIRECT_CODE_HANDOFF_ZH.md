# RA-LSBO 151,040-call confirmation: direct code handoff to the Windows workstation

This stacked PR is based on PR #41. It supplies **code and Windows CI**, not a replacement for the scientific qualification process and not a production execution approval. No aircraft or RCAIDE calls are executed by any newly added file.

## Implemented here

- `confirm12/structured_record_ledger.py`: new write-once SQLite transaction/event ledger for authentic `DesignRecord` structures. Claim a unique budget ticket **before** settling evidence; ensure shared 12 initials are fully settled before any online trial; enforce sequential 18-step online prefixes; bind candidate/source contract for 20s and 5s, verify record identity and event hash chain, reject duplicate tickets/contract drift, and preserve unknown/incomplete results. `RECORD_CONTRACT_TEST_ONLY` is enforced; **record shape is not proof that a physical evaluation happened**.
- `confirm12/offline_ledger_trial.py`: actual frozen H1 `OfflineHistoricalTrial` `propose/accept` wrapped with the above ledger. Reads exactly 12 settled shared initial records and prevents another arm's outcome being injected; supplied feedback is audited but no `evaluate` method is called. One historical `random` full one-step proposal/accept is tested; it cannot be claimed as seven-arm confirmation.
- `tests/test_confirm12_structured_record_ledger.py`: 12-way competing ticket tests, duplicate/over-budget rejection, truth-record identity and source-contract checks, unknown, tampering, incomplete ledgers.
- `tests/test_confirm12_offline_ledger_trial.py`: test real H1 random optimizer with the ledger, including a wrong-feedback stop and no-evaluator guards.
- `tests/test_confirm12_real_onehot_gp_offline.py`: actual frozen `onehot_stable` H1 GP fitting and SLSQP/propose/accept smoke **on fabricated offline observations only**. The test is opt-in on Windows (`CONFIRM12_TEST_GP_ONEHOT=1`) and does not constitute real model qualification.
- `.github/workflows/ra-confirm12-structured-evidence.yml`: Windows regression suite with aircraft calls disabled.

## For the workstation: concrete immediate integration task

1. Pull PR #42's HEAD into an **isolated new worktree** based on PR #41, without overwriting previous results or PR #36's qualified plan. Review the new ledger's state transitions and actual source identity before reusing; keep it separate from the old `SIMULATED_ONLY` ledgers.
2. Execute the Windows regression tests, including `CONFIRM12_TEST_GP_ONEHOT=1` and confirm source SHA and outcomes; report **actual** memory and wall time for true H1 onehot GP/SLSQP fitting, not only random or hash simulation.
3. Next write the **real `DesignRecord` production adapter** for the original reduced-aircraft evaluator, with an external immutable phase-approval checker and independent physical evidence cross-audit. The new ledger is a contract-validation component, **not** yet a physical provenance ledger.
4. Use actual H1 `hybrid_onehot`, `corpus_stable`, `onehot_stable` and `random` interfaces, but scientifically freeze the mapping from these historical names to the confirmation methods. Complete registered reverse and interval-as-missing full closed loops; **CBOB must be a faithful qualified adaptation**, not a synthetic/relabelled random control.
5. Complete seed-history exclusion, q_A qualification (requires independent capped true calls and review), LF/CRLF checksum compatibility without mutating historical evidence, frozen independent nested roster, and 12-worker actual seven-method stability. No 24-hour deadline.
6. Produce a single **genuine** PyCharm entry with READY/BLOCKED gate: all method, task, source, plan, seed, budget and authority conditions must pass before any airplane call. No hidden `--force`. Keep original P0–P6 and PR #36 records immutable. No RCAIDE or physical calls in the current development task.

**Why PR #41 was not production-ready:** its `execute` mode raised unconditionally, the seven labels still mapped to a synthetic policy, and all evidence ledgers were `SIMULATED_ONLY`. Passing 73 or more tests on mock systems does not qualify real GP/VAE methods, CBOB adaptation or the 151,040-call budget. This PR closes **part** of the real-evidence interface gap and supplies a real-policy-to-ledger test. It does **not** remove the other blockers.

When work is complete, report code SHA/PR, verified true-policy tests, per-arm qualification, real-call ledger authority proof, seed and task receipts, 12-worker memory, and whether a qualified full confirmation plan may start.
