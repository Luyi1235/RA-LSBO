# RA-LSBO-H2 P1 Windows smoke

- Executed code: `7ca420613ccb50e138dc3c9d4292ac8e863c4ecf`.
- Windows Rcaide_vsp; seed 11061; six methods on 100 km ordinary and 140 km sparse tasks.
- 12 trials, 24 online proposals, 43 settled simplified-model calls: 12 shared initial, 24 online, seven 5-second checks. RCAIDE MDA: zero.
- Independent audit passed; the read-only `smoke_receipt` numerical gate passed. Seven incumbent verifications passed. Five sparse trials had no nominal feasible incumbent, so no verification was run for them.
- This is a single-seed, two-online-step infrastructure smoke, not a statistical comparison or physical validation. The four-score warmup cannot be crossed within this smoke; adaptive-weight performance remains untested in this run.
- The full original output including ledger, lock, proposals, observations, checkpoints, source snapshots and console logs is provided in the associated GitHub Release asset. No prior pilot was resumed.
- `RA-LSBO-H2-P1-smoke-evidence-20261102_01.zip` SHA-256: `76fa4f10b853d070bcc66e73d4d4a5f72aa5d036709311ffa9c75a368ca8ca1d`.
- `SHA256SUMS.txt` lists the individual original evidence files. The archive itself and this manifest are hashed separately in the release.
- The future 1,008-call pilot was neither started nor authorized by this smoke.
