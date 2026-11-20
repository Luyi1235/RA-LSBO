# PR149 full-budget external comparison — confirmation batch

Fixed code SHA: `2393ff261e39e461ac1b68e15b2a9458791a2f6d`.
Runtime: Python 3.12.10; imported torch `2.8.0+cpu`; `importlib.metadata.version('torch')` reports `2.15.0` (see `runtime-*.txt`).

| scenario | status | seeds | paid_calls | rcaide_mda_calls |
|---|---|---:|---:|---:|
| payload | `VERIFIED_FULL_BUDGET_EXTERNAL_COMPARISON confirmation-payload 24 9024` | 24 | 9,024 | 0 |
| headwind | `VERIFIED_FULL_BUDGET_EXTERNAL_COMPARISON confirmation-headwind 24 9024` | 24 | 9,024 | 0 |

Each seed ran initial sizes 8 and 16 and five arms:
`sign_greedy`, `interval_greedy`, `interval_active`, `logcei_sign`, `archsbo_screened`.
Sealed ledgers are under `payload/<seed>/` and `headwind/<seed>/`.
