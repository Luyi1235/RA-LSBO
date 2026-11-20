# PR149 full-budget external comparison pilot

Fixed code SHA: `2393ff261e39e461ac1b68e15b2a9458791a2f6d`
Runtime: Python 3.12.10, torch 2.8.0+cpu; see `runtime.txt` and `environment.txt`.

| scenario | status | paid_calls | rcaide_mda_calls |
|---|---|---:|---:|
| payload | `VERIFIED_FULL_BUDGET_EXTERNAL_COMPARISON full-pilot-payload 1 376` | 376 | 0 |
| headwind | `VERIFIED_FULL_BUDGET_EXTERNAL_COMPARISON full-pilot-headwind 1 376` | 376 | 0 |

Each scenario ran one seed with initial sizes 8 and 16, and five arms:
`sign_greedy`, `interval_greedy`, `interval_active`, `logcei_sign`, `archsbo_screened`.
The sealed run directories are `payload/29060100/` and `headwind/29070100/`.

This is a pilot, not a formal multi-seed comparison.
