# X-57 engineering comparison v2

Code `7d106a807559a564d4b4065a935fb8abe8b7e265`, mission 45 nmi, seed `20261125`. The run freezes eight initial designs with four per layout and shares their ordered results across the three arms. Each arm may make up to 12 online calls. The physical limit is 44 calls and the worker concurrency limit is 12. All four thread settings are fixed at one.

## Checks before RCAIDE

- `offline_report.json` contains 12 matched historical calls at 50 nmi and 20 at 55 nmi. It records the fixed candidate pool, both rankings and the per-call evidence alignment. The SHA256 is `9585e86cbe46629e48b156ca57f0011a4176ef0ab1b169c51b10676db9cc7360`.
- The full-budget mock in `mock_report.json` settled 44 subprocess calls. It charged the eight initial calls once and 12 calls per arm for online evaluation. Both RA arms fitted the same synthetic partial observation, using a violation sign and a numeric lower bound respectively.
- Both protocol tests passed, including the matched mock and a fixture that exercises the native consecutive-missing-objective stop.

## Reproduction

Use the interpreter and four frozen input paths in the [previous PowerShell reproduction script](https://github.com/Luyi1235/RA-LSBO/blob/613983af769de05ce7ca8a15a7c0be7bd38963aa/results/x57-engineering-energy-pilot-07a95dec/reproduce.ps1). At the code commit above, run `offline` with those inputs and a fresh output directory. Run the mock and real `pilot` commands with `--seed 20261125 --workers 12 --online 12`. The real command takes `--execute-rcaide --offline-report $proof --offline-report-sha256 $proofHash`, using the offline report from this same checkout and its computed SHA256.
