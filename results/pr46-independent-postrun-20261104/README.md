# PR #46 independent read-only PR45 postrun audit — 2026-11-04

Status: `COMPLETE_AUDITED_POSTRUN` (independent postrun review only; **not** a task or scientific qualification).

## Provenance
- PR #46 head: `65cd6904713f3341b282ef75570ca4f3e42178fb`
  *Bind independent postrun certificates to the exact uploaded plan, real ledger, source and archive SHA256 identities*
- PR #45 runner source SHA: `386082beeacac8ae43d0dfb1ccda5c5ad3b356cf`
- PR #45 archived results commit: `d42ac115231372a39e0557aa8f61948b7783f61b`
- Original result directory (read-only): `D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z`
- Entry point: `run_pr45_postrun_audit_pycharm.py`, executed directly with the `Rcaide_vsp` interpreter (equivalent to the PyCharm Run configuration).

## Environment
- Python 3.9.25 (`C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe`), SQLite 3.51.2
- `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- Command:
```
python -u run_pr45_postrun_audit_pycharm.py \
  --root "D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z" \
  --output "D:\LSBO\RA-LSBO-pr45-independent-postrun-20261104-verify1"
```
- No aircraft, RCAIDE or simplified-aircraft evaluation was executed. The SQLite ledger was opened with `mode=ro`.

## Original evidence unchanged (pre/post SHA256 identical)
| file | SHA256 |
|---|---|
| real_calls.sqlite | `a30b0dadb5d619be9b6c0d70c5aa29fe115f86b85d62a449e4acf05325d6f1b7` |
| audit.json | `81e5cf82ae40e8bffd5ae085a9be5fca5849cd48ed50028f7913bdb1302adf17` |
| completed.json | `6bb032d927b39c77532a6147c1970aee7872685cce8a66e7a846ed54c800973c` |
| plan.json | `ac5f1424f3c8d52eff9ffd835cb5004224be965afee8ec15069e1fea38c43012` |
| execution_status.json | `e374396fe85c3c8529d3a0c6c9dad5f17524ffd91b4a08bd3e27d00831ce86e1` |

## Key results
- 151,006 charged calls = 24,576 initial + 119,808 online + 6,622 verification; 34 optional verifications unused; 2,048/2,048 groups; 6,656 method `result.json`; 6,622 `verification.json`.
- No duplicate call ids; event chain head recomputed as `10e49c0a8f6c89932cddc1ae844e3cfbb9ab2d05fe407902f42eaa240673f44e`; no errors.
- Shared-initial 20 s `q_A` feasibility:
  - ordinary: 5,340 / 12,288 = 43.46% (95% CI 42.58–44.34%), passes the original diagnostic (lower CI > 20%);
  - sparse: 1,198 / 12,288 = 9.75% (95% CI 9.23–10.29%), **does not pass** the original diagnostic (upper CI < 10%). The 140 km sparse-domain claim is therefore not statistically established by this posthoc sample.
- Paired RA-LSBO vs full_latent, 1,024 pre-selected seeds, posthoc descriptive:
  - ordinary energy: mean −20.26 Wh (95% CI −40.25 to −0.28), median −1.53 Wh; 526 vs 497 lower-energy wins, 1 tie;
  - sparse energy: 1,022 paired feasible; mean −176.54 Wh (95% CI −305.85 to −47.22), median −2.97 Wh; 526 vs 496 lower-energy wins, 0 ties;
  - sparse feasible discovery: 1,024 vs 1,022 (RA-only 2, control-only 0; exact McNemar p = 0.5).
- All p-values and confidence intervals are posthoc exploratory, not preregistered. No formal seven-arm equivalence or scientific superiority is claimed.
- The original ledger status remains `RUNNING_REAL`; the audit does not rewrite the historical terminal state.

## Files
| file | SHA256 |
|---|---|
| `postrun_report.json` | `e44eeb189a94593f89d632871e36669ad0f04540a49d527c0243dea94c915bd5` |
| `paired_main_by_seed.csv` | `d24b60872a834eff27970f7f8af157d0bbd93c619d879b9539824e5cbaf6252b` |
| `console.log` | `93d8e2964921112564918d5086e2f67be44369d712917889e608a42f129e2bf3` |
| `pre_run_sha256.txt` | `ab2d8b45e0f46fa9202c1f26fdcf47200187bd10c709d78e8811f138d46f0bd1` |
