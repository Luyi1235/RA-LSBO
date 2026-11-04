# PR #50 risk-event semantics audit — 2026-11-04

Status: `COMPLETE_READ_ONLY_LABEL_AUDIT` over all 151,006 PR #45 reduced-aircraft records. **0 new aircraft/RCAIDE calls**; the original ledger was opened in SQLite read-only mode and is byte-identical before and after.

## Provenance
- Analysis code SHA: `6e7bb5be99900d76814e49eaa45221672ced4b0b`
- PR #45 source SHA: `386082beeacac8ae43d0dfb1ccda5c5ad3b356cf`
- Source: `D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z`
- Entry point: `run_ra_lsbo_risk_semantics_pycharm.py` (Rcaide_vsp, Python 3.9.25)
- Smoke: `--max-records 1000` → `PARTIAL_DIAGNOSTIC`, 1,000 rows.
- Full: no arguments → fresh output dir, 151,006 rows, exit 0.

## Source integrity (pre-run and post-run identical)
| file | SHA256 |
|---|---|
| plan.json | `ac5f1424f3c8d52eff9ffd835cb5004224be965afee8ec15069e1fea38c43012` |
| real_calls.sqlite | `a30b0dadb5d619be9b6c0d70c5aa29fe115f86b85d62a449e4acf05325d6f1b7` |
| audit.json | `81e5cf82ae40e8bffd5ae085a9be5fca5849cd48ed50028f7913bdb1302adf17` |
| completed.json | `6bb032d927b39c77532a6147c1970aee7872685cce8a66e7a846ed54c800973c` |
| execution_status.json | `e374396fe85c3c8529d3a0c6c9dad5f17524ffd91b4a08bd3e27d00831ce86e1` |

`real_calls.sqlite` remains 856,600,576 bytes, mtime unchanged; ledger state remains `RUNNING_REAL`.

## Headline semantic finding
- All 151,006 records have `observation.status = completed`; `status_completion = 1` for every row. There are **0 hidden failures and 0 infrastructure errors**, so the old status-based completion label has no negative examples in this campaign.
- `mission_completed`: **116,331 true (77.04%)**, **34,675 false (22.96%)**, 0 unknown.
- Every one of the 34,675 false rows has `termination_reason = physical_soc_reserve_abort`, an interval objective, and SOC sign false. There is no mission-true row with an interval target and no mission-false row with an exact target.
- `status_completed_mission_incomplete = 34,675` (22.96%): this is the key gap between evaluation-record status and actual mission completion. It is a counted label difference, not a rule violation; `semantic_issue_counts` is empty.

## Exact objective coverage
| scope | exact targets | calls | coverage |
|---|---|---|---|
| all | 116,331 | 151,006 | 77.04% |
| aircraft_ordinary | 68,435 | 75,520 | 90.62% |
| aircraft_sparse | 47,896 | 75,486 | 63.45% |
| role=initial | 11,969 | 24,576 | 48.70% |
| role=online | 97,740 | 119,808 | 81.58% |
| role=verification | 6,622 | 6,622 | 100.00% |

## SOC and other constraint evidence (exact / interval / missing; sign true / false / unknown)
| constraint | exact | interval | missing | true | false | unknown |
|---|---|---|---|---|---|---|
| soc | 116,331 | 34,675 | 0 | 116,331 | 34,675 | 0 |
| power | 116,331 | 34,675 | 0 | 116,331 | 0 | 34,675 |
| current | 116,331 | 0 | 34,675 | 116,331 | 0 | 34,675 |
| voltage | 116,331 | 0 | 34,675 | 116,331 | 0 | 34,675 |
| stall | 151,006 | 0 | 0 | 120,524 | 30,482 | 0 |
| root_bending | 151,006 | 0 | 0 | 138,107 | 12,899 | 0 |
| packing | 151,006 | 0 | 0 | 151,006 | 0 | 0 |
| span | 151,006 | 0 | 0 | 151,006 | 0 | 0 |

SOC is never missing: it is exact on 77.04% of calls and an interval on the 22.96% aborted calls, where the interval resolves to a known violation. Current/voltage are missing exactly on the aborted calls; power becomes interval on the same calls.

## Physical and usable feasibility
| scope | physical true | physical false | physical unknown | usable feasible | calls | usable share |
|---|---|---|---|---|---|---|
| all | 78,973 | 72,033 | 0 | 78,973 | 151,006 | 52.30% |
| aircraft_ordinary | 45,951 | 29,569 | 0 | 45,951 | 75,520 | 60.85% |
| aircraft_sparse | 33,022 | 42,464 | 0 | 33,022 | 75,486 | 43.75% |
| role=initial | 6,538 | 18,038 | 0 | 6,538 | 24,576 | 26.60% |
| role=online | 65,813 | 53,995 | 0 | 65,813 | 119,808 | 54.93% |
| role=verification | 6,622 | 0 | 0 | 6,622 | 6,622 | 100.00% |

`usable_feasible` equals `mission_and_usable_feasible` on all rows: status completed ∧ target exact ∧ all eight constraints known satisfied. Physical feasibility is never unknown because every row with censored/missing constraints also has a known SOC violation; individual constraint unknowns still exist in 34,675 rows.

At unique-design level (repeated evaluations collapsed): 141,775 distinct designs, 71,732 usable (**50.60%**); ordinary 42,350/71,275 (59.42%), sparse 29,382/70,575 (41.63%).

## Termination reasons
- `complete`: 116,331
- `physical_soc_reserve_abort`: 34,675 (ordinary 7,085; sparse 27,590)

## Caveats
- Retrospective label audit only; no model fitting, calibration, hidden-failure generalization or optimization-benefit claim.
- `status=completed` is not evidence of mission completion or final physical feasibility; 22.96% of records are physically aborted flights recorded as completed evaluations.
- Initial versus online records have different selection mechanisms; the 100% verification share is a post-selection artifact.
- The 140 km sparse task has not passed independent sparse-domain qualification.

## Files
| file | SHA256 |
|---|---|
| risk_semantics_report.json | `c0015335ec2636730c7f33940088ae19965730d2fbaf9864fa2d3c05dec12dc0` |
| risk_semantics_by_group.csv | `5d88f119227598dd9be9c24111e92dcab70055b58af0fcc94c9ae534ae11a785` |
| risk_event_records.csv | `2179d878b8b5a7e29d691a3c416ef7af197dfdf8fb624605e3aba84ceb9823a1` |
| console.log | `824b6ad4812379088a81990756b3a1737eae13d58c8ae4cc2938cda00eb715c4` |
| smoke-console.log | `c7ed8c8c5b1d718bf916f87d8634ce6aff6b097b75047568faed797d1e10e01a` |
| pre_run_sha256.txt | `6560985b6f9445b20b149bba8041763f7483b262da5b0558b98eca820df5fce5` |
| output_sha256.txt | `0c6599064e9c95a88416ed6a684d27d3a5907d4209bedd3fc7cbdf44e90783b3` |
