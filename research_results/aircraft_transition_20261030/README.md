# Aircraft transition qualification, 2026-10-30

Execution commit: `b8a94c324cd23a0ef9f15c668fe9a247e4ad9a00`. The
independent execution worktree had clean tracked sources. The existing
`Rcaide_vsp` environment passed `pip check`; all 264 specified regression tests
passed with zero failures, errors, or skips. Python 3.9.25; NumPy 2.0.2;
SciPy 1.13.1; PyTorch 2.8.0+cpu; GPyTorch 1.11; BoTorch 0.10.0;
scikit-learn 1.6.1. Test-generated cheap evaluations are separate from the
research budget.

Smoke: 36/36 settled; qualification: 316/316 settled; unsettled: 0;
new RCAIDE MDA calls: 0. Both software audits passed. No optimization pilot
was run. An audit is not physical validation or pilot authorization.

The independent, geometry-accepted panels contain 128 designs per case:

| Case | Usable feasible | Known infeasible | Unresolved | Usable fraction | 95% binomial interval |
| --- | ---: | ---: | ---: | ---: | --- |
| ordinary, 100 km | 65 | 63 | 0 | 50.78% | 41.80%-59.72% |
| historical sparse, 140 km | 19 | 109 | 0 | 14.84% | 9.18%-22.21% |

The old `aircraft_sparse` name is not evidence that feasibility is below 10%.
These estimates describe the registered accepted sampling distribution, not
an unconditional design-space probability or safety guarantee. The result
JSON and report include all 12 branch strata and output kinds. Panel numerical
failures were zero. Complete/SOC reserve abort counts were 102/26 and 41/87,
respectively; aborts yielded partial observations, not imputed exact values.

Integration screening: ordinary 10 passed and 4 inconclusive; sparse 2 passed
and 12 inconclusive, with no `needs_review` statuses. Inconclusive means partial
outputs, not a pass. Anchors use 20/10/5 s; branch probes use 20/5 s. Among
passed screens, maximum relative energy ranges were 1.64e-14 and 5.67e-15.
For exactly observed constraints across available steps, the maximum absolute
range of normalized power was 0.002219 (ordinary, 10 checks) and 0.000724
(sparse, 2 checks); maximum SOC range was 6.58e-7 and 1.82e-7. Other exact
constraint differences and full per-design observations can be recomputed
from the unmodified ledger in the full archive.

Full archive: GitHub Release `aircraft-transition-20261030-1955`, asset
`aircraft_entry_20261030_01_full_records.zip`; 1,344,478 bytes; SHA-256
`ea1359fb7c76ea8641c54432e8e901fc4de065be6dc643590c585f6c1d7d7789`.
It contains 382 files, including complete ledgers, proposals, snapshots, locks,
registrations, logs, environment and JUnit records. Every archived file was
verified byte-for-byte against the original before upload, and the server asset
digest matched. This branch contains lightweight reports and audit records.
