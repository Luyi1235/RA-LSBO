# Nominal aircraft pilot: seven methods and eight development seeds

Execution commit: `e4fbcd37ca4d096bbdac03fc57b0f122f006aea3`.
The independent execution checkout retained clean tracked sources. The existing
Rcaide_vsp environment passed dependency checks and all 327 specified regression
tests, without failures, errors or skips. Six local launcher regression tests also
passed. Python 3.9.25; NumPy 2.0.2; SciPy 1.13.1; PyTorch 2.8.0+cpu;
GPyTorch 1.11; BoTorch 0.10.0; scikit-learn 1.6.1.

## Completion and budget

- Smoke: 54/54 calls settled; 14 trials and 28 online proposals audited.
- Pilot: 3,664/3,664 calls settled; 112 trials and 3,360 online proposals audited.
- Pilot roles: 192 shared initial calls, 3,360 online calls and 112 held-out
  incumbent checks. Total new research calls: 3,718/3,718.
- Unsettled calls: zero. New RCAIDE MDA calls: zero. Native LSBO early stops:
  zero. No unused budget was reassigned.
- All 112 nominal incumbents were feasible and remained usable feasible at 5 s;
  all 112 numerical screens passed. No online numerical or physical-unknown
  outcomes were reported. Physical infeasible and partially observed samples
  remain in the original ledgers.
- Smoke and pilot audits passed. Original audit hashes were independently
  rechecked after completion (92 and 709 sealed evidence files, respectively).

The complete previous qualification was read only. It was not re-evaluated or
used as optimizer training labels. Fixed seeds 9911-9918, 100/140 km missions,
12 common initial designs per seed and 30 online evaluations per method were
retained. The final 5 s checks did not enter optimization or select replacements.

## Observed outcomes

Conditional final objective medians in Wh; lower is better. Every cell has eight
feasible runs, and every method achieved 8/8 nominal and 8/8 refined feasibility
in both missions.

| Method | 100 km median Wh | 140 km median Wh |
| --- | ---: | ---: |
| Random | 29957.277009 | 39823.553099 |
| Phase RA latent (`ev_latent`) | 28199.401685 | 37666.122316 |
| Matched hierarchical one-hot (`ev_onehot`) | 26244.281053 | 33471.303715 |
| Exact-value view (`phase_exact`) | 28704.026084 | 40427.555557 |
| Sign view (`phase_sign`) | 27090.337390 | 39350.953411 |
| Random then latent | 28199.401685 | 37138.024894 |
| Original LSBO | 26260.605646 | 35535.543737 |

Cold initial sets were limited to seed 9916 in the 100 km mission and seeds
9915/9916 in the 140 km mission. All methods discovered feasibility in these
sets. The remaining seeds were warm. These small cold denominators do not
support strong cold-start superiority claims.

The latent reference improved the observed median over Random in both missions,
but its medians were worse than the matched one-hot control. These outcomes do
not establish the incremental advantage of the structured latent representation.
Native LSBO retained its own optimization effort, so it is not effort-matched.
Paired warm comparisons and all denominators are retained in the reports and
machine-readable summary. Eight development seeds are not independent
confirmation, and the historical sparse task name does not establish below-10%
feasibility. Numerical screening is not physical model calibration or safety
validation. No subsequent campaign was started.

## Complete records

Release tag: `aircraft-nominal-pilot-20261030-233225`.
Archive: `aircraft_nominal_20261030_01_full_records.zip`.
Size: 176,852,525 bytes; file count: 1,164.
SHA-256: `35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd`.

The archive contains the entire new evidence directory: all initialization,
observations, proposals, evaluation ledgers, plans, registrations, source
snapshots, retained locks, numerical diagnostics, timings, reports and logs.
Earlier failed launcher attempts and the final launcher snapshot are included.
Those earlier attempts failed before pilot registration and consumed no pilot
evaluations. The old qualification directory is not duplicated.

Every archived file was verified against its local original. The separate
`archive_manifest.json` lists relative paths, sizes and SHA-256 values. This
Git branch contains original lightweight reports/audits and an explicitly derived
`summary.json`; it does not reconstruct or replace any raw proposal or ledger.
Machine-specific source plans and raw environment logs remain in the archive.
