# Local aircraft representation diagnostic

Fixed execution commit: `5f33e4b9174b23e232ebe5d5ce53f7859a58e1ed`.
Independent worktree, clean tracked sources; original Windows Rcaide_vsp
environment. Python 3.9.25, Windows-10-10.0.26200-SP0, PyTorch 2.8.0+cpu,
NumPy 2.0.2. The package snapshot matches the original pilot environment.
No package installation or algorithm change was made.

Input: the original complete nominal aircraft pilot ZIP, SHA-256
`35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd`.
Its digest was checked before and after the diagnostic and remained unchanged.
Historical records, plans, locks, snapshots and identities were not rewritten.

## Scope completed

- Exactly 20 diagnostic tests passed, zero failures, errors or skips.
- Exactly 64 encoder refits: all 16 latent trials at history prefixes 0 and 29,
  each refitted at 30 and 120 epochs (32 fits for each training budget).
- Exactly 32 one-hot conversions; 96 unique output rows overall.
- 156 source-text bindings checked against sealed historical snapshots, without
  executing snapshot code. New model evaluations, GP fits and RCAIDE calls: zero.
  No optimizer replay or new optimization was run.

## Original identity comparison

Eligibility is strictly `representation=latent`, `epochs=30` and a nonempty
`saved_feature_identity`. All **29/29** eligible rows match their original
identities. There are zero eligible mismatches.

The three excluded original coverage steps had no saved identity:
ordinary/9916/prefix0, constrained/9915/prefix0 and constrained/9916/prefix0.
120-epoch and one-hot rows are not original-identity reproduction tests.
Their raw diagnostic fields are retained without changing any hashes.

The original identities therefore reproduce in this Windows environment. The
cause of the previously reported Linux differences remains undetermined; this
result does not establish that floating-point differences alone explain them.

## Reconstruction sensitivity

Aggregates below include every registered prefix, without selecting failures.
These are grammar reconstruction measurements, not held-out prediction scores.

| Epochs | Encoder fits | Mean exact active-grammar reconstruction | Median | Range |
| --- | ---: | ---: | ---: | --- |
| 30 | 32 | 13.28125% | 8.33333% | 8.33333%-33.33333% |
| 120 | 32 | 50.78125% | 50% | 33.33333%-75% |

Mean active battery-category accuracy increased from 35.15625% to 74.73958%.
Mean layout accuracy was 83.07292%/98.69792%, and propeller-count accuracy was
35.41667%/60.06944%, for 30/120 epochs respectively. Full per-prefix feature
identities, latent ranges, normalization and battery-pair kernel diagnostics are
in the unmodified `diagnostic.json`.

120 epochs remains only the fixed sensitivity diagnostic; it was not adopted
as an optimizer setting. Reconstruction alone does not establish the cause of
the earlier BO performance differences.

## Records

Release tag: `aircraft-representation-20261031`.
Archive: `aircraft_representation_20261031_01_records.zip`, 95,118 bytes,
9 files, SHA-256
`9b514cdcfa92dcab33345429e2a73f521a022a3f2fae041aec98ef83d578ea64`.
Every archive entry was checked against its local original.

The Release includes complete diagnostic, environment, platform/build metadata,
JUnit, test/refit logs, fixed commit and the explicitly derived summary.
This branch contains the original diagnostic plus the derived summary and
file manifest. The original 176 MB pilot archive is not duplicated.
Execution stopped after this diagnostic; no later experiment was started.
