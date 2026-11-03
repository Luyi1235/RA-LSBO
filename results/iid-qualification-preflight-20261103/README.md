# IID aircraft qualification: zero-call preflight

Date: 2026-11-03. Related research review: PR #35, comment 6597274474. This is a plan-only, unapproved preflight. No aircraft model, RCAIDE, GP fitting, VAE training, optimizer or prior pilot was executed. Planned 600 evaluations are NOT authorized. The implementation commit is `f6e72443c30da8a8bfbff193b65b2f84513dfe01`; this evidence is committed afterwards, so the PR head is necessarily a different commit. The historical model reference is `57e390fc8f6280db9f201940aec14089e81c104c`.

## Immutable plan and tests

- `plan.json` SHA-256: `919d092ac675d9c854bffda08c7e397ee9f41415902471757194d14e916f66e2`. It binds code commit, raw source hashes, historical 20-second model contracts, Python/NumPy/platform/thread settings, 12 configurations, bounds/defaults, PCG64 seeds, all ordered candidate identities and screen values, budget, thresholds and stop rules.
- Ordinary seed 21001: 300 accepted, 325 proposed, 25 geometry rejections, 0 observed duplicates. Sparse seed 21002: 300 accepted, 317 proposed, 17 rejections, 0 observed duplicates. No duplicates happened in these draws; the sampler has no deduplication and retains/charges repeated designs by index.
- Ordinary ordered candidate-digest-list SHA-256 `4aa5eae27b8a7fced97917c26449548abc6eba85b523daf7b37c24062688509f`; sparse `27be8f2c67fd39d7d9a871188898d31e407cdb33f78bb321c6a8ace984ad999b`. The plan SHA above, rather than either subdigest, binds the full plan.
- Windows `Rcaide_vsp`: `python -m pytest -q tests/test_iid_qualification_preflight.py --junitxml=results/iid-qualification-preflight-20261103/tests.xml`: **13 passed**, no skipped or failed in final run. XML SHA-256 `531acabc1af844c8104895b98ca9d8e13bd495b407ab0b85cae7a72dd427093d`. The first development run exposed an existing package initializer importing the problem module (11 passed, one failed); module relocation removed that dependency, then all 13 passed. It did not evaluate the problem.
- Tests cover two deterministic PCG64 streams and prefix preservation (not a mathematical proof of independence of pseudorandom streams), finite discrete grammar, active bounds/canonical defaults, original analytic geometry via the original geometry function with evaluator patched to fail, duplicate preservation, failed/invalid screen, attempt exhaustion, no label access, unknown/physical-abort contracts, fixed confidence threshold, import isolation, and write-once plan. The plan command does not expose a run action.
- Qualification rule if separately approved: 140 km 1–19 known-feasible of 300, zero unknown, and the two-sided 95% Clopper–Pearson upper bound < 0.10; 0 does not establish existence, 20 or more fails the sparsity rule. Ordinary uses the frozen lower-bound > 0.20 rule. Unknown/numerical/infrastructure or unsettled call stops and remains evidence; no silent retries, deletion or imputation.

## Seed-history scope and blocker

No confirmed use of 21001 or 21002 was found in the checked local history. A structured scan of local plan/manifest/registration/protocol JSON files found 343 parseable records, zero seed-field hits, one unreadable old `plan-validation.json` and no files skipped for size. A separate archive scan listed 88 ZIPs and parsed 10,572 matching plan/registration/manifest/config/result JSON members; no seed-field hits. Five ZIPs were invalid or incomplete: `RA-LSBO-H-pilot-9961-9964-release-download-20261102.zip`, `RA-LSBO-H2-P3-pilot-download-verification-20261103_01.zip`, `RA-LSBO-H2-P2-upload-verification-20261102_02/api-download.zip`, `RA-LSBO-H2-P2-upload-verification-20261102_02/RA-LSBO-H2-P2-pilot-evidence-20261102_02.zip`, `tmp/soc-release-download.zip`.

The archive scan did not inspect JSONL, nested ZIP contents or members above 50 MB; the text search is not a proof across inaccessible GitHub history, ignored files or unregistered executions. The unreadable items must be reconciled against independently hash-verified valid originals and historical source/seed manifests reviewed before any execution. The plan honestly says `incomplete_review_pending`, not `seed_unique=true`. An actual conflict blocks the gate and must not trigger automatic seed replacement.

## CBOB adaptation gaps: no implementation in this PR

The local reference README identifies its code as the authors' GPy implementation with a separate GPflow/Trieste archive; provenance, immutable upstream commit, licensing and dependencies have not been independently certified. CBOB-adapted has no qualified interface here. Before acceptance, freeze and review: (1) reference objective/constraint likelihood and EICB, normalization, EP and stopping rules; (2) a legal hierarchical mixed-space candidate generator, activation, category mapping and analytic screen; (3) identical visible exact/interval/sign/missing and failure contracts, preserving unknown rather than converting it to violation; (4) shared initialization, equal cost, acquisition optimization, tie handling, deterministic seeds, maximum proposal time and single-thread resource accounting; (5) zero-truth reference cases and reproducibility, then separately approved independent small-budget validation. An approximation must be labelled and documented, never presented as an unmodified published baseline. The repository's SCBO/SEGO adapted implementations do not close this gap.

## P3 cost evidence and seven-arm extrapolation

Read-only P3 source evidence: `plan.json` SHA-256 `4340e7528b139400b4521a33ef1964c9fccf928185f7c09558003ffcde996da7`, `calls.jsonl` `fce05ecd984842e6be014a2546a695ca69fe002a83356e89570c3f06a4c80c5f`, `results.json` `234b5074e1a119ea3a86bed5261fb5e5a39b51c2eff31894b888b8a4d1e0ce91`. No measurements were re-run.

| P3 measured component | Observed |
|---|---:|
| 32 optimizer trials, 576 online proposals | proposal wall-clock sum 1,345.673 s, mean 2.336 s/proposal |
| Same 32 trials, reported optimization wall time | 1,397.561 s total; includes proposals and online evaluation, excludes shared initial and 5 s final verification |
| 32 VAE training summaries | 96.744 s total, included in proposal time, not additive |
| 192 shared initial calls | 10.766 s total, 0.05607 s/call |
| 576 online calls | 39.097 s total, 0.06788 s/call |
| 32 five-second verification calls | 9.017 s total, 0.28178 s/call |

Purely illustrative, **not a qualified forecast**: applying those same P3 component means uniformly to seven different methods and 1,024 seeds per task gives 258,048 proposals at 602,861.5 s, 24,576 shared initial calls at 1,378.0 s, 258,048 online calls at 17,515.5 s, and 14,336 verifications at 4,039.6 s, a serial subtotal approximately **173.8 hours**. This includes proposal-time VAE/GP/local search and avoids adding the VAE time twice. It excludes initialization outside those measures, logging, source checks, audit, failures, orchestration, queuing and machine contention. Actual costs of random, reverse allocation, interval-missing and CBOB-adapted are unknown; P3 measured only two H2 variants, not seven independent algorithms. GP and local-search times are not separately measured in these records. GPU utilization, memory and availability were not logged and cannot be estimated from the package version. Qualification must measure per-arm runtime/hardware before any compute reservation.

## Remaining gates

Independent review of finite-precision RNG/distribution and model schema binding, authoritative all-history seed clearance, explicit approval of the precommitted plan and 600-call budget, source/environment re-verification and no unknown/unsettled status are still required. The plan's source SHA cannot be changed to the later evidence-only commit. Further CBOB acceptance, mechanical-ablation qualification, cost feasibility and RCAIDE mapping have separate gates. **No simulator execution is authorized by this PR.**
