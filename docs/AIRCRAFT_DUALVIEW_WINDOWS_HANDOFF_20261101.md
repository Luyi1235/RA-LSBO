# Windows handoff — new RA-LSBO dual-view aircraft experiment

This is an **independent new research trial**, not a way to resume or complete
the interrupted 2026-10-31 corpus pilot. It must preserve that pilot's
2,212 settled calls, uncompleted status, original execution lock, source
snapshots, full ZIP and SHA256 receipts.

Original code ancestor: b5744736c942eadf0a56f79d811f7e7b8ca3d52b.
New branch: research/ra-lsbo-risk-dualview-20261101.
**PIN the reviewed head SHA supplied with the handoff**, not the floating
branch name. The same single source commit is required by fresh Windows
smoke and pilot. Do not merge into main.

Prospective roster and limits:
- Cases: aircraft_ordinary 100km and aircraft_sparse 140km.
- Arms: random, onehot_stable, corpus_stable, hybrid_onehot, hybrid_guard.
- Smoke seed 9960, 6 shared initial per case, 2 online per method, at most
  42 model evaluations, **0 RCAIDE**.
- Pilot seeds 9961-9964 (four fresh independent seeds), 12 initial per
  case, 18 online per method, at most **856** simplified aircraft
  evaluations, **0 RCAIDE**.
- Both stages combined: at most 898 new calls, separate from original
  interrupted 2,212 and from prior Linux CI-only attempts.

## Scientific design

The three preregistered method ablations separately test:
1. Onehot risk constraints vs corpus latent risk, with latent energy.
2. A deterministic per-candidate minimum-SOC-feasibility-probability dual
   view vs onehot SOC only.
3. Corpus latent energy vs onehot energy, both with onehot constraints.

All model arms use the same original grouped kernel grid 4x2 and stable
extreme-tail EP moments. In both smoke and pilot, **smoke does not reduce
the grid**. If an individual required response-head GP has no valid
kernel posterior, the algorithm records all eight numeric failures and
uses deterministic least-visited candidate-branch coverage instead of a
fabricated posterior. Real simulator unknown/numerical outcomes still
settle in the ledger and terminate the experiment with locks retained.
All model failure fallbacks are separately counted in trial summaries.

The 2026-11-01 Linux *code-only integration* has a completed audited smoke:
10/10 trials, 20/20 online observations, 37/42 possible simplified calls,
0 RCAIDE. The 140km two-step smoke had no feasible incumbents in its five
arms; hence no 5s checks were fabricated. The numerical qualification
criterion now explicitly distinguishes no feasible incumbent from a
failed numerical refinement. This gate amendment was made AFTER
observing the short Linux smoke and is disclosed, not hidden. It does NOT
change seeds, physics or any earned objective value. Linux results cannot
be used as the Windows same-code qualification receipt.

Historical Linux smoke attempts are distinct:
- Attempt 1: 9 started/9 settled, aborted by kernel-grid protocol mismatch.
- Attempt 2: 15 started/15 settled, aborted by missing corpus audit feature
  identity.
- Attempt 3: 37 started/37 settled, full result and audit passed.
TOTAL 61 simplified calls in those 3 independent Linux runs. The 9 and 15
are not missing data for the successful 37-call smoke. Do not delete or
rename the historical attempt artifacts.

## Original Windows/Rcaide_vsp preflight

Set $ErrorActionPreference='Stop'. Use the ORIGINAL existing Windows
Rcaide_vsp environment with unchanged dependencies. Create a brand-new
detached Git worktree at the exact reviewed SHA and a brand-new output root.
Do not reuse any 2026-10-31 or 2026-11-01 output directories.
Never run scripts from the original interrupted pilot directory.

Select:
- $Repo: the existing local repository root (do not modify its uncommitted
  contents).
- $Code: the exact reviewed commit SHA (from final handoff).
- $Worktree: a NEW directory, for example
  D:\LSBO\RA-LSBO-dualview-20261101_01
- $Root: a NEW evidence output directory outside that worktree.
- $Qualification: the existing COMPLETE, reviewed, 316-call aircraft
  qualification directory from the prior protocol, NOT just a report ZIP.
- $Py: C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe

PowerShell template (replace angle-bracket paths and fixed SHA):

    $ErrorActionPreference='Stop'
    $Repo='<existing repo root>'
    $Worktree='D:\LSBO\RA-LSBO-dualview-20261101_01'
    $Root='D:\LSBO\RA-LSBO-dualview-output-20261101_01'
    $Qualification='<existing COMPLETE reviewed aircraft qualification folder>'
    $Py='C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe'
    $Code='<FIXED_REVIEWED_SHA>'
    if ((Test-Path $Worktree) -or (Test-Path $Root)) {throw 'A new untouched worktree/output directory is required'}
    git -C $Repo fetch origin research/ra-lsbo-risk-dualview-20261101
    if ($LASTEXITCODE -ne 0) {throw 'Git fetch failed'}
    git -C $Repo worktree add --detach $Worktree $Code
    if ($LASTEXITCODE -ne 0) {throw 'Git worktree failed'}
    Set-Location $Worktree
    if ((git rev-parse HEAD).Trim() -ne $Code) {throw 'Wrong code revision'}
    if (git status --porcelain --untracked-files=no) {throw 'Tracked execution source is dirty'}
    if (!(Test-Path $Qualification) -or !(Test-Path $Py)) {throw 'Missing prereq or original Python'}
    $env:OMP_NUM_THREADS='1'
    $env:MKL_NUM_THREADS='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:PYTHONHASHSEED='0'
    $env:MPLBACKEND='Agg'
    $env:PYTHONUNBUFFERED='1'
    New-Item -ItemType Directory -Path $Root | Out-Null
    & $Py -m pip check
    if ($LASTEXITCODE -ne 0) {throw 'Environment invalid'}
    & $Py -m pytest -q tests/test_aircraft_dualview.py tests/test_ep_stability_candidate.py tests/test_risk_core_mixed_gp.py tests/test_aircraft_corpus_protocol.py tests/test_aircraft_corpus_optimizer.py
    if ($LASTEXITCODE -ne 0) {throw 'Regression failed'}

**Windows smoke only**, fresh and source-bound:

    $Module='ra_lsbo.experiments.simplified_aircraft.dualview_pilot'
    $Smoke=Join-Path $Root 'smoke_9960'
    & $Py -m $Module prepare --profile smoke --output $Smoke
    if ($LASTEXITCODE -ne 0) {throw 'Smoke registration failed'}
    & $Py -u -m $Module run --output $Smoke --approved-call-limit 42 2>&1 | Tee-Object -FilePath (Join-Path $Root 'smoke_console.log')
    if ($LASTEXITCODE -ne 0) {throw 'Smoke failed: preserve lock and records'}
    & $Py -m $Module audit --output $Smoke
    if ($LASTEXITCODE -ne 0) {throw 'Smoke audit failed'}
    & $Py -m $Module report --output $Smoke

Any no-feasible-incumbent trial is explicitly recorded as such; it is
NOT numerically failed and does NOT trigger a fake 5s call. A physically
unusable 5s comparison of an actually nominal-feasible incumbent DOES
block pilot registration. Keep every failed Linux/Windows result.

**Independent new pilot**, ONLY if that exact Windows smoke and
qualification gates pass and the user has approved the complete
prospective 856-call budget:

    $Pilot=Join-Path $Root 'pilot_9961_9964'
    & $Py -m $Module prepare --profile pilot --qualification $Qualification --smoke $Smoke --output $Pilot
    if ($LASTEXITCODE -ne 0) {throw 'Pilot registration blocked'}
    & $Py -u -m $Module run --output $Pilot --approved-call-limit 856 2>&1 | Tee-Object -FilePath (Join-Path $Root 'pilot_console.log')
    if ($LASTEXITCODE -ne 0) {throw 'Pilot stopped; preserve new pilot lock and all evidence'}
    & $Py -m $Module audit --output $Pilot
    if ($LASTEXITCODE -ne 0) {throw 'Pilot audit failed'}
    & $Py -m $Module report --output $Pilot

Do not silently retry, resume, delete a lock, or roll new seeds. If a
new fault appears, preserve the already-settled calls and use a NEW
diagnostic branch, rather than changing the model mid-study.

## Evidence handoff

For smoke and pilot separately preserve:
- Git commit, profile, source snapshot, plan.json and registration.json.
- Full calls.jsonl started/settled ledger, every trial manifest, proposals,
  observations, timing and 5s records.
- Complete persistent VAE checkpoints, candidate/EP/head failure metadata
  and prequential SOC exact-vs-interval scores.
- results.json, audit.json, report.md, console logs and preserved locks.
- Independent whole archive SHA-256 and per-file digests.

The 4-seed pilot is exploratory and not final confirmation. Report
per-case/per-seed energy and feasible status, GP degradation counts,
constraint scoring (exact and censored separate), total compute including
VAE, and nominal-to-5s sensitivity. It is not evidence of certified
aircraft performance.
