# RA-LSBO-H2-M: pending Windows 48-call smoke handoff

**Do not run without separate approval of 48 NEW simplified-aircraft
evaluations.** This is not a continuation of the H1 856-call pilot,
the 384-GP cross-policy panel, or the interrupted 2,212-call pilot.

GitHub branch: research/ra-lsbo-h2-causal-risk-20261102

Entrypoint: ra_lsbo.experiments.simplified_aircraft.h2_pilot

Before running, read:
- docs/RA_LSBO_H2_CAUSAL_RISK_PROTOCOL_20261102.md
- docs/RA_LSBO_H2_PROSPECTIVE_AIRCRAFT_PROTOCOL_20261102.md

**A complete fixed SHA will be supplied after the final CI review.**
Do not execute a floating branch checkout or modify the H1 main sources.

## Exact H2-M smoke

- Cases: aircraft_ordinary 100km and aircraft_sparse 140km.
- Fixed new smoke seed: 11061.
- Fixed six arms: random, onehot_stable, corpus_stable,
  hybrid_onehot, hybrid_guard, h2_causal.
- 6 common initial candidates/case, 2 online per arm/case.
- At most 12 step-5s verification evaluations.
- Complete hard cap **48** simplified-aircraft calls, **0 RCAIDE**.
- Controller: fixed clipped prior log-likelihood ratio H2-M
  with Gaussian-component SOC mixture; the synthetic-only
  H2-B hard Brier head is NOT a flight arm.

Use original Windows/Rcaide_vsp Python, unchanged 67-dependency
environment, physical equations, OpenVSP search paths and single-thread
settings. Create a NEW detached clean worktree at the final SHA. The
new evidence directory must not already exist.

PowerShell setup template:

    $ErrorActionPreference='Stop'
    $Repo='<EXISTING_REPOSITORY_ROOT>'
    $Code='<EXACT_FINAL_H2_COMMIT>'
    $Worktree='D:\LSBO\RA-LSBO-H2-M-code-20261102_01'
    $Out='D:\LSBO\RA-LSBO-H2-M-smoke-20261102_01'
    $Log='D:\LSBO\RA-LSBO-H2-M-smoke-console-20261102_01.log'
    $Py='C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe'
    if ((Test-Path $Worktree) -or (Test-Path $Out) -or (Test-Path $Log)) {
        throw 'New source, smoke and log paths required'
    }
    git -C $Repo fetch origin research/ra-lsbo-h2-causal-risk-20261102
    if ($LASTEXITCODE -ne 0) { throw 'Git fetch failed' }
    git -C $Repo worktree add --detach $Worktree $Code
    if ($LASTEXITCODE -ne 0) { throw 'Worktree creation failed' }
    Set-Location $Worktree
    if ((git rev-parse HEAD).Trim() -ne $Code) { throw 'Wrong source SHA' }
    if (git status --porcelain --untracked-files=no) {
        throw 'Dirty tracked source prohibited'
    }
    $env:OMP_NUM_THREADS='1'
    $env:MKL_NUM_THREADS='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:PYTHONHASHSEED='0'
    $env:MPLBACKEND='Agg'
    $env:PYTHONUNBUFFERED='1'
    & $Py -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Environment invalid' }

First run all H2 and relevant frozen GP/corpus regression tests in
the same environment. Do not patch failures in-place.

Smoke registration (zero model calls):

    $Module='ra_lsbo.experiments.simplified_aircraft.h2_pilot'
    & $Py -m $Module prepare --profile smoke --output $Out
    if ($LASTEXITCODE -ne 0) { throw 'Source-bound smoke plan failed' }

Before executing, use the registered load_plan function to verify
the complete source snapshot, correct Windows runtime and physical
contracts, exact seed/arms, cap=48, RCAIDE=0, and no preexisting
execution.lock/calls.jsonl.

**Only once the user separately approves the 48-call smoke budget**:

    & $Py -u -m $Module run --output $Out --approved-call-limit 48 2>&1 |
        Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) {
        throw 'Preserve failed run, lock and all started/settled evidence'
    }

On success alone, run audit then report under the same fixed worktree.
Audit H2 saved forecast commitments, exact mixture SOC scores,
training prefixes, VAE checkpoints, numerical 5s verification, and
all evaluated labels. Never substitute a moment-matched single-Gaussian
NLL for the mixture's exact density/mass. Retain separate logs and
every original ZIP/member SHA256.

**No automatic retry or resume**. A real model or infrastructure error
requires retaining locks and all partial evidence. No RCAIDE or MDA.
Do not modify original H1/incomplete corpus releases or ledger.

## Future independent H2 development pilot — NOT AUTHORIZED

Requires a separately approved 1,008-call registration after the
fixed-source Windows smoke and full reviewed 316-call qualification
receipt pass. Pilot seeds 11062–11065, two cases, six arms, 18 online
evaluations per arm, 96 shared initial, up to 48 same-incumbent 5s
checks, 0 RCAIDE. This is a four-seed developmental comparison only,
not a powered superiority study or physical safety validation.
