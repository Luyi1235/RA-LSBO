param(
    [string]$Repository = 'D:\RA-LSBO\x57-cold-start',
    [string]$Archive = 'D:\RA-LSBO\x57-cold-debug-publish\results\x57-cold-start-debug-fccf94c6',
    [string]$QualificationPlan = 'D:\RA-LSBO\x57-qualification-v1-final\qualification_plan.json',
    [string]$PhysicalPython = 'C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe',
    [string]$BaselinePython = 'D:\RA-LSBO\x57-baseline-py311\Scripts\python.exe',
    [string]$OutputRoot = "D:\RA-LSBO\x57-formal-prefix-repro-$(Get-Date -Format 'yyyyMMdd-HHmmss')",
    [switch]$InstallBaselineDependencies,
    [switch]$ExecuteRcaide
)

$ErrorActionPreference = 'Stop'
$runner = '8a40729177d3473b17f05655e3819167c943fe93'
$adapters = '508398032f98b389be0701b8004aa5279c00b485'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'

function Assert-Exit {
    param([int]$Code, [string]$Step)
    if ($Code -ne 0) { throw "$Step exited with $Code. Outputs are under $OutputRoot." }
}

git -C $Repository fetch origin refs/heads/dev/x57-formal-prefix-qualification
Assert-Exit $LASTEXITCODE 'git fetch'
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$runnerTree = Join-Path $OutputRoot 'runner-code'
$adapterTree = Join-Path $OutputRoot 'adapter-code'
git -C $Repository worktree add --detach $runnerTree $runner
Assert-Exit $LASTEXITCODE 'physical runner checkout'
git -C $Repository worktree add --detach $adapterTree $adapters
Assert-Exit $LASTEXITCODE 'external adapter checkout'

if ($InstallBaselineDependencies) {
    $venv = Split-Path -Parent (Split-Path -Parent $BaselinePython)
    if (!(Test-Path -LiteralPath $BaselinePython -PathType Leaf)) {
        py -3.11 -m venv $venv
        Assert-Exit $LASTEXITCODE 'baseline venv'
    }
    & $BaselinePython -m pip install -r (Join-Path $PSScriptRoot 'environment\baseline_requirements.txt')
    Assert-Exit $LASTEXITCODE 'baseline dependencies'
}

$common = @('--archive', (Resolve-Path -LiteralPath $Archive).Path,
            '--qualification-plan', (Resolve-Path -LiteralPath $QualificationPlan).Path)
Push-Location -LiteralPath $adapterTree
try {
    & $BaselinePython -B run_x57_external_acceptance.py @common --component adapters `
        --output (Join-Path $OutputRoot 'adapters')
    Assert-Exit $LASTEXITCODE 'ArchSBO and LogCEI qualification'
    & $PhysicalPython -B run_x57_external_acceptance.py @common --component ra-replay `
        --output (Join-Path $OutputRoot 'ra-replay')
    Assert-Exit $LASTEXITCODE 'RA same-prefix replay'
    & $PhysicalPython -B run_x57_external_acceptance.py @common --component native-checkpoints `
        --output (Join-Path $OutputRoot 'native-checkpoints')
    Assert-Exit $LASTEXITCODE 'native checkpoint replay'
} finally { Pop-Location }

Push-Location -LiteralPath $runnerTree
try {
    $run = @('--workers', '12', '--output', (Join-Path $OutputRoot 'prefix'))
    if ($ExecuteRcaide) { $run += '--execute-rcaide' }
    & $PhysicalPython -B run_x57_formal_prefix.py @common @run
    Assert-Exit $LASTEXITCODE 'frozen prefix campaign'
} finally { Pop-Location }
