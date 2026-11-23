param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('verify', 'mock', 'real')]
    [string] $Mode,
    [Parameter(Mandatory = $true)]
    [string] $Output,
    [string] $Verification,
    [string] $MockReport
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

$frozen = @(
    '--qualification-plan', 'D:\RA-LSBO\x57-qualification-v1-final\qualification_plan.json',
    '--qualification-ledger', 'D:\RA-LSBO\x57-distance-input-838c85fe\results\x57-rcaide-qualification-20261122\calls.jsonl',
    '--paired-archive', 'D:\RA-LSBO\x57-paired-distance-publish-20261122\results\x57-paired-distance-feedback-20261122',
    '--initialization-ledger', 'D:\RA-LSBO\x57-init40-publish-20261122\results\x57-initialization-40nmi-20261122\calls.jsonl',
    '--old-pilot-archive', 'D:\RA-LSBO\x57-engineering-comparison-v2-publish-7d106a8\results\x57-engineering-comparison-v2-7d106a8-seed20261125',
    '--distance-diagnostic', 'D:\RA-LSBO\x57-distance-publish-20261122\results\x57-distance-diagnostic-20261122\calls.jsonl',
    '--engineering-pilot', 'D:\RA-LSBO\x57-energy-pilot-publish-07a95dec\results\x57-engineering-energy-pilot-07a95dec\calls.jsonl',
    '--workers', '12', '--online', '12', '--output', $Output
)
$entryMode = if ($Mode -eq 'verify') { 'verify' } else { 'campaign' }
$execution = @()
if ($Mode -eq 'real') {
    $verificationHash = (Get-FileHash -LiteralPath $Verification -Algorithm SHA256).Hash.ToLowerInvariant()
    $mockHash = (Get-FileHash -LiteralPath $MockReport -Algorithm SHA256).Hash.ToLowerInvariant()
    $execution = @(
        '--execute-rcaide', '--verification', $Verification, '--verification-sha256', $verificationHash,
        '--mock-report', $MockReport, '--mock-report-sha256', $mockHash
    )
}

Push-Location -LiteralPath $repo
try {
    & $python -B "$repo\run_x57_cold_start_debug.py" $entryMode @frozen @execution
    if ($LASTEXITCODE -ne 0) {
        throw "X-57 $Mode exited with code $LASTEXITCODE. Artifacts at $Output"
    }
} finally {
    Pop-Location
}
