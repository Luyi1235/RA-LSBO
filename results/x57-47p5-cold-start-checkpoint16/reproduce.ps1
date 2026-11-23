param(
    [string]$Code = 'D:\RA-LSBO\x57-cold-start',
    [string]$Archive = 'D:\RA-LSBO\x57-cold-debug-publish\results',
    [string]$Qualification = 'D:\RA-LSBO\x57-qualification-v1-final\qualification_plan.json',
    [string]$Python = 'C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe',
    [string]$ExternalPython = 'D:\RA-LSBO\x57-baseline-py311\Scripts\python.exe',
    [Parameter(Mandatory = $true)][string]$Output,
    [switch]$Resume,
    [switch]$ExecuteRcaide
)

$ErrorActionPreference = 'Stop'
$sha = '282647a90655714b251b47d3163b8b68c7210fbb'
$env:OMP_NUM_THREADS = $env:OPENBLAS_NUM_THREADS = $env:MKL_NUM_THREADS = $env:NUMEXPR_NUM_THREADS = $env:NUMBA_NUM_THREADS = '1'
Push-Location -LiteralPath $Code
try {
    if ((git rev-parse HEAD) -ne $sha) { throw "Check out $sha in $Code" }
    $common = @('-B', 'run_x57_five_arm.py', '--archive', $Archive, '--qualification-plan', $Qualification,
                '--external-python', $ExternalPython, '--workers', '12', '--output', $Output)
    if ($ExecuteRcaide) {
        $mode = @('--execute-rcaide')
        if ($Resume) { $mode += '--resume' }
        & $Python @common --checkpoint 16 @mode
        if ($LASTEXITCODE -ne 0) { throw "Physical run stopped. Inspect ledgers at $Output." }
    } else {
        if (!$Resume) {
            & $Python @common --checkpoint 10
            if ($LASTEXITCODE -ne 0) { throw "Mock run stopped before the resume check." }
        }
        & $Python @common --checkpoint 12 --resume
        if ($LASTEXITCODE -ne 0) { throw "Mock replay stopped. Inspect ledgers at $Output." }
        $report = Get-Content -LiteralPath (Join-Path $Output 'report.json') -Raw | ConvertFrom-Json
        $matches = @($report.next_proposal_replay_verified.PSObject.Properties |
                     Where-Object { $_.Value -eq $true })
        if ($report.actual_rcaide_calls -ne 0 -or $report.worker_starts -ne 28 -or $matches.Count -ne 5) {
            throw "Mock replay did not reproduce all five next proposals."
        }
    }
} finally {
    Pop-Location
}
