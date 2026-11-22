param([switch]$ExecuteRcaide)

$ErrorActionPreference = 'Stop'
$runCode = '07a95dec9787c05601a64728d0b18cd073090db4'
$repo = 'D:\RA-LSBO\x57-cold-start'
$py = 'C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe'
$inputs = @(
    '--qualification-plan', 'D:\RA-LSBO\x57-qualification-v1-final\qualification_plan.json'
    '--qualification-ledger', 'D:\RA-LSBO\x57-distance-input-838c85fe\results\x57-rcaide-qualification-20261122\calls.jsonl'
    '--paired-archive', 'D:\RA-LSBO\x57-paired-distance-publish-20261122\results\x57-paired-distance-feedback-20261122'
    '--initialization-ledger', 'D:\RA-LSBO\x57-init40-publish-20261122\results\x57-initialization-40nmi-20261122\calls.jsonl'
)
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$prefix = "D:\RA-LSBO\x57-engineering-energy-$stamp"

Push-Location -LiteralPath $repo
try {
    $headSha = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $headSha -ne $runCode) { throw "Check out $runCode before running" }

    & $py -B -u run_x57_engineering_energy_pilot.py offline @inputs --output "$prefix-offline"
    if ($LASTEXITCODE -ne 0) { throw 'Offline replay failed' }
    & $py -B -u run_x57_engineering_energy_pilot.py pilot @inputs --output "$prefix-mock"
    if ($LASTEXITCODE -ne 0) { throw 'Mock pilot failed' }

    if ($ExecuteRcaide) {
        $proof = Join-Path "$prefix-offline" 'offline_report.json'
        $proofHash = (Get-FileHash -LiteralPath $proof -Algorithm SHA256).Hash.ToLowerInvariant()
        $out = "$prefix-real"
        & $py -B -u run_x57_engineering_energy_pilot.py pilot @inputs --output $out `
            --offline-report $proof --offline-report-sha256 $proofHash --execute-rcaide `
            1> "$out.stdout.log" 2> "$out.stderr.log"
        if ($LASTEXITCODE -ne 0) { throw "Pilot stopped. Inspect $out and its logs." }
    }
}
finally {
    Pop-Location
}
