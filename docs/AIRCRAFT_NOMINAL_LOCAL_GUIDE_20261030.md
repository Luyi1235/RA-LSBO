# 简化飞机七组名义条件 pilot：本地模型执行指南

日期：2026-10-30。此指南取代“准备下一版pilot”的讨论规格，不要求补做旧transfer诊断，不重跑已经完成的352次资格。所有实验均为简化模型，RCAIDE MDA上限为0。

## 1. 固定版本与工作范围

仓库：Luyi1235/RA-LSBO。
分支：`research/aircraft-nominal-pilot-20261030`，PR #8，尚未合并。
固定运行SHA：`e4fbcd37ca4d096bbdac03fc57b0f122f006aea3`。
后续指南/回执提交不改变这份运行代码；不能直接pull main或使用移动HEAD。
入口：`python -m ra_lsbo.experiments.simplified_aircraft.nominal_pilot`。

完整计划：2个原任务（100/140km）、7方法、8个新开发种子9911–9918、每题每种子12共同初始点、每方法30在线评估。保留所有种子、失败及原LSBO原生停止。
方法：random、ev_latent、ev_onehot、phase_exact、phase_sign、ev_random_then_latent、original_lsbo。
没有自适应权重，没有新的采集函数，没有官方CBOB复现或安全保证。

预算：smoke上限54=12共同初值+28在线+至多14细步长复核；pilot上限3664=192共同初值+3360在线+至多112复核。本机本轮总研究上限3718。之前讨论40次smoke未包含细步长检查，执行配置补上了14次。历史352次、云端验收及单元测试内部廉价求值另计。没有可行解不复核，原LSBO早停不补跑，未用预算不重新分配。

## 2. 先检查现有文件，不动历史记录

从上一轮执行资格检查的工作树根目录开始（即本地已存在的`RA-LSBO-aircraft-entry-20261030_01`）。
完整旧资格目录相对于该根目录为`runs/aircraft_entry_20261030_01/qualification`。
它必须包含原plan、registration、audit、results、calls和source_snapshot。不要用GitHub摘要目录代替，不修改路径、换行或哈希，不添加JSON文件到旧目录。新prepare会只读校验固定原件，并封存28个固定设计的积分敏感性诊断；不重新计算飞机，不执行旧快照。

在原仓库检查`git status --short`、当前进程和运行目录；不终止无关程序、不stash/reset/覆盖用户修改。发现本次目标实验已有运行时停止操作，不能创建第二个写入者。下面命令只在相关实验没有运行时执行。路径已存在则保留并选择全新的工作目录，绝不复用失败运行。

## 3. PowerShell：独立worktree与环境

以下变量可按实际仓库挂载位置核实，不能修改SHA、种子、算法或预算。

```powershell
$ErrorActionPreference = 'Stop'
$Repo = (Get-Location).Path  # 必须先进入上轮资格工作树根目录
$Qualification = Join-Path $Repo 'runs\aircraft_entry_20261030_01\qualification'
$Branch = 'research/aircraft-nominal-pilot-20261030'
$Sha = 'e4fbcd37ca4d096bbdac03fc57b0f122f006aea3'
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
$Worktree = Join-Path (Split-Path $Repo -Parent) ('RA-LSBO-aircraft-pilot-' + $Stamp)
if (Test-Path $Worktree) { throw '工作目录已存在，不覆盖。' }
if (!(Test-Path (Join-Path $Qualification 'calls.jsonl'))) { throw '缺少完整原资格目录。' }
git -C $Repo status --short
if ($LASTEXITCODE -ne 0) { throw '原仓库不可用。' }
git -C $Repo remote get-url origin
if ($LASTEXITCODE -ne 0) { throw '缺少origin。' }
# 核实origin确实指向Luyi1235/RA-LSBO，不要改其他仓库。
git -C $Repo fetch origin $Branch
if ($LASTEXITCODE -ne 0) { throw 'fetch失败。' }
git -C $Repo cat-file -e "$Sha^{commit}"
if ($LASTEXITCODE -ne 0) { throw '固定提交未取得。' }
git -C $Repo worktree add --detach $Worktree $Sha
if ($LASTEXITCODE -ne 0) { throw '创建worktree失败。' }
Set-Location $Worktree
if ((git rev-parse HEAD).Trim() -ne $Sha) { throw 'SHA不匹配。' }
if (git status --porcelain --untracked-files=no) { throw '新worktree受跟踪源码不干净。' }

# 先正常激活已有Rcaide_vsp；激活失败则停止，不能偷偷切换Python。
conda activate Rcaide_vsp
if (!$env:CONDA_PREFIX -or (Split-Path $env:CONDA_PREFIX -Leaf) -ne 'Rcaide_vsp') {
    throw '没有激活已有Rcaide_vsp。'
}
$Python = Join-Path $env:CONDA_PREFIX 'python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$env:PYTHONHASHSEED = '0'
$Evidence = Join-Path $Worktree ('runs\aircraft_nominal_' + $Stamp)
New-Item -ItemType Directory -Path $Evidence | Out-Null
function Run-Python([string[]]$Arguments, [string]$LogName) {
    $SavedPreference = $ErrorActionPreference
    try {
        # 原生程序stderr可包含普通警告；记录原文，依据真实退出码决定是否停止。
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $Evidence $LogName)
        $Code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $SavedPreference }
    if ($Code -ne 0) { throw "Python失败($Code)：$($Arguments -join ' ')；停止后续执行。" }
}
Run-Python @('--version') 'python_version.txt'
Run-Python @('-m','pip','check') 'pip_check.txt'
Run-Python @('-m','pip','freeze') 'environment.txt'
Run-Python @('-c','import numpy,scipy,torch,gpytorch,botorch,sklearn; print({m.__name__:m.__version__ for m in (numpy,scipy,torch,gpytorch,botorch,sklearn)})') 'model_versions.txt'
git rev-parse HEAD | Out-File (Join-Path $Evidence 'commit.txt') -Encoding utf8
```

上轮本机主要数值环境为Python3.9.25、numpy2.0.2、scipy1.13.1、torch2.8.0+cpu、gpytorch1.11、botorch0.10.0、scikit-learn1.6.1。CI用Python3.11。不要直接执行全量升级或用CI环境覆盖既有RCAIDE环境。缺依赖或pip check失败时保留日志并停止；修复环境需独立说明后重新登记，不能在实验已登记后更改环境。

## 4. 新回归测试：全部通过才继续

```powershell
$Tests = @(
 'tests/test_aircraft_nominal_pilot.py',
 'tests/test_aircraft_numerical_checks.py',
 'tests/test_aircraft_transition.py',
 'tests/test_discovery_stages.py',
 'tests/test_discovery_stages_evidence.py',
 'tests/test_discovery_stages_phase.py',
 'tests/test_paper_experiments.py',
 'tests/test_soc_revision_acquisition.py',
 'tests/test_soc_revision_posterior.py',
 'tests/test_numerical_safeguards.py'
)
$Junit = Join-Path $Evidence 'junit.xml'
Run-Python (@('-m','pytest','-q') + $Tests + @("--junitxml=$Junit")) 'pytest.txt'
[xml]$JUnitXml = Get-Content -Raw $Junit
$Suites = @($JUnitXml.testsuites.testsuite)
$TotalTests = ($Suites | ForEach-Object { [int]$_.tests } | Measure-Object -Sum).Sum
if ($TotalTests -ne 327) { throw '测试收集数不同，先检查，不启动pilot。' }
foreach ($Suite in $Suites) {
 if ([int]$Suite.failures -ne 0 -or [int]$Suite.errors -ne 0 -or [int]$Suite.skipped -ne 0) {
   throw '测试有失败、错误或跳过，不能启动研究实验。'
 }
}
```

本次指定范围预期327项（新增63项、原相关264项），必须无失败、无错误、无跳过；以实际JUnit核对，不能把旧264项当作新代码验收。测试不是性能实验；内部人工标签和廉价求值单独披露。

## 5. 新七方法smoke，上限54次

```powershell
$Module = 'ra_lsbo.experiments.simplified_aircraft.nominal_pilot'
$Smoke = Join-Path $Evidence 'smoke'
Run-Python @('-m',$Module,'prepare','--profile','smoke','--output',$Smoke) 'smoke_prepare.txt'
$SmokePlan = Get-Content -Raw (Join-Path $Smoke 'plan.json') | ConvertFrom-Json
if ($SmokePlan.budget.maximum_model_calls -ne 54) { throw 'smoke预算不匹配。' }
Run-Python @('-m',$Module,'run','--output',$Smoke,'--approved-call-limit','54') 'smoke_run.txt'
Run-Python @('-m',$Module,'audit','--output',$Smoke) 'smoke_audit.txt'
Run-Python @('-m',$Module,'report','--output',$Smoke) 'smoke_report.txt'
$SmokeAudit = Get-Content -Raw (Join-Path $Smoke 'audit.json') | ConvertFrom-Json
if (!$SmokeAudit.passed -or $SmokeAudit.trial_count -ne 14 -or $SmokeAudit.checked_proposals -ne 28) {
 throw 'smoke试验或提案不完整。'
}
```

实际调用可以小于54：没有名义可行incumbent时不做5秒复核，不能为了凑预算重试。查看各复核结果；若出现数值异常，程序会停止，不能进入pilot。物理可行性标签改变是需要回报的差异，不得删掉或替换设计。smoke每组仅2步，禁止按其算法排名改配置。

## 6. 新探索性pilot，上限3664次

```powershell
$Pilot = Join-Path $Evidence 'pilot'
Run-Python @('-m',$Module,'prepare','--profile','pilot','--qualification',$Qualification,'--output',$Pilot) 'pilot_prepare.txt'
$Plan = Get-Content -Raw (Join-Path $Pilot 'plan.json') | ConvertFrom-Json
if ($Plan.budget.maximum_model_calls -ne 3664 -or $Plan.arms.Count -ne 7 -or $Plan.profile.seeds.Count -ne 8) {
 throw 'pilot范围不匹配。'
}
if (($Plan.profile.seeds -join ',') -ne '9911,9912,9913,9914,9915,9916,9917,9918') {
 throw '种子不匹配，不能修改计划。'
}
if (!(Test-Path (Join-Path $Pilot 'qualification_diagnostics.json'))) { throw '没有旧资格数值诊断回执。' }
Run-Python @('-m',$Module,'run','--output',$Pilot,'--approved-call-limit','3664') 'pilot_run.txt'
Run-Python @('-m',$Module,'audit','--output',$Pilot) 'pilot_audit.txt'
Run-Python @('-m',$Module,'report','--output',$Pilot) 'pilot_report.txt'
$Audit = Get-Content -Raw (Join-Path $Pilot 'audit.json') | ConvertFrom-Json
if (!$Audit.passed -or $Audit.trial_count -ne 112) { throw '试验未完整验收。' }
$Result = Get-Content -Raw (Join-Path $Pilot 'results.json') | ConvertFrom-Json
if ($Result.charged_model_calls -gt 3664 -or $Result.new_rcaide_mda_calls -ne 0) { throw '调用边界异常。' }
# 此处停止，不自动启动下一批、确认试验、场景验证或MDA。
```

最大3360在线提案；原LSBO原生早停可能减少实际步数，审计会逐条确认其依据。不得要求原LSBO补足30步或将剩余费用给其他方法。每个trial的5秒复核不进入训练，不修改名义最优，不按复核结果选替代设计。失败运行和无可行结果保留。

## 7. 故障处置与回传

任何非零退出码：立即停止后续命令，保留执行锁、未结算调用、计划、源码快照和日志。不自动重试，不删锁，不续跑，不更换种子，不通过改哈希绕过检查。未经重新评审不能复制部分结果启动另一个“补跑”目录。

全部正常完成后回传本轮完整Evidence目录（日志、测试、smoke、pilot、qualification_diagnostics、账本、提案、观测、manifest、summary、verification、timing、计划和快照）。摘要应列实际SHA、环境、测试数、各预算角色实际调用、原LSBO停止数、冷暖分层、名义及5秒可行率、条件目标样本数、复核差异和审计是否通过。

大档案放GitHub Release，记录文件清单、ZIP大小与SHA256；小摘要新建结果分支，不覆盖main或历史结果。不将机器专用editable路径、完整凭据或大型产物重复提交普通Git。原资格目录已经发布，不需要再次上传。允许保留本机原始环境日志在受控原始档案，但公开轻量摘要应去掉机器专用绝对路径；不要修改已封存运行证据。

当前pilot属于八种子开发研究，不证明普遍优势、风险校准或物理安全。稀疏标识不等于低于10%已证实，现有区间主要是单侧界。CBOB正式参照、确认性统计和RCAIDE工程MDA仍是后续单独任务。
