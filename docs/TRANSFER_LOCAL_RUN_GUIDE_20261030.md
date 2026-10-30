# 本地大模型执行交接：phase评审后的表示信任筛查

日期2026-10-30。直接同步GitHub实现，不重新写补丁，不改旧记录。
固定运行提交：`21505bba5f9485287a3e34f4ce50dc849edf4e04`。
分支：`research/phase-evidence-20261030`，PR #5，未合并main。
工作机本轮只运行transfer smoke及pilot；不是再运行phase/旧representation/baselines。

## 保护原件与明确预算

必须保存完整旧phase目录（含calls.jsonl、全部optimization记录、source_snapshot），而非仅research_results汇总目录：
`runs/phase_pilot_20261029_205231_7f4f459f/`。
其运行版本155f0b0与新版本不同是正常现象；新只读导入器核验固定旧字节，不改旧Windows路径、换行或哈希，也不执行旧快照。

新pilot六组、两个稀疏算例、8个新分配开发种子9801–9808、初值12、在线30，96trial、2880在线调用，上限3108。
先跑四算例一seed两步smoke，112调用、24trial、48在线；本机研究总上限3220。
单元测试内部廉价/人工求值及历史费用另计。新增RCAIDE MDA必须为0。
保留所有失败与unknown，不能换种子、排除失败、挑选任务赢家或用已知可行初值替换随机初值。
8种子只是机制筛查，冷启动样本数不足也要报告，不许重抽。

## A. 同步、固定版本及环境

先识别用户现有仓库路径，执行git status --short并记录未提交修改。不要stash/reset/clean旧工作区，不停止无关进程。
以下PowerShell中$RepoRoot必须由本地模型从用户机器实际定位，不要照抄一个虚构绝对路径。

```powershell
$ErrorActionPreference = 'Stop'
function Invoke-Checked {
    param([string]$Exe, [string[]]$ArgList)
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "命令失败，停止后续执行：$Exe $ArgList" }
}
# 在已定位的原仓库中执行（不改文件）：
$RepoRoot = (git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw '当前目录不是目标仓库' }
Invoke-Checked git @('-C',$RepoRoot,'remote','get-url','origin')
# 确认origin确为Luyi1235/RA-LSBO；不同则停止，不能覆盖origin。
Invoke-Checked git @('-C',$RepoRoot,'status','--short')
Invoke-Checked git @('-C',$RepoRoot,'fetch','origin','research/phase-evidence-20261030')
$Sha = '21505bba5f9485287a3e34f4ce50dc849edf4e04'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Work = Join-Path (Split-Path $RepoRoot -Parent) ('RA-LSBO-transfer-' + $Stamp)
if (Test-Path $Work) { throw '新worktree已存在，不可覆盖' }
Invoke-Checked git @('-C',$RepoRoot,'worktree','add','--detach',$Work,$Sha)
Set-Location $Work
$Actual = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $Actual -ne $Sha) { throw '固定提交不符' }
```

使用现有Rcaide_vsp或已验证的独立环境，记录python路径和版本、pip freeze、pip check、硬件和线程设置。不要擅自升级/卸载原MDA环境。依赖不符或pip check失败时先停止报告；必要时另建隔离环境，不能为通过审计改requirements或结果。
CI用Linux/Python3.11与torch2.8.0+cpu；工作机Windows须重新验证。当前代码支持Python3.9+，不是承诺任意依赖组合可用。

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:MPLBACKEND='Agg'; $env:PYTHONHASHSEED='0'
Invoke-Checked python @('-c','import sys; print(sys.executable); print(sys.version)')
Invoke-Checked python @('-m','pip','check')
$Check = Join-Path $Work ('runs/transfer_local_check_' + $Stamp)
New-Item -ItemType Directory -Path $Check | Out-Null
python -m pip freeze | Set-Content -Encoding utf8 (Join-Path $Check 'environment.txt')
if ($LASTEXITCODE -ne 0) { throw '环境记录失败' }
```

$Check用于新检查记录，不写进旧实验。后续命令一旦非零退出立即停止，保留日志；不要隐式继续跑pilot。

## B. 本机回归与smoke

```powershell
$Tests = @('tests/test_evidence_transfer.py','tests/test_discovery_stages.py',
 'tests/test_discovery_stages_evidence.py','tests/test_discovery_stages_phase.py',
 'tests/test_numerical_safeguards.py','tests/test_external_comparison.py',
 'tests/test_comparison_reliability.py','tests/test_comparison_entrypoints.py',
 'tests/test_comparison_visualization.py','tests/test_paper_experiments.py',
 'tests/test_grouped_partial_gp.py')
Invoke-Checked python (@('-m','pytest','-q') + $Tests + @('--junitxml=' + (Join-Path $Check 'pytest.xml')))
# 验收280 passed、0 failures/errors/skipped；不以退出码0但跳过模型测试为通过。
$Module = 'ra_lsbo.experiments.discovery_stages.compare'
$Smoke = Join-Path $Work ('runs/transfer_smoke_' + $Stamp)
Invoke-Checked python @('-m',$Module,'prepare','--stage','transfer','--profile','smoke','--output',$Smoke)
Invoke-Checked python @('-m',$Module,'run','--output',$Smoke,'--approved-call-limit','112')
Invoke-Checked python @('-m',$Module,'audit','--output',$Smoke)
Invoke-Checked python @('-m',$Module,'report','--output',$Smoke)
```

检查112/112调用、24trial、48在线、审计通过、MDA0。读取提案确认冷阶段保存两个expert预测，prequential更新只发生在结算后，warm不继续双专家；注意smoke只有两步，不能当成性能证据。不得用旧smoke代替新工作机验收。

## C. 新pilot：先决目录只读，然后新登记

本地模型将$OldPhase设为真实完整旧phase目录的绝对路径。不要设为research_results摘要目录。保留旧execution.lock是正常的，导入器不把它当作新实验锁。

```powershell
# $OldPhase = 已定位的完整旧phase目录；不是一个需要用户再次下载的占位凭据。
if (-not (Test-Path (Join-Path $OldPhase 'calls.jsonl'))) { throw '旧phase完整账本缺失' }
if (-not (Test-Path (Join-Path $OldPhase 'source_snapshot'))) { throw '旧phase源码快照缺失' }
$Pilot = Join-Path $Work ('runs/transfer_pilot_' + $Stamp)
Invoke-Checked python @('-m',$Module,'prepare','--stage','transfer','--profile','pilot',
 '--prerequisite',$OldPhase,'--output',$Pilot)
$Plan = Get-Content -Raw (Join-Path $Pilot 'plan.json') | ConvertFrom-Json
if ($Plan.source_commit -ne $Sha -or $Plan.budget.maximum_model_calls -ne 3108) { throw '新计划身份/预算不符' }
if (($Plan.profile.seeds -join ',') -ne '9801,9802,9803,9804,9805,9806,9807,9808') { throw '新计划种子不符' }
Invoke-Checked python @('-m',$Module,'run','--output',$Pilot,'--approved-call-limit','3108')
Invoke-Checked python @('-m',$Module,'audit','--output',$Pilot)
Invoke-Checked python @('-m',$Module,'report','--output',$Pilot)
```

实际运行前还核验六组ID、两个算例、initial12/horizon30、所有MDA上限0。不要下载CI的Linux准备计划直接运行。登记后不git pull、不改源码/依赖/计划/快照。
新版无resume。失败时保留已结算/未结算调用与锁，停止回传，不删锁续跑、不自动重试、不拼接跨版本轨迹。出现未知结果不能免费重试。

## D. 回传

完成后停止，不自动进入下一阶段。回传固定SHA、环境、JUnit、实际调用数/时间、全部原始提案/观测/manifest、calls.jsonl、plan/registration/source_snapshot/audit/results/report及检查日志。
关注冷发现分母、首次可行删失成本、warm配对、等权与自适应差异、onehot与latent差异、简单随机后开发参照和额外拟合时间。不得只报告成功目标中位数。
完整原始ZIP放GitHub Release并保存SHA256，轻量摘要放独立results分支；不要把数百MB压缩包再次提交普通Git历史，不包含凭据/本机缓存/无关模型数据。

新机制尚未证明有效。若不超过等权或简单混合策略，应保留负面证据，不按结果修改种子、阈值或协议。该轮不是独立确认、一般有限区间实验、官方CBOB复现或飞机MDA。
