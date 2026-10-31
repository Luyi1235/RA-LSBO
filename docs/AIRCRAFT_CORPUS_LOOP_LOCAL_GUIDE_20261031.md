# 工作机执行：五组简化飞机2,000构型持久表示闭环

日期：2026-10-31。不是重复离线重构任务；这是新入口、新计划、新种子的受控开发性闭环。

## 1. 固定版本和不可变范围

分支 `research/aircraft-corpus-loop-20261031`；PR #13，尚未合并。
固定运行提交：`9243aca17d262945e426a61b2e7a6b50a1ef2de3`。
指南/回执在后续文档提交中；不要因此把运行SHA换成移动HEAD。
新入口：`python -m ra_lsbo.experiments.simplified_aircraft.corpus_pilot`。
不要运行旧nominal_pilot、transition、transfer或离线study_structure_corpus_2000。

使用独立worktree及原Windows/Rcaide_vsp环境。保留所有旧运行目录、锁、账本、来源快照及哈希。
不得安装/升级包以绕过验证；依赖不满足或与原环境不符时，停止并报告。
没有自动续跑、删锁重试或替换失败种子的权限。不启动RCAIDE/场景/确认性实验。

## 2. 三种文件的用途——不要再把交接包当作结果输入

**本次交接包**：`RA_LSBO_aircraft_corpus_loop_handoff_20261031.zip`。
它装指南、协议和云端验证证据，不是 `--corpus-evidence` 的输入。

**所需旧离线结果ZIP**：`RA-LSBO-structure-corpus-output-20261031_01_records.zip`（约5.2 MB）。
SHA-256：`d7953cdcc91891eff8fa0ff7c1fc1f7103ccdb7d24a39cbd94651b4ef2a2628f`。
这是刚完成的160次编码器/320头离线任务的原始结果，用作只读来源核验；不导入旧权重或训练标签。
原文件已有时直接使用，无需重新上传或下载；不能替换成上一轮handoff ZIP或摘要JSON。

**所需旧资格目录**：首次316次简化飞机资格检查的完整 `qualification/` 目录。
必须保留 plan.json、registration.json、audit.json、results.json、calls.jsonl 和 source_snapshot/ 等原文件。
不能使用轻量research_results摘要目录。若旧完整目录不可用，使用既有封存ZIP恢复到新的只读目录：
`aircraft_entry_20261030_01_full_records.zip`，SHA256
`ea1359fb7c76ea8641c54432e8e901fc4de065be6dc643590c585f6c1d7d7789`。
不要修改旧计划中Windows路径或行尾去强制通过；导入器已支持只读核验。

本次不要求原176 MB nominal pilot ZIP、旧transfer目录或重复运行之前的诊断。

## 3. 配置、计算预算与停止边界

五组：random、ra_old30、ra_old480、ra_corpus、ra_onehot。
主候选ra_corpus：每trial固定2,000条合法构型有放回抽样，重复不删除；与当前评估前缀结构共同训练。
t=0训练30轮/480更新；t=8、16、24各增量20轮/320更新；每trial总1,440更新。
只有真实可信精确目标作性能监督。无标签构型不含能耗、SOC或可行标签，不调用飞机评估器。
即使覆盖步骤跳过GP，也按固定日程维护表示。网络和私有随机状态保留；Adam动量不保留。
不临时追加轮数、改尺度/核/采集器、按重构结果删种子或换方法。

smoke：两题，seed9920，6共同初值、2在线/方法、每trial最多1次同点5s复核。
最大42=12初值+20在线+10复核。两个在线步骤不测试后续t=8更新；对应状态链由合成集成测试覆盖。
pilot：两题，seeds9921--9928，12共同初值、30在线/方法；80trial。
最大2,672=192初值+2,400在线+80复核。
**本机总研究上限2,714次简化飞机评价，RCAIDE MDA上限0。** 历史费用、云端smoke与测试内部廉价求值另计。
无可行设计时省略该次复核，不能补造调用或将剩余额度转给其他trial。

每个trial先固定名义最优可行设计，再同点5s复核一次；复核不回灌、不重新选择替代设计。
主比较为ra_corpus对ra_onehot及ra_old480：先看5s可用可行性，再看同一点的5s能耗，按seed配对。
全部失败和条件分母保留。smoke的性能或重构率不得用于调整pilot。
任何unknown/infra/数值失败在结算留证后停止整个campaign，不能自动重试。
程序非零退出码立即停止后续步骤；保留已发生成本和不完整记录。

## 4. PowerShell执行模板

本地模型根据已存在的仓库、原Python及两个输入的真实路径填写下面五个变量。
不要猜测机器绝对路径，也不要覆盖既有目录。输出不能位于旧结果目录内。

```powershell
$Repo = '<现有RA-LSBO Git仓库目录>'
$Worktree = '<尚不存在的新worktree目录>'
$Py = '<原Rcaide_vsp环境的python.exe完整路径>'
$Qualification = '<原完整316调用qualification目录>'
$CorpusArchive = '<RA-LSBO-structure-corpus-output-20261031_01_records.zip完整路径>'
$Output = '<尚不存在的新结果总目录>'
$Commit = '9243aca17d262945e426a61b2e7a6b50a1ef2de3'
$ErrorActionPreference = 'Stop'

function CheckExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed, exit $LASTEXITCODE" }
}
function Invoke-Py {
    param([string]$Step, [string]$Log, [string[]]$CommandArgs)
    & $Py @CommandArgs 2>&1 | Tee-Object -FilePath $Log
    $Exit = $LASTEXITCODE
    if ($Exit -ne 0) { throw "$Step failed, exit $Exit" }
}
if ((Test-Path $Worktree) -or (Test-Path $Output)) { throw '新worktree与输出目录必须不存在' }
if (!(Test-Path $Py) -or !(Test-Path $CorpusArchive) -or !(Test-Path $Qualification)) { throw '所需输入不存在' }
$ArchiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CorpusArchive).Hash.ToLowerInvariant()
if ($ArchiveHash -ne 'd7953cdcc91891eff8fa0ff7c1fc1f7103ccdb7d24a39cbd94651b4ef2a2628f') { throw '离线结果ZIP身份不符' }
foreach ($Name in @('plan.json','registration.json','audit.json','results.json','calls.jsonl','source_snapshot')) {
    if (!(Test-Path (Join-Path $Qualification $Name))) { throw "旧资格目录缺少 $Name" }
}
git -C $Repo fetch origin research/aircraft-corpus-loop-20261031
CheckExit 'fetch'
git -C $Repo worktree add --detach $Worktree $Commit
CheckExit 'worktree'
Set-Location $Worktree
$Head = git rev-parse HEAD
CheckExit 'rev-parse'
if ($Head.Trim() -ne $Commit) { throw '未固定到验收版本' }
$Dirty = git status --porcelain --untracked-files=no
CheckExit 'git status'
if ($Dirty) { throw '受跟踪源码不干净' }
New-Item -ItemType Directory -Path $Output | Out-Null
$Logs = Join-Path $Output 'logs'
New-Item -ItemType Directory -Path $Logs | Out-Null
$Head | Set-Content -Encoding utf8 (Join-Path $Output 'execution_commit.txt')
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONHASHSEED='0'
$env:MPLBACKEND='Agg'
Invoke-Py -Step 'dependency check' -Log (Join-Path $Logs 'pip_check.txt') -CommandArgs @('-m','pip','check')
Invoke-Py -Step 'environment' -Log (Join-Path $Output 'environment.txt') -CommandArgs @('-m','pip','freeze')
Invoke-Py -Step 'platform' -Log (Join-Path $Output 'platform.txt') -CommandArgs @('-c','import sys,platform,torch,numpy; print(sys.version); print(platform.platform()); print(torch.__version__); print(numpy.__version__); print(torch.__config__.show())')
$Tests = @(
 'tests/test_aircraft_corpus_loop.py',
 'tests/test_structure_corpus.py',
 'tests/test_structured_encoder.py',
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
$JUnit = Join-Path $Output 'junit.xml'
$TestArgs = @('-m','pytest','-q') + $Tests + @('--junitxml',$JUnit)
Invoke-Py -Step 'affected regressions' -Log (Join-Path $Logs 'pytest.txt') -CommandArgs $TestArgs
```

应核对JUnit中 **432项全部通过，0失败、0错误、0跳过**。不是只看最后退出码。
测试包含人工观测、VAE训练与廉价合成任务，不能描述为完全没有计算；没有额外授权RCAIDE任务。
只有回归通过后继续：

```powershell
$Module = 'ra_lsbo.experiments.simplified_aircraft.corpus_pilot'
$Smoke = Join-Path $Output 'smoke'
Invoke-Py -Step 'smoke prepare' -Log (Join-Path $Logs 'smoke_prepare.txt') -CommandArgs @('-m',$Module,'prepare','--profile','smoke','--output',$Smoke)
Invoke-Py -Step 'smoke run' -Log (Join-Path $Logs 'smoke_run.txt') -CommandArgs @('-m',$Module,'run','--output',$Smoke,'--approved-call-limit','42')
Invoke-Py -Step 'smoke audit' -Log (Join-Path $Logs 'smoke_audit.txt') -CommandArgs @('-m',$Module,'audit','--output',$Smoke)
Invoke-Py -Step 'smoke report' -Log (Join-Path $Logs 'smoke_report.txt') -CommandArgs @('-m',$Module,'report','--output',$Smoke)
$Pilot = Join-Path $Output 'pilot'
Invoke-Py -Step 'pilot prepare' -Log (Join-Path $Logs 'pilot_prepare.txt') -CommandArgs @('-m',$Module,'prepare','--profile','pilot','--qualification',$Qualification,'--corpus-evidence',$CorpusArchive,'--output',$Pilot)
$Plan = Get-Content -Raw -LiteralPath (Join-Path $Pilot 'plan.json') | ConvertFrom-Json
if ($Plan.budget.maximum_model_calls -ne 2672 -or $Plan.profile.horizon -ne 30 -or $Plan.profile.initial -ne 12 -or (($Plan.profile.seeds -join ',') -ne '9921,9922,9923,9924,9925,9926,9927,9928') -or (($Plan.arms -join ',') -ne 'random,ra_old30,ra_old480,ra_corpus,ra_onehot')) { throw '登记计划与固定协议不符' }
Invoke-Py -Step 'pilot run' -Log (Join-Path $Logs 'pilot_run.txt') -CommandArgs @('-m',$Module,'run','--output',$Pilot,'--approved-call-limit','2672')
Invoke-Py -Step 'pilot audit' -Log (Join-Path $Logs 'pilot_audit.txt') -CommandArgs @('-m',$Module,'audit','--output',$Pilot)
Invoke-Py -Step 'pilot report' -Log (Join-Path $Logs 'pilot_report.txt') -CommandArgs @('-m',$Module,'report','--output',$Pilot)
$AfterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CorpusArchive).Hash.ToLowerInvariant()
if ($AfterHash -ne $ArchiveHash) { throw '旧离线档案发生变化' }
$Dirty = git status --porcelain --untracked-files=no
CheckExit 'final source status'
if ($Dirty) { throw '运行源码发生变化' }
```

不要拿CI的Linux计划搬到Windows运行；在固定本机checkout中按上面prepare新计划。
执行中断或出错后不能再次调用run，即使错误看似很小；先保留日志与账本并报告。
不要由性能smoke胜负判断是否换种子；科学结果没有胜过对照时同样完整保留。

## 5. 产物与回传

保存整个新Output，包括tests/logs、smoke和pilot的plan、registration、execution.lock、calls.jsonl、
source_snapshot、optimization内的manifest/proposals/observations/summary/timing/verification，
以及ra_corpus的representation/corpus.json、events.jsonl和全部state.npz/metadata.json/grammar.json。
不能只上传最优设计或成功运行。所有GP候选失败、表示训练失败和未结算记录均保留。
源码快照和锁要保留；不提交缓存、凭据或机器专用日志到算法源分支。
轻量报告可放独立results分支，大档案放Release，记录固定代码SHA、ZIP SHA256、文件清单及每文件哈希。
不用再次打包旧资格或旧176 MB pilot档案。报告真实调用数，不按上限伪称实际发生。
回传后停止，不自动启动更多种子、尺度/epoch网格、第三层RCAIDE或“挑一个赢家再跑”的任务。

## 6. 研究边界

本次新种子仍是开发性验证。只改善训练重构或只胜过old30不构成最终创新证据。
old480不是与持久VAE严格FLOPs/累计训练匹配；实际训练和整体提案耗时必须同时报告。
主比较保持onehot强对照，不能按算例分别改默认方法。全部失败、核候选异常及未使用额度保留。
5秒复核只是数值一致性检查，不是物理校准、安全认证、机会约束或CVaR风险验证。
审计恢复VAE数值状态并检查特征前向，但不重训、不拟合GP、不独立重放优化器。
