# 给本地大模型：阶段覆盖复验同步和执行说明

2026-10-29；仓库Luyi1235/RA-LSBO。
分支`research/discovery-review-20261029`；PR #4。
**运行固定提交：`155f0b0237183d3feefa124e430c01bcd5db1450`。**
本说明为后续文档交付；不要把移动HEAD代替固定运行版本。

## 任务和限制

同步已经提交的代码，不自行按文字重新实现。创建独立worktree；保留已完成discovery pilot原目录、计划/来源/账本/锁/日志。不要重跑旧pilot，不覆盖旧目录，不删除锁续跑，不改历史哈希或换行，不剔除失败种子，不自动免费重试未结算调用。

本轮仅：本机回归 → 新phase smoke(120次) → 新phase pilot(上限8916次) → 审计/报告 → 停止回传。研究调用上限9036次；测试内部求值另计。不要启动representation、baselines、飞机pilot、confirm或RCAIDE MDA；也不要用旧版本绕过此暂停要求。

原因：新discovery数据的覆盖包改善冷启动，但Rosenbrock14个暖启动中13个最终目标劣于原RA-v2。下一轮检验是否应在首次可行后关闭新增覆盖，近重复保护始终保留；并非已证明新方案更好。

## 1. 定位原始完整先决证据

本机原pilot目录名：`discovery_pilot_20261029_134131_9366a4b6`。
需要该目录完整806项封存证据及source_snapshot，不是GitHub只上传摘要的research_results目录，也不是外层打包目录，不是smoke。让本地模型从已知旧工作区及原运行日志定位并给变量`$Prerequisite`赋真实绝对路径；不向旧目录写任何文件。

关键原件SHA256：
- plan.json：3b2013aadd6cc999483c04225e7e09760abadcac5f6178690ba0a97087c00af4
- audit.json：a66d885395f6584efad2a78127c67bba9ff3fd733e840b6efaaddf3be4aa1b44
- results.json：b54612217d5d48174ec4c5ba49ac7de38228112ee0de6968c46375eb98704398
- registration.json：c1c0519d0736671fd6f092ec852c463e6b9d16f4547e9fbfa4008febdb2ca6c6

若从发布附件取得，原ZIP SHA256必须为0867c1fe7eaeaa06e0930700bea658d6f4e10fd4e48466439e58efee6655494f；保留原目录结构，选择内部`runs/discovery_pilot_20261029_134131_9366a4b6`。发布附件地址见评审文档。文件不匹配时停止，不能重写JSON、重封存或绕过哈希检查。

## 2. 固定代码的独立worktree

在已确认origin为目标仓库的现有Git仓库根执行PowerShell：

```powershell
$ErrorActionPreference = 'Stop'
$CodeCommit = '155f0b0237183d3feefa124e430c01bcd5db1450'
$Old = (git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw '请先定位正确Git仓库' }
git remote -v
git status --short
# 检查origin确为Luyi1235/RA-LSBO；不得覆盖未提交修改。
git fetch origin research/discovery-review-20261029
if ($LASTEXITCODE -ne 0) { throw 'fetch失败' }
git cat-file -e "${CodeCommit}^{commit}"
if ($LASTEXITCODE -ne 0) { throw '固定提交不存在' }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$New = Join-Path (Split-Path $Old -Parent) ('RA-LSBO-phase-' + $Stamp)
if (Test-Path $New) { throw '新目录已存在' }
git worktree add --detach $New $CodeCommit
if ($LASTEXITCODE -ne 0) { throw 'worktree创建失败' }
Set-Location $New
if ((git rev-parse HEAD).Trim() -ne $CodeCommit) { throw '提交不匹配' }
if (git status --porcelain) { throw '新工作区不是干净状态' }
```

仅`pull main`不能取得本修订。禁止reset --hard、git clean、覆盖本地修改或在旧运行目录切换分支。不需要等待PR合并。

## 3. 环境及先决证据导入检查

激活已经验证过的`Rcaide_vsp`；不擅自升级现有物理软件环境。若依赖冲突，停止并考虑独立环境，不改旧环境使历史审计失效。

```powershell
conda activate Rcaide_vsp
python -c "import sys; print(sys.executable); print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw '解释器错误' }
python -c "import importlib.metadata as m; print({n:m.version(n) for n in ['numpy','scipy','torch','gpytorch','botorch','scikit-learn','matplotlib','pytest']})"
if ($LASTEXITCODE -ne 0) { throw '缺依赖' }
python -m pip check
if ($LASTEXITCODE -ne 0) { throw '依赖冲突' }
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$env:PYTHONHASHSEED = '0'
# $Prerequisite由第1步定位赋值，不要把下述检查改成忽略。
if (-not $Prerequisite -or -not (Test-Path (Join-Path $Prerequisite 'calls.jsonl'))) { throw '缺完整旧pilot' }
$env:RA_DISCOVERY_PARENT = $Prerequisite
python -c "import os; from ra_lsbo.experiments.discovery_stages.reviewed_evidence import verify_reviewed_discovery; print(verify_reviewed_discovery(os.environ['RA_DISCOVERY_PARENT']))"
if ($LASTEXITCODE -ne 0) { throw '旧证据不匹配；禁止修改历史文件修复' }
```

预期主要依赖numpy2.0.2/scipy1.13.1/torch2.8.0/gpytorch1.11/botorch0.10.0/sklearn1.6.1/matplotlib3.9.4；CI pytest8.3.5。旧本机Python3.9与CI3.11不同，必须完成本机验证。导入器只验证固定原始证据，不会重写Windows历史路径；不要用新代码的严格audit直接加载旧计划。

## 4. 本机回归

```powershell
$Validation = Join-Path 'runs' ('phase_validation_' + $Stamp)
New-Item -ItemType Directory -Path $Validation | Out-Null
python -m pip freeze | Out-File (Join-Path $Validation 'environment.txt') -Encoding utf8
git rev-parse HEAD | Out-File (Join-Path $Validation 'commit.txt') -Encoding utf8
python -m pytest -q tests/test_discovery_stages.py tests/test_discovery_stages_evidence.py tests/test_discovery_stages_phase.py tests/test_numerical_safeguards.py tests/test_external_comparison.py tests/test_comparison_reliability.py tests/test_comparison_entrypoints.py tests/test_comparison_visualization.py tests/test_paper_experiments.py tests/test_grouped_partial_gp.py --junitxml="$Validation/pytest.xml"
if ($LASTEXITCODE -ne 0) { throw '回归失败，停止并保存证据' }
```

核对JUnit：CI为262项、0fail/error/skip。本机若跳过依赖测试或收集项数不符，停止核查，不能将skip称为验证通过。保存控制台日志。测试是影响范围验证，不是所有历史测试。

## 5. 全新phase smoke，上限120次

```powershell
$Smoke = Join-Path 'runs' ('phase_smoke_' + $Stamp)
python -m ra_lsbo.experiments.discovery_stages.compare prepare --stage phase --profile smoke --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke prepare失败' }
python -m ra_lsbo.experiments.discovery_stages.compare run --output $Smoke --approved-call-limit 120
if ($LASTEXITCODE -ne 0) { throw 'smoke run失败，保留锁和账本，不重用目录' }
python -m ra_lsbo.experiments.discovery_stages.compare audit --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke audit失败' }
python -m ra_lsbo.experiments.discovery_stages.compare report --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke report失败' }
```

预期28trial、56在线、总120；新三组保护策略的阈值内近重复0，cold-scoped组warm_coverage=0。检查未知/失败状态，不只看进程退出码。不得根据smoke排名选种子或调参数。

## 6. 全新phase pilot，上限8916次

```powershell
$Pilot = Join-Path 'runs' ('phase_pilot_' + $Stamp)
python -m ra_lsbo.experiments.discovery_stages.compare prepare --stage phase --profile pilot --prerequisite $Prerequisite --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot prepare失败' }
Get-Content (Join-Path $Pilot 'registration.json')
# 核对schema=ra_lsbo_phase_discovery_v1，7组，2算例，9701-9720，12初值，30在线。
python -m ra_lsbo.experiments.discovery_stages.compare run --output $Pilot --approved-call-limit 8916
if ($LASTEXITCODE -ne 0) { throw 'pilot失败/中断，保留所有证据；不删锁续跑' }
python -m ra_lsbo.experiments.discovery_stages.compare audit --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot审计失败，禁止最终报告' }
python -m ra_lsbo.experiments.discovery_stages.compare report --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot报告失败' }
```

280trial、8400在线+480初值+32面板+4锚点=8916。方法：random、ra_v2、ra_coverage、ra_discovery、ra_guard_only、ra_phase_coverage、ra_phase_discovery。保持全部种子，不按任务使用不同赢家。从prepare到report不改登记源码/环境，不并发使用同一结果目录。模型拟合数量较多，不能按真实函数调用数推算运行时间。

## 7. 交付与停止

完成后停止，不自动启动消融/基线/MDA。回传实际SHA、环境、测试结果、smoke/pilot实际调用、审计与报告，并上传完整原始运行记录及SHA256清单。目录中保留calls.jsonl、每trial manifest/proposals/observations/summary、来源快照、plan/registration/audit/results、日志。大ZIP优先GitHub Release附件，仓库保留摘要和哈希，避免提交大型生成文件到普通Git历史。

下一评审同时看每题冷发现率与相同暖启动种子的最终目标，检查是否回收开发能力；仅显示warm_improved不足以通过。所有报告必须保留无解和失败分母。这是已查看过的开发种子，不宣称独立确认或安全保证。
