# 给本地大模型：从数学层转入简化飞机（不补做旧诊断）

本轮只执行新的飞机接口 smoke 和模型资格检查。不要再执行旧
`TRANSFER_REVIEW_LOCAL_GUIDE_20261030.md`，也不需要旧transfer原始ZIP作先决目录。
保留所有历史结果，不重跑数学pilot，不修改算法参数，不启动RCAIDE。

## 版本、范围与费用

仓库：Luyi1235/RA-LSBO。
分支：`research/aircraft-transition-20261030`，PR #7（未合并）。
固定执行提交：`b8a94c324cd23a0ef9f15c668fe9a247e4ad9a00`。
后续交付文档提交只补说明，不要以移动HEAD代替固定执行版本。

新入口：`python -m ra_lsbo.experiments.simplified_aircraft.transition`。
旧`compare.py`不是这次入口。该入口没有完整pilot、confirm或MDA命令。

本机研究上限352次简化飞机调用：36次smoke + 316次资格。
测试内部人工/廉价求值另计，RCAIDE MDA上限0。
不重做云端64次CI缩小资格配置；本机资格使用`--size full`。

## 0. 不修改旧工作区/已有仿真环境

先确定本机仓库位置和当前正在运行的进程。若有其他任务在跑，不切换或覆盖其工作区。
不得`git reset --hard`、`git clean`、删除旧锁或强制杀掉不相关进程。
不在已登记实验目录修改代码、依赖或计划。网络/依赖问题不是允许免费重试评价器的理由。

在现有仓库目录用PowerShell建立新worktree；路径相对于实际仓库，不复制机器专用路径：

```powershell
$ErrorActionPreference = 'Stop'
$Expected = 'b8a94c324cd23a0ef9f15c668fe9a247e4ad9a00'
$Repo = (git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw '无法确定仓库根目录' }
git -C $Repo remote -v
# 确认origin指向Luyi1235/RA-LSBO；不擅自改写remote。
git -C $Repo status --short
# 旧工作区允许保留未提交内容，但绝不将其复制到新执行worktree。
git -C $Repo fetch origin research/aircraft-transition-20261030
if ($LASTEXITCODE -ne 0) { throw 'fetch失败' }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Work = Join-Path (Split-Path $Repo -Parent) ('RA-LSBO-aircraft-entry-' + $Stamp)
if (Test-Path $Work) { throw '工作目录已存在' }
git -C $Repo worktree add --detach $Work $Expected
if ($LASTEXITCODE -ne 0) { throw '建立独立worktree失败' }
Set-Location $Work
if ((git rev-parse HEAD).Trim() -ne $Expected) { throw '提交不匹配' }
if (git status --porcelain --untracked-files=no) { throw '受跟踪源码不干净' }
```

不依赖PR合并，也不要只`git pull main`。

## 1. 环境核查和影响范围回归

激活既有`Rcaide_vsp`（以本机实际名称和解释器为准），确认不是系统Python。
先检查，不直接用pip覆盖原MDA环境。若版本/导入不兼容，停止并报告，或另建独立环境；
不得为了运行新实验破坏既有RCAIDE/OpenVSP/CFD环境。

```powershell
$Python = (Get-Command python).Source
& $Python -c "import sys, numpy, scipy, torch, gpytorch, botorch, sklearn; print(sys.executable); print(sys.version); print(numpy.__version__, scipy.__version__, torch.__version__, gpytorch.__version__, botorch.__version__, sklearn.__version__)"
if ($LASTEXITCODE -ne 0) { throw '模型依赖导入失败' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw '依赖冲突，禁止直接覆盖仿真环境' }
$Out = Join-Path $Work ('runs/aircraft_entry_' + $Stamp)
New-Item -ItemType Directory -Path $Out | Out-Null
& $Python -m pip freeze | Out-File (Join-Path $Out 'environment.txt') -Encoding utf8
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MPLBACKEND='Agg'
$env:PYTHONHASHSEED='0'
& $Python -m pytest -q tests/test_aircraft_transition.py tests/test_discovery_stages.py tests/test_discovery_stages_evidence.py tests/test_discovery_stages_phase.py tests/test_paper_experiments.py tests/test_soc_revision_acquisition.py tests/test_soc_revision_posterior.py tests/test_numerical_safeguards.py --junitxml="$Out/local_tests.xml"
if ($LASTEXITCODE -ne 0) { throw '回归失败，不进入模型评估' }
```

读取JUnit确认264项测试、0失败、0错误、0跳过；“缺依赖跳过”不算通过。已验证的CI环境见验证回执。
新测试使用人工标签，没有把它们当成真实飞机研究调用。

## 2. 六组算法接口 smoke：36次

不含自适应权重。方法为Random、阶段式单潜空间RA、匹配层级one-hot、
精确值训练视图、随机发现后相同潜空间开发、冻结原LSBO。两任务各六共同初值，
每方法两次在线。只能验证接口，不能评价性能排名。

```powershell
$Entry = 'ra_lsbo.experiments.simplified_aircraft.transition'
$Smoke = Join-Path $Out 'smoke'
& $Python -m $Entry prepare --stage smoke --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke准备失败' }
# 读取plan/registration：必须为36次、MDA0。
& $Python -m $Entry run --output $Smoke --approved-call-limit 36
if ($LASTEXITCODE -ne 0) { throw 'smoke执行失败，保留所有证据，不重试/删锁' }
& $Python -m $Entry audit --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke审计失败' }
& $Python -m $Entry report --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke报告失败' }
```

## 3. 独立模型资格检查：316次

100km普通任务及140km历史“sparse”任务均保留原设置，不调航程、储备或约束阈值。
256次独立面板 + 12次锚点积分步长检查 + 48次离散分支探针检查。
所有面板结果保留；未知、数值异常、物理违规、精确/区间/缺失输出分别报告。
锚点和刻意选定分支探针不进入随机可行率分母，也不作优化器初始标签。

```powershell
$Qual = Join-Path $Out 'qualification'
& $Python -m $Entry prepare --stage qualification --size full --output $Qual
if ($LASTEXITCODE -ne 0) { throw '资格准备失败' }
# 读取plan/registration：必须为316次、每任务128面板点、MDA0。
& $Python -m $Entry run --output $Qual --approved-call-limit 316
if ($LASTEXITCODE -ne 0) { throw '资格执行失败，保留账本/未结算调用/锁，不自动重试' }
& $Python -m $Entry audit --output $Qual
if ($LASTEXITCODE -ne 0) { throw '资格审计失败' }
& $Python -m $Entry report --output $Qual
if ($LASTEXITCODE -ne 0) { throw '资格报告失败' }
```

**到此停止。** 不自动启动多种子性能比较、不启动场景CVaR验证、不启动RCAIDE MDA。

## 4. 结果回传与研究决策

保留两个完整目录（账本、全部提案/观测、计划/注册、来源快照、审计、结果、报告），
以及环境/JUnit/控制台日志。提交轻量摘要到独立results分支；大归档优先Release附件，
记录文件清单、字节数和SHA256。不要把缓存、凭据、原机器路径副本或旧全量ZIP再提交。

回传固定代码SHA、实际调用数、未结算数、全部异常，以及每任务：
面板可用可行/已知违规/未知数量及区间；各输出kind和终止原因；按构型分层计数；
20/10/5秒能量及约束差异；所有inconclusive/needs_review项。

`aircraft_sparse`只是旧任务名，不能当作稀疏性证据。部分输出的积分检查不会自动判通过；
即使软件审计passed，也不是物理模型已经校准或正式实验已获准。

接下来依据资格结果，冻结新的飞机pilot预算、指标和对照；如模型存在真问题，
先版本化修正，不覆盖本次结果。CBOB参考、正式统计、扰动验证与RCAIDE均单独安排。

## 可以直接给本地模型的简短任务

无需补做旧transfer诊断。按本指南同步固定提交`b8a94c324cd23a0ef9f15c668fe9a247e4ad9a00`，独立worktree核对环境和回归，
随后执行36次飞机接口smoke和316次独立资格检查，共352次简化模型调用，完成审计/报告后停止。
保留旧工作区和所有历史证据，不换参数或种子，不删锁续跑，不自动重试，不启动完整pilot或RCAIDE。
