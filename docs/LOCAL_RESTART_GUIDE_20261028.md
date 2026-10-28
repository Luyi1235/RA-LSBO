# RA-LSBO comparison v2：本地同步、验证与重新实验说明

日期：2026-10-28。仓库：Luyi1235/RA-LSBO。
分支：`fix/comparison-reliability-20261028`；PR： https://github.com/Luyi1235/RA-LSBO/pull/1
修复前基线：`8e141c821702b052df4b916c0e4d86a275432497`。
**运行固定代码提交：`aea7607f0edc25f4a424d1d735d85bbb27ebfd68`。**
本文为后续文档提交；不要因文档提交号不同而使用移动的最新HEAD替代固定运行版本。

## 给本地大模型的任务

同步固定代码，不要按说明重新实现算法。识别并停止用户刚启动的数学比较进程，完整保留旧实验。在独立工作目录核查现有 Rcaide_vsp、执行回归测试和一个新的数学 smoke；全部通过后创建新的数学 pilot 计划并运行。不要自动启动完整飞机 pilot 或 RCAIDE MDA。失败时保存证据并停止，不改变种子、不剔除失败运行、不自动重试终态未知调用。

## 已验证的状态

GitHub Actions run 40690256091 已成功：185 passed，0 failures/errors/skipped；数学 smoke 112次、简化飞机 smoke 108次，均完成运行、审计、报告，0次RCAIDE MDA。测试环境为Linux/Python3.11.17，并不替代Windows本机验证。14条Matplotlib/Pyparsing弃用告警保留。详细记录见 `VALIDATION_RECEIPT_20261028.md`。

## 1. 警告与修复含义

SLSQP warning 表示内部试探步越界后 SciPy 执行裁剪，不能仅凭该警告认定最终设计越界或实验失败。本次重跑还针对确定的基线和审计问题，不是为了消除控制台输出。

实际变更：局部捕获并记录精确的 clipping 告警；验证有限坐标/采集值/支持量，拒绝实质越界或非有限端点，保留局部求解状态；稳定log-EI；区分严格正区间违规与未知；SCBO初始与历史最优维护；可信精确目标充分利用；SEGO显式候选资格；完整trial/账本/面板/场景审计和报告前重验。详细版本协议为 `COMPARISON_RELIABILITY_REVISION_20261028.md`。

没有修改site-packages、全局屏蔽警告、放宽物理约束、改变实验种子或任务。四个外部适配器仍不是论文官方实现。RA概率乘积仍无联合概率保证。数学算例的分支权衡限制未在本次bug修订中调整。

## 2. 停止并保留旧实验

核对进程命令行、PID、解释器及输出路径。优先在该实验终端Ctrl+C，只结束已确认的数学比较进程。禁止全局停止所有Python进程，不能影响其他LSBO/MDA工作。

保留原目录及其中 `plan.json`、`registration.json`、`source_snapshot/`、`calls.jsonl`、`execution.lock`、所有optimization记录和控制台日志。停止且不再写入后，在独立说明中记下started/settled数量、未结算call_id和中止原因。已启动但无终态的调用不免费重试，不补造settled记录。旧记录是中止v1探索实验，不与v2拼接，也不能删除旧成本。

不要改历史哈希、删锁续跑、改源码让旧审计假通过。保留旧工作区绝对路径，方便之后历史核查。

## 3. 独立worktree取得固定代码

在已定位的旧仓库根目录，用PowerShell只读检查origin、HEAD和未提交修改。确认origin确为目标仓库后执行：

```powershell
$ErrorActionPreference = 'Stop'
$CodeCommit = 'aea7607f0edc25f4a424d1d735d85bbb27ebfd68'
$Old = (git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw '请先进入正确的Git仓库' }
git remote -v
git status --short
git rev-parse HEAD
# 确认origin正确后继续。
git fetch origin fix/comparison-reliability-20261028
if ($LASTEXITCODE -ne 0) { throw 'fetch失败' }
git cat-file -e "${CodeCommit}^{commit}"
if ($LASTEXITCODE -ne 0) { throw '固定提交不存在' }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$New = Join-Path (Split-Path $Old -Parent) ('RA-LSBO-reliability-' + $Stamp)
if (Test-Path $New) { throw '新目录已存在，换一个新目录' }
git worktree add --detach $New $CodeCommit
if ($LASTEXITCODE -ne 0) { throw 'worktree创建失败' }
Set-Location $New
if ((git rev-parse HEAD).Trim() -ne $CodeCommit) { throw '代码身份不匹配' }
git status --short
```

不使用`reset --hard`、`git clean`、覆盖本地修改或在旧运行目录切换分支。若旧工作区存在必要修改，单独列出差异并决定其用途；不能无声迁入新版本。必要的新改动必须单独提交、测试、重新登记，且不能仍称运行上述固定提交。冻结基线已在Git中，不复制旧大规模runs来补代码。

## 4. 核查现有环境和回归测试

```powershell
conda activate Rcaide_vsp
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import importlib.metadata as m; print({n:m.version(n) for n in ['numpy','scipy','torch','gpytorch','botorch','scikit-learn','matplotlib','pytest']})"
python -m pip check
```

仓库主要版本：numpy2.0.2/scipy1.13.1/torch2.8.0/gpytorch1.11/botorch0.10.0/sklearn1.6.1/matplotlib3.9.4；CI pytest8.3.5。若缺依赖或冲突，停止并报告，优先新建或克隆隔离环境，禁止擅自升级现有物理环境。解释器须来自预期conda环境。不要把依赖缺失导致的skip当作通过。

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$env:PYTHONHASHSEED = '0'
$Validation = Join-Path 'runs' ('local_validation_' + $Stamp)
New-Item -ItemType Directory -Path $Validation | Out-Null
python -m pip freeze | Out-File (Join-Path $Validation 'environment.txt') -Encoding utf8
git rev-parse HEAD | Out-File (Join-Path $Validation 'commit.txt') -Encoding utf8
python -m pytest -q tests/test_numerical_safeguards.py tests/test_external_comparison.py tests/test_comparison_reliability.py tests/test_comparison_entrypoints.py tests/test_comparison_visualization.py tests/test_paper_experiments.py tests/test_grouped_partial_gp.py --junitxml="$Validation/pytest.xml"
if ($LASTEXITCODE -ne 0) { throw '测试失败：禁止启动smoke/pilot，保留日志' }
```

这是影响范围回归，不是整个历史仓库全测试验证。测试内人工标签和廉价评价与研究账本分开记录。

## 5. 新数学smoke：上限112次，0次MDA

```powershell
$Smoke = Join-Path 'runs' ('math_smoke_v2_' + $Stamp)
python -m ra_lsbo.experiments.mathematical_benchmarks.compare prepare --profile smoke --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke prepare失败' }
python -m ra_lsbo.experiments.mathematical_benchmarks.compare run --output $Smoke --approved-call-limit 112
if ($LASTEXITCODE -ne 0) { throw 'smoke run失败，不复用目录或自动重试' }
python -m ra_lsbo.experiments.mathematical_benchmarks.compare audit --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke audit失败' }
python -m ra_lsbo.experiments.mathematical_benchmarks.compare report --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke report失败' }
```

应有4个数学ordinary/sparse算例、六方法、1种子、每方法2在线：24trial、48在线提案、112总调用。核对v2 schema、审计通过、报告生成和未知/失败统计。局部告警与端点诊断在`proposals.jsonl`的`metadata.local_reports`，包含`slsqp_clipping_warnings`、`bound_projections`、`max_callback_bound_excess`、`success/status/message/local_error`。SciPy内部裁剪前幅度不可由计数推断。

## 6. 新数学pilot：完整上限7,716次

保持9701–9720、12共同初值、30在线、128候选、每题16面板，算例为branin_sparse和rosenbrock_sparse。240trial、7,200在线调用，加4锚点、32面板、480初值，总7,716。数学smoke+pilot新增研究上限7,828；旧中止费用和测试调用另计，不自动追加飞机pilot。

```powershell
$Pilot = Join-Path 'runs' ('math_pilot_v2_' + $Stamp)
python -m ra_lsbo.experiments.mathematical_benchmarks.compare prepare --profile pilot --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot prepare失败' }
# 只读核对新plan.json的schema、seeds、cases、arms、预算和来源。
Get-Content (Join-Path $Pilot 'registration.json')
python -m ra_lsbo.experiments.mathematical_benchmarks.compare run --output $Pilot --approved-call-limit 7716
if ($LASTEXITCODE -ne 0) { throw 'pilot中断，保留目录和费用，禁止删锁续跑' }
python -m ra_lsbo.experiments.mathematical_benchmarks.compare audit --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot audit失败，禁止最终报告' }
python -m ra_lsbo.experiments.mathematical_benchmarks.compare report --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot report失败' }
```

从prepare至报告完成不修改登记文件或环境，不对同一目录并发启动，不按中途排名改参数。无参数运行compare现在只prepare，不能当作已经开始优化。确认实验confirm暂不通过外部comparison入口运行。

## 7. 本地交付回执

记录：实际提交和差异、解释器/依赖、旧进程停止与未结算调用、回归pass/fail/skip、smoke账本及审计、pilot计划哈希/目录/预算、实际调用和未知失败状态。尚未完整结束只能称已登记或运行中，不能称完成或有统计优势。全部失败种子保留，条件目标统计必须带可行分母。
