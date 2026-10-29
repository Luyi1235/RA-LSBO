# RA-LSBO 分阶段实验：工作机执行说明

日期：2026-10-29。仓库：Luyi1235/RA-LSBO。
分支：`research/discovery-stages-20261029`；PR： https://github.com/Luyi1235/RA-LSBO/pull/3
**固定运行提交：`cfdf7c30216585b3920f23723236708859435ce7`。**
本文是后续文档提交，不应使用移动的最新HEAD替代固定运行版本。

## 给本地大模型的任务

直接同步已提交代码，不重新实现补丁。使用独立worktree，保护旧计划、账本、锁、源码快照和原始ZIP。核查环境并执行影响范围回归；通过后先运行discovery smoke（104次），再运行discovery pilot（6,516次），完成审计与报告后停止并回传结果。首轮本机研究上限6,620次，测试内部人工/廉价评价另计。不自动运行其余完整pilot、确认实验或RCAIDE MDA，不改种子、不剔除失败、不免费重试未知调用。

最终CI run40819328407实际通过229项测试及三套数学smoke，详见`STAGED_VALIDATION_RECEIPT_20261029.md`。这不是Windows本机验证，也不证明新策略性能优势。

## 阶段、预算与限制

入口：`python -m ra_lsbo.experiments.discovery_stages.compare`。
Schema：`ra_lsbo_discovery_stages_v1`。旧comparison/core入口和算法源码不变，不能复用旧计划。

| stage | 组数 | pilot上限 | smoke上限 | 当前动作 |
|---|---:|---:|---:|---|
| discovery | 5 | 6,516 | 104 | 先执行这一套 |
| representation | 8 | 10,116 | 128 | 第一阶段评审后另行授权 |
| baselines | 8 | 10,116 | 128 | 匹配消融评审后另行授权 |

Pilot仍用两个原数学稀疏算例、9701–9720、12共同初值、30在线、128候选、每题16面板点。第一阶段200trial、6,000在线，加4锚点、32面板、480初值，总6,516；不要沿用旧六组的240trial/7,716计数。开发种子已经看过，不是独立确认。

发现阶段：Random、原RA-v2、仅冷启动禁用局部增广、仅覆盖/近重复保护、组合策略。禁用范围包括局部随机候选和SLSQP，首次得到可用可行解后恢复。保护组每5轮安排绕过代理门控的分支覆盖抽样，最终合格候选退化为单点时另行覆盖回退；同构型活跃归一化连续维度的近重复阈值1e-8。原边界探索槽仍保留，新增20%覆盖槽不代表总探索比例仅20%，更不是风险预算或安全保证。原design_id、观测和物理约束不变。容差不是物理等价判据，不直接推广到有噪声的主动重复测量。

表示阶段：latent/onehot × exact/sign/interval六组，加Random和random_then_local。仅改变模型训练视图，原始记录和已知可行incumbent保留。当前数学区间主要是符号，sign/interval可能相同，不能证明一般有限区间的增量。

基线阶段包括冻结LSBO、历史保守SEGO（sego_conservative_v2）和符号修正UTB（sego_utb_v3）；各适配器不冒充官方实现。冻结LSBO保留连续5次无目标的原生停止规则，未使用额度不重分配，停止轨迹保留到名义预算；它不是搜索工作量匹配的消融。

CBOB正式集成、全新分支权衡/有限区间任务、独立确认性统计和真实飞机MDA仍需另立方案，当前入口不提供confirm/MDA开关。后两套pilot必须传入前一套完整pilot的审计来源；这只是来源门槛，不是科学晋级批准。

6,516是真实数学评价上限，不是GP拟合次数，也不保证墙钟少于旧实验。四组RA相关代理比原比较中一组RA带来更多拟合；smoke的训练轮数和核网格较小，不按smoke耗时线性外推pilot。记录optimization_seconds，确认磁盘空间，不删旧证据。预期产物是完整记录和发现/开发分层诊断，而不是预设取胜。

## 1. 独立worktree同步

在已确认的旧仓库根目录使用PowerShell。禁止reset --hard、git clean、覆盖本地修改或在旧运行目录切分支。

```powershell
$ErrorActionPreference = 'Stop'
$CodeCommit = 'cfdf7c30216585b3920f23723236708859435ce7'
$Old = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw '请进入正确Git仓库' }
$Old = $Old.Trim()
$Origin = git remote get-url origin
if ($LASTEXITCODE -ne 0 -or $Origin -notmatch 'github\.com[:/]Luyi1235/RA-LSBO(\.git)?$') { throw 'origin不是目标仓库' }
git status --short
git fetch origin research/discovery-stages-20261029
if ($LASTEXITCODE -ne 0) { throw 'fetch失败' }
git cat-file -e "${CodeCommit}^{commit}"
if ($LASTEXITCODE -ne 0) { throw '固定提交不存在' }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$New = Join-Path (Split-Path $Old -Parent) ('RA-LSBO-staged-' + $Stamp)
if (Test-Path $New) { throw '目标目录已存在' }
git worktree add --detach $New $CodeCommit
if ($LASTEXITCODE -ne 0) { throw 'worktree创建失败' }
Set-Location $New
if ((git rev-parse HEAD).Trim() -ne $CodeCommit) { throw '提交身份错误' }
if (git status --porcelain) { throw '新工作区有未提交修改，先核查' }
```

不要只pull main或旧修复分支。没有需停止的旧实验时，不操作其他Python/MDA进程。

## 2. 环境和回归

先核查现有环境；缺依赖或冲突时停止，必要调整优先新建/克隆隔离环境，不擅自升级物理工作环境。

```powershell
conda activate Rcaide_vsp
if ($LASTEXITCODE -ne 0) { throw '环境激活失败' }
python -c "import sys; print(sys.executable); print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw 'Python检查失败' }
python -c "import importlib.metadata as m; print({n:m.version(n) for n in ['numpy','scipy','torch','gpytorch','botorch','scikit-learn','matplotlib','pytest']})"
if ($LASTEXITCODE -ne 0) { throw '缺少依赖' }
python -m pip check
if ($LASTEXITCODE -ne 0) { throw '依赖冲突' }
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$env:PYTHONHASHSEED = '0'
$Validation = Join-Path 'runs' ('staged_local_validation_' + $Stamp)
New-Item -ItemType Directory -Path $Validation | Out-Null
python -m pip freeze | Out-File (Join-Path $Validation 'environment.txt') -Encoding utf8
Start-Transcript -Path (Join-Path $Validation 'console.txt')
python -m pytest -q tests/test_discovery_stages.py tests/test_discovery_stages_evidence.py tests/test_numerical_safeguards.py tests/test_external_comparison.py tests/test_comparison_reliability.py tests/test_comparison_entrypoints.py tests/test_comparison_visualization.py tests/test_paper_experiments.py tests/test_grouped_partial_gp.py --junitxml="$Validation/pytest.xml"
if ($LASTEXITCODE -ne 0) { throw '回归失败，不启动smoke/pilot' }
```

主依赖沿用requirements-models.txt：numpy2.0.2/scipy1.13.1/torch2.8.0/gpytorch1.11/botorch0.10.0/sklearn1.6.1/matplotlib3.9.4；CI pytest8.3.5、Linux/Python3.11.17。最终CI229通过、0跳过；本机计数不同需查明，不能把缺依赖skip当通过。

## 3. 全新discovery smoke：104次

```powershell
$Smoke = Join-Path 'runs' ('discovery_smoke_' + $Stamp)
python -m ra_lsbo.experiments.discovery_stages.compare prepare --stage discovery --profile smoke --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke prepare失败' }
python -m ra_lsbo.experiments.discovery_stages.compare run --output $Smoke --approved-call-limit 104
if ($LASTEXITCODE -ne 0) { throw 'smoke run失败，保留锁/账本，不自动重试' }
python -m ra_lsbo.experiments.discovery_stages.compare audit --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke audit失败' }
python -m ra_lsbo.experiments.discovery_stages.compare report --output $Smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke report失败' }
```

应为4数学算例、1seed、5方法、每方法2步：20trial、40在线、104总调用。只用于流程，不可作为后续pilot的完整前置证据。

## 4. 新discovery pilot：6,516次

```powershell
$Pilot = Join-Path 'runs' ('discovery_pilot_' + $Stamp)
python -m ra_lsbo.experiments.discovery_stages.compare prepare --stage discovery --profile pilot --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot prepare失败' }
Get-Content (Join-Path $Pilot 'registration.json')
# 只读核对plan的stage、5方法、20seed、2算例、12初值、30在线、6516上限。
python -m ra_lsbo.experiments.discovery_stages.compare run --output $Pilot --approved-call-limit 6516
if ($LASTEXITCODE -ne 0) { throw 'pilot中断，保留全部证据，不删锁续跑' }
python -m ra_lsbo.experiments.discovery_stages.compare audit --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot审计失败，不生成最终报告' }
python -m ra_lsbo.experiments.discovery_stages.compare report --output $Pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot报告失败' }
Stop-Transcript
```

prepare至report结束不修改源码、协议、依赖、提交、源文件换行。不并发使用一个结果目录。程序不支持中断续跑；未结算started保留未知，不补造settled或删费用。需要代码修订必须新版本、新计划。

## 5. 回传结果与后续门槛

保留plan/registration/source_snapshot/calls/lock/qualification、所有optimization的manifest/proposals/observations/summary、results/audit/report/summary.csv及环境日志。按原字节压缩，不从汇总重建日志。大ZIP优先Release附件/独立存档，仓库保留小报告和SHA256清单。

按任务报告冷暖分母、冷发现率、首次可行删失、暖开发、全样本成对状态和条件目标分母。`decision_diagnostics`由逐步记录重算近重复、覆盖、单候选回退、局部选择、SLSQP告警。不同旧适配器未记录的字段不等于从未发生对应行为。

保护组应没有阈值内近重复送评，覆盖槽不受代理门控阻断；软件规则通过不等于性能提升。新组合若未改善发现，或损害暖开发，先评审而非晋级。

后续representation/baselines分别需要本版前一阶段已审计的完整pilot作为`--prerequisite`，并另行核准10,116次上限。本机先做对应128次smoke。形式：

```text
python -m ra_lsbo.experiments.discovery_stages.compare prepare --stage representation --profile pilot --prerequisite <已审计discovery-pilot目录> --output <新目录>
python -m ra_lsbo.experiments.discovery_stages.compare run --output <同一新目录> --approved-call-limit 10116
```

继续audit/report，不能把示例尖括号当实际路径。前置审计要求同一冻结checkout和环境；源码变动后不能改旧哈希强行通过，应先版本迁移评审。当前数学任务不足以证明一般有限区间和复杂分支权衡效果。
