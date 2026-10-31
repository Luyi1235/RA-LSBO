# 本机交接：2,000构型持久表示五组闭环候选

## 当前交付性质

本包是**可直接应用的Git补丁**，不是已上传GitHub的固定算法提交。远端基础为：
`analysis/structure-corpus-review-20261031` / `725bd0e8131f977c912b55e2fe95e87760864d3c`。
本地模型只应用补丁，不重新编写算法。新提交SHA必须由实际git commit产生并回传，不能沿用基础SHA冒称新代码。

已在本会话通过91项主测试及5个子测试，但本环境缺少完整运行依赖，未完成整套集成测试及真实smoke。
不要将此前78/327项、其他版本的CI通过记录当成本版验收。本机完整回归及新smoke均是硬门槛。
详细范围见同目录`AIRCRAFT_CORPUS_VALIDATION_20261031.md`。

## 数据与预算

五组固定为random、old30、old480、corpus_persistent、onehot。任务100/140km不变。
smoke种子9930、6共同初值/任务、每方法2次在线，最多42次（12+20+10）。
pilot种子9931–9938、12共同初值/任务、每方法30次在线，最多2672次（192+2400+80），80trial。
本机合计上限2714次简化飞机评价，新增RCAIDE=0。无可行最优点时不补造5秒检查，不转移预算。
模型运行时保留全部12个分支，不再引入自适应专家或尺度扫描。

输入为之前完成的**完整qualification目录**（316次、执行b8a94c3、结果59643c3d），而非摘要目录。
不重跑旧352/3718次，不需要重新执行VAE离线full，也不把资格面板或旧pilot标签加到新优化历史。
新的初始样本前瞻生成；不得按可行性筛种子。

使用原Windows/Rcaide_vsp，依赖不安装不升级。输入旧记录、计划、锁、来源快照全部只读保留。
输出在新目录；代码应用/提交完成后才prepare新计划。运行中不得改代码、commit或依赖。

## 1. 应用与提交补丁（PowerShell）

从新交接包读取`DELIVERY_MANIFEST.json`，核对补丁SHA256；不要使用同名的旧交付包。
填入的路径必须来自实际机器；`$Qualification`是已有完整资格目录。

```powershell
$ErrorActionPreference = 'Stop'
$Repo = '<已有RA-LSBO仓库路径>'
$Worktree = '<尚不存在的新worktree路径>'
$Delivery = '<本次交接包解压根目录>'
$Out = '<新的实验输出根目录>'
$Py = '<原Rcaide_vsp的python.exe>'
$Qualification = '<原完整qualification目录>'
$Base = '725bd0e8131f977c912b55e2fe95e87760864d3c'
$Branch = 'research/aircraft-corpus-pilot-20261031-local'
$Patch = Join-Path $Delivery 'aircraft_corpus_closedloop.patch'
$Manifest = Get-Content -Raw (Join-Path $Delivery 'DELIVERY_MANIFEST.json') | ConvertFrom-Json
function CheckExit([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed: $LASTEXITCODE" } }
if (Test-Path $Worktree) { throw '新worktree已存在' }
if (Test-Path $Out) { throw '新输出根目录已存在' }
if (!(Test-Path $Py) -or !(Test-Path $Qualification)) { throw '原环境或完整资格目录不存在' }
$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Patch).Hash.ToLowerInvariant()
if ($Hash -ne $Manifest.patch_sha256) { throw '补丁哈希不一致' }
if ($Manifest.base_commit -ne $Base) { throw '交付基础提交不一致' }

git -C $Repo fetch origin analysis/structure-corpus-review-20261031
CheckExit 'fetch base'
git -C $Repo worktree add --detach $Worktree $Base
CheckExit 'new worktree'
git -C $Worktree switch -c $Branch
CheckExit 'new branch'
git -C $Worktree apply --check $Patch
CheckExit 'patch applicability'
git -C $Worktree apply $Patch
CheckExit 'apply patch'
$Files = @($Manifest.patch_files)
git -C $Worktree add -- $Files
CheckExit 'stage exact patch files'
git -C $Worktree diff --cached --check
CheckExit 'patch whitespace'
git -C $Worktree commit -m 'Add opt-in five-arm aircraft corpus closed-loop candidate with state and budget audits'
CheckExit 'commit patch'
$Commit = (git -C $Worktree rev-parse HEAD).Trim()
CheckExit 'read real new commit'
Set-Location $Worktree
if (git status --porcelain --untracked-files=no) { throw '受跟踪代码不干净' }
New-Item -ItemType Directory -Path $Out | Out-Null
Set-Content -LiteralPath (Join-Path $Out 'execution_commit.txt') -Value $Commit -Encoding utf8
Start-Transcript -Path (Join-Path $Out 'console.log')
```

若git author身份未配置或分支重名，停止并报告；不伪造作者、不覆盖既有分支、不改写历史。

## 2. 原环境完整验收

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONHASHSEED='0'
$env:MPLBACKEND='Agg'
& $Py -m pip check
CheckExit 'dependency check'
& $Py -m pip freeze | Set-Content -Encoding utf8 (Join-Path $Out 'environment.txt')
CheckExit 'environment record'
& $Py -c 'import sys,platform,torch,numpy,scipy,botorch,gpytorch; print(sys.version); print(platform.platform()); print(torch.__version__,numpy.__version__,scipy.__version__,botorch.__version__,gpytorch.__version__); print(torch.__config__.show())' | Set-Content -Encoding utf8 (Join-Path $Out 'platform.txt')
CheckExit 'platform record'
$Tests = @('tests/test_structure_corpus.py','tests/test_structured_encoder.py',
 'tests/test_risk_core_mixed_gp.py','tests/test_risk_core_observations.py',
 'tests/test_aircraft_corpus_protocol.py','tests/test_aircraft_corpus_optimizer.py',
 'tests/test_aircraft_corpus_pilot.py')
$JUnit = Join-Path $Out 'junit.xml'
& $Py -m pytest -q @Tests --junitxml $JUnit
CheckExit 'complete regression'
& $Py tools/verify_aircraft_corpus_tests.py $JUnit
CheckExit 'no failed skipped or missing regression modules'
```

不要安装新包来掩盖环境不一致；缺依赖、测试失败或跳过都停止后续执行，回传日志。
测试含人工观测/合成训练，不能把它们标成真实研究评价。不能删断言或替换测试模块。

## 3. 新smoke，最多42次

```powershell
$Module = 'ra_lsbo.experiments.simplified_aircraft.corpus_pilot'
$Smoke = Join-Path $Out 'smoke'
& $Py -m $Module prepare --profile smoke --output $Smoke
CheckExit 'prepare smoke'
& $Py -m $Module run --output $Smoke --approved-call-limit 42
CheckExit 'run smoke'
& $Py -m $Module audit --output $Smoke
CheckExit 'audit smoke'
& $Py -m $Module report --output $Smoke
CheckExit 'report smoke'
```

smoke不说明性能优势。每个任务必须至少存在一个5秒复核，所有已复核设计均可用可行且数值筛查通过；
否则下一步prepare会拒绝。不得换smoke种子或增评到“通过”。

## 4. 门槛通过后，执行一次新pilot，最多2672次

```powershell
$Pilot = Join-Path $Out 'pilot'
& $Py -m $Module prepare --profile pilot --output $Pilot --qualification $Qualification --smoke $Smoke
CheckExit 'prepare registered pilot and verify prerequisites'
& $Py -m $Module run --output $Pilot --approved-call-limit 2672
CheckExit 'run registered pilot'
& $Py -m $Module audit --output $Pilot
CheckExit 'audit pilot'
& $Py -m $Module report --output $Pilot
CheckExit 'report pilot'
Stop-Transcript
```

任何非零退出码立即停止，保留已有锁、账本和终态未知记录。不能删除锁续跑、免费重试、补足未使用预算。
如果中止，不改代码后接续旧计划，也不擅自再开一次完整pilot。回传完整失败证据再评审。
5秒复核不回灌训练、不用于另选结果。未找到可行设计的运行不排除。

## 5. 上传与回传

完整回归通过后可将**已验证的真实新commit**推到独立分支，不合并main，不强推：

```powershell
git -C $Worktree push -u origin $Branch
CheckExit 'push validated implementation branch'
```

此补丁的CI工作流仅允许手动触发，不会因push自动新增研究调用。本轮无需触发；另行触发会额外执行最多42次，
必须单列成本，不包括在工作机2714上限内。不得把尚未运行的工作流写成CI已通过。

回传：新commit、回归JUnit、环境、完整smoke和pilot目录（包括全部NPZ/JSON检查点、训练事件、
账本、提案、原始观测、时间、来源快照、计划、锁、审计、报告）。压缩包计算SHA256并逐文件清单核对。
结果上传新结果分支/Release；不提交大档案到普通git历史，不夹带原工作机凭据。
旧352/3718目录、旧离线full与原176MB档案无需重复打包。全部完成后停止，不运行MDA或后续实验。

## 研究解释

这是新候选闭环开发试验，不是已经证明更优的最终RA。主对照是onehot和old480，不能只宣传战胜欠训练old30。
持久表示的累计更新和计算费用与old480不严格匹配，需报告而非暗示公平FLOPs。
保留两个任务所有种子、所有失败和数值复核差异，不为结果好看删分支或改任务。
