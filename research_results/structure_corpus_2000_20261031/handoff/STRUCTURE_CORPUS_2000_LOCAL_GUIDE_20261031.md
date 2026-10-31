# 工作机执行指南：2,000条无仿真构型训练与同历史预测验证

日期：2026-10-31。此任务不是再做旧表示复现，也不是重新运行飞机pilot。

## 本轮范围

使用2,000条随机生成、允许重复的合法构型，与当前已评估设计的结构一起训练VAE。
结构样本不提供目标值或可行标签，不调用飞机评估器或RCAIDE。
五组为 old30、old480、corpus_reset、corpus_persistent、onehot。
初始联合训练30轮；增量组每8条新增在线观测更新20轮，保留网络及私有随机状态。
这是原LSBO训练思路的明确适配，未宣称还原原X-57的抽样种子/分布或所有参数。
新模块未替换任何旧pilot默认优化器；原论文文字和历史结果不改。

固定代码版本：`08eed076f54fc6ed155f285a9bb695ff6d857bf8`。
分支：`research/structure-corpus-2000-20261031`。PR #11，不需要等合并。
不要只pull main，不使用移动HEAD，不让本地模型重新实现补丁。

唯一输入为原完整飞机pilot ZIP：
`aircraft_nominal_20261030_01_full_records.zip`
SHA-256：`35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd`。
使用现有文件，不必重新上传或重新下载176MB档案。历史审计、计划、原件、锁和快照不能改。
不需要重新执行旧20项表示诊断或旧327项飞机优化器回归。

## 环境与拟合上限

使用原Windows/Rcaide_vsp环境。Python3.9.25、NumPy2.0.2、SciPy1.13.1、PyTorch2.8.0+cpu、
GPyTorch1.11、BoTorch0.10.0及scikit-learn1.6.1应与旧记录核对。
本指南不授权安装、升级、修改原环境。依赖或环境不同则停止报告。

本地先运行78项新结构语料及相关回归（38新测试＋40相关测试），再直接运行一次full离线流程；不要求重复云端smoke。
full读取两个任务×9911–9918全部历史，比较原提案0和29之前的共同历史。
- 编码器fit共160次，其中包括初始化和固定增量调用；不是160次飞机仿真。
- one-hot转换32次；GP头fit320次，每次至多8个核候选，共至多2,560个候选GP拟合。
- 每个历史trial的未来已记录候选共5个，16trial合计80个已有设计用于预测评分。
- 新研究飞机/数学真值/RCAIDE调用全部为0，优化器运行0。
- 单元测试有人工数据训练，以及旧回归的廉价SyntheticProblem测试；这些与研究调用分开记录。

fit次数并不表示模型训练计算为零。累计梯度步数、训练时间和所有失败均保留。
不能添加种子、扩大epoch、去重2,000条样本、重新抽样或接着启动BO/MDA。

## PowerShell步骤

路径由本地模型依据已有仓库、ZIP及原Python解释器填写，不能猜造；新目录必须不存在。
新的输出目录不能位于任何旧实验记录目录内部。

```powershell
$Repo = '<现有RA-LSBO Git仓库>'
$Worktree = '<尚不存在的新worktree>'
$Archive = '<已有aircraft_nominal_20261030_01_full_records.zip>'
$Py = '<原Rcaide_vsp的python.exe>'
$OutputRoot = '<新的诊断输出父目录>'
$Commit = '08eed076f54fc6ed155f285a9bb695ff6d857bf8'
$ExpectedArchive = '35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd'
$ErrorActionPreference = 'Stop'

function CheckExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit $LASTEXITCODE" }
}
if ((Test-Path $Worktree) -or (Test-Path $OutputRoot)) { throw 'New directories must not exist' }
if (!(Test-Path $Archive) -or !(Test-Path $Py)) { throw 'Missing input archive or interpreter' }
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedArchive) {
    throw 'Original archive hash mismatch'
}

git -C $Repo fetch origin research/structure-corpus-2000-20261031
CheckExit 'fetch'
git -C $Repo worktree add --detach $Worktree $Commit
CheckExit 'worktree'
Set-Location $Worktree
$Head = git rev-parse HEAD
CheckExit 'git rev-parse'
if ($Head.Trim() -ne $Commit) { throw 'Wrong pinned commit' }
$Dirty = git status --porcelain --untracked-files=no
CheckExit 'git status'
if ($Dirty) { throw 'Tracked source files changed' }

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
Start-Transcript -LiteralPath (Join-Path $OutputRoot 'console.log')
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONHASHSEED='0'
$env:MPLBACKEND='Agg'

& $Py -m pip check
CheckExit 'pip check'
& $Py -m pip freeze | Set-Content -Encoding utf8 (Join-Path $OutputRoot 'environment.txt')
CheckExit 'pip freeze'
& $Py -c 'import sys,platform,numpy,scipy,torch,gpytorch,botorch,sklearn; print(sys.version); print(platform.platform()); print(numpy.__version__,scipy.__version__,torch.__version__,gpytorch.__version__,botorch.__version__,sklearn.__version__); print(torch.__config__.show())'
CheckExit 'platform record'

$JUnit = Join-Path $OutputRoot 'junit.xml'
& $Py -m pytest -q tests/test_structure_corpus.py tests/test_structured_encoder.py tests/test_risk_core_mixed_gp.py tests/test_risk_core_observations.py --junitxml $JUnit
CheckExit 'structural and affected regression tests'
```

检查JUnit为78项全部通过且无跳过；依赖缺失导致skip不能当作验收通过。测试失败则停，不改断言、不去掉失败测试。
旧编码器回归的仓库外基线路径已由本提交改为仓库内冻结基线，无需手工创建外部LSBO目录。

```powershell
$Study = Join-Path $OutputRoot 'full'
& $Py tools/study_structure_corpus_2000.py $Archive $Study --profile full --approved-encoder-fits 160 --approved-gp-heads 320
CheckExit 'full offline corpus and forecast study'

& $Py -c 'import sys; from tools.study_structure_corpus_2000 import audit_output; print(audit_output(sys.argv[1]))' $Study
CheckExit 'read-only output audit'
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedArchive) {
    throw 'Archive changed during execution'
}
$Dirty = git status --porcelain --untracked-files=no
CheckExit 'final source status'
if ($Dirty) { throw 'Tracked sources changed during study' }
Stop-Transcript
```

任何非零退出码立即停止后续执行。失败时仍保留输出、execution.lock及fit_attempts.jsonl，不删除锁重试、不继续扩网格。
若控制台操作失败，应先判断是否发生拟合，并披露已发生计算；不得把部分运行说成完整。

## 结果核对与解释

full正常结束应有160个case/seed/prefix/arm结果、320个头结果、160次编码器fit、32次onehot。
记录 old30 的29个非空历史身份；原身份缺失的3项不算失败。若原环境中身份不匹配，停止后续新研究，保留差异。

结构训练样本必须为固定2,000条 + 当次已评价设计结构。初始化2012行/16批/30轮=480次梯度更新。
后缀reset2041行仍16批/轮，持久组按0/8/16/24更新；29处编码器前缀36条，GP前缀41条，这是声明的周期更新。

`raw_mean_active_exact`是编码均值下原始分类头完整重构率；`raw_sampled_active_exact`是32次后验采样结果。
不要把规则修复后的合法率混入。`mean_roundtrip_target_met`是训练语法目标，不是留出泛化或自动BO授权。
old480匹配corpus_reset的梯度步数，但批大小/MMD估计及每步工作量不同，不是墙钟/FLOPs严格匹配。
persistent最终累计1440次更新，不与old480匹配；不能把其差异单独解释成“增量学习必然更好”。

`feature_fit_seconds`对持久组只计快照提取，不含已完成的历次训练。
真实持久训练费用在对应`persistent_checkpoint/metadata.json`的`fit_seconds`列表中，按每条训练事件分别报告。
跨前缀列表存在累计重叠，汇总时每条fit只算一次，不能既漏计又双计。reset亦有检查点计时。

预测分数按task/head/observation kind分开：精确值NLL是密度，区间NLL是概率质量，不能直接混为总体排名。
同一历史原算法选出的设计不是新算法闭环轨迹；80个已有设计也不是80个独立优化重复。
结构重构改善不能替代预测收益，预测改善不能替代BO收益。已有历史被用于开发，不称独立确认。

## 回传与停止

保留并打包完整新OutputRoot，包括plan、audit、rows、scores、summary、fit_attempts、checkpoint、环境、JUnit和日志。
使用新的results分支/Release，给出完整提交、档案SHA-256和文件清单；无需再次上传旧176MB输入ZIP。
不可只上传平均重构率，必须保留全部前缀、全部方法、失败与预测分数。
执行完成即停止；不启动nominal_pilot、数学比较、transfer或RCAIDE。
本轮只完成2,000构型训练修订的离线验证，不授权新的真实评估。
