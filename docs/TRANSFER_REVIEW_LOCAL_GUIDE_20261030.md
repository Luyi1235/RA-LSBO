# 给本地大模型：Transfer复核与暂停扩跑说明

## 当前决定

不要启动新pilot、表示消融、MDA，也不要重跑刚结束的3,108次实验。自适应组合的480个送评候选与等权组合完全相同，当前没有新增选点价值证据。保留原实现和负面结果；本轮只运行可选的零评估诊断，不重新实现补丁，不改权重温度/下限或覆盖周期。

本轮新增GitHub代码是只读诊断工具，不是新优化器版本：
- 分支：`research/transfer-review-20261030`
- 固定诊断提交：`8c1ee02131124eb82be62acf261a7f97830e8b37`
- 原实验算法：`21505bba5f9485287a3e34f4ce50dc849edf4e04`
- 原结果提交：`497f810260fe8828189974f540ff1e5c5000db7e`

## 1. 原件保护与目录选择

保留原工作区、环境、计划、锁、日志和源码快照。不要在原实验目录切换分支、执行git clean/reset或重写换行。新建独立worktree。历史结果目录仍使用原件，只读。

本地模型应根据实际工作区定位仓库和已经存在的完整目录：
`runs/transfer_pilot_20261030_141720_80f5ce0d/`。
它必须含`calls.jsonl`、全部`optimization/`、`source_snapshot/`及封存计划/审计/结果，不能用`research_results/`下的汇总目录代替。不要猜测盘符或从汇总重建缺失日志。

若使用归档，先检查原ZIP：
`90d13623d1175374d442b19c431493393617016a1267baa65db2cc2203f2fca1`。
再解压到新的目录，不覆盖原件。工具自己还会检查固定头文件、390项证据及170项快照；哈希不一致必须停止，不能更新预期值使其通过。

## 2. 同步固定诊断提交

以下PowerShell从已存在的RA-LSBO仓库目录运行；每个原生命令后都必须核对退出码。

```powershell
$ErrorActionPreference = 'Stop'
$Repo = (git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw '当前目录不是已识别的仓库' }
$Sha = '8c1ee02131124eb82be62acf261a7f97830e8b37'
git -C $Repo fetch origin research/transfer-review-20261030
if ($LASTEXITCODE -ne 0) { throw 'fetch失败；不继续' }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Work = Join-Path (Split-Path $Repo -Parent) ('RA-LSBO-transfer-review-' + $Stamp)
if (Test-Path $Work) { throw '新worktree路径已存在' }
git -C $Repo worktree add --detach $Work $Sha
if ($LASTEXITCODE -ne 0) { throw 'worktree创建失败' }
$Actual = (git -C $Work rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $Actual -ne $Sha) { throw '固定提交不匹配' }
Set-Location $Work
```

不用`git pull main`代替同步；无需等待PR合并。不要修改任何正在运行的其他工作区。

## 3. 本机纯数值测试与只读复核

使用已经核验的Python解释器；不要为此更新原RCAIDE/MDA环境。诊断仅依赖numpy、scipy、pytest。CI版本为numpy2.0.2/scipy1.13.1/pytest8.3.5、Python3.11。缺依赖先停止并报告，可使用单独诊断环境，不要未经确认改动物理环境。

```powershell
python -c "import sys,numpy,scipy,pytest; print(sys.executable); print(sys.version); print(numpy.__version__,scipy.__version__,pytest.__version__)"
if ($LASTEXITCODE -ne 0) { throw '诊断依赖不可用' }
python -m pytest -q tests/test_transfer_decision_effect.py
if ($LASTEXITCODE -ne 0) { throw '诊断回归失败；不继续' }
```

33项测试应通过且不跳过。使用实际完整旧目录填写`$Pilot`，不得把下述占位文字直接执行。输出选在旧目录之外、全新路径，例如新worktree的`runs/review_transfer_<唯一标识>`。

```powershell
$Pilot = '替换成已核验的完整旧transfer pilot绝对路径'
if (-not (Test-Path (Join-Path $Pilot 'calls.jsonl'))) { throw '缺完整调用账本' }
$Out = Join-Path $Work ('runs/review_transfer_' + $Stamp)
if (Test-Path $Out) { throw '输出目录已存在' }
python tools/review_transfer_pilot.py $Pilot $Out
if ($LASTEXITCODE -ne 0) { throw '只读复核失败；保留错误，不改历史哈希' }
```

这是旧数据的独立关联与保存预测算术复核，不是在新代码/路径下重执行旧严格环境审计，也不运行旧源码快照。它不会调用评估器，不会重新训练模型。

## 4. 应回传的产物和边界

回传固定提交、解释器/依赖版本、33项测试结果、派生输出目录中的`facts.json`、`groups.csv`、`paired_outcomes.csv`、`weights.csv`、`decision_effects.csv`及终端日志。大型原始ZIP不必再次上传。

重点核对：3,108已结算、96trial、2,880提案、390证据、170来源；等权/自适应480个候选相同，103个冷阶段中模型首选变3次、实际变0次；新数学调用0、MDA0。浮点库不同可能导致末位权重不同，不重写原件。

诊断结束后停止。后续优先准备真正有分支权衡、合法有限区间与匹配对照的机制任务方案，再登记预算。当前未实现新任务/官方CBOB复现，也未授权新完整实验。不要为了让自适应模块改变选点而直接移除已有覆盖保护。
