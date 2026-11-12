# PR86-E：既有轨迹只读机制诊断

## 固定输入与范围

- 原实验代码：`6e5cbd0bafc25cf4e36f9a5675401d6991165416`。
- 结果提交：`311ef606d173d6fe0bb3663ca18f9560c1ca5586`。
- 输入目录：该结果 worktree 的 `results/pr86d-full-20261111-6e5cbd0/full/`。
- 10 个已用数学任务、30 个 task/repeat 单元、300 条轨迹、9000 个在线选点。
- 本轮不是新优化算法，也不是新 Full；不调用数学评价器、不拟合 GP、不改变采集权重或历史数据。
- 24 点为后续主要开发设置，12 点仍用于冷启动对照；本诊断同时读取两组，不选择性排除不利结果。

## 三个待检验问题（尚无诊断结论）

1. **同状态选点分歧**：在每个已存档的预测状态上，分别重算冻结的 CEI 和 0.65/0.35 排名加权规则，记录候选身份、预测可行概率和 CEI 代理量变化。按来源策略、任务、初始样本量和冷/暖状态分组，不能把不同历史状态的策略输出直接当作单因素对照。
2. **事前预测误差**：只用该次实际选中候选的观测反馈，计算目标残差和约束 Brier。目标缺失时不计为数值标签；region 满足而 resource 不满足时，精确目标可以计入模型误差，但不能成为可行最优值。所有误差仅为 on-policy、目标可观测条件下的诊断，不是独立测试泛化性能。
3. **平局身份**：以已观测精确可行目标为准，区分同一已评价候选、不同候选相同目标值、双方均无可行解。进一步记录共享的最优候选是否来自共同初始点，不能仅从目标值持平推断算法行为相同。

## 观测与证据边界

- 诊断只消费 `observed` 反馈包、候选特征、历史前缀、选点前预测及分数；不读取 `truth` 字段、held-out 标签或未选候选的真实目标。
- 未选中的替代候选只有模型预测，没有反事实结果；`cei_proxy_fraction_lost` 只是相对 CEI 代理量变化，不是实际 regret，也不是多步信息价值。
- `no_objective_gp`、`no_feasible_incumbent`、`objective_ready` 分开报告。冷状态中 EI=1 是冻结规则的回退值，不解释成目标改进。
- 标准化残差使用已存的目标 GP 潜在方差，不冒充包含观测噪声的预测区间覆盖率。
- 9000 个选点存在轨迹内依赖，不是9000次独立重复；按任务报告目标误差，不跨任务混合原始目标量纲。
- 文件按 unit 顺序读取；不需要把约2GB记录整体读入内存，也不读取740MB的 `progress.jsonl`。对实际消费的 plan/report/source/unit 文件逐一核对原始 manifest 哈希；这不是重新核验全部 journal/LFS 的声明。

## 程序与测试

`pr86e_diagnostics.py` 仅依赖 Python 标准库，独立复现冻结分数公式。测试包括：合成9000决策端到端流程、输入哈希与选点身份篡改拦截、冷启动和无 incumbent 区分、隐藏 truth/test 标签不可用时结果不变、共享初始最优候选识别，以及与仓库实际 `pr86c_acquisition.py` 的数值一致性对照。

CI 只运行这些纯代码和合成记录测试，不安装 PyTorch，不重跑旧数学实验。合成样本不作为科研结果。

## 工作机执行（代码测试通过后；不运行 Full）

准备两个独立、干净的 worktree：

- 诊断代码分支：`research/pr86e-readonly-diagnostics-20261112`，检出 PR 公布的精确 analyzer SHA。
- 已有结果分支：固定在上述 `311ef606...` 结果提交。直接复用本地已完整保存的 unit 文件，无须重新计算或上传原始2GB。

在诊断代码根目录运行（替换路径和公布的 analyzer SHA）：

```powershell
$Sha = "<PR公布的诊断代码完整SHA>"
$InputFull = "<冻结结果worktree>\results\pr86d-full-20261111-6e5cbd0\full"
$Output = "D:\LSBO\pr86e-readonly-diagnostics-20261112"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -S pr86e_diagnostics.py --input $InputFull --output $Output --expected-analyzer-sha $Sha
if ($LASTEXITCODE -ne 0) { throw "只读诊断失败，保留日志，不修改原始记录" }
```

输出目录必须全新且位于两个 worktree 之外，禁止覆盖。代码会检查 analyzer HEAD、结果 HEAD、干净工作区和原始数据哈希；元数据检查通过不代表9000条真实诊断已在本PR完成。

## 交付

只上传新生成的 `diagnostic_summary.json`、`decision_diagnostics.csv`、`incumbent_identity.json`、`manifest.json`，另附运行日志和简短 README 到独立 `results/pr86e-diagnostics-20261112-<短SHA>` 分支。

不得重传旧原始数据、发 ZIP、修改算法/main/旧结果或调节权重。报告新结果 commit、9000 decisions / 300 trajectories / 240 identity pairs，以及 `new_evaluator_calls=0`、`new_gp_fits=0`。

拿到诊断后再决定一个单因素的算法改动；既有数据上的探索性发现必须用后续预先冻结的新实验验证，不能包装成确认性优势。
