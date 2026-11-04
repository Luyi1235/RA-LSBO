# PR55：真实 GP/VAE 在线闭环的风险预算独立验证

固定实验代码：`d947db5f0b544739c9db286a75d99c0308713fdb`  
父级 PR54 代码：`fd5492d33b21421ff8f54ad8bba453543caf39fe`  
父级 PR54 结果分支：`41db5094f29645cf5afd17b08e65ff62544d3f41`  
环境：`Rcaide_vsp` / Python 3.9.25。

本分支只上传独立实验产物；没有修改 PR45–PR54 的代码或历史证据。

## 运行范围与身份

- 四个解析任务：Branin ordinary、Branin sparse、Rosenbrock ordinary、Rosenbrock sparse。
- 每任务 16 个独立种子，`996101–996116`，与 PR54 种子块不重叠。
- 三个算法：`legacy`、`semantic_only`、`budgeted`。
- 12 个共享初始评价、最多 12 次在线评价、最多 6 次已观察不可行停止。
- 全量上限 3,072 次新解析评价；实际 **2,838** 次，192 个 trial。
- 实际 GP/VAE 拟合与真实解析函数反馈；**飞机 0 次、RCAIDE 0 次**。
- 不重放 PR45–PR54 历史 truth；`actual_on_policy_feedback=true`。
- 192/192 条路径均遵守 6 次不可行停止；`all_paths_respect_observed_stop=true`。
- 13 项依赖测试通过（0.11 s）。

Smoke：72 次评价，2 trials/task-arm，`COMPLETE_INDEPENDENT_ON_POLICY_ANALYTICAL_VALIDATION`，exit 0。  
Full：2,838 次评价，192 trials，`COMPLETE_INDEPENDENT_ON_POLICY_ANALYTICAL_VALIDATION`，exit 0；无 traceback/error。

## 最终可行目标（条件于最终存在可行解的种子）

| 任务 | 算法 | 最终可行/16 | 在线调用均值 | 不可行均值 | 触发 6 次停止/16 | 最优可行中位数 | 最优可行均值 |
|---|---|---:|---:|---:|---:|---:|---:|
| Branin ordinary | legacy | 16 | 12.00 | 1.94 | 0 | 1.9564 | 2.010 |
| Branin ordinary | semantic_only | 16 | 12.00 | 2.06 | 0 | 1.8801 | 2.178 |
| Branin ordinary | budgeted | 16 | 12.00 | 2.06 | 0 | 1.8801 | 2.553 |
| Branin sparse | legacy | 12 | 9.06 | 5.50 | 12 | 6.5682 | 8.561 |
| Branin sparse | semantic_only | 12 | 8.62 | 5.50 | 13 | 6.5682 | 8.714 |
| Branin sparse | budgeted | 12 | 10.19 | 5.25 | 9 | **2.1913** | **2.434** |
| Rosenbrock ordinary | legacy | 16 | 12.00 | 2.50 | 0 | 3.4048 | 5.824 |
| Rosenbrock ordinary | semantic_only | 16 | 12.00 | 2.75 | 0 | 3.4048 | 6.077 |
| Rosenbrock ordinary | budgeted | 16 | 12.00 | 2.69 | 0 | 3.9682 | 6.282 |
| Rosenbrock sparse | legacy | 13 | 9.75 | 5.31 | 11 | **6.5759** | **15.099** |
| Rosenbrock sparse | semantic_only | 13 | 9.50 | 5.25 | 10 | 8.4242 | 19.805 |
| Rosenbrock sparse | budgeted | 13 | 10.25 | 5.00 | 7 | 8.4242 | 19.436 |

所有任务三种算法的最终可行计数完全相同；没有任何 `candidate-only` 或 `control-only` 可行发现。

## 配对差（candidate − control；仅双方都有最终可行的种子）

| 任务 | 对比 | n | 改善/恶化/持平 | 均值 Δ | 中位数 Δ | Δ调用 | Δ不可行 |
|---|---|---:|---:|---:|---:|---:|---:|
| Branin ordinary | budgeted − semantic | 16 | 0/1/15 | +0.3752 | 0 | 0 | 0 |
| Branin ordinary | semantic − legacy | 16 | 2/3/11 | +0.1675 | 0 | 0 | +0.125 |
| Branin ordinary | budgeted − legacy | 16 | 2/3/11 | +0.5428 | 0 | 0 | +0.125 |
| Branin sparse | budgeted − semantic | 12 | 6/2/4 | **−6.2801** | **−2.5248** | +1.562 | −0.250 |
| Branin sparse | semantic − legacy | 12 | 4/1/7 | +0.1534 | 0 | −0.438 | 0 |
| Branin sparse | budgeted − legacy | 12 | 7/2/3 | −6.1267 | −2.6210 | +1.125 | −0.250 |
| Rosenbrock ordinary | budgeted − semantic | 16 | 0/1/15 | +0.2050 | 0 | 0 | −0.062 |
| Rosenbrock ordinary | semantic − legacy | 16 | 1/3/12 | +0.2529 | 0 | 0 | +0.250 |
| Rosenbrock ordinary | budgeted − legacy | 16 | 1/4/11 | +0.4579 | 0 | 0 | +0.188 |
| Rosenbrock sparse | budgeted − semantic | 13 | 2/0/11 | −0.3695 | 0 | +0.750 | −0.250 |
| Rosenbrock sparse | semantic − legacy | 13 | 2/5/6 | **+4.7062** | 0 | −0.250 | −0.062 |
| Rosenbrock sparse | budgeted − legacy | 13 | 3/4/6 | +4.3367 | 0 | +0.500 | −0.312 |

Δ 为正表示 candidate 目标更差；Δ调用/Δ不可行为 candidate 减 control。

## 风险预算回退与预测风险

| 任务 | 算法 | receipt | 有预测 | 回退 | 回退中真实失败 | 预测风险均值（有预测） | 实际失败率（有预测） |
|---|---|---:|---:|---:|---:|---:|---:|
| Branin sparse | semantic_only | 138 | 123 | 0 | — | 0.3822 | 0.5935 |
| Branin sparse | budgeted | 163 | 148 | 17 | 12/17 (70.6%) | 0.3098 | 0.4662 |
| Rosenbrock sparse | semantic_only | 152 | 137 | 0 | — | 0.3314 | 0.5182 |
| Rosenbrock sparse | budgeted | 164 | 149 | 22 | 16/22 (72.7%) | 0.3115 | 0.4497 |

Ordinary 任务：预算从未回退（0 fallback）；Branin ordinary 预测均值 0.1581、实际 0.1719；Rosenbrock ordinary 0.1685、实际 0.2240。

预算确实把稀疏任务的选点移向较低预测风险，实际失败率也低于 semantic-only；但预测仍然系统性乐观，且 17/22 次回退里 12/16 次真实失败。**回退不是安全网。**

## 解发现

| 任务 | 初始无可行种子的冷启动组 | 各算法首次发现 |
|---|---:|---|
| Branin ordinary | 0 | — |
| Branin sparse | 4 | 三臂均为 0/4 |
| Rosenbrock ordinary | 0 | — |
| Rosenbrock sparse | 5 | 三臂均在种子 996107 第 5 步、996115 第 4 步发现；完全相同 |

因此本轮没有任何“预算/语义修正改善冷启动发现”的证据；Branin sparse 的 4 个冷启动种子在三臂中都没有找到可行解。

## 解释与限制

1. **普通任务上预算几乎不改变行为。**  
   Branin ordinary 只有 1/16 路径改变，Rosenbrock ordinary 3/16；15/16 最终目标持平，各 1 个种子变差。均值分别 +0.375 和 +0.205（正为更差）。这与 PR54 的“普通任务无收益”一致。

2. **Branin sparse 出现强但不等工作的目标改善。**  
   最终可行的 12 个种子中，budgeted vs semantic 为 6 改善、2 恶化、4 持平；中位数从 6.568 降到 2.191（−4.38），均值从 8.714 降到 2.434。  
   但 budgeted 平均多 1.56 次在线调用、不可行少 0.25 次，且 6 次停止触发从 13/16 降到 9/16。改善至少部分来自“少触发停止、继续搜索”，不是同等调用预算下的纯效率提升。

3. **Rosenbrock sparse 的预算收益很小，且 legacy 仍是该任务最强。**  
   budgeted vs semantic 为 2 改善、0 恶化、11 持平；均值仅 −0.370，中位数不变；10/16 路径相同。  
   但 budgeted vs legacy 均值 +4.337（更差），4 个恶化、3 个改善、6 个持平；legacy 的中位数 6.576，明显优于 budgeted/semantic 的 8.424。  
   这说明语义修正本身在该任务上损害目标，预算只能部分恢复，不能恢复至 legacy。

4. **semantic-only 不是“免费修正”。**  
   它改变了几乎所有路径（16/16、16/16、16/16、14/16），但最终目标没有一致改善；在 Rosenbrock sparse 上相对 legacy 均值 +4.706，5 个恶化、2 个改善、6 个持平。语义门控改变了搜索轨迹，但并未证明更优。

5. **预算减少失败次数，但不是安全保证。**  
   Branin sparse 总失败 84 vs 88，Rosenbrock sparse 80 vs 84；但调用数同时增加，且预测风险仍低于实际。回退点的失败率约 70%。不能把 budgeted 解释为满足失败预算或安全约束的策略。

6. **没有发现收益，最终可行计数不变。**  
   四种任务的最终可行计数在三臂间完全相同；Branin sparse 的 4 个冷启动组全部未发现；Rosenbrock sparse 的 2 个发现步数在三臂间相同。本实验支持“预算影响已可行种子的后续目标”，不支持“预算改善可行性发现”。

7. **统计为描述性。**  
   每任务 16 个种子，稀疏任务最终可行对只有 12/13 个；没有做多重比较校正或确认性检验。均值差和配对计数应作为效应量描述，而非显著性结论。

## 报告汇总字段异常（不影响执行）

`method_summary_by_case.*.predicted_risk_receipt_coverage` 与 `predicted_allowance_gate_fallbacks` 在 budgeted 稀疏任务中显示 0，但原始 `budget_receipts` 明确记录 17/22 次回退。原因是汇总字段检查 `forecast_available`，而 runner 写入的是 `prediction_available_for_selected`。  
这是**报告聚合字段的缺陷**，不改变选点或真实结果；本 README 的回退统计直接取原始 receipts。完整报告与 `analysis.txt` 均保留，工作机未修改代码。

## 文件

- `risk_budget55_report.json`：2,838 次评价、192 trials、配对、预算 receipt 与诊断。
- `plan.json`、`source_receipt.json`、`progress.jsonl`：计划、源码身份与逐 trial 进度。
- `trials.zip`：192 个 trial 的完整 `proposals.jsonl`、`observations.jsonl`、`result.json` 等。
- `shared_initial.zip`：四个任务的共享初始评价证据。
- `smoke_report.json`、`smoke_plan.json`、`smoke_source_receipt.json`、`smoke_progress.jsonl`、`smoke-console.log`：smoke 证据。
- `full-console.log`、`tests.log`、`analysis.txt`：完整日志、测试日志与分析脚本输出。
- `fixed_sha.txt`、`fixed_sha256.txt`：固定 SHA、源码身份与产物 SHA256。
- `README.md`、`.gitattributes`：本说明与字节保真设置。
