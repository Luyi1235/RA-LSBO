# PR60：分支级 Beta 风险不确定性与信息回退的六臂等成本验证

固定代码：`8675ee94b23bc06779788907dafa3a068c226c29`  
环境：`Rcaide_vsp` / Python 3.9.25。

本分支只上传 PR60 新生成的实验结果；没有读取或修改 PR45–PR59 的飞机/SQLite 历史数据。

## 实验范围与完整性

- 4 个自定义层级数学任务：Branin ordinary/sparse、Rosenbrock ordinary/sparse。
- 每任务 12 个新种子：`997101–997112`，与 PR54/PR55/PR58 种子块不重叠。
- 6 个算法：`legacy`、`semantic_only`、`budgeted`、`branch_mean`、`branch_dispersion`、`branch_explore`。
- 每个 case/seed 共享 12 个真实初始评价；随后每臂严格执行 **6 次**在线评价。
- 共 **2,304 次**真实解析评价（576 初始 + 1,728 在线），**288 条** on-policy trial。
- 新增飞机、RCAIDE 评价：**0**；新增 GP/VAE 拟合：真实执行。
- 原始 PR54 三个对照臂字节一致（`original_pr54_three_control_arms_byte_identical=true`），所有臂每次在线成本一致。
- 测试：`12 passed in 0.14s`。
- 完整状态：`COMPLETE_FRESH_FIXED_SIX_ON_POLICY`，exit 0。

六臂的固定定义：

- `branch_mean`：以 `max(r_GP(x), m_b)` 替代 GP 风险代理，其余预算机制不变。
- `branch_dispersion`：以 `max(r_GP(x), min(1, m_b+d_b))` 替代风险代理。
- `branch_explore`：同 `branch_dispersion`，但无候选满足预测额度时选择 `d_b` 最大的分支，作为**高风险信息性回退**，不假装通过门槛。
- `m_b=(1+f)/(2+f+s)`，`d_b=sqrt((1+f)(1+s)/((2+f+s)^2(3+f+s)))`，Beta(1,1) 先验、1 个后验标准差，均为事先固定的启发式，不是统计置信上界。

## 主结果：feasible-first 配对胜负

表中为 `候选 vs 对照` 的胜/负/平；仅在双方都有精确可行目标时比较目标差。每格 12 个种子。

| case | branch_mean vs budgeted | branch_dispersion vs branch_mean | branch_explore vs branch_dispersion |
|---|---:|---:|---:|
| Branin ordinary | 6 / 5 / 1 | 3 / 5 / 4 | 4 / 6 / 2 |
| Branin sparse | 1 / 5 / 6 | 2 / 1 / 9 | 1 / 2 / 9 |
| Rosenbrock ordinary | 2 / 6 / 4 | 2 / 4 / 6 | 0 / 6 / 6 |
| Rosenbrock sparse | 2 / 4 / 6 | 1 / 2 / 9 | 0 / 2 / 10 |
| **合计（48 对）** | **11 / 20 / 17** | **8 / 12 / 28** | **5 / 16 / 27** |

逐项消融没有出现正向结果：

- `branch_mean` 相对 `budgeted`：11 胜、20 负；只有 Branin ordinary 为 6/5 的轻微胜势。
- 加入离散度：`branch_dispersion` 相对 `branch_mean` 为 8 胜、12 负、28 平；多数时候连候选路径都不变。
- 信息性回退：`branch_explore` 相对 `branch_dispersion` 为 5 胜、16 负、27 平；Rosenbrock 两个任务合计 0 胜、8 负。

作为补充：

- `branch_dispersion vs budgeted`：10 / 24 / 14；
- `branch_explore vs budgeted`：5 / 29 / 14；
- `branch_explore vs legacy`：4 / 30 / 14；
- `budgeted vs semantic_only`：0 / 0 / 48（可行首次胜负全平，但语义臂多 6 次实际失败）。

## 实际工作量、预测覆盖与回退

每个臂 288 次在线评价（4 case × 12 seed × 6 call）。

| arm | 最终可行 seed | 冷启动新可行 | 实际在线不可行总数 | 有风险预测 | 回退次数 | 回退后失败 | 回退失败率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 43/48 | 3 | 110/288 | 0 | 0 | 0 | — |
| semantic_only | 43/48 | 3 | 114/288 | 266 | 0 | 0 | — |
| budgeted | 43/48 | 3 | 108/288 | 266 | 22（8.3%） | 19 | 86.4% |
| branch_mean | 42/48 | 2 | 164/288 | 265 | 109（41.1%） | 92 | 84.4% |
| branch_dispersion | 41/48 | 1 | 179/288 | 264 | 175（66.3%） | 131 | 74.9% |
| branch_explore | 41/48 | 1 | 247/288 | 264 | 231（87.5%） | 213 | 92.2% |

预测覆盖率约 92%（264–266/288）；剩余是 scheduled/degenerate coverage 等无模型概率调用，未伪造概率，但其实际评价费用全部保留在失败总数中。

**分任务看，损失集中在稀疏任务：**

- Branin sparse：六个臂最终都只有 9/12 可行；但实际在线不可行数从 `budgeted` 的 4.083 升到 `branch_mean` 5.333、`branch_dispersion` 5.500、`branch_explore` 5.750；回退数分别为 50/56/56。
- Rosenbrock sparse：`budgeted` 最终 10/12 可行、冷启动发现 2 个；`branch_mean` 降到 9 个，`branch_dispersion`/`branch_explore` 降到 8 个，冷启动发现只有 1/0/0；实际在线不可行数从 3.333 升到 5.167/5.667/6.000。
- Branin ordinary 与 Rosenbrock ordinary 最终仍为 12/12 可行，但新分支臂的实际失败数也明显更高：Branin ordinary 由 0.667 升到 1.333/1.917/4.667；Rosenbrock ordinary 由 0.917 升到 1.833/1.833/4.167。

## 双方可行时的目标差

`mean (median)`，正值为候选更差；仅在双方都有精确可行目标时统计：

| case | branch_mean − budgeted | branch_dispersion − branch_mean | branch_explore − branch_dispersion |
|---|---:|---:|---:|
| Branin ordinary | −0.603 (−0.375) | +1.673 (0) | +2.359 (+0.017) |
| Branin sparse | +3.825 (+1.299) | −0.198 (0) | +1.671 (0) |
| Rosenbrock ordinary | −0.296 (+0.883) | +19.68 (0) | +130.6 (+1.138) |
| Rosenbrock sparse | +67.58 (0) | −6.261 (0) | +26.46 (0) |

- `branch_mean` 在 Branin ordinary 有小的目标改善，但在两个稀疏任务明显变差，且中位数多为 0，说明收益集中在少数种子/离群值。
- 加入离散度没有稳定目标收益；Rosenbrock ordinary 的均值 +19.68 也提示长尾。
- 信息性回退在四个任务上的均值全部为正，即双方都可行时它选到更差目标；Rosenbrock ordinary 的 +130.6 尤其明显。

## 判读

**本轮没有验证出分支级 Beta 风险不确定性的算法价值，反而显示三类改动都会增加实际失败与回退。**

- `branch_mean` 通过 `max(r_GP,m_b)` 让风险代理更保守，但没有改善可行首次胜负，反而把回退从 22 次推到 109 次；实际不可行总数增加 56 次（164 vs 108）。
- `branch_dispersion` 在均值上再加一个标准差，进一步收紧门槛；回退 175 次、失败 179 次，均比 `branch_mean` 更差，且 28/48 对为平局，说明很多种子根本没有改变选择轨迹。
- `branch_explore` 的信息性回退主动选择 `d_b` 最大的分支，实际变成高风险探索：87.5% 的在线调用触发回退，92.2% 的回退点不可行，总失败 247/288；它不是风险控制策略。
- 稀疏任务中初始可行点少，回退很容易把预算消耗在不可行区域；普通任务虽然仍能找到可行点，但失败费用同样增加。
- 本轮所有结果都是 12 种子/任务的机制探索，Beta 离散度依赖自适应采样且不是置信域；不能据此声称安全、收敛或对飞机任务的泛化。

## 文件

- `fixed_six_onpolicy_report.json`：完整 trial、配对胜负、目标差、回退与覆盖统计。
- `plan.json`、`source_receipt.json`、`progress.jsonl`：固定协议、代码/引擎身份和 288 条逐 trial 进度。
- `pr60-branin_ordinary-trials.zip`、`pr60-branin_sparse-trials.zip`、`pr60-rosenbrock_ordinary-trials.zip`、`pr60-rosenbrock_sparse-trials.zip`：每任务完整的 `shared_initial/` 与 `trials/`（各 444 个条目，已通过 ZIP 完整性检查）。
- `smoke-console.log`、`full-console.log`、`tests.log`：两轮控制台与 pytest 日志。
- `fixed_sha.txt`、`artifact_sha256.txt`：代码 SHA、协议/runner/引擎 SHA256，以及报告、计划、收据、进度和 ZIP 的 SHA256。
