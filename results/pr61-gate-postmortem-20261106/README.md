# PR61：PR60 分支 Beta 风险下限导致预算门失效的只读根因审计

固定分析代码：`931101444c083c89e6f10dbf40c1f20d3eb585f2`  
输入：PR60 `fixed_six_onpolicy_report.json`（commit `b175daf7...`，Git BLOB `508ba9bb...`，
SHA256 `18032f45...`，2,248,878 B）  
环境：`Rcaide_vsp` / Python 3.9.25。

本分支只上传只读审计产物，**没有**重新运行 GP/VAE、数学评价器或飞机/RCAIDE，
也没有修改 PR60 报告、ZIP 或 PR45 原始数据。

## 结论摘要

PR60 的分支后验标准差不只让风险代理变得更保守；它把大量候选推到预算门之外，
随后触发的回退才是实际失败的主要通道。

- `branch_mean`：Beta 下限绑定 **80.4%** 的已评分分支调用；预算门失败 **109/265**（41.1%），
  回退后 **92/109（84.4%）** 实际不可行，总在线失败 **164/288**。
- `branch_dispersion`：绑定 **87.5%**；门失败 **175/264**（66.3%），回退失败 **131/175（74.9%）**，
  总失败 **179/288**。
- `branch_explore`：绑定 **81.1%**；门失败 **231/264**（87.5%），回退失败 **213/231（92.2%）**，
  总失败 **247/288**。
- 对照 `budgeted`：门失败仅 **22/266**（8.3%），回退失败 **19/22（86.4%）**，总失败 **108/288**。

因此，把“分支 Beta 下限”当作风险控制，实际效果是把原预算机制推向无候选可过门的回退；
回退又偏好高不确定/高失败区域，尤其在 feasible discovery 和稀疏任务上。

## 输入身份与复算范围

审计逐条读取 PR60 的 **288 条 trial、1,728 条在线 receipt**，只使用原始已选设计与已观察标签：

- 校验报告 Git BLOB SHA1 与 SHA256；
- 校验 11 个臂 × 288 条 receipt 对齐，四种 arm（legacy/semantic/budgeted/三个 branch）齐全；
- 复算 Beta 后验公式、`m_b`、`m_b+d_b`、风险增量、预算门结果；
- 只对“实际已选点”分析，不推断未选候选的结果。

运行结果：`COMPLETE_RETROSPECTIVE_BRANCH_BETA_GATE_COLLAPSE_AUDIT`，exit 0；
新增飞机/ RCAIDE/GP/VAE/解析评价均为 **0**；原始数据 `modified=false`。
测试：`5 passed`。

## 各臂总体

每臂 288 条在线评价。

| arm | 有预测调用 | Beta 下限绑定 | 绑定率 | 平均风险增量 | 平均分支 n | n≤3 | 预算门失败 | 回退次数 | 回退后失败 | 覆盖率调用/失败 | 总失败 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 0 | 0 | — | — | — | — | 0 | 0 | 0 | — | 110 |
| semantic_only | 266 | 0 | — | — | — | — | 0 | 0 | 0 | — | 114 |
| budgeted | 266 | 0 | — | — | — | — | 22 | 22 | 19 | 22 / 21 | 108 |
| branch_mean | 265 | 213 | 80.4% | 0.149 | 4.37 | 98 | 109 | 109 | 92 | 23 / 22 | 164 |
| branch_dispersion | 264 | 231 | 87.5% | 0.244 | 4.94 | 78 | 175 | 175 | 131 | 24 / 23 | 179 |
| branch_explore | 264 | 214 | 81.1% | 0.188 | 2.98 | 186 | 231 | 231 | 213 | 24 / 23 | 247 |

说明：

- 绑定率分母是“有 Beta 历史且被评分”的分支调用；覆盖率调用没有模型概率，未伪造。
- 三个 branch 臂的 `n_predicted_gate_failed` 与 `fallback_calls` 完全相等，说明这里的回退全部由“没有候选低于剩余预算”触发。
- `branch_explore` 的绑定率不是最高，但它的回退最多，因为其信息性回退会主动选择 `d_b` 最大的分支。
- 非回退调用也有失败：`branch_mean` 72/179（40.2%）、`branch_dispersion` 48/113（42.5%）、
  `branch_explore` 34/57（59.6%）；回退点的失败率仍显著更高。

## 按任务分布

| arm / case | 有预测 | Beta 绑定 | 绑定率 | 平均增量 | 平均 n | 门失败 | 回退失败 | 覆盖调用/失败 | 总失败 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| branch_mean / Branin ordinary | 72 | 55 | 76.4% | 0.273 | 4.85 | 3 | 0/3 | 0 / 0 | 16 |
| branch_mean / Branin sparse | 59 | 47 | 79.7% | 0.142 | 4.41 | 50 | 44/50 | 13 / 12 | 64 |
| branch_mean / Rosenbrock ordinary | 72 | 56 | 77.8% | 0.205 | 4.33 | 7 | 3/7 | 0 / 0 | 22 |
| branch_mean / Rosenbrock sparse | 62 | 55 | 88.7% | 0.113 | 3.81 | 49 | 45/49 | 10 / 10 | 62 |
| branch_dispersion / Branin ordinary | 72 | 62 | 86.1% | 0.370 | 4.94 | 30 | 11/30 | 0 / 0 | 23 |
| branch_dispersion / Branin sparse | 59 | 51 | 86.4% | 0.226 | 4.86 | 56 | 52/56 | 13 / 12 | 66 |
| branch_dispersion / Rosenbrock ordinary | 72 | 60 | 83.3% | 0.342 | 4.79 | 28 | 11/28 | 0 / 0 | 22 |
| branch_dispersion / Rosenbrock sparse | 61 | 58 | 95.1% | 0.160 | 5.20 | 61 | 57/61 | 11 / 11 | 68 |
| branch_explore / Branin ordinary | 72 | 56 | 77.8% | 0.283 | 3.11 | 60 | 51/60 | 0 / 0 | 56 |
| branch_explore / Branin sparse | 59 | 46 | 78.0% | 0.219 | 2.78 | 56 | 55/56 | 13 / 12 | 69 |
| branch_explore / Rosenbrock ordinary | 72 | 57 | 79.2% | 0.216 | 3.26 | 54 | 46/54 | 0 / 0 | 50 |
| branch_explore / Rosenbrock sparse | 61 | 55 | 90.2% | 0.210 | 2.69 | 61 | 61/61 | 11 / 11 | 72 |

关键分布：

- **普通任务仍然能收敛到可行点，但失败费用显著上升。**  
  Branin ordinary 的五个臂最终都是 12/12 可行，但 `branch_mean` 的在线失败从 `budgeted` 的 8 升到 16，
  `branch_dispersion` 23，`branch_explore` 56。
- **稀疏任务是回退集中区。**  
  Branin sparse 和 Rosenbrock sparse 的门失败分别为 50/59、49/62（`branch_mean`），
  56/59、61/61（`branch_dispersion`），56/59、61/61（`branch_explore`）。
  `branch_explore` 在 Rosenbrock sparse 的 61 次回退全部失败，最终只有 8/12 初始可行。
- **风险增量在普通任务也不小。**  
  `branch_dispersion` 的 Branin ordinary 平均增量 0.370，Rosenbrock ordinary 0.342；
  这意味着很多候选在未超过 `r_GP` 时就因 `m_b+d_b` 被推到预算之外。

## 按优化阶段分布（`branch_mean`）

| mode | 调用 | Beta 绑定 | 绑定率 | 门失败 | 回退失败 | 覆盖调用/失败 |
|---|---:|---:|---:|---:|---:|---:|
| scheduled_coverage | 14 | 0 | — | 0 | — | 14 / 14 |
| degenerate_gate_coverage | 9 | 0 | — | 0 | — | 9 / 8 |
| feasible_discovery | 19 | 4 | 21.1% | 19 | 18/19 | 0 / 0 |
| probability_weighted_feasible_improvement | 204 | 187 | 91.7% | 74 | 59/74 | 0 / 0 |
| budgeted_boundary_information | 42 | 22 | 52.4% | 16 | 15/16 | 0 / 0 |

同样模式在另外两臂更严重：

- `branch_dispersion`：feasible_discovery 21 次调用全部回退、全部失败；boundary 24/42 回退、20 次失败；
  PW 130/201 回退、90 次失败。
- `branch_explore`：discovery 20/20 回退且 20/20 失败；boundary 38/43 回退、33 次失败；
  PW 173/201 回退、160 次失败。

两个“覆盖”阶段没有模型风险预测，不会进入预算门，但本身几乎全失败
（scheduled 14/14、degenerate 8/9 等）；它们解释了部分模型覆盖之外的失败，不能混入回退失败率。

## 机制判读

1. **Beta 下限不是“偶尔生效”的保险，而是大多数调用都会改变风险值。**  
   绑定率 80–88%，平均增量 0.15–0.24，90 分位约 0.40–0.55；这足以让原本可过门的候选被推到剩余预算之外。

2. **失效的直接通道是预算门回退，而不是阈值本身。**  
   floor 生效后，候选的 `max(r_GP, m_b)` 或 `max(r_GP, m_b+d_b)` 超过 allowance；
   若所有候选都超过，就回退。回退按原评分选择，往往落到高失败区域。  
   `branch_mean` 的 109 次门失败全部回退，其中 92 次实际失败；`branch_dispersion` 175 次回退，131 次失败；
   `branch_explore` 231 次回退，213 次失败。

3. **小样本分支是风险放大器。**  
   `branch_mean` 平均只有 4.37 条分支历史，98 次调用 n≤3；`branch_explore` 平均 2.98 条，186 次 n≤3。
   自适应分支计数还混合了不同连续设计，因此 `m_b`/`d_b` 不是逐设计的校准失败概率，也不是置信上界。
   `branch_explore` 在 177 个回退点上只有 ≤3 条历史，却用最大 `d_b` 作为回退依据，等于把探索预算投向最不确定的区域。

4. **稀疏任务上，回退几乎无法自救。**  
   Branin sparse 初始可行点少，回退后的候选仍大多不可行；`branch_explore` 的 56 次回退中 55 次失败，
   `branch_dispersion` 56 次回退中 52 次失败。普通任务虽然保持最终可行，但代价是更多失败评价。

5. **这不是“floor 直接导致失败”的因果证明。**  
   本轮只观察原始已选点的结果；回退规则同时改变了候选选择，可能选中本身更难的设计。
   “floor 绑定—门失败—回退—实际失败”的链式关联很强，但需要新的、预先注册的反事实选择实验才能分离
   floor 与回退各自的作用。本报告不提供这种因果估计。

## 限制

- 只评估 PR60 已选设计；没有未观测候选的真实结果，不能推断换一个阈值会选中哪个候选。
- Beta 后验来自自适应分支计数，不是逐设计校准概率；`m_b+d_b` 没有覆盖保证。
- 不重建完整 trial ZIP 的分支成员关系；分支计数来自原始记录的概率/候选来源字段。
- 全部是自定义 Branin/Rosenbrock 层级任务，不是 100/140 km 飞机或 RCAIDE 验证。
- 不声称安全、风险预算保证或对 PR60 算法的事后改进；PR60 的等六次 / 288 trial 比较结论仍成立。

## 文件

- `branch_floor_gate_audit.json`：逐 trial/arm/case 的 1,728 条 receipt 与完整聚合、分布和复算校验。
- `branch_floor_selected_receipts.csv`：可直接复核的逐在线选择行（含 Beta 计数、增量、门结果、实际标签）。
- `tests.log`、`full-console.log`：pytest 与完整运行日志。
- `fixed_sha.txt`：分析代码 SHA、PR60 报告 SHA256/BLOB/大小、计数与零新增评价声明。
- `pre_run_source_sha256.txt` / `post_run_source_sha256.txt`：PR60 输入报告运行前后哈希（一致）。
- `pr45_sqlite_post.sha256`：顺带记录的 PR45 SQLite/审计文件哈希；本次没有修改它们。
- `artifact_sha256.txt`：本目录关键产物与日志的 SHA256。
