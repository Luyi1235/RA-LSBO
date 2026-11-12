# PR86-E：PR86-D 9000 条已发生决策的只读诊断

诊断代码 SHA：`2091e0e4c9d3d86a88c55715d2a23a2cd0f36db4`  
输入 PR86-D 结果 HEAD：`311ef606d173d6fe0bb3663ca18f9560c1ca5586`  
输入实验代码 SHA：`6e5cbd0bafc25cf4e36f9a5675401d6991165416`  
输入清单 SHA256：`b93001bc7f4c4cd3385812fdf9bd3196c98028aabe49d00ebf1243328a71f4aa`

本目录只含只读诊断输出；不改动 PR86-D 原始记录。

## 1. 范围与完整性

| 项目 | 结果 |
|---|---|
| 状态 | `COMPLETE_PR86E_READONLY_DIAGNOSTICS` |
| decisions / trajectories | 9,000 / 300 |
| identity pairs | 240 |
| 读取的输入文件 | 33（unit/report/plan/source receipt；不读 740 MB progress） |
| 新评价 / 新 GP 拟合 | **0 / 0** |
| 本机回归 | 15 tests passed，0 skipped（7.55 s） |
| 冻结权重检查 | `W_OBJECTIVE=0.65`、`W_INFO=0.35` 通过 |
| 源/输入 worktree | 干净，HEAD 与固定 SHA 一致 |

15 项测试包括合成端到端、冻结 `pr86c_acquisition` 对照、manifest 篡改、前缀/反馈绑定、分数篡改、零方差和不可行资源等。

## 2. 同状态 CEI 与 0.65/0.35 加权规则的分歧

同一决策的 CEI 首候选与加权首候选：

| 子集 | 决策数 | 规则分歧率 |
|---|---:|---:|
| 全部 | 9,000 | **56.8%** |
| `no_objective_gp` | 1,397 | 41.5% |
| `objective_ready` | 7,573 | **59.8%** |
| `no_feasible_incumbent` | 30 | 33.3% |

- 有 objective GP 后分歧明显上升；这与“目标/信息权重开始实际改变选择”一致。
- 分歧时加权选择相对 CEI 选择损失的 CEI proxy：

| 指标 | 全部 | 分歧子集 |
|---|---:|---:|
| 平均损失 | 21.5% | **37.9%** |
| 中位损失 | 6.3% | 36.3% |
| CEI proxy 为 0 | 0% | 0% |

- `source_replay_choice_diff=0`：诊断重放与原始已选点完全一致，没有源记录绑定错误。
- 但这只是**模型内 CEI proxy 损失**，不是真实目标 regret；不能据此断定 0.65/0.35 规则更差。

## 3. 实际已选点的结果与误差

| 指标 | 全部 9,000 |
|---|---:|
| 返回精确目标 | 12.7% |
| 返回可行点 | 11.4% |
| 首次可行发现 | 0.48% |
| 选中点 joint Brier 均值 | .1485 |
| 选中点 region Brier 均值 | .1674 |
| `objective_zero_variance` | 0% |

### 有精确目标时的误差（988 条，按任务分层，不混合原始尺度）

| 任务 | n | 平均误差 | 中位误差 | RMSE |
|---|---:|---:|---:|---:|
| legacy_branin_sparse | 149 | +1.990 | +0.734 | 11.48 |
| legacy_rosenbrock_sparse | 159 | +4.698 | −22.264 | **171.24** |
| tradeoff_s0 | 74 | −0.508 | −0.281 | 2.14 |
| tradeoff_s1 | 106 | −0.101 | −0.231 | 1.09 |
| tradeoff_s2 | 56 | −0.047 | −0.224 | 1.86 |
| tradeoff_s3 | 92 | −0.322 | −0.203 | 2.27 |
| tradeoff_s4 | 75 | −0.079 | +0.058 | 2.01 |
| tradeoff_s5 | 119 | +0.569 | +0.227 | 4.04 |
| tradeoff_s6 | 71 | +1.193 | +0.035 | 4.67 |
| tradeoff_s7 | 87 | +1.093 | +0.434 | 5.43 |

- 误差不是全局常数偏移：Rosenbrock sparse 的 RMSE 远大于 tradeoff 任务，且中位误差为 −22.3。
- 因此“目标 GP 误差”是**任务相关**问题；不能在任务间池化一个平均 RMSE 作为统一根因。
- 只有 988/9,000 个选中点有精确目标，其余没有可比较的误差，不能把缺失目标补 0。

## 4. 风险预测

| 子集 | joint Brier | region Brier |
|---|---:|---:|
| 全部 | .1485 | .1674 |
| `no_objective_gp` | .0952 | .1170 |
| `objective_ready` | .1585 | .1766 |
| 规则一致 (3,885) | .1404 | — |
| 规则分歧 (5,115) | .1546 | — |

- 风险 Brier 在 objective-ready 后升高，分歧样本也略差；但不是“完全失效”（仍低于 0.25 的无信息二分类区间）。
- 该 Brier 是选中样本上的事后值，不是全候选池校准；不能与 PR86-A 的随机样本 Brier 直接比较。

## 5. 最终 incumbent 身份

240 个 identity pair（objective_weighted 与四个对照，每 task/repeat/anchor）：

| 类别 | 数量 |
|---|---:|
| 同一已观察候选 | 107 |
| 不同候选且不同值 | **124** |
| 一方不可行 | 5 |
| 双方都不可行 | 4 |

- 51.7% 的最终 incumbent 数值不同；44.6% 是同一候选。
- 只有 9/240 涉及可行性差异；多数差异是“值不同”，不是“有没有可行解”。
- 这支持“采集规则确实改变路径/最终最优值”，但不证明哪条规则的真实收益更好。

## 6. 判读：根因不是单点

1. **采集规则分歧是最大的系统性信号**：56.8% 决策分歧，分歧时加权选择损失 37.9% 的 CEI proxy；且最终 incumbent 124/240 数值不同。
2. **目标代理误差是任务依赖的第二信号**：Rosenbrock sparse 的 RMSE 171，tradeoff 任务 1–5；统一目标模型不是全局充分。
3. **风险预测是中等强度问题**：joint Brier .1485，objective-ready .1585，分歧样本 .1546；没有出现零方差或全零代理。
4. **没有发现记账/重放/绑定错误**：`source_replay_choice_diff=0`、`cei_proxy_zero=0`、`objective_zero_variance=0`。
5. **不能把 CEI proxy 损失当作真实 regret，也不能从未选候选的事前预测推断其真实效果**；9000 决策不是 9000 个独立样本。

因此，下一轮若继续，应优先做**目标代理的任务分层误差控制**和**采集规则的同状态反事实/受控消融**，而不是继续增加种子或笼统调 0.65/0.35 权重。本轮不启动任何新 Full，也不修改算法或采集权重。

## 7. 文件

- `diagnostic_summary.json`：9000 决策的 task/anchor/method/phase 分层汇总。
- `decision_diagnostics.csv`：逐决策 CEI/加权候选、proxy 损失、选中点误差、风险 Brier、身份字段。
- `incumbent_identity.json`：300 endpoints、240 pairs 的身份类别。
- `manifest.json`：四个诊断输出文件的 SHA256。
- `logs/tests.log`、`logs/diagnostics.log`、`logs/import-check.log`、`environment.json`、`fixed_sha.txt`。
- `.gitattributes`：`* -text`，防止换行转换破坏 CSV/JSON。
