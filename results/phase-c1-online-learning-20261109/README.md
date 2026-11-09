# PR80 Phase C1：等评价预算下的风险引导与代理更新

固定代码 SHA：`62b46f7a30ab5c3475cd59b86c7f3ad51974a1da`  
入口：`run_c1_online.py`  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本轮是**新的在线数学优化实验**：四个 tradeoff 任务、8 个新种子、六臂、每臂每组 24 次在线评价。没有飞机、RCAIDE、VAE 训练或幻想拟合；GP 头与核构造是真实计算。

## 1. 执行与费用

- 预检：34 项 core 测试、15 项 native 测试，0 跳过；compileall / help / plan 全部 exit 0。
- smoke：24 trials，264 次数学评价（192 初始 + 72 在线）；verify 11 文件；8 类 tamper 全部拒绝。
- full：192 trials，6,144 次数学评价（1,536 共享初始 + 4,608 在线）；status `COMPLETE_C1_ONLINE_PILOT`；elapsed 4,136.8 s；verify 39 文件、192 trials、0 new math/GP；8 类 tamper 全部拒绝。
- 每臂每组 24 次在线 + 48 次初始；初始只支付一次。单臂从零成本 72，已有初始条件下新增 24。
- GP：5,984 heads / 47,872 kernels / 2,976 predictions，started=completed，0 errors，0 kernel rejects；VAE / 幻想 / 飞机 / RCAIDE 均为 0。
- 0 个模型失败、0 个未结算评价、0 条轨迹缺失；`model_failure_calls=0`。

## 2. 六臂总览（k24，initial_warm，n=32 每组 8）

| arm | 最终可行/32 | 改善初始/32 | 新增精确目标 | 在线不可行/768 |
|---|---:|---:|---:|---:|
| `ei_frozen` | 32 | 19 | 222 | 605 |
| `ei_updated` | 32 | 16 | 84 | 722 |
| `risk_ei_frozen` | 32 | 29 | 358 | 410 |
| `risk_ei_updated` | 32 | 31 | 223 | 545 |
| `risk_only_updated` | 32 | 16 | 724 | 44 |
| `uniform_hash` | 32 | 14 | 88 | 696 |

所有 32 个 case/seed 都是 `initial_warm`，0 个 `initial_cold`；所有臂最终 32/32 可行，0 条无解轨迹。**冷启动机制未被本轮触发**，因此不能据此评价冷启动回退。

## 3. 主比较：`risk_ei_updated` vs `risk_ei_frozen`

k24、initial_warm、每任务 8 个配对：

| 任务 | W/L/T | Δ 均值 | Δ 中位数 | 改善（更新/冻结） | 失败（更新/冻结） | 新标签（更新/冻结） |
|---|---:|---:|---:|---:|---:|---:|
| s0 | 5/2/1 | −0.308343 | −0.254090 | 8/6 | 154/106 | 38/86 |
| s1 | 6/1/1 | −0.035829 | −0.032565 | 8/8 | 118/74 | 74/118 |
| s2 | 5/0/3 | −0.106297 | −0.017269 | 7/7 | 119/108 | 73/84 |
| s3 | 5/2/1 | −0.183037 | −0.070060 | 8/8 | 154/122 | 38/70 |
| **合计** | **21/5/6** | **−0.158376** | — | **31/29** | **545/410** | **223/358** |

- 四个任务的平均差全部为负（更新臂更好），总 21 胜、5 负、6 平。
- 更新臂的新增精确目标反而少于冻结臂（223 vs 358），失败数更高（545 vs 410），却取得更低目标；这说明收益不是“更多标签”直接带来的，而来自更新后选点分布的变化，不能把标签数量当因果中介。
- 32/32 路径都发生变化；所有最终可行数相同（32/32），因此差异只在目标值与在线失败上。

### 6 / 12 / 24 步

| 步数 | W/L/T | Δ 均值 |
|---|---:|---:|
| 6 | 18/4/10 | −0.124580 |
| 12 | 19/5/8 | −0.092820 |
| 24 | 21/5/6 | −0.158376 |

三个检查点均为正方向，且 24 步比 6 步更强；仍是同一批轨迹的连续观察，不是独立重复。

## 4. 其他匹配对照（k24）

| 比较 | W/L/T | Δ 均值 | 说明 |
|---|---:|---:|---|
| `ei_updated` vs `ei_frozen` | 4/9/19 | **+0.129541** | 单纯在线更新平均更差；s2/s3 尤其明显 |
| `risk_ei_updated` vs `ei_updated` | 25/3/4 | −0.418260 | 风险引导+更新相对普通 EI 更新有大幅改善 |
| `risk_ei_frozen` vs `ei_frozen` | 17/7/8 | −0.130342 | 仅风险引导（不更新）也更好 |
| `risk_ei_updated` vs `risk_only_updated` | 29/1/2 | −0.450282 | risk_only 标签最多、失败最少，但目标最差 |
| `risk_ei_updated` vs `uniform_hash` | 26/5/1 | −0.406024 | 相对随机哈希基线更好 |

关键判读：

- **单纯在线更新不够**：`ei_updated` 对 `ei_frozen` 为 4/9/19，平均 +0.1295，说明更新本身不是收益来源。
- **风险引导本身有信号**：`risk_ei_frozen` 对 `ei_frozen` 为 17/7/8，平均 −0.1303。
- **风险+更新最强**：对 `ei_updated` 25/3/4、对 `risk_only` 29/1/2、对 `uniform` 26/5/1。
- `risk_only_updated` 获得 724 个新精确目标、仅 44 次在线失败，但目标最差；这支持“更多反馈/更少失败 ≠ 更好设计”，不能把标签数当作唯一目标。

## 5. 机制与完整性

- 同一起点：32/32 case/seed 为 `initial_warm`；六臂共享同一 48 条初始记录，初始只计费一次。
- 冻结/更新首动作：`ei_frozen` vs `ei_updated`、`risk_ei_frozen` vs `risk_ei_updated` 均为 **0 个首动作不一致**（32/32 相同）；差异从第二步后出现。
- 路径变化：主比较 32/32 全部改变；所有匹配对照也在各任务 8/8 改变（`ei_updated` vs `ei_frozen` 为 27/32）。
- 首次可行位置：全部为 **0**（初始已有可行解）；没有 1–24 的在线首次发现，也没有 null。
- 构型覆盖：六臂在每个 unit 都覆盖 4/4 branches（均值 4.0），没有覆盖差异。
- 回退：0 个模型失败、0 个冷启动回退、0 个无目标模型回退；`modes` 只有正常 `ei/risk_ei/risk_only/uniform_hash`。
- 部分观测：中止目标保持缺失，未补造；没有 VAE/幻想/特征附件。
- 源 worktree 和结果 worktree 均干净；PR75–PR79 证据未修改。

## 6. 逐任务机制补充

- s0：风险更新 8/8 改善初始，冻结 6/8；新标签 38 vs 86，失败 154 vs 106。
- s1：双方都 8/8 改善；更新臂标签更少（74 vs 118）、失败更多（118 vs 74），但目标仍更好。
- s2：双方 7/8 改善；更新臂 73 vs 84 标签、119 vs 108 失败。
- s3：双方 8/8 改善；更新臂 38 vs 70 标签、154 vs 122 失败。
- `risk_only` 在 s3 的改善为 0/8，说明“只追风险/信息”可能牺牲目标；其 724 标签与 44 失败是诊断信息，不是推荐策略。

## 7. 解释边界

- 全部 32 组 `initial_warm`，没有 cold 层；冷启动、首次可行发现、冷回退均未被检验。
- 8 个种子/任务，仍属探索性；不是 held-out 任务确认。
- 冻结臂是“冻结后验”，不是冻结当前最优或候选调度；当前最佳仍随自身评价更新。
- 等真实调用不等于等模型计算成本；n 个决策不是独立种子重复。
- 新增精确目标更多是诊断量，不是因果中介；不能断言“标签数导致目标改善”。
- 没有飞机/RCAIDE 数据；不能外推工程安全或飞行性能。
- 主比较为正，但仍是简化数学任务上的机制信号；不得直接提升为默认方法。

## 8. 文件

- `smoke/`、`full/`：完整原始输出；`units/<task>/r<repeat>/records.jsonl` 保存 ref/value 记录（初始、模型、候选池、预测、proposal、observation、结果）。
- `full/plan.json`、`source_receipt.json`、`environment.json`、`report.json`、`events.jsonl`、`evaluations.jsonl`、`progress.jsonl`、`manifest.json`。
- `logs/`：预检、smoke/full 运行、verify、8 类 tamper 检查的完整 stdout/stderr 与退出码。
- `analysis_summary.json`、`case_table.csv`、`checkpoint_table.csv`、`pair_deltas.csv`：逐任务/逐配对汇总与 32 条主比较逐重复差。
- `integrity_check.json`、`fixed_sha.txt`、`environment.txt`、`environment_packages.txt`、`instructions.txt`、`protocol.txt`、`sha256_manifest.txt`。
- `.gitattributes`：`* -text`。
