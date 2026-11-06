# PR68：真实 GP 两步前瞻与中止区间信息消融

固定代码 SHA：`21a1267ee907774e6958a12539da98c0840cc1d9`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/68>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本目录是工作机原始结果的完整副本：`smoke/`、`full/`、全部日志、测试、环境、协议与 SHA256 清单。分析脚本只读报告/证据，没有再次拟合 GP/VAE，也没有调用飞机或 RCAIDE。

## 1. 执行范围与完整性

- 预检：`22` 个 stage3 数值测试通过；`8` 个真实运行测试通过；`compileall` 与 `--help` 退出码均为 0。
- smoke：4 任务 × 种子 `998791`，`48` 初始 + `32` 在线 = **80** 次解析评价，状态 `COMPLETE_SMOKE_TECHNICAL_ONLY`；verify 校验 105 个文件。
- full：4 任务 × 种子 `998801–998804`，`192` 初始 + `384` 在线 = **576** 次解析评价，**64 条** trial，状态 `COMPLETE_GP_FANTASY_PILOT`；verify 校验 405 个文件，`manifest_sha256=29b39c0c...d6eb`。
- 每臂每 case/seed 恰好 6 次在线评价；四个 arm 的初始前缀逐 case/seed 相同（每 case 4 个唯一初始 hash）。
- 飞机/RCAIDE 调用均为 **0**；GP/VAE 真实拟合；全程 `0` 个未知/未结算评价（576 started / 576 settled）。
- 源 worktree 在运行后仍为干净（`git status --porcelain` 为空）；模块哈希与 `source_receipt.json` 一致。
- 无 GP 数值失败：`fantasy_fallback=0`；每个 depth=2 决策的幻想拟合 started=completed。

## 2. 四个 arm 的总览

| arm | 最终可行 /16 | 冷启动发现 | 实际在线不可行 /96 | 模型无覆盖 | 幻想拟合 | 回退 | 提议耗时 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `soft_gp_eligible` | 13 | 1 | 34 | 13 | 0 | 0 | 81.8 |
| `cei_myopic` | 12 | 0 | 36 | 0 | 0 | 0 | 88.1 |
| `cei_two_step` | 12 | 0 | 44 | 0 | 5,476 | 0 | 182.4 |
| `cei_two_step_no_abort_update` | 12 | 0 | 45 | 0 | 4,216 | 0 | 123.4 |

- 两步臂相对 myopic 的提议耗时约 **2.07×**，`no_abort` 约 **1.40×**。
- 幻想总耗时：两步 `92.7 s`，`no_abort` `33.6 s`；两者均无数值回退。
- `soft_gp_eligible` 有 13 次 scheduled coverage/无模型调用；三个 CEI 臂均为 0，因此它不是完全同覆盖的“同表示”对照。

## 3. 主对比：`cei_two_step` vs `cei_myopic`

每格 4 个种子；W/L/T 为候选对对照的可行优先胜负平；Δ 仅在双方都可行时统计。

| case | W/L/T | 双方可行 | 路径改变 | Δ 均值 | Δ 中位数 | 最终可行（两步/对照） | 在线失败（两步/对照） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Branin ordinary | 1/0/3 | 4 | 3/4 | −0.158 | 0 | 4/4 | 3/2 |
| Branin sparse | 0/1/3 | 2 | 3/4 | +2.512 | +2.512 | 2/2 | 17/15 |
| Rosenbrock ordinary | 0/0/4 | 4 | 2/4 | 0 | 0 | 4/4 | 4/4 |
| Rosenbrock sparse | 0/2/2 | 2 | 4/4 | +33.307 | +33.307 | 2/2 | 20/15 |
| **合计** | **1/3/12** | **12** | **12/16** | — | — | **12/12** | **44/36** |

- **没有任何一个种子因为两步前瞻而多得到最终可行解**：`only_candidate=0`（0/16），`only_control=0`；最终可行总数完全相同。
- 12 条路径改变只换来 1 个 BO 的极小目标改善（−0.158）和 3 个目标劣化（+2.512、+33.307）；其余 12 对目标打平。
- 实际在线失败从 36 增到 44（+8），其中 Branin sparse +2、Rosenbrock sparse +5、Branin ordinary +1、Rosenbrock ordinary 不变。
- 两个稀疏任务的 2 条无解轨迹仍然无解；两步前瞻没有带来冷启动发现。

## 4. 中止区间信息消融：`cei_two_step` vs `no_abort_update`

| case | W/L/T | 路径改变 | Δ 均值 | 最终失败（两步/消融） |
|---|---:|---:|---:|---:|
| Branin ordinary | 0/0/4 | 0/4 | 0 | 3/3 |
| Branin sparse | 0/0/4 | 3/4 | 0 | 17/18 |
| Rosenbrock ordinary | 0/0/4 | 0/4 | 0 | 4/4 |
| Rosenbrock sparse | 0/0/4 | 2/4 | 0 | 20/20 |
| **合计** | **0/0/16** | **5/16** | **0** | **44/45** |

`no_abort_update` 只在 5/16 条路径上改变了选择，但没有改变任何最终可行/目标胜负；失败数只差 1。**在本 pilot 中，是否用中止区间更新 region GP 没有可检测的收益，也没有明显损害。**

## 5. 其他对照

### `cei_myopic` vs `soft_gp_eligible`

| case | W/L/T | 路径改变 | Δ 均值 | 最终可行（myopic/soft） | 失败（myopic/soft） |
|---|---:|---:|---:|---:|---:|
| Branin ordinary | 0/0/4 | 4/4 | 0 | 4/4 | 2/1 |
| Branin sparse | 1/0/3 | 4/4 | −4.593 | 2/2 | 15/17 |
| Rosenbrock ordinary | 0/3/1 | 4/4 | +4.549 | 4/4 | 4/2 |
| Rosenbrock sparse | 1/1/2 | 4/4 | −11.630 | 2/3 | 15/14 |
| **合计** | **2/4/10** | **16/16** | — | **12/13** | **36/34** |

`soft_gp` 多 1 个冷启动可行（Rosenbrock sparse），但它在 4 个 case 上有 13 次无模型覆盖；这是预算内可用的覆盖策略，不是纯 GP 比较。

### `cei_two_step` vs `soft_gp_eligible`

合计 **2/6/8**，失败 **44 vs 34**，最终可行 **12 vs 13**；四个 case 的 Δ 分别为 −0.158、−2.081、+4.549、+21.678。两步前瞻没有超过 old soft-GP 基线。

## 6. 逐任务诊断

### Branin ordinary

- 四臂最终都 4/4 可行；在线失败 `soft 1 / myopic 2 / two_step 3 / no_abort 3`。
- 两步 vs myopic：1 胜、0 负、3 平；3/4 路径改变，唯一非平目标差 −0.158（两步略好），其余目标相同。
- 冷启动无新可行（初始均已可行）；两步没有改变最终可行集合，但多 1 次在线失败。
- 本实验没有候选池真值审计，不能回答“eligible 是否排除了好点/原效用前八是否含好点”；只能确认已选路径没有产生可行收益。

### Branin sparse

- 四臂最终都只有 2/4 可行；失败 `17/15/17/18`；每臂 2 条轨迹完全无解。
- 两步 vs myopic：0 胜、1 负、3 平；3/4 路径改变；唯一双方可行目标差 +2.512（两步更差）。
- 无冷启动发现；soft 的 7 次无模型覆盖也没有提高最终可行数（仍 2/4）。
- 候选池/eligible/前八名的机会审计在本轮未执行；不能把“路径改变”解释为错过了池内好点。

### Rosenbrock ordinary

- 四臂最终都 4/4 可行；失败 `soft 2 / myopic 4 / two_step 4 / no_abort 4`。
- 两步 vs myopic：0/0/4；仅 2/4 路径改变，全部目标相同；没有收益也没有目标损害。
- 该任务的主要失败来自在线评价本身，而非最终可行性；两步前瞻没有降低失败数。

### Rosenbrock sparse

- 最终可行 `soft 3 / myopic 2 / two_step 2 / no_abort 2`；失败 `14/15/20/20`；两步臂 2 条轨迹完全无解。
- 两步 vs myopic：0 胜、2 负、2 平；4/4 路径改变；双方可行目标差 +33.307（两步明显更差）。
- soft 在此任务有 1 个冷启动发现；三个 CEI 臂均无冷启动发现。两步前瞻没有复制该发现。
- 该任务是本轮最明确的负结果：路径大幅改变、失败增加、没有新增可行。

## 7. 风险预测与幻想机制

### 预测风险 vs 实际失败

| arm | 有预测决策 | 平均预测失败率 | 实际失败率 | Brier | ECE | p≥0.8 接受 / 其中假可行 |
|---|---:|---:|---:|---:|---:|---:|
| `soft_gp_eligible` | 83 | 0.229 | 0.265 | 0.1171 | 0.0900 | 11 / 0 |
| `cei_myopic` | 96 | 0.334 | 0.375 | 0.1012 | 0.0919 | 24 / 0 |
| `cei_two_step` | 96 | 0.360 | 0.458 | 0.1202 | 0.0984 | 24 / 0 |
| `no_abort_update` | 96 | 0.362 | 0.469 | 0.1211 | 0.1067 | 24 / 0 |

两步前瞻的预测失败率与实际失败率偏差更大，Brier 比 myopic 差（0.1202 vs 0.1012）；p≥0.8 没有假可行，但接受的全是真实失败。风险概率不是安全保证。

### 中止/完成的幻想契约

- 每个两步臂 80 个 depth=2 决策；共 **2,544** 个幻想结果/臂：`1,264` 中止、`1,280` 完成。
- 中止结果只暴露 `region>0` 的区间、精确 `resource`、`objective=missing`；完成但不可行结果暴露精确 `region≤0`、精确 `resource` 和精确 `objective`；冷启动完成不暴露 objective。
- 逐条核对 **0** 个契约违规；中止分支的 `abort_interval_used` 与 arm 设置一致（两步为 True，`no_abort` 为 False）。
- 最后一步退化：16/16 条两步轨迹的最后决策 `depth=1`；80 个 depth=2、16 个 depth=1 决策。
- 幻想 GP 拟合：两步 `5,476` 次 started=completed，`no_abort` `4,216` 次；`fantasy_fallback=0`。

## 8. 判读

**PR68 没有显示两步真实 GP 前瞻或中止区间信息带来任何可复现的收益。**

- 主对比 1/3/12，最终可行 12/12，失败 44/36；唯一“胜”是 0.158 的极小目标差，随后在稀疏任务被 +2.5 与 +33.3 的负差抵消。
- 中止区间消融 0/0/16；路径只在 5/16 变化，最终指标不变；中止信息在当前实现中没有可检测价值。
- 提议耗时翻倍（2.07×）且失败更多；这不是“用更多计算换更好结果”的正面证据。
- `soft_gp` 的 13 次无模型覆盖是唯一多出冷启动可行的臂；它本身不是两步前瞻的贡献，且不是同覆盖比较。
- 本 pilot 只覆盖 4 任务 × 4 种子 × 6 次在线；没有候选池真值审计，不能判断 eligible/前八名是否错过机会，也不能外推到飞机、RCAIDE 或安全保证。

## 9. 文件与校验

- `smoke/`、`full/`：完整原始输出，包含 `report.json`、`plan.json`、`source_receipt.json`、`progress.jsonl`、`evaluations.jsonl`、`manifest.json`、`shared_initial/`、`trials/`（含 `representation/`）。
- `logs/`：环境、22+8 测试、compileall、help、smoke/full 控制台及 verify 日志。
- `analysis_summary.json`：机器可读汇总；`integrity_check.json`：逐项完整性结论；`sha256_manifest.txt`：全部上传文件 SHA256。
- `protocol.txt`：固定协议原文；`instructions.txt`：本轮任务指令副本。
- `.gitattributes`：`* -text`，防止 Git 换行转换改变证据字节。

### 已知限制

- 4 个自定义数学任务、每任务 4 个种子、6 次在线，统计功效有限；不是 PR45/67 的飞机任务。
- 没有独立候选池真值审计，无法回答“池内机会/eligible/前八名”的问题。
- `soft_gp` 有 scheduled coverage，其他 CEI 臂没有；表示与覆盖不是完全正交。
- 两步前瞻未接入完整飞机 GP 前瞻优化；结果不能声称“新两步 GP 算法胜出”或“飞机安全已验证”。
