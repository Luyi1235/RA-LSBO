# PR70：下一轮候选池一致性隔离实验

固定代码 SHA：`275aac2c5d5d6a8c8f6278c0d556de3de5867f74`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/70>  
上一阶段审计结果 HEAD：`ff8da0895e302ccf24c0ed5c2d5de72d8f9b5b2f`  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本目录是工作机原始输出的完整副本：`smoke/`、`full/`、全部日志、测试、环境、协议与 SHA256 清单。没有重新运行 PR68/69，也没有新增飞机或 RCAIDE 调用。

## 1. 执行与完整性

- 预检：协议测试 `13 passed`，真实 EP/候选一致性测试 `13 passed`，0 skipped；`compileall` 与 `--help` 退出码均为 0。
- smoke：4 任务 × 种子 `998891`，`48` 初始 + `24` 在线 = **72** 次数学评价；状态 `COMPLETE_SMOKE_TECHNICAL_ONLY`；verify 校验 81 个文件，`future_pool_checks=4`。
- full：4 任务 × 种子 `998901–998904`，`192` 初始 + `288` 在线 = **480** 次数学评价，**48 条** trial；状态 `COMPLETE_NEXT_POOL_PILOT`；verify 校验 309 个文件，`future_pool_checks=80`。
- 每臂每 case/seed 恰好 6 次在线评价；初始前缀逐 case/seed 三臂相同。
- 飞机/RCAIDE 调用均为 **0**；552 started / 552 settled 是 smoke+full 的真实评价账本，无未知调用。
- 源 worktree 运行后为干净；模块哈希与 `source_receipt.json` 一致。

## 2. 三臂总览

| arm | 最终可行 /16 | 冷启动发现 | 实际在线不可行 /96 | future_pool_checks | 幻想拟合 started/completed | 回退 | 无模型调用 | 提议耗时 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cei_myopic` | 13 | 2 | 35 | 0 | 0 / 0 | 0 | 0 | 111.4 |
| `cei_two_step` | 14 | 3 | 31 | 0 | 5,604 / 5,604 | 0 | 0 | 210.3 |
| `cei_two_step_next_pool` | 14 | 3 | 30 | 80 | 5,594 / 5,594 | 0 | 0 | 237.8 |

- 相同真实调用数不等于相同墙钟：next_pool 的提议耗时是 two_step 的 **1.13×**、myopic 的 **2.13×**；full 总墙钟 569.9 s。
- 幻想拟合 started 与 completed 完全一致，数值回退 **0**；最后一步均无幻想拟合。

## 3. 主比较：`cei_two_step_next_pool` vs `cei_two_step`

| case | W/L/T | 双方可行 | 路径改变 | Δ 均值 | 逐种子 Δ |
|---|---:|---:|---:|---:|---|
| Branin ordinary | 2/1/1 | 4 | 4/4 | −0.902 | +0.402, −1.991, 0, −2.021 |
| Branin sparse | 1/1/2 | 3 | 4/4 | +0.664 | 0, +3.572, −1.581, 无解/无解 |
| Rosenbrock ordinary | 1/1/2 | 4 | 3/4 | −0.005 | 0, +0.579, −0.601, 0 |
| Rosenbrock sparse | 1/0/3 | 3 | 4/4 | −100.396 | 无解/无解, 0, 0, −301.187 |
| **合计** | **5/3/8** | **14** | **15/16** | — | — |

- 最终可行 **14/14** 相同；实际失败 **30 vs 31**，只少 1 次。
- 胜负相对接近，8/16 为平；Δ 均值被 Rosenbrock sparse 的单个种子 `998904`（−301.187）主导，不能当作普遍收益。
- Branin ordinary 是 2 胜 1 负，但幅度只有 0.4–2.0；Branin sparse 有 +3.572 的反向差。
- 这是“修复候选支持”的有限改善，**不是**两步前瞻优越的统计证据；4 种子仅探索性。

## 4. 次比较：`cei_two_step_next_pool` vs `cei_myopic`

| case | W/L/T | 双方可行 | 路径改变 | Δ 均值 | 逐种子 Δ |
|---|---:|---:|---:|---:|---|
| Branin ordinary | 1/0/3 | 4 | 2/4 | −0.655 | 0, −2.619, 0, 0 |
| Branin sparse | 1/1/2 | 2 | 3/4 | +2.689 | 仅 next 可行, 0, +5.377, 无解/无解 |
| Rosenbrock ordinary | 1/1/2 | 4 | 3/4 | +0.860 | +4.041, 0, −0.601, 0 |
| Rosenbrock sparse | 1/0/3 | 3 | 3/4 | −36.746 | 无解/无解, 0, 0, −110.238 |
| **合计** | **4/2/10** | **13** | **11/16** | — | — |

- 最终可行 **14 vs 13**；实际失败 **30 vs 35**；冷启动发现 **3 vs 2**。
- 主要可行性改善在 Branin sparse：next_pool 最终 3/4，myopic 2/4；另有冷启动发现差异。
- 目标差仍由少数稀疏种子主导；Rosenbrock ordinary 的 +4.041 是反向差。

## 5. 新种子参照：`cei_two_step` vs `cei_myopic`

合计 **5 胜 / 4 负 / 7 平**；最终可行 **14 vs 13**，失败 **31 vs 35**，冷启动 **3 vs 2**。  
也就是说，本轮原两步臂本身已优于 myopic；next_pool 相对 myopic 的部分差异可能继承自两步规则，不能全部归因于候选支持修复。主比较仍是 next_pool vs two_step，其结论是“有限、混合、不足以宣称优势”。

## 6. 逐任务细节

### Branin ordinary

- 最终三臂均 4/4；失败 `myopic 2 / two_step 1 / next_pool 2`。
- next vs two：2/1/1，路径 4/4 改变，Δ −0.902；998901 反向 +0.402，998902/998904 为 −1.991/−2.021，998903 平。
- next vs myopic：1/0/3，仅 998902 −2.619；其余平。
- 无冷启动、无无解轨迹；first feasible 全为初始已可行（位置 0）。

### Branin sparse

- 最终 `2 / 3 / 3`；失败 `16 / 15 / 12`；无解轨迹 `2 / 1 / 1`；冷启动 `1 / 2 / 2`。
- next vs two：1/1/2，Δ +0.664；998902 +3.572（next 更差），998903 −1.581（next 更好），998901/4 平；998904 双方均无解。
- next vs myopic：1/1/2，Δ +2.689；主要来自 998903 +5.377；998901 next 可行而 myopic 无解。
- first feasible：998901 `None / 3 / 3`，998902 `1 / 1 / 1`，998903 `0 / 0 / 0`，998904 `None / None / None`。

### Rosenbrock ordinary

- 最终均 4/4；失败 `2 / 2 / 1`；无无解。
- next vs two：1/1/2，Δ −0.005；998902 +0.579（next 稍差），998903 −0.601（稍好），其余平。
- next vs myopic：1/1/2，Δ +0.860；998901 +4.041 为反向差。
- first feasible 全为 0。

### Rosenbrock sparse

- 最终 `3 / 3 / 3`；失败 `15 / 13 / 15`；无解轨迹各 1；冷启动各 1。
- next vs two：1/0/3，Δ −100.396；998904 −301.187 是唯一非平，998901 双方无解，998902/3 平。
- next vs myopic：1/0/3，Δ −36.746；同样由 998904 −110.238 主导。
- first feasible：998901 `None / None / None`，998902 `3 / 3 / 3`，998903 `0 / 0 / 0`，998904 `0 / 0 / 0`。

## 7. 候选一致性与单步退化

- `future_pool_checks`：smoke 4、full 80；每个 next_pool trial 5 次（6 次在线，最后一次 depth=1 不检查）。
- 原始池、过滤后池、候选列表和上一轮 `selected_future_support` 的对应由 `--task verify` 只读核验；full verify 状态通过，**候选池不一致记录 0**。
- 这是程序正确性验收，不是优化收益指标；100% 一致不能当作方法胜出。
- 最后一步：`16/16` 的 next_pool 轨迹末尾 `depth=1`，`fantasy_head_fits_started=0`；two_step 同样 16/16。
- 无模型覆盖：三臂均 0；没有伪造概率或补造检查记录。
- 幻想拟合：two_step `5,604`、next_pool `5,594`，均 started=completed、回退 0；fantasy 总耗时分别 101.9 s、114.7 s。

## 8. 判读

**候选池完全一致这一正确性目标已经达到；但它只带来有限、任务间不一致的性能变化，不能宣称两步前瞻已经优越。**

- 主比较 5/3/8，最终可行相同，失败只少 1；目标差由单个稀疏种子主导，Branin sparse 还有 +3.572 的反向差。
- 相对 myopic 的 4/2/10、14 vs 13 可行、30 vs 35 失败是有限的可行性改善；但 two_step 本身也优于 myopic，因此不能把这部分全部归因于候选支持修复。
- 若把 next_pool 的候选一致性称为成功，它成功的是“实现一致性”，不是“信息价值已被证明”。
- 仍需单独检验数值积分精度与模型误差；本轮没有提高求积节点、未加风险惩罚、未增加覆盖，也没有解决这些近似。
- 4 个种子/任务、每臂 6 次在线，仅探索性；任何单种子差值都不是统计显著。

## 9. 文件与上传

- `smoke/`、`full/`：完整原始输出（plan、source_receipt、report、progress、evaluations、manifest、shared_initial、trials、raw_pool/protected_pool/future_supports）。
- `logs/`：预检、smoke/full 控制台、verify、测试与 compileall 日志及退出码。
- `analysis_summary.json`：机器可读汇总；`environment.txt`、`fixed_sha.txt`、`commands.txt`、`instructions.txt`、`protocol.txt`。
- `README.md`：本说明；`.gitattributes`：`* -text`。
- 结果分支：`results/pr70-next-pool-20261106`；结果 HEAD 见分支提交。

## 10. 限制

- 4 个自定义数学任务、4 种子、6 次在线；不是飞机/RCAIDE 验证。
- 共识候选池一致只保证下一步候选可生成，不保证未来后验重选或滚动重规划会选同一第二动作。
- 冻结参数幻想不是未来重拟合的精确后验；不能称精确动态规划或安全保证。
- 未混合不同任务尺度计算总目标均值；单种子差值不作为显著性结论。
