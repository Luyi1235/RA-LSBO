# PR69：PR68 决策只读复核与归档

固定审计代码：`fd56186bb75b29b0df96bb2c27d67791c5f0d73b`  
固定输入结果 HEAD：`2823c6204de7da0ffa537548e0dbbe411e41e22a`  
输入结果对应原实验代码：`21a1267ee907774e6958a12539da98c0840cc1d9`  
原 PR：<https://github.com/Luyi1235/RA-LSBO/pull/68>  
审计 PR：<https://github.com/Luyi1235/RA-LSBO/pull/69>

本目录是 PR69 只读审计的完整归档。没有重新运行 PR68，也没有新增 GP/VAE/数学/飞机/RCAIDE 调用。

## 1. 完成状态与零新增调用

- 状态：`COMPLETE_READ_ONLY_PR68_DECISION_AUDIT`
- 新增数学评价：`0`
- 新增 GP 拟合：`0`
- 新增 VAE 拟合：`0`
- 飞机调用：`0`
- RCAIDE 调用：`0`
- 原 full 证据复核：`405` 个文件、`64` 条 trial、`192` 次初始评价、`384` 次在线评价、`576 started / 576 settled`
- 原 full manifest：`29b39c0cbcfc6dcad1a46207271c1de8647cbb02b8227f1f54f1a2d83312d6eb`
- 审计只读：证据 worktree 与代码 worktree 的 `git status --porcelain` 均为空；原结果 HEAD 未变化。

输出 `audit/` 包含 `report.json`、`decisions.json`、`manifest.json`；`logs/` 在审计目录之外，未插入已封存输出。

## 2. 测试与入口

- `python -S -m unittest discover -s tests -p "test_pr69*.py" -v`：**24 tests passed**，0 skip。
- `compileall`：exit 0。
- `--help`：exit 0。
- 正式命令：`python -S run_pr69_decision_audit.py --evidence-root <结果worktree根> --expected-sha fd56186... --output <外部audit目录>`：exit 0，状态如上。
- 解释器：`Rcaide_vsp` Python 3.9.25；`-S` 禁用 site 初始化，审计只依赖标准库，没有导入科学库或替换假模块。

## 3. 逐任务/逐臂汇总

`depth2` 为该 case/arm 的两步决策数；`改选`为相对同一历史状态下 own-myopic 最优的改选次数；`删点敏感`为四节点删一后首动作改变的决策数；`平均牺牲`为全部 depth2 决策的即时收益牺牲比例；`最终可行`、`在线失败`为该臂在该 case 的 4 个种子合计。

| case | arm | depth2 | 改选 | 删点敏感 | 平均牺牲 | 最终可行 | 在线失败 |
|---|---|---:|---:|---:|---:|---:|---:|
| Branin ordinary | `cei_myopic` | 0 | 0 | 0 | — | 4/4 | 2 |
| Branin ordinary | `cei_two_step` | 20 | 7 | 10 | 5.51% | 4/4 | 3 |
| Branin ordinary | `cei_two_step_no_abort_update` | 20 | 7 | 10 | 5.51% | 4/4 | 3 |
| Branin ordinary | `soft_gp_eligible` | 0 | 0 | 0 | — | 4/4 | 1 |
| Branin sparse | `cei_myopic` | 0 | 0 | 0 | — | 2/4 | 15 |
| Branin sparse | `cei_two_step` | 20 | 6 | 8 | 3.16% | 2/4 | 17 |
| Branin sparse | `cei_two_step_no_abort_update` | 20 | 12 | 10 | 5.84% | 2/4 | 18 |
| Branin sparse | `soft_gp_eligible` | 0 | 0 | 0 | — | 2/4 | 17 |
| Rosenbrock ordinary | `cei_myopic` | 0 | 0 | 0 | — | 4/4 | 4 |
| Rosenbrock ordinary | `cei_two_step` | 20 | 3 | 8 | 0.64% | 4/4 | 4 |
| Rosenbrock ordinary | `cei_two_step_no_abort_update` | 20 | 3 | 8 | 0.64% | 4/4 | 4 |
| Rosenbrock ordinary | `soft_gp_eligible` | 0 | 0 | 0 | — | 4/4 | 2 |
| Rosenbrock sparse | `cei_myopic` | 0 | 0 | 0 | — | 2/4 | 15 |
| Rosenbrock sparse | `cei_two_step` | 20 | 8 | 7 | 3.97% | 2/4 | 20 |
| Rosenbrock sparse | `cei_two_step_no_abort_update` | 20 | 10 | 7 | 7.54% | 2/4 | 20 |
| Rosenbrock sparse | `soft_gp_eligible` | 0 | 0 | 0 | — | 3/4 | 14 |
| **合计** | `cei_two_step` | **80** | **24** | **33** | **3.32%** | **12/16** | **44** |
| **合计** | `cei_two_step_no_abort_update` | **80** | **32** | **35** | **4.88%** | **12/16** | **45** |

- 两步臂的 24 次改选集中在 Branin ordinary（7）、Branin sparse（6）、Rosenbrock sparse（8）；Rosenbrock ordinary 只有 3 次。
- 在 24 次改选中，即时收益牺牲均值 **11.07%**、中位数 **7.48%**、最大 **32.25%**；`no_abort` 32 次改选，均值 **12.21%**、最大 **47.99%**。
- 删点敏感在两步臂为 **33/80 = 41.3%**，`no_abort` 为 **35/80 = 43.8%**。这只是 3-of-4 节点敏感性，不是求积收敛检验、误差界或置信区间。
- **下一轮候选池精确保持质量为 0**：所有 case/arm 的 `next_pool_retention_mass=0`，即没有一次被选中的幻想第二动作精确出现在实际下一轮池中。这不等于附近没有替代点，也不能据此推断性能差异的因果原因。
- `own_myopic_overrides` 是保存的 four-node 评分重算后的改选，不是新在线结果；没有新增评价。

## 4. 主对比：两步 vs 自身单步（逐种子）

负值 = 两步目标更好；正值 = 两步目标更差；`无`=双方没有同时可行的最终目标。仅列 `cei_two_step` 相对 `cei_myopic` 的最终目标差，以及相对 `no_abort` 的差（后者全部为 0）。

| case | seed | 两步 − myopic | 两步 − no_abort |
|---|---:|---:|---:|
| Branin ordinary | 998801 | 0 | 0 |
| Branin ordinary | 998802 | 0 | 0 |
| Branin ordinary | 998803 | 0 | 0 |
| Branin ordinary | 998804 | **−0.6316** | 0 |
| Branin sparse | 998801 | 0 | 0 |
| Branin sparse | 998802 | **+5.0239** | 0 |
| Branin sparse | 998803 | 无（双方无解） | 0 |
| Branin sparse | 998804 | 无（双方无解） | 0 |
| Rosenbrock ordinary | 998801 | 0 | 0 |
| Rosenbrock ordinary | 998802 | 0 | 0 |
| Rosenbrock ordinary | 998803 | 0 | 0 |
| Rosenbrock ordinary | 998804 | 0 | 0 |
| Rosenbrock sparse | 998801 | 无（双方无解） | 0 |
| Rosenbrock sparse | 998802 | **+2.3437** | 0 |
| Rosenbrock sparse | 998803 | 无（双方无解） | 0 |
| Rosenbrock sparse | 998804 | **+64.2712** | 0 |

- 逐种子胜/负/平：**1/3/12**（负=两步更差）。
- 唯一显著负差是 Branin ordinary 998804 的 −0.6316；三个正差为 +5.0239、+2.3437、+64.2712。
- **不能把任务内均值写成单种子差**：例如 Branin sparse 的 +5.0239 只属于 998802，998803/998804 双方都无解。
- `no_abort` 相对两步在全部 16 个种子上最终目标差为 0；中止区间 update 没有改变最终目标。

## 5. 同前缀比较

仅当历史 hash、候选池和原始预测全部相同才比较：

| 对比 | 可比较前缀数 | 选择不同 |
|---|---:|---:|
| 两步 vs myopic | 53 | 12 |
| 两步 vs no_abort | 76 | 5 |
| **合计** | **129** | **17** |

- 在同一历史/池/预测下，两步仍有 12/53 次改选；说明保存的幻想评分确实会改变动作，但没有带来最终可行或目标收益。
- 中止信息消融只在 5/76 个同前缀上改变动作；与“0/0/16 的最终胜负”一致，影响很小。

## 6. 风险概率方向纠正

`predicted_failure` 是**失败概率**，不是可行概率。正确方向如下（own selected points，四个 case 合计）：

| arm | 有预测行 | 预测失败概率 ≤0.2：实际失败/可行 | 预测失败概率 ≥0.8：实际失败/可行 |
|---|---:|---:|---:|
| `cei_myopic` | 96 | 7 / 44（n=51） | 24 / 0（n=24） |
| `cei_two_step` | 96 | 8 / 41（n=49） | 24 / 0（n=24） |
| `cei_two_step_no_abort_update` | 96 | 8 / 41（n=49） | 24 / 0（n=24） |
| `soft_gp_eligible` | 83 | 6 / 43（n=49） | 11 / 0（n=11） |

- **PR68 README 把 `p≥0.8` 写成“假可行”是方向错误**；正确解释是：预测失败概率≥0.8 的 24 个点实际全部失败，没有一个是可行点。
- 低风险桶（≤0.2）仍有 7–8 个实际失败，说明低预测失败概率不等于安全。
- 这里只核对了保存的在线选点；不同策略的自适应采样不是受控校准比较，不能据此宣称某个模型更校准。

## 7. 判读

- 两步前瞻确实改变选择：80 次 depth2 决策中 24 次改选；但最终可行总数仍为 12/16，在线失败为 44/96 对 36/96。
- 即时收益牺牲在改选中平均 11.1%，最大 32.2%；Rosenbrock sparse 出现 +64.27 的单种子目标劣化，说明两步估计在稀疏任务上不稳定。
- 删点敏感性 41.3% 表明当前 four-node 评分的动作对节点删除并非稳定；但这不是求积收敛检验，也不能单独证明高精度节点会带来收益。
- 下一轮精确保持 0/80，说明幻想第二动作与实际候选池存在明显代际差；不能据此声称“附近有替代点”或“没有替代点”。
- 中止区间消融：同前缀 5/76 改变动作，但最终目标全部相同；没有可检测收益。
- **结论：PR69 没有发现两步信息价值稳定转化为实际搜索收益的证据；也没有证明所有信息价值方法无效。** 下一步若要继续，应优先解决“候选池代际不一致”和“删点敏感”这两个可复核问题，而不是直接扩大在线种子或增加风险惩罚。

## 8. 对比与解释边界

- 原 PR68 full 证据只读引用固定 HEAD，没有修改其 README、汇总或原始数据。
- 本审计不是新在线算法结果；`immediate_sacrifice_fraction`、删点敏感和 next-pool retention 都是保存决策上的诊断。
- 删点敏感是 3-of-4 节点敏感性，不是高精度求积、误差界、置信区间或收敛性检验。
- 精确 next-pool 保持为 0 不证明附近没有替代候选，也不证明它是性能下降的因果原因。
- 不同策略的自适应选点不能当作模型校准的受控比较；风险方向表只描述已选点。
- 没有飞机/RCAIDE 评价；不能外推到真实飞机或安全保证。

## 9. 文件

- `audit/report.json`：完整逐 case/arm 汇总、同前缀、逐种子配对、风险方向与限制。
- `audit/decisions.json`：288 个决策明细（96 myopic + 96 two-step + 96 no_abort），含改选、牺牲、删点敏感与 next-pool 字段。
- `audit/manifest.json`：report/decisions 的 SHA256（已在本地逐项核对）。
- `logs/`：测试、compileall、help、正式审计的 stdout/stderr/exit 码，环境与命令记录。
- `audit_summary.json`：机器可读的汇总；`environment.txt`：解释器/平台/依赖记录。
- `instructions.txt`：本轮任务指令副本；`sha256_manifest.txt`：全部上传文件 SHA256。
- `.gitattributes`：`* -text`，防止换行转换改变证据字节。

## 10. CI 与未知记录

- 工作机没有 CI 原始日志；无法逐行比对 CI 输出。工作机 `24/24` 测试通过、compileall/--help 为 0、审计状态与 PR69 声明一致，未发现平台相关导入差异。
- 未知/未结算记录：原 full `576 started / 576 settled`，审计新增 0；无 skip、无未捕获异常、无未解释的候选关联失败。
- 原始证据 405 文件、64 trial、576 次历史调用均通过 manifest/账本/初始与在线前缀校验。
