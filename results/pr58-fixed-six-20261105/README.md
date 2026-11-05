# PR58：新种子、真实等六次调用的风险预算策略闭环验证

固定代码：`7d8e5edb0cd9db6bc6525a8f9989591388c71af5`  
父提交：PR57 `916d7ac3cf1d961358b094a9f227dd246c57728c`  
环境：`Rcaide_vsp` / Python 3.9.25。

本分支只上传**新的数学实验产物**和只读汇总；不修改 PR45–PR57 的任何历史证据。

## 实验身份

- 4 个自定义层级数学任务：Branin 普通/稀疏、Rosenbrock 普通/稀疏。
- 24 个全新种子：`997001–997024`（与 PR54 的 995401–995408、PR55 的 996101–996116 不重叠）。
- 每个 case/seed 共享 12 个初始评价；三个算法 `legacy`、`semantic_only`、`budgeted` 各自进行 **6 次真实在线解析评价**，共 2,880 次新解析评价。
- 内部 12 次规划视界、PR54 冻结的 acquisition/预算策略均未改动；六个已观察不可行停止上限等于六次实际调用，因此任何算法都不可能在第六次调用之前被截断。
- 288 个 on-policy trials；每条 trial 都从自己的真实观察历史重新拟合。
- 新增飞机评价 0，RCAIDE 调用 0；GP/VAE 为真实拟合并非零。
- 状态：`COMPLETE_FRESH_FIXED_SIX_ON_POLICY`。

## 运行

| 阶段 | 输出 | 结果 |
|---|---|---|
| 测试 | `tests.log` | 7 passed in 0.06s |
| smoke：`--profile smoke` | `smoke-console.log` | 72 次调用，2 次/算法，`SMOKE_TWO_CALLS_NOT_FULL_STUDY`，exit 0 |
| full：默认 | `full-console-v2.log` | 2,880 次调用，288 trials，exit 0，无 traceback |

运行目录：

- full：`D:\LSBO\RA-LSBO-pr58-fixed-six-full-20261105-v2`
- 完整 `trials/` 与 `shared_initial/`：4 个 ZIP，按 case 分包，见下文
- 原始证据清单：`evidence_manifest.sha256`（1,824 个文件，245,856 B，逐文件 SHA256）

## 结果概览

### 按任务/算法

| case / arm | init 可行 | final 可行 | 冷启动新发现 | 条件均值目标 | 条件中位目标 | 平均在线不可行数 |
|---|---:|---:|---:|---:|---:|---:|
| branin_ordinary / legacy | 24/24 | 24/24 | 0 | 2.496 | 1.818 | 0.708 |
| branin_ordinary / semantic_only | 24/24 | 24/24 | 0 | 2.393 | 1.749 | 0.875 |
| branin_ordinary / budgeted | 24/24 | 24/24 | 0 | 2.393 | 1.749 | 0.833 |
| branin_sparse / legacy | 13/24 | 17/24 | 4 | 8.705 | 6.240 | 3.875 |
| branin_sparse / semantic_only | 13/24 | 19/24 | 6 | 10.658 | 7.581 | 4.042 |
| branin_sparse / budgeted | 13/24 | 19/24 | 6 | 9.707 | 4.544 | 3.750 |
| rosenbrock_ordinary / legacy | 24/24 | 24/24 | 0 | 9.187 | 2.212 | 1.167 |
| rosenbrock_ordinary / semantic_only | 24/24 | 24/24 | 0 | 9.300 | 2.212 | 1.167 |
| rosenbrock_ordinary / budgeted | 24/24 | 24/24 | 0 | 15.719 | 2.212 | 1.167 |
| rosenbrock_sparse / legacy | 17/24 | 21/24 | 4 | 90.097 | 22.349 | 3.542 |
| rosenbrock_sparse / semantic_only | 17/24 | 21/24 | 4 | 91.404 | 22.349 | 3.583 |
| rosenbrock_sparse / budgeted | 17/24 | 21/24 | 4 | 87.893 | 9.675 | 3.542 |

条件目标只在最终存在已知可行解的种子上统计；Rosenbrock 的均值受少数极端值影响，必须与中位数一起看。

### 配对主指标（feasible-first wins/losses/ties）

| case | 对比 | W/L/T | 仅候选可行 / 仅对照可行 | 改变选点路径 | 平均目标差 | 中位目标差 | 平均不可行数差 |
|---|---|---:|---:|---:|---:|---:|---:|
| branin_ordinary | budgeted vs legacy | 1/1/22 | 0/0 | 24/24 | −0.103 | 0 | +0.125 |
| branin_ordinary | budgeted vs semantic | 0/0/24 | 0/0 | 1/24 | 0 | 0 | −0.042 |
| branin_sparse | budgeted vs legacy | 5/1/18 | 2/0 | 21/24 | −0.698 | 0 | −0.125 |
| branin_sparse | budgeted vs semantic | 3/1/20 | 0/0 | 9/24 | −0.951 | 0 | −0.292 |
| rosenbrock_ordinary | budgeted vs legacy | 3/2/19 | 0/0 | 23/24 | +6.533 | 0 | 0 |
| rosenbrock_ordinary | budgeted vs semantic | 0/1/23 | 0/0 | 3/24 | +6.419 | 0 | 0 |
| rosenbrock_sparse | budgeted vs legacy | 5/4/15 | 0/0 | 21/24 | −2.204 | 0 | 0 |
| rosenbrock_sparse | budgeted vs semantic | 3/3/18 | 0/0 | 11/24 | −3.511 | 0 | −0.042 |

合并：

- **budgeted vs legacy：14W / 8L / 74T**，2 个仅 budgeted 可行、0 个仅 legacy 可行；96 个配对种子中 89 个选点路径改变。
- **budgeted vs semantic_only：6W / 5L / 85T**，0/1 仅一方可行；只有 24/96 个路径改变。
- **semantic_only vs legacy：9W / 4L / 83T**，2/0 仅一方可行；87/96 个路径改变。

目标差的均值被少数极端种子支配：所有 12 个配对的中位差都是 **0**；Rosenbrock 普通任务 budgeted−legacy 的均值 +6.53 中，最大单种子差 +154.07，24 个种子里 19 个完全相同。不能把均值差解读为稳定改进。

## 风险门控与回退

- budgeted 的 576 个在线风险 receipt 中，**47 个（8.2%）发生回退**（`no_candidate_below_allowance_min_predicted_failure`）。
- 回退全部集中在稀疏任务：Branin 24/144（16.7%），Rosenbrock 23/144（16.0%）；普通任务 0 次。
- 回退后的实际不可行：**42/47（89.4%）**（Branin 23/24，Rosenbrock 19/23）。
- 预测不可行概率 vs 实际不可行率（仅模型覆盖的 receipt）：
  - Branin 普通：0.131 vs 0.139；
  - Branin 稀疏：0.390 vs 0.561；
  - Rosenbrock 普通：0.167 vs 0.194；
  - Rosenbrock 稀疏：0.320 vs 0.544。

因此预算门控没有降低总体实际不可行数：budgeted 与 legacy 的平均差为 0，且稀疏任务中门控在 16% 的步骤找不到低于允许值的候选，只能回退，而回退选择几乎全部失败。边际 GP 概率乘积本身偏乐观，不构成物理可靠性的校准，更不是安全保证。

## 判读限制

- 这是看到 PR54–PR57 后冻结的新种子、六次调用设计，属于**探索性机制验证**，不是预注册确认性试验。
- 仅自定义层级 Branin/Rosenbrock，不是标准 benchmark 复刻，也不是飞机/RCAIDE/飞行器验证。
- 每个算法确实做了相同次数的真实评价，但选的是不同设计、用各自历史重新拟合；这是 on-policy 比较，不是同点反事实。
- legacy 的 completion factor 在 PR45 飞机数据上是单类，在本数学任务上不是；因此 legacy vs semantic 是策略变化，不是普适概率定理。
- 六次失败上限不能被“在第六次之前截断”；这只保证评价次数相同，不保证安全。
- 140 km 任务独立稀疏资格和 RCAIDE 验证仍未建立。

## 文件与哈希

| 文件 | SHA256 / 说明 |
|---|---|
| `fixed_six_onpolicy_report.json` | `67a6a965e090af77f0fea4f2cbe0b47e5a826b981a401d8a649a73574f1564ff` |
| `plan.json` | `c1610a8b5ba844bf294d3c783a137d29688011ed6d0a517127477aec54be1439` |
| `source_receipt.json` | `9daca178cd6e61d48c2289dab371901f6fbca72819dd570c475a83d4785c3644` |
| `progress.jsonl` | `4bd50639eb93fc2240571c9914735371ba61e0651a526d0d05d51bfd9f7c096c` |
| `evidence_manifest.sha256` | `1673342c7f185d088e4c6aac832fed2408147c75a3a1613efc13e253b15bfefe` |
| `trials_branin_ordinary.zip` | `cde00266090d7df0d7a5de39357fc8f15ff7003e376e7311458b37049ee2861f` |
| `trials_branin_sparse.zip` | `06c3d45ddfcc897dfa53d68ede14f5687437d652f72cc86bf2f2ed74011029c7` |
| `trials_rosenbrock_ordinary.zip` | `5ccc105f9a962faaf9802e84040f98b02ee02915dd7e3f6894c810ebc10b5ec3` |
| `trials_rosenbrock_sparse.zip` | `b2180bddaa6f8728054f486c0080867cde19d0aade7afdddebb5ed8015befb66` |
| `smoke-console.log` | `a3efb572f80350c72d181246cf6efdbf49bd3174a84ae33e1e242b1d9bec8c24` |
| `full-console-v2.log` | `bbe28dd56b66f0d103da1a4ae5f1617e20b8683e876c500c738d16b1c533865c` |
| `tests.log` | `5fc6a184cef674ea791ce5bcd07164c4cf7e0866eff0932d82be5420a00f7dc5` |

`evidence_manifest.sha256` 列出 `trials/` 与 `shared_initial/` 全部 1,824 个文件的相对路径、大小无关的 SHA256；四个 ZIP 已逐一解压回读并与清单校验一致（每个 ZIP 456 个文件）。
