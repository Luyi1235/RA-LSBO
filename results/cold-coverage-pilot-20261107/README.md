# PR74：冷启动覆盖探索的匹配对照

固定代码 SHA：`491dbf6f1fd340864f5b5332bc2b75d75e9a6a9e`  
代码分支：`research/cold-coverage-isolation-20261107`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/74>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本目录是 PR74 的完整原始结果副本：`smoke/`、`full/`、全部日志、测试、环境、协议与 SHA256 清单。没有安装 ZIP、没有下载特征参考、没有读取旧结果标签，也没有修改代码或依赖。

## 1. 执行与完整性

- 28 项 `test_cold_coverage_core.py` 通过，0 skip。
- 12 项 `test_cold_coverage_runtime.py` 通过，0 skip；原生测试使用合成观测，可能拟合 GP/VAE，不是正式在线实验，也不是零计算测试。
- `compileall` 与 `--help` 均为 exit 0。
- smoke：72 calls = 48 initial + 24 online，12 trials，`COMPLETE_SMOKE_TECHNICAL_ONLY`；verify 81 files，`initially_cold_groups=1`，`initially_warm_matched_paths_verified=3`，manifest `68e87a90...430b`。
- full：960 calls = 384 initial + 576 online，96 trials，`COMPLETE_COLD_COVERAGE_PILOT`，577.0 s；verify 613 files，`initially_cold_groups=13`，`initially_warm_matched_paths_verified=19`，manifest `7e5e86fc...25f6`。
- 每臂 32 trials、每组恰好 6 次在线评价；每 case/seed 三臂共享同样的 12 个初始评价。
- 新增飞机 / RCAIDE：**0 / 0**；幻想拟合：**0**；未知/未结算：**0 / 0**（960 started / 960 settled）。
- 工作机 smoke+full = **1,032** 次数学评价；CI 的 72 次另记，未重复 `--profile ci`。

真实历史 GP/VAE 更新正常执行；原程序没有独立的 VAE 训练计数字段，本报告不补造该数字。计划覆盖调用会跳过 GP 后验拟合，因此“同真实调用数”不等于“同 GP 计算成本”。

## 2. 三个臂的总览

| arm | 最终可行 /32 | 无解 | 冷启动发现 | 实际失败 /960 | scheduled | model-free | degenerate | 提议耗时 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cei_myopic` | 21 | 11 | 2 | 123 | 0 | 0 | 0 | 201.3 |
| `cei_myopic_cold_coverage` | **25** | **7** | **6** | **115** | 22 | 22 | 0 | 181.5 |
| `soft_gp_eligible` | 24 | 8 | 5 | 110 | 22 | 31 | 9 | 179.8 |

- 覆盖臂比 myopic 多恢复 4 个最终可行组（25 vs 21），少 8 次实际失败（115 vs 123）。
- `soft_gp` 仍少 5 次失败（110），但最终可行比覆盖臂少 1（24 vs 25），并使用了 9 次 degenerate coverage。
- 覆盖臂的 22 次 scheduled coverage 全部发生在 cold 状态；myopic 为 0。

## 3. 分层：initial_cold / initial_warm

| 分层 | 组数 | 臂 | 最终可行 | 无解 | 发现 | 实际失败 | scheduled | degenerate | 耗时 s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cold | 13 | myopic | 2 | 11 | 2 | 70 | 0 | 0 | 87.7 |
| cold | 13 | coverage | 6 | 7 | 6 | 62 | 22 | 0 | 67.1 |
| cold | 13 | soft_gp | 5 | 8 | 5 | 62 | 22 | 9 | 66.5 |
| warm | 19 | myopic | 19 | 0 | 0 | 53 | 0 | 0 | 113.6 |
| warm | 19 | coverage | 19 | 0 | 0 | 53 | 0 | 0 | 114.4 |
| warm | 19 | soft_gp | 19 | 0 | 0 | 48 | 0 | 0 | 113.3 |

- **warm 匹配核验：19/19 组完全一致**；coverage vs myopic 的 W/L/T 为 **0/0/19**，目标差均值 0，路径改变 0，scheduled coverage 0。
- cold 分母只有 13（Branin sparse 8，Rosenbrock sparse 5），必须按小样本报告，不能扩种或事后筛选。

## 4. 主比较：coverage vs myopic

### 总体与分层

| 范围 | W/L/T | 只覆盖臂发现 | 只对照臂发现 | 双方可行 | 双方无解 | 路径改变 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 32 组 | **4/1/27** | 4 | 0 | 2 | 7 | 13 |
| initial_cold 13 组 | **4/1/8** | 4 | 0 | 2 | 7 | 13 |
| initial_warm 19 组 | **0/0/19** | 0 | 0 | 19 | 0 | 0 |

- 全部 32 组：最终可行 **25 vs 21**，无解 **7 vs 11**，发现 **6 vs 2**，实际失败 **115 vs 123**。
- 冷启动分层：发现 **6/13 vs 2/13**；**只覆盖臂发现 4 组，只 myopic 发现 0 组**。
- 唯一“负”的冷启动对是 Branin `999212`：coverage 第 3 次发现、myopic 第 1 次发现，coverage 最终目标差 **+10.686**（更差）；Rosen `999216` 双方都发现，目标差 0。
- warm 层完全一致，说明主改动没有改变已有可行解的轨迹。

### 逐任务

| case | cold 组数 | 臂 | 最终可行 cold / all | 无解 cold / all | 失败 cold / all | scheduled | 发现数 |
|---|---:|---|---:|---:|---:|---:|---:|
| Branin sparse | 8 | myopic | 1 / 9 | 7 / 7 | 44 / 66 | 0 | 1 |
| Branin sparse | 8 | coverage | 4 / 12 | 4 / 4 | 41 / 63 | 14 | 4 |
| Branin sparse | 8 | soft_gp | 3 / 11 | 5 / 5 | 41 / 64 | 14 | 3 |
| Rosenbrock sparse | 5 | myopic | 1 / 12 | 4 / 4 | 26 / 57 | 0 | 1 |
| Rosenbrock sparse | 5 | coverage | 2 / 13 | 3 / 3 | 21 / 52 | 8 | 2 |
| Rosenbrock sparse | 5 | soft_gp | 2 / 13 | 3 / 3 | 21 / 46 | 8 | 2 |

- Branin cold：coverage 4/8 vs myopic 1/8；只覆盖臂发现 3 组（`999204`、`999215`、`999216`）；`999212` 是唯一双方可行且 coverage 目标更差的组。
- Rosen cold：coverage 2/5 vs myopic 1/5；只覆盖臂发现 1 组（`999208`）；`999216` 双方可行且目标相同。
- 两任务不能合并原始目标均值；上表只并列计数。

### 首次可行调用位置

`0` 表示初始已可行；`null` 表示预算内未发现。

| task | cold 组 | arm | 发现数 | 首次调用位置（seed@call） |
|---|---:|---|---:|---|
| Branin sparse | 8 | myopic | 1 | `999212@1`；其余 null |
| Branin sparse | 8 | coverage | 4 | `999204@6`、`999212@3`、`999215@6`、`999216@2`；其余 null |
| Branin sparse | 8 | soft_gp | 3 | `999212@3`、`999215@6`、`999216@2`；其余 null |
| Rosenbrock sparse | 5 | myopic | 1 | `999216@1`；其余 null |
| Rosenbrock sparse | 5 | coverage | 2 | `999208@1`、`999216@2`；其余 null |
| Rosenbrock sparse | 5 | soft_gp | 2 | `999208@1`、`999216@2`；其余 null |

覆盖臂的 4 个“只其发现”中，有 2 个发生在第 **6** 次（最后一次）调用，1 个在第 2 次，1 个在第 1 次；不能把最后一次救回单独当作稳定机制证据。

## 5. 次要比较：与 soft_gp_eligible

| case | cold coverage vs soft W/L/T | cold 发现（cov/soft） | all coverage vs soft W/L/T | all 最终可行（cov/soft） | all 失败（cov/soft） |
|---|---:|---:|---:|---:|---:|
| Branin sparse | 1/2/5 | 4/3 | 5/3/8 | 12/11 | 63/64 |
| Rosenbrock sparse | 0/1/4 | 2/2 | 2/4/10 | 13/13 | 52/46 |

- 覆盖臂的 cold 发现 6 vs soft 5；覆盖臂多出的是 Branin `999204`，soft 未发现。
- 但 soft 整体实际失败更少（110 vs 115），主要来自 warm 层的 48 vs 53；它还有 9 次 degenerate coverage，和覆盖臂的 22 次 scheduled coverage 不是同一机制。
- 覆盖臂没有超过 soft_gp 的总体失败数；不能把 cold 层的一次优势升级为默认方法。

`cei_myopic` vs `soft_gp` 的分层结果：Branin 5/3/8，Rosen 2/5/9；soft 最终 24 vs myopic 21，失败 110 vs 123，仍是更强的旧基线。

## 6. 覆盖动作与成本

- `cei_myopic_cold_coverage`：22 次 scheduled coverage，全部 `cold_before=true`；迭代 0 有 13 次、迭代 5 有 9 次；`model_free_calls=22`；`degenerate_coverage_calls=0`；`model_failure_calls=0`。
- `cei_myopic`：0 次 scheduled/degenerate/model-free。
- `soft_gp_eligible`：22 次 scheduled、31 次 model-free、**9 次 degenerate**（迭代 4 有 7 次、迭代 3 有 2 次）；模型失败 0。
- 覆盖周期保持原有的 5；只在尚无可行解时启用；没有新增风险惩罚、阈值、信息奖励、提前停止或候选生成规则。
- 提议耗时：myopic 201.3 s，coverage 181.5 s，soft 179.8 s。coverage 更快是因为计划覆盖调用跳过 GP 后验拟合；真实调用数相同，GP 成本不同，不能把耗时差解释为算法效率优势。

## 7. 异常与完整性

- 所有 960 次评价均 started=settled；0 未知、0 中断、0 未结算。
- `model_failure_calls=0`；没有注册的幻想数值失败（本实验 fantasy_fits=0）。
- warm 匹配 19/19 通过：路径、最优值曲线、失败次数、首次可行调用位置均与 myopic 相同；覆盖臂在 warm 层没有 scheduled coverage。
- 源 worktree：`D:\LSBO\RA-LSBO-pr74-20261107-164658`，HEAD 等于固定 SHA，`git status --porcelain` 为空。
- 结果 worktree：`D:\LSBO\RA-LSBO-pr74-results-worktree-20261107`；证据目录使用 `* -text`。

## 8. 判读

- **在 13 个初始冷启动组中，覆盖臂发现 6 个，myopic 发现 2 个；只覆盖臂发现 4 个，只 myopic 发现 0 个。** 这是本轮唯一明确的正面信号，但分母只有 13，且 2 个只覆盖发现发生在最后一次调用。
- 覆盖臂最终可行 25 vs 21、失败 115 vs 123；warm 层完全一致，说明收益集中在 cold，而不是改变了已有可行解的搜索。
- 唯一双方可行且 coverage 更差的组（Branin `999212`，+10.686）说明“发现更晚/更差”也可能发生；不能只报成功救回。
- 覆盖臂没有超过 soft_gp：最终 25 vs 24，但失败 115 vs 110；soft 的 degenerate coverage 与覆盖臂的 scheduled coverage 机制不同，不能混为同一种覆盖。
- 这是**诊断性基线**，不是新算法；cold 组只有 8+5=13，不能作为确认性证据，也不能外推到飞机/RCAIDE。
- 计划覆盖只在 cold 的迭代 0/5 触发，说明它确实是一个“无可行解时的后备探索”，而不是普遍探索奖励。

## 9. 文件

- `smoke/`、`full/`：完整原始输出，含 `report.json`、`plan.json`、`source_receipt.json`、`progress.jsonl`、`evaluations.jsonl`、`manifest.json`、`shared_initial/`、`trials/` 及 representation checkpoints。
- `logs/`：28+12 测试、compileall、help、smoke/full 控制台与 verify。
- `analysis_summary.json`、`case_arm_table.csv`、`pair_table.csv`、`cold_seed_table.csv`：机读汇总。
- `integrity_check.json`、`sha256_manifest.txt`、`fixed_sha.txt`、`commands.txt`、`environment.txt`、`environment_packages.txt`、`instructions.txt`、`protocol.txt`。

### 限制

- cold 分母 13（8+5），不是确认性样本；不能扩种或事后筛选。
- 每个 case/seed 只有 6 次在线评价；最后一次救回可能只是预算边界。
- 计划覆盖跳过 GP 后验拟合，真实调用数相同但模型计算成本不同。
- 没有飞机/RCAIDE；没有单独 VAE 计数器；不能把本结果写成工程或安全结论。
