# PR86-C：目标感知的冷启动采集与风险校准诊断

固定代码 SHA：`cfc0db47e2c9161ac6cfac0a4bf7a538e5b0ff5f`  
分支：`research/phase-pr86c-objective-aware-20261111`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/88>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

这是**固定预算的离线/前瞻式采样诊断**，不是在线飞机优化，也不声称任何方法可直接作为默认方法。测试集与训练集隔离；缺失目标值不填 0，也不参与可行目标统计。

## 1. 实际预算

| 项目 | 数值 |
|---|---:|
| 任务 | 10（`legacy_branin_sparse`、`legacy_rosenbrock_sparse`、`tradeoff_s0–s7`） |
| 每任务重复 | 3 |
| case/repeat 单元 | 30 |
| 每单元共享初始锚点 | 12 |
| 方法 | 5（iid_random、boundary、marginal_entropy、constrained_ei、objective_weighted_boundary） |
| 每臂在线评价 | 36 |
| 每单元测试点 | 512 |
| 数学评价总数 | 21,120（360 anchor + 5,400 online + 15,360 test） |
| 实际 trial | 150 |

`full` 状态：`COMPLETE_PR86C_OBJECTIVE_AWARE`，5,290.7 s。  
独立 verify：`VERIFIED_PR86C_OBJECTIVE_AWARE`，36 files、150 trials、21,120 calls，`new_math_calls=0`、`new_gp_fits=0`。  
Analysis：`COMPLETE_PR86C_ANALYSIS`，450 outcome rows、120 paired rows、1,350 risk rows。

## 2. 计算账本

| 项目 | 数值 |
|---|---:|
| GP heads started/completed/errors | 11,100 / 11,100 / 0 |
| GP kernels started/completed/errors | 88,800 / 88,800 / 0 |
| Kernel candidates rejected | 0 |
| Objective GP heads started/completed/errors | 4,286 / 4,286 / 0 |
| Objective GP kernels started/completed/errors | 34,288 / 34,288 / 0 |
| Risk refits | 5,550 |
| Math started/settled | 21,120 / 21,120 |
| VAE / fantasy / aircraft / RCAIDE | 0 / 0 / 0 / 0 |

5,550 次 risk refit 是构建风险模型的实际重拟合；11,100 GP heads 包含 region/resource 头；4,286 个 objective GP heads 是目标模型。没有 VAE 训练或飞机评价。

## 3. 可行发现率（30 个 task/repeat 单元）

| n | iid_random | boundary | marginal_entropy | constrained_ei | objective_weighted_boundary |
|---:|---:|---:|---:|---:|---:|
| 12 | 21/30 | 21/30 | 21/30 | 21/30 | 21/30 |
| 24 | 27/30 | 25/30 | 25/30 | 24/30 | 23/30 |
| 48 | **29/30** | 27/30 | 28/30 | 27/30 | 27/30 |

- n12 五方法完全相同，因为它们只共享 12 个初始锚点，尚未做在线选择。
- n24 objective_weighted_boundary 的可行发现最差（23/30）；n48 为 27/30，低于 iid 的 29/30。
- 目标感知方法没有提高“找到可行设计”的比例；它改变的是找到后的目标质量。

## 4. n48 各任务最优已观测可行目标

每格是该任务 3 个重复的中位数；`—` 表示该任务 3 个重复都没有精确可行目标。不同任务的目标尺度不同，不跨任务合并。

| 任务 | iid | boundary | marginal | constrained_ei | objective_weighted_boundary |
|---|---:|---:|---:|---:|---:|
| Branin sparse | 7.562 | 5.814 | 1.755 | 2.169 | 9.035 |
| Rosenbrock sparse | 25.633 | 26.936 | 20.449 | **4.516** | 20.449 |
| tradeoff_s0 | 0.839 | **0.272** | 1.054 | **0.272** | 0.301 |
| tradeoff_s1 | 1.076 | **0.207** | 0.572 | **0.207** | **0.207** |
| tradeoff_s2 | 2.914 | 1.151 | **0.527** | 1.151 | 1.151 |
| tradeoff_s3 | 0.660 | 1.053 | 1.107 | **0.488** | 1.053 |
| tradeoff_s4 | 3.345 | 2.891 | 6.355 | 2.836 | **1.323** |
| tradeoff_s5 | 4.175 | 3.358 | 6.378 | 6.343 | 6.343 |
| tradeoff_s6 | 7.336 | 25.243 | 25.243 | 11.385 | 25.243 |
| tradeoff_s7 | 20.329 | 20.329 | **12.644** | 17.770 | 17.770 |

objective_weighted_boundary 只在 `tradeoff_s4` 明显最好；在 Branin sparse、s6 明显更差；其余任务与 boundary 相同或略差。不存在跨任务稳定的目标优势。

## 5. 精确目标标签数量

每单元平均 `exact_objective_count`（30 个单元）：

| n | iid | boundary | marginal | constrained_ei | objective_weighted_boundary |
|---:|---:|---:|---:|---:|---:|
| 12 | 1.33 | 1.33 | 1.33 | 1.33 | 1.33 |
| 24 | 2.73 | 2.37 | 2.37 | 3.37 | 2.40 |
| 48 | 5.37 | 5.30 | 5.83 | **5.77** | 4.57 |

目标感知方法在 n48 获得的精确目标标签最少（4.57/单元），iid 与 constrained_ei 最高。它更集中在“目标加权边界”，但这没有转化为更多可行发现。

## 6. 配对胜负（objective_weighted_boundary 为候选）

每格为 30 个单元中 `W/L/T`；仅当双方都有精确可行目标时计算目标差；缺失不填 0。

| n | vs iid_random | vs boundary | vs marginal_entropy | vs constrained_ei |
|---:|---|---|---|---|
| 12 | 0/0/30 | 0/0/30 | 0/0/30 | 0/0/30 |
| 24 | 8/11/11 | 5/8/17 | **11/3/16** | 3/10/17 |
| 48 | **15/7/8** | 6/5/19 | **12/5/13** | 4/9/17 |

- n48 objective_weighted_boundary 明显优于 iid（15/7）与 marginal_entropy（12/5），对 boundary 仅 6/5，弱于 constrained_ei（4/9）。
- n24 没有稳定优势；对 iid 还是 8/11。
- 因此它不能替代 constrained_ei，也不能作为默认采集策略。

## 7. 风险预测（joint feasibility，n48 六任务均值）

| 方法 | Brier | Brier Skill | AUROC | AUPRC | ECE | False-feasible |
|---|---:|---:|---:|---:|---:|---:|
| iid_random | .0819 | .0159 | **.8331** | **.5106** | .1126 | .0514 |
| boundary | .0826 | −.0048 | .7890 | .4798 | .1152 | **.0272** |
| marginal_entropy | .0873 | −.0514 | .7917 | .4204 | .1137 | .0362 |
| constrained_ei | **.0751** | **.0904** | .7806 | .4632 | **.0886** | .0283 |
| objective_weighted_boundary | .0759 | .0838 | .7685 | .4326 | .0879 | .0266 |

- iid 的 AUROC/AUPRC 最高；objective_weighted_boundary 的排序能力弱于 iid。
- constrained_ei 与 objective_weighted_boundary 的 Brier/ECE 较好，false-feasible 较低；但这只是同一风险模型的路径差异，不是校准算法差异。
- n12 五方法风险指标完全相同（共享锚点）；n24 才出现差异。

## 8. 结论

1. **没有可行发现优势**：objective_weighted_boundary 在 n48 为 27/30，低于 iid 的 29/30；n24 为 23/30，最差。
2. **目标质量只在部分任务/部分对照上有优势**：n48 对 iid 15/7、对 marginal 12/5，但对 constrained_ei 4/9；per-task 结果方向不一致。
3. **精确标签更少**：n48 为 4.57/单元，低于 iid 5.37、constrained 5.77；不能用“目标更好”掩盖标签效率下降。
4. **风险排序没有同步改善**：AUROC .7685 低于 iid .8331；Brier/ECE 略好于 iid，但不构成机制证据。
5. 缺失目标保持缺失，未填 0；双方无解的目标差不计算。  
6. 这是 10 任务 ×3 重复的固定预算诊断，不是在线飞机实验，也不是新默认方法的证据。

## 9. 文件与限制

- `full/`：raw units、`progress.jsonl`（LFS）、`report.json`、`plan.json`、`source_receipt.json`、`manifest.json`、`evaluations.jsonl`、`environment.json`。
- `analysis/`：`analysis.json`、`objective_learning_curves.csv`、`paired_objective.csv`、`risk_learning_curves.csv`、`manifest.json`。
- `logs/`：plan、full、verify、analysis 控制台日志。
- `SHA256SUMS.txt`、`.gitattributes`（`progress.jsonl` LFS，其余 `* -text`）。

限制：三个重复只是固定种子的重复，不是独立实验室复现；目标值跨任务不可比，未做跨任务汇总；`objective_weighted_boundary` 只在 48 次观测内测试；没有飞机/RCAIDE；没有安全或收敛保证。
