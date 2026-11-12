# PR86-F：相同 48 次评价预算下的排序与幅度组合

固定代码 SHA：`cda1aa24c3b87b6c139d967b09966dbf234e5f37`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/95>  
环境：`Rcaide_vsp` / Python 3.9.25。

本实验比较三个**仅采集规则不同**的方法：`constrained_ei`、`objective_weighted_boundary`（rank/value 组合）与 `direct_weighted_boundary`（用直接预测目标值替换排序分量）。三臂共享 24 个初始锚点、相同的候选池、相同的模型与相同的每臂 24 次在线评价；不含 VAE、幻想、飞机或 RCAIDE 调用。

## 1. 执行与费用

| 项目 | 数值 |
|---|---:|
| 已知任务 | 10 |
| 重复 | 3 |
| 方法 | 3 |
| 轨迹 | 90 |
| 数学评价 | 18,240（720 anchor + 2,160 online + 15,360 held-out） |
| 每臂总观测 | 48（24 初始 +24 在线） |
| 风险 GP heads / kernels | 4,500 / 36,000 |
| 目标 GP heads / kernels | 2,108 / 16,864 |
| 零错误 / 零拒绝 | 是 |
| VAE / 幻想 / 飞机 / RCAIDE | 0 / 0 / 0 / 0 |
| 运行时间 | 2,364.4 s |

独立 verify：36 files、18,240 calls、2,160 decisions、0 new calls/fits，`VERIFIED_PR86F`。  
分析：270 outcome/paired rows、90 task-summary rows，`COMPLETE_PR86F_ANALYSIS`。

## 2. 主比较（feasible-first，W/L/T）

比较是“候选方法 vs 对照”；双方均可行时按精确可行目标比较。

| 比较 | n=24 | n=36 | n=48 |
|---|---:|---:|---:|
| `direct_weighted_boundary` vs `objective_weighted_boundary` | 0/0/30 | 10/4/16 | 10/3/17 |
| `direct_weighted_boundary` vs `constrained_ei` | 0/0/30 | 3/2/25 | 3/3/24 |
| `objective_weighted_boundary` vs `constrained_ei` | 0/0/30 | 4/11/15 | 3/12/15 |

n=24 三臂都还没有在线评价，全部平局是设计结果。

n=48 的任务内平均目标差（候选−对照）中位数：

- direct vs objective：`−0.0316`；任务范围 `−16.99`（Rosenbrock sparse）到 `+4.30`（Branin sparse）。
- direct vs constrained：`0`；任务范围未混合总体均值。
- objective vs constrained：`+0.0582`；中位为正，即组合臂略差。

## 3. 可行发现、目标标签与风险

| n | 可行发现/30 |
|---:|---:|
| 24 | 28 |
| 36 | 28 |
| 48 | 30 |

n=48 精确目标标签数（每臂 90 条轨迹合计）：constrained 6.97、direct 6.13、objective 5.33。  
n=48 联合风险：

| method | Brier | false-feasible | feasible recall |
|---|---:|---:|---:|
| constrained_ei | .0774 | .0324 | .4875 |
| objective_weighted_boundary | .0800 | .0333 | .4533 |
| direct_weighted_boundary | .0821 | .0383 | .4851 |

`constrained_ei` 的 Brier 与假可行率最好，recall 也略高；直接加权臂拿到更多精确目标标签，但联合风险最差。

“无可用目标 GP”阶段计数（总计 720 个决策）：constrained 40、objective 58、direct 44；没有“无可行 incumbent”的阶段。

## 4. 判读

- n=24 三种方法没有在线差异；n=36/48 的差异来自采集排序，而不是初始样本。
- **rank/value 组合没有稳定超过其直接加权消融**：n48 为 10/3/17，任务中位差 −0.0316，方向混合；Rosenbrock sparse 的 −16.99 与 Branin sparse 的 +4.30 同时存在。
- **rank/value 组合也没有超过 constrained EI**：n48 为 3/12/15，约束臂在 12 个任务/重复对上更好。
- 直接加权臂获得更多精确目标标签，但这没有转化为更好的风险 Brier 或假可行率；这是“排序分量是否值得”的负结果，而不是正向证据。
- 可行发现率在 n24/36 均 28/30，n48 30/30；三臂都未显示可行性优势。
- 所有结果都是 10 个已知任务、3 重复的探索性比较；不是新算法胜利，也不涉及飞机/RCAIDE。

## 5. 文件

- `full/`：90 个 `unit.json`、`evaluations.jsonl`（15.9 MB）、`progress.jsonl`（210 MB，Git LFS）、`report.json`、`plan.json`、`source_receipt.json`、`manifest.json`、`environment.json`。
- `analysis/`：`analysis.json`、`learning_curves.csv`、`paired_methods.csv`、`paired_initial_size.csv`、`risk_learning_curves.csv`、manifest。
- `logs/`：plan/full/verify/analyze 完整日志。
- `environment.json`、`fixed_sha.txt`、`instructions.txt`、`SHA256SUMS.txt`。

原始目录未做任何截断或重写；`SHA256SUMS.txt` 记录上传文件校验值。`progress.jsonl` 使用 Git LFS。
