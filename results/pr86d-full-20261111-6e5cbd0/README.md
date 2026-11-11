# PR86-D：初始样本 12 vs 24 的嵌套配对消融

固定代码 SHA：`6e5cbd0bafc25cf4e36f9a5675401d6991165416`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/93>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本实验只改变**初始共享样本分配**：`n0=12` 与 `n0=24`，两者每臂总观测均为 48；在线评价分别为 36 与 24。其余算法、种子、候选池、测试集和风险/目标 GP 规则完全一致。不是新采集策略，也不是飞机/RCAIDE 结果。

## 1. 预算与完整性

- 10 个数学任务 × 3 个重复 × 5 种方法 × 2 个 cohort = **300 trials / 30 单元**。
- 12 点 cohort 只看到共享 24 个 anchor 的前 12 个；13–24 的标签被 mask，未用于拟合或采集。
- 25,080 次数学评价：720 shared anchors + 9,000 online + 15,360 held-out。
- 9,300 次 risk refit；18,600 risk GP heads / 148,800 kernels，0 errors/rejects。
- objective GP：7,895 heads / 63,160 kernels，0 errors/rejects；按精确标签单独计费。
- verify：`VERIFIED_PR86D_INITIAL_SAMPLE_ABLATION`，36 files，300 trials，25,080 calls，0 new calls/fits；`nested_anchor_masking_verified=true`、`test_labels_fit_isolated=true`、`objective_missing_not_imputed=true`。
- analysis：`COMPLETE_PR86D_ANALYSIS`，300 initial rows、1,050 objective rows、150 initial-size pairs、180 method pairs、3,150 risk rows。
- 未结算/未知：0；VAE/fantasy/aircraft/RCAIDE：0/0/0/0；源 worktree 干净。

## 2. 初始样本量对可训练性的影响

| cohort | 可训练 objective GP | 精确 objective 标签均值 | 初始可行单元 | 首次 objective GP 拟合中位 n | 首次可行中位 n |
|---|---:|---:|---:|---:|---:|
| n0=12 | **11/30 (36.7%)** | 1.37 | 20/30 | 13–16.5 | 12 |
| n0=24 | **25/30 (83.3%)** | 2.97 | 30/30 | 24 | 24 |

- 12 个初始点只产生 0–2 个精确可行目标；多数单元 objective GP 因少于 2 个精确标签而不可训练（`objective_gp_single_label_prior=false`）。
- 24 个初始点在 30/30 单元至少有一个可行点，25/30 达到 objective GP 最小训练量。
- 这是初始样本“+12 个真实观测”的直接结果，不是算法差异。

## 3. n=24/36/48 的可行发现率

| cohort | method | n24 | n36 | n48 |
|---|---|---:|---:|---:|
| n0=12 | iid_random | .833 | 1.000 | 1.000 |
| n0=12 | boundary | .833 | .900 | .933 |
| n0=12 | marginal_entropy | .767 | .900 | .967 |
| n0=12 | constrained_ei | .833 | .900 | .933 |
| n0=12 | objective_weighted_boundary | .833 | .900 | .933 |
| n0=24 | iid_random | 1.000 | 1.000 | 1.000 |
| n0=24 | boundary | 1.000 | 1.000 | 1.000 |
| n0=24 | marginal_entropy | 1.000 | 1.000 | 1.000 |
| n0=24 | constrained_ei | 1.000 | 1.000 | 1.000 |
| n0=24 | objective_weighted_boundary | 1.000 | 1.000 | 1.000 |

n0=24 在 n24 已全部 30/30 可行；n0=12 在 n24 仍有 5/30 未发现可行点。

## 4. 精确可行目标数量（每单元均值）

| cohort | iid | boundary | marginal | constrained_ei | objective_weighted_boundary |
|---|---:|---:|---:|---:|---:|
| n0=12, n48 | 4.47 | 5.13 | 3.73 | **6.53** | 5.20 |
| n0=24, n48 | 4.77 | 5.93 | 4.77 | **6.73** | 5.30 |

- 两种 cohort 下 `constrained_ei` 都收集到最多精确可行目标；`objective_weighted_boundary` 不是最多。
- n0=24 只把每方法 n48 精确可行目标提高约 0.1–0.8 个。

## 5. 初始样本配对：n0=24 vs n0=12（最终 n48）

W/L/T 为 n0=24 相对 n0=12；Δ 仅在双方都有精确可行目标时计算，负值表示 n0=24 更好。

| method | W/L/T | 双方可行 | Δ 均值 | Δ 中位数 | 精确标签差 |
|---|---:|---:|---:|---:|---:|
| iid_random | 8/3/19 | 30 | −3.136 | 0 | +0.40 |
| boundary | 13/8/9 | 28 | −0.555 | 0 | +1.03 |
| marginal_entropy | **15/9/6** | 29 | −3.963 | 0 | +0.70 |
| constrained_ei | 9/6/15 | 28 | −0.974 | 0 | +0.47 |
| objective_weighted_boundary | 13/8/9 | 28 | −0.960 | 0 | +0.43 |

n0=24 多数方法胜多负少，但中位差均为 0；改善集中在少数长尾任务，且精确标签只多 0.4–1.0 个。

## 6. 五种方法的最终配对（n48，objective_weighted_boundary 为候选）

| cohort | 对照 | W/L/T | 双方可行 | Δ 均值 | Δ 中位数 |
|---|---|---:|---:|---:|---:|
| n0=12 | iid_random | 12/13/5 | 28 | −3.064 | 0 |
| n0=12 | boundary | 9/1/20 | 28 | −0.771 | 0 |
| n0=12 | constrained_ei | 3/8/19 | 28 | +0.171 | 0 |
| n0=24 | iid_random | 13/8/9 | 30 | −0.618 | 0 |
| n0=24 | boundary | 11/3/16 | 30 | −1.098 | 0 |
| n0=24 | constrained_ei | 3/9/18 | 30 | +0.172 | 0 |

- 对 iid 在 n0=24 为 13/8/9，在 n0=12 为 12/13/5；方向不稳定。
- 对 boundary 明显占优（11/3/16、9/1/20），但对 constrained_ei 明显落后（3/9/18、3/8/19）。
- 大量平局（中位差 0）说明多数任务最终目标相同，差异集中在少数任务。

## 7. 风险预测（joint，n48，30 单元均值）

| cohort | method | Brier | AUROC | False-feasible | Recall |
|---|---|---:|---:|---:|---:|
| n0=12 | iid_random | .0823 | **.8448** | .0545 | **.5572** |
| n0=12 | boundary | .0878 | .8035 | .0305 | .4465 |
| n0=12 | marginal_entropy | .0973 | .7977 | .0334 | .3892 |
| n0=12 | constrained_ei | .0843 | .8058 | .0359 | .4533 |
| n0=12 | objective_weighted_boundary | **.0779** | .8008 | .0340 | .4092 |
| n0=24 | iid_random | .0788 | **.8550** | .0517 | **.5737** |
| n0=24 | boundary | **.0721** | .8432 | .0275 | .5481 |
| n0=24 | marginal_entropy | .0736 | .8309 | .0308 | .4794 |
| n0=24 | constrained_ei | .0776 | .8359 | .0432 | .5504 |
| n0=24 | objective_weighted_boundary | .0727 | .8393 | .0416 | .5449 |

- 目标感知方法在 n0=12 的 Brier 最低（.0779），但 AUROC/Recall 最低；不是“同时更好”。
- n0=24 时 boundary 的 Brier 最低（.0721）、iid 的 AUROC/Recall 最高；objective 方法居中。
- 假可行率与召回率方向相反：低假可行通常伴随低召回，必须同时报告。

## 8. 判读

- **初始 24 个点的主要收益是“让 objective GP 可训练、初始可行覆盖完整”**：可训练比例 36.7%→83.3%，初始可行 20/30→30/30。
- 它并没有让最终目标在所有任务一致更好：多数配对平局，中位 Δ=0，只有少数长尾任务获益。
- `objective_weighted_boundary` 仍不是默认方法：对 iid 混合，对 constrained_ei 明显落后；风险排序也不占优。
- 12 点 cohort 的 13–24 标签确实被 mask；这不是“12 点也看到了 24 个标签”。
- 10 任务 ×3 重复仍是已知任务族内的描述性复制；不能外推到飞机、RCAIDE 或安全保证。

## 9. 文件

- `full/`：30 units 的原始 report、plan、progress、evaluations、source receipt、manifest、`progress.jsonl`（740 MB，Git LFS）。
- `analysis/`：`analysis.json`、`initial_trainability.csv`、`objective_learning_curves.csv`、`paired_initial_size.csv`、`paired_methods.csv`、`risk_learning_curves.csv`、manifest。
- `logs/`：plan、full、verify、analysis 全日志。
- `environment_packages.txt`、`fixed_sha.txt`、`README.md`、`SHA256SUMS.txt`、`.gitattributes`（`full/progress.jsonl` LFS；`* -text`）。
