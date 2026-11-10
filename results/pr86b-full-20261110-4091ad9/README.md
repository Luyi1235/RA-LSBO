# PR86-B Full：预算约束下的序贯风险信息获取

固定代码 SHA：`4091ad9176df63aef0c7a3a0403bc5d2e1ede5b0`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/86>  
环境：`Rcaide_vsp` / Python 3.9.25 / torch 2.8.0+cpu / Windows 10。

本实验是在线数学优化实验，不是 PR72 的历史重建，也不是飞机/RCAIDE 验证。

## 1. 执行与预算

| 项目 | 设置 |
|---|---:|
| 任务 | 6（Branin sparse、Rosenbrock sparse、tradeoff_s0–s3） |
| 每任务独立重复 | 3 |
| 方法 | 5 |
| 每臂总观测 | 48（12 共享 anchor + 36 online） |
| 测试点/任务/重复 | 512 |
| Trials | 90 |
| 数学评价 | 12,672（72 anchor + 6,480 online + 6,120 test） |
| 模型重拟合 | 3,330 |
| GP heads | 6,660 started = 6,660 completed |
| GP kernels | 53,280 started = 53,280 completed |
| Head / kernel errors | 0 / 0 |
| Kernel rejects | 0 |
| VAE / fantasy / aircraft / RCAIDE | 0 / 0 / 0 / 0 |
| Full 用时 | 3,113.2 s |

五种方法：`iid_random`、`maximin`、`latent_uncertainty`、`boundary`、`calibration_aware_voi`。  
所有方法共享同一 task/repeat 的 12 个 anchor；anchor 只计费一次。VOI 每 4 次在线选择中有 1 次来自独立随机校准池（36 次中 9 次），其余为 BALD 风格主动信息获取。

## 2. 完整性

- Full：`COMPLETE_PR86B_SEQUENTIAL_RISK_INFORMATION`。
- 独立 verify：`VERIFIED_PR86B_SEQUENTIAL_RISK_INFORMATION`，24 files，90 trials，0 new math/GP，`test_labels_fit_isolated=true`。
- Analysis：`COMPLETE_PR86B_ANALYSIS`，540 curve rows、72 paired rows。
- 所有 `math_started = math_settled = 12,672`；0 未知/未结算。
- 五个方法均完成 12/24/48 三个检查点。
- 源 worktree 干净；固定 SHA 与报告 `code_sha` 一致。
- `manifest.json` 与 `SHA256SUMS.txt` 均已核验。

## 3. 联合风险学习曲线（anchor-calibrated，六任务均值）

| n | 方法 | Brier | Brier Skill | AUROC | ECE | False-feasible |
|---:|---|---:|---:|---:|---:|---:|
| 12 | 全部五种 | .0855 | — | .628 | .0636 | .0162 |
| 24 | iid_random | .0787 | — | .7198 | .0616 | .0179 |
| 24 | maximin | .0934 | — | .6474 | .0769 | .0377 |
| 24 | latent_uncertainty | .0935 | — | .6318 | .0782 | .0357 |
| 24 | boundary | **.0763** | — | .6795 | **.0546** | **.0100** |
| 24 | calibration_aware_voi | .0840 | — | .6506 | .0613 | .0159 |
| 48 | iid_random | .0650 | .2143 | **.8040** | **.0443** | .0152 |
| 48 | maximin | .0765 | .0872 | .7271 | .0574 | .0192 |
| 48 | latent_uncertainty | .0851 | .0215 | .6818 | .0729 | .0308 |
| 48 | boundary | **.0620** | **.2490** | .7864 | .0540 | .0133 |
| 48 | calibration_aware_voi | .0716 | .1408 | .7360 | .0512 | **.0109** |

- n12 五方法完全相同是共享 anchor 的设计结果。
- n24：boundary Brier/ECE/false-feasible 最好，但 AUROC 低于 iid。
- n48：boundary Brier/技能分最好，iid AUROC/ECE 最好，VOI false-feasible 最低但 Brier/AUROC 均不如 iid 和 boundary。
- VOI 相对 maximin/latent 有改善，但没有超过 iid/boundary。

## 4. n48 逐任务 Brier / AUROC

| 任务 | iid | maximin | latent | boundary | VOI |
|---|---|---|---|---|---|
| Branin sparse | .0638 / .835 | .0678 / .828 | .0938 / .645 | **.0469 / .948** | .0723 / .677 |
| Rosenbrock sparse | **.0496 / .916** | .0732 / .860 | .0844 / .780 | .0539 / .901 | .0696 / .824 |
| tradeoff_s0 | .0820 / .714 | .0862 / .569 | .0939 / .554 | .0736 / .599 | **.0706 / .746** |
| tradeoff_s1 | **.0558 / .860** | .0691 / .708 | .0772 / .666 | .0670 / .733 | .0676 / .686 |
| tradeoff_s2 | .0785 / .731 | .0918 / .681 | .0827 / .725 | **.0718 / .762** | .0801 / .698 |
| tradeoff_s3 | .0605 / .769 | .0707 / .717 | .0785 / .722 | **.0589 / .775** | .0695 / .786 |

VOI 仅在 s0 的 Brier、s3 的 AUROC 上有局部亮点；其余任务均落后于 iid 或 boundary。

## 5. VOI 的 n48 配对 Brier

`W/L/T` 为任务内 3 个重复的候选对对照；Δ = VOI − 对照，负值表示 VOI 更好。

| 对照 | W/L/T | 平均 Δ Brier | 判读 |
|---|---:|---:|---|
| iid_random | 8/10/0 | +0.00659 | VOI 更差 |
| boundary | 5/13/0 | +0.00961 | VOI 更差 |
| maximin | 11/7/0 | −0.00487 | VOI 更好 |
| latent_uncertainty | 14/4/0 | −0.01347 | VOI 更好 |

VOI 低于 maximin/latent，但没有超过 iid/boundary；false-feasible 最低是唯一一致的正向信号，但不足以支持默认采用。

## 6. 按任务观察

- **Branin sparse**：boundary 明显最好（Brier .0469，AUROC .948）；VOI .0723/.677，假可行率 .0028 最低。
- **Rosenbrock sparse**：iid 最好（.0496/.916）；VOI .0696/.824。
- **tradeoff_s0**：VOI Brier 最好（.0706）且 AUROC .746；boundary 次之。
- **tradeoff_s1**：iid 最好（.0558/.860）；VOI .0676/.686。
- **tradeoff_s2**：boundary 最好（.0718/.762）；VOI .0801/.698。
- **tradeoff_s3**：boundary Brier 最好（.0589），VOI AUROC 最好（.786）但 Brier .0695 较差。

## 7. 解释边界

- 这是 6 任务 × 3 重复、每臂 36 次在线选择的探索性实验，不是确认性试验。
- VOI 的校准池只有 9/36 次；其随机性本身会影响小样本比较。
- 风险模型/信息分数来自当前 GP，不是真值；测试集只用于事后评分。
- 不能把 false-feasible 最低写成安全保证；也不能把 VOI 超过 maximin/latent 写成超过 iid/boundary。
- 未接入飞机/RCAIDE；结果不外推到 100/140 km 或真实设计。
- 不根据本结果追加种子或调参；下一步由 ChatGPT 阅读原始记录决定。

## 8. 文件

- `full/`：原始输出（plan、source_receipt、report、evaluations、progress、units、manifest、environment）。
- `analysis/`：`analysis.json`、`learning_curves.csv`、`manifest.json`。
- `logs/`：plan、full 控制台、verify、analysis、environment、pip freeze。
- `README.md`、`fixed_sha.txt`、`environment.json`、`instructions.txt`。
- `SHA256SUMS.txt`、`.gitattributes`（`progress.jsonl` 使用 Git LFS，其余 `* -text`）。
