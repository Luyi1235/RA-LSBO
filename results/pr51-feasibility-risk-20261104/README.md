# PR51 工程可行风险概率与校准诊断结果

分析代码固定 SHA：`5ccc87b2c356069c122b6df3e047eb3c20080954`
源 PR45 数据 commit：`386082beeacac8ae43d0dfb1ccda5c5ad3b356cf`
PR45 plan SHA256：`ac5f1424f3c8d52eff9ffd835cb5004224be965afee8ec15069e1fea38c43012`
PR45 SQLite SHA256（运行前后一致）：`a30b0dadb5d619be9b6c0d70c5aa29fe115f86b85d62a449e4acf05325d6f1b7`
PR50 全库语义审计结果 commit：`ec76c4102a19c751081dcfaa92c18b702aba26f3`

新增飞机/RCAIDE 调用：0；原始 SQLite 只读且 mtime/哈希未变。
模型配置：20/20 完整；GP 拟合异常：0。
测试预测行：19200（2 任务 × 2 fold × 5 规模 × 2 视图 × 5 概率定义 × 96 测试评价）。

## 分割与标签

训练前缀：种子列表索引 0–15（fold 0）与 16–31（fold 1），规模 12/24/48/96/192。
校准：索引 384–391（fold 0）与 392–399（fold 1），每 fold 96 条真实 qA 评价。
测试：索引 512–519（fold 0）与 520–527（fold 1），每 fold 96 条；同一测试设计在全部规模/视图重复使用。
训练、校准、测试的真实设计 ID 两两不重叠（代码强制）。

| case | test feasible positives/negatives (192) | test mission positives/negatives (192) |
|---|---:|---:|
| ordinary | 97 / 95 | 141 / 51 |
| sparse | 17 / 175 | 47 / 145 |

`status_completion` 训练标签在全部 20 个配置中均为 1（`original_status_completion_training_is_all_one=true`）。

## n=192 指标（校准 / 原始 / 训练流行率基线）

| case | proxy | view | Brier cal | Brier raw | Brier base | LogLoss cal | LogLoss raw | ECE cal | FP/neg |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary | legacy_completion_product | latent | 0.0225 | 0.0218 | 0.2600 | 0.0746 | 0.0737 | 0.0252 | 3/95 |
| ordinary | legacy_completion_product | onehot | 0.0214 | 0.0212 | 0.2600 | 0.0734 | 0.0739 | 0.0132 | 4/95 |
| ordinary | constraint_product | latent | 0.0226 | 0.0217 | 0.2600 | 0.0705 | 0.0702 | 0.0148 | 3/95 |
| ordinary | constraint_product | onehot | 0.0213 | 0.0211 | 0.2600 | 0.0693 | 0.0705 | 0.0132 | 4/95 |
| ordinary | soc_stall_bending | latent | 0.0226 | 0.0217 | 0.2600 | 0.0705 | 0.0702 | 0.0148 | 3/95 |
| ordinary | soc_stall_bending | onehot | 0.0213 | 0.0211 | 0.2600 | 0.0693 | 0.0705 | 0.0132 | 4/95 |
| ordinary | mission_status_proxy | latent | 0.2451 | 0.2618 | 0.1964 | 1.6667 | 1.3294 | 0.2163 | 51/51 |
| ordinary | mission_status_proxy | onehot | 0.2428 | 0.2616 | 0.1964 | 1.6966 | 1.3089 | 0.2089 | 51/51 |
| ordinary | mission_soc_proxy | latent | 0.0169 | 0.0176 | 0.1964 | 0.0580 | 0.0621 | 0.0188 | 3/51 |
| ordinary | mission_soc_proxy | onehot | 0.0164 | 0.0171 | 0.1964 | 0.0603 | 0.0658 | 0.0195 | 4/51 |
| sparse | legacy_completion_product | latent | 0.0367 | 0.0417 | 0.0839 | 0.1152 | 0.1330 | 0.0368 | 7/175 |
| sparse | legacy_completion_product | onehot | 0.0234 | 0.0247 | 0.0839 | 0.0854 | 0.0903 | 0.0296 | 0/175 |
| sparse | constraint_product | latent | 0.0366 | 0.0420 | 0.0839 | 0.1139 | 0.1335 | 0.0350 | 7/175 |
| sparse | constraint_product | onehot | 0.0233 | 0.0248 | 0.0839 | 0.0847 | 0.0902 | 0.0288 | 0/175 |
| sparse | soc_stall_bending | latent | 0.0366 | 0.0420 | 0.0839 | 0.1139 | 0.1335 | 0.0350 | 7/175 |
| sparse | soc_stall_bending | onehot | 0.0233 | 0.0248 | 0.0839 | 0.0848 | 0.0903 | 0.0288 | 0/175 |
| sparse | mission_status_proxy | latent | 0.7030 | 0.7431 | 0.1849 | 2.6257 | 3.7020 | 0.7190 | 145/145 |
| sparse | mission_status_proxy | onehot | 0.7133 | 0.7424 | 0.1849 | 2.7910 | 3.6533 | 0.7264 | 145/145 |
| sparse | mission_soc_proxy | latent | 0.0582 | 0.0761 | 0.1849 | 0.1937 | 0.2372 | 0.0363 | 6/145 |
| sparse | mission_soc_proxy | onehot | 0.0371 | 0.0446 | 0.1849 | 0.1267 | 0.1450 | 0.0338 | 2/145 |

## n=12 指标（校准 / 原始 / 训练流行率基线）

| case | proxy | view | Brier cal | Brier raw | Brier base | LogLoss cal | LogLoss raw | ECE cal | FP/neg |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary | legacy_completion_product | latent | 0.2388 | 0.2637 | 0.2997 | 0.7123 | 0.7582 | 0.1057 | 36/95 |
| ordinary | legacy_completion_product | onehot | 0.2182 | 0.2266 | 0.2997 | 0.6557 | 0.6764 | 0.1035 | 29/95 |
| ordinary | constraint_product | latent | 0.2415 | 0.2597 | 0.2997 | 0.7232 | 0.7503 | 0.1369 | 36/95 |
| ordinary | constraint_product | onehot | 0.2182 | 0.2206 | 0.2997 | 0.6560 | 0.6567 | 0.0895 | 26/95 |
| ordinary | soc_stall_bending | latent | 0.2306 | 0.2391 | 0.2997 | 0.6619 | 0.6780 | 0.0832 | 41/95 |
| ordinary | soc_stall_bending | onehot | 0.2048 | 0.2025 | 0.2997 | 0.5994 | 0.5943 | 0.0476 | 24/95 |
| ordinary | mission_status_proxy | latent | 0.2095 | 0.2247 | 0.2500 | 0.6189 | 0.7041 | 0.0915 | 51/51 |
| ordinary | mission_status_proxy | onehot | 0.2093 | 0.2243 | 0.2500 | 0.6178 | 0.7012 | 0.0897 | 51/51 |
| ordinary | mission_soc_proxy | latent | 0.1561 | 0.1989 | 0.2500 | 0.4753 | 0.5640 | 0.0403 | 28/51 |
| ordinary | mission_soc_proxy | onehot | 0.1274 | 0.1475 | 0.2500 | 0.3937 | 0.4481 | 0.0193 | 24/51 |
| sparse | legacy_completion_product | latent | 0.0785 | 0.0897 | 0.0925 | 0.2840 | 0.3564 | 0.0057 | 0/175 |
| sparse | legacy_completion_product | onehot | 0.0767 | 0.0878 | 0.0925 | 0.2801 | 0.3610 | 0.0086 | 2/175 |
| sparse | constraint_product | latent | 0.0786 | 0.0919 | 0.0925 | 0.2843 | 0.3584 | 0.0049 | 0/175 |
| sparse | constraint_product | onehot | 0.0768 | 0.0900 | 0.0925 | 0.2801 | 0.3627 | 0.0086 | 2/175 |
| sparse | soc_stall_bending | latent | 0.0797 | 0.1056 | 0.0925 | 0.2910 | 0.3721 | 0.0299 | 0/175 |
| sparse | soc_stall_bending | onehot | 0.0790 | 0.1008 | 0.0925 | 0.2861 | 0.3523 | 0.0297 | 3/175 |
| sparse | mission_status_proxy | latent | 0.4729 | 0.6078 | 0.1887 | 4.0617 | 1.7560 | 0.4448 | 71/145 |
| sparse | mission_status_proxy | onehot | 0.4743 | 0.6031 | 0.1887 | 4.0455 | 1.7307 | 0.4441 | 71/145 |
| sparse | mission_soc_proxy | latent | 0.1875 | 0.1975 | 0.1887 | 0.5576 | 0.5811 | 0.0336 | 2/145 |
| sparse | mission_soc_proxy | onehot | 0.1719 | 0.1783 | 0.1887 | 0.5200 | 0.5319 | 0.0331 | 5/145 |

## 全约束乘积的 Brier 学习曲线（校准后）

| case | view | n=12 | n=24 | n=48 | n=96 | n=192 |
|---|---|---:|---:|---:|---:|---:|
| ordinary | latent | 0.2415 | 0.1260 | 0.0694 | 0.0412 | 0.0226 |
| ordinary | onehot | 0.2182 | 0.1053 | 0.0475 | 0.0237 | 0.0213 |
| sparse | latent | 0.0786 | 0.0825 | 0.0796 | 0.0753 | 0.0366 |
| sparse | onehot | 0.0768 | 0.0755 | 0.0517 | 0.0336 | 0.0233 |

## 旧完成头的实际输出（legacy / constraint 原始概率比）

| case | view | mean status probability n=12 | n=24 | n=48 | n=96 | n=192 |
|---|---|---:|---:|---:|---:|---:|
| ordinary | latent | 0.8935 | 0.9428 | 0.9699 | 0.9848 | 0.9921 |
| ordinary | onehot | 0.8922 | 0.9395 | 0.9683 | 0.9840 | 0.9918 |
| sparse | latent | 0.8940 | 0.9403 | 0.9698 | 0.9844 | 0.9919 |
| sparse | onehot | 0.8902 | 0.9373 | 0.9681 | 0.9835 | 0.9915 |

## 关键观察

- **旧完成概率乘积并未携带任务失败信息。** PR45 的 status 标签全为 1，完成头只学到“记录已结算”，平均原始概率从 n=12 的 0.89 升到 n=192 的 0.99。它与全约束乘积几乎成常数比例：n=192 时 ordinary Brier 0.0214（legacy, onehot）对 0.0213（constraint），sparse 0.0234 对 0.0233；假可行次数也相同。数值上不差，是因为约束头在承担全部区分度，不是因为 completion 因子有效。
- **任务完整性头（mission_status_proxy）失效。** n=192 时 ordinary Brier 0.245、LogLoss 1.67，sparse Brier 0.703、LogLoss 2.63，均劣于训练流行率基线（0.196 / 0.185）；假可行 51/51 与 145/145（全部真实负类被预测为 ≥0.5），校准截距也无法挽救。相反，SOC 代理（mission_soc_proxy）n=192 Brier 为 ordinary 0.0164、sparse 0.0371，支持 PR50 的“该历史数据中任务完成 ≡ SOC 满足”经验事实，但这不是通用因果恒等式。
- **全约束乘积可用，但 12 条训练评价不足以支撑稀疏任务。** n=12 时 sparse 校准 Brier 0.0768，仅略优于基线 0.0925；n=96 降到 0.0336，n=192 降到 0.0233。ordinary 在 n=192 为 0.0213，远优于基线 0.260。
- **One-hot 一致优于潜空间，稀疏任务差距更大。** n=192 全约束乘积：ordinary 0.0213 vs 0.0226，sparse 0.0233 vs 0.0366（约低 36%）；0.5 阈值假可行 sparse 为 0/175 vs 7/175。n=12 时 ordinary 0.2182 vs 0.2415，sparse 0.0768 vs 0.0786。
- **三项约束近似没有稳定增益。** n≥48 时它与全约束乘积数值相同（其余约束头概率接近 1）；n=12 时 ordinary 0.2048 vs 0.2182（更好），sparse 0.0790 vs 0.0768（更差）。它是看到 PR50 全库标签后选择的探索性对照，不能作为独立优势证明。
- **校准主要帮助小样本与稀疏任务，n≥96 后接近中性。** 全约束 onehot：n=12 ordinary 0.2206→0.2182、sparse 0.0900→0.0768；n=192 sparse 0.0248→0.0233，而 ordinary 0.0211→0.0213（轻微变差）。n=192 ECE 为 ordinary 0.013、sparse 0.029（onehot）。
- **假可行仍存在。** n=192 ordinary onehot 4/95、latent 3/95；sparse onehot 0/175、latent 7/175。n=12 ordinary onehot 26/95；sparse 2/175。校准不提供安全保证。

## 限制

- 回顾性 qA 种子，不是在线优化器前瞻试验；结论不能外推为优化性能或安全保证。
- 训练流行率基线来自各 fold 的训练前缀；测试种子块与其分布不完全相同。
- 边际约束乘积假设独立，不是联合可行概率的严格校准。
- 三项约束子集是 PR50 审计后的探索性选择；12 条以上训练评价是已付出的跨种子真实数据。
- 测试设计在规模/视图间重复使用（配对），未报告 p 值或朴素 IID 置信区间。
- 140 km 任务仍未通过独立稀疏资格。

## 文件

- `feasibility_risk_report.json`：完整指标、分割、fit receipts、限制。
- `feasibility_risk_predictions.csv`：19,200 条测试预测（原始/校准/基线）。
- `smoke-console.log` / `full-console.log`：两轮完整控制台日志。
- `tests.log`：`tests/test_feasibility_risk_calibration.py`，7 passed。
- `pr51_pre_run.sha256` / `pr51_output.sha256`：源文件运行前后哈希与输出哈希。
