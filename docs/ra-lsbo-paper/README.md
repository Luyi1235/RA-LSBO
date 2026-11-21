# RA-LSBO 论文方法与可视化数据

[方法章节草案](方法章节.md)包括问题定义、区间反馈的编码、构型层级核、两阶段采集机制和可支持的创新主张。

## 浏览和重画

```powershell
python plot_results.py --output figures
```

脚本需要 Python 和 Matplotlib 3.6 或更新版本，输出 PDF 和 PNG。CSV 可以用 Python 读入，也可以解压 `.csv.gz` 后用 Excel 或 Origin 打开。

| 图 | 口径 |
|---|---|
| [01 冷启动达标曲线](figures/01_cold_start_target_attainment.pdf) | 32 次端到端内部对比和 32 次外部对比。两组各自采用其固定冷启动种子。 |
| [02 外部预算敏感性](figures/02_external_budget_sensitivity.pdf) | 64 次重跑的完整达标曲线，以及第 32、48、64 次的最佳可行能耗配对差。 |
| [03 风险预测消融](figures/03_completion_risk_ablation.pdf) | 独立确认实验中，冷启动种子的留出构型均衡 Brier 分数差。 |

| 数据 | 每行代表什么 |
|---|---|
| `data/seed_trajectories.csv.gz` | 一个工况、种子、算法、累计评价次数。含历史最低可行能耗、是否达标、可用设计累计数和该次评价的停机原因。无可行设计时能耗留空。 |
| `data/target_attainment.csv.gz` | 一个实验、工况、种子组、算法和累计评价次数。含达标数、分母、达标比例和找到可行设计的种子数。 |
| `data/paired_energy_checkpoints.csv` | RA-LSBO 减对照的最佳可行能耗配对差，单位 Wh。每个节点使用双方均有可行设计的种子对。 |
| `data/mechanism_ablation.csv` | 相对区间非结构化对照及结构化符号对照的 Brier 差和到第 3 个可用设计的截尾评价次数差。 |

`cold` 表示 8 个共享初始设计均未产生可用结果。`all` 包含该工况的全部种子，`initially_usable` 包含其余种子。逆风和增载预设能耗目标分别为 40 和 38 kWh。所有横轴都包含前 8 次评价。64 次实验复用 32 次外部对比的种子，从初始化重新运行。图 02 中的 32 次点来自 64 次轨迹。

已复核四个归档的 SHA256 清单。导出时按定版 `DesignRecord.usable_feasible` 从封存账本重建轨迹，逐种子核对了原分析中的达标评价次数、第三个可用设计出现时点和已登记预算节点的能耗。

[固定提交与原始分析](https://github.com/Luyi1235/RA-LSBO/blob/2735da170c38c07ff7e8d769a92a917b0ddea047/docs/AIRCRAFT_STRUCTURED_RISK_OPTIMIZATION_EVIDENCE_20261121.md#fixed-sources)
