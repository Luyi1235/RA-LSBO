# RA-LSBO 论文绘图数据

## 按预算取数

| 目录 | 实验 | 工况和算法 |
|---|---|---|
| [32-evaluations](32-evaluations) | 32 次主实验。保留 `internal_32` 和 `external_32` 两个实验标签。 | 逆风、增载。内部对比包括 RA-LSBO、原 LSBO、区间反馈对照。外部对比包括 RA-LSBO、Screened ArchSBO、Adapted LogCEI。 |
| [64-evaluations](64-evaluations) | 64 次预算敏感性试验，仅含 `external_64`。 | 逆风、增载。三种外部对比算法。 |

两个目录各含以下三张表。CSV 用 UTF-8，压缩表可用 Python 的 `gzip.open` 直接读取。

| 文件 | 字段与用途 |
|---|---|
| `seed_trajectories.csv.gz` | 每个实验、工况、种子、算法和累计评价次数一行。画单种子轨迹或按组汇总时，用 `best_feasible_energy_kWh` 表示截至该次评价的最佳可行能耗。空值表示该种子此时尚无可行解。 |
| `target_attainment.csv.gz` | 达标曲线数据。横轴 `evaluation`，纵轴 `target_fraction`，分子 `target_hits`，分母 `seeds`。按 `experiment`、`scenario`、`cohort` 和 `arm` 分组。 |
| `paired_energy_checkpoints.csv` | 同预算最佳可行能耗的配对差。 `mean_delta`、`ci95_low`、`ci95_high` 单位为 Wh。差值为 RA-LSBO 减对照，负值有利于 RA-LSBO。 `pairs` 是该节点双方均已有可行解的种子对数。 |

## 绘图口径

- 逆风 `headwind` 的目标为 40 kWh，增载 `payload` 为 38 kWh，同步数据已包含目标值。累计评价次数包含共享的前 8 次初始评价。
- `cohort=cold` 指 8 个初始设计均未产生可用结果，`cohort=all` 指全部种子。逐种子表中的 `cold=1` 对应冷启动组。32 次内部和外部对比各自成图或分面。
- 32 次主实验优先绘制冷启动达标阶梯曲线及第 32 次的能耗配对差。64 次实验绘制至第 64 次的达标曲线，以及第 32、48、64 次的能耗配对差和 95% 配对 bootstrap 区间。64 次运行使用外部 32 次试验的种子并从初始化重新运行，其第 32 次节点取自本轮轨迹。
- 能耗空值保留为缺失。汇总能耗时同时报告可行种子数或配对数。外部比较使用实验中实际运行的筛选版 ArchSBO 和适配版 LogCEI。

## 数据校验

每个预算目录包含 9,216 行逐种子轨迹和 1,152 行达标汇总。32 次和 64 次分别包含 72 行和 60 行配对能耗记录。筛分后已逐项用轨迹核对达标分子、分母、配对数量与配对均值。置信区间沿用原始分析。

[固定数据源](https://github.com/Luyi1235/RA-LSBO/tree/1a0b8d819ca505a91a3dc4f5858f6a8d72524968/docs/ra-lsbo-paper/data) · [实验归档与分析说明](https://github.com/Luyi1235/RA-LSBO/blob/2735da170c38c07ff7e8d769a92a917b0ddea047/docs/AIRCRAFT_STRUCTURED_RISK_OPTIMIZATION_EVIDENCE_20261121.md)
