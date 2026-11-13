# PR100：局部 vs 全局 CEI 候选增强（飞机风险冷启动）

固定代码 SHA：`19080e0b3f150217549958b6f75196d7c5080a6f`  
分支：`research/aircraft-risk-local-search-20261113`  
环境：Rcaide_vsp / Python 3.9.25

本实验在同一 `cei_latent` 共享搜索轨迹上，比较局部增强与全局 CEI 增强；不是新的端到端优化算法，也不调用 RCAIDE。

## 1. 预算与完整性

| 项目 | 结果 |
|---|---:|
| 种子 | 12（28421000–28422100，步长 100） |
| 臂 | `cei_latent`、`cei_latent_local` |
| 每臂观测 | 48（12 初始 +36 在线） |
| Smoke | 26 次评价，`VERIFIED_AIRCRAFT_RISK_COLD_START` + `COMPLETE_AIRCRAFT_RISK_ANALYSIS` |
| Pilot | 1,104 次 =144 初始 +576 在线 +384 留出 |
| Pilot verify | `VERIFIED_AIRCRAFT_RISK_COLD_START`，1,104 calls |
| Pilot analysis | `COMPLETE_AIRCRAFT_RISK_ANALYSIS` |
| RCAIDE / MDA | 0 |
| 本地增强点 | 47 |
| 全局增强点（baseline） | 42 |
| shared-rule fallback | 17 /臂（对称） |

## 2. 联合风险（joint）

| n | `cei_latent` Brier / AUC / recall / FF | `cei_latent_local` Brier / AUC / recall / FF |
|---:|---|---|
| 12 | .11888 / .5765 / .0675 / .0558 | 同左（尚未切换） |
| 24 | .07726 / .8653 / .4921 / .0408 | 同左 |
| 36 | .06543 / .9264 / .6944 / .0502 | 同左 |
| 48 | .05383 / .9382 / .6706 / .0499 | **.05144 / .9595 / .6548 / .0411** |

n48 局部增强相对基线：

- Brier **−0.00239**（相对降低约 4.4%）；
- AUROC **+0.0213**；
- feasible recall **−1.58 个百分点**（.6548 vs .6706）；
- false-feasible **−0.88 个百分点**（.0411 vs .0499）。

所以局部增强改善了排序/校准和假可行率，但可行召回略降；不是“全面更好”。

## 3. 能量配对（12 个种子）

`energy_pairs.csv` 为 12 条同种子配对：

| 指标 | 值 |
|---|---:|
| 胜 / 负 / 平 | **10 / 1 / 1** |
| 平均 `delta_Wh` | **−403.47 Wh** |
| 中位 `delta_Wh` | **−589.14 Wh** |
| 范围 | −845.55 到 **+711.36 Wh** |
| 在线可行数变化 | 平均 **+2.92** |
| 在线不可行数变化 | 平均 **−2.92** |

- 12 个种子里 10 个局部增强的最终最优能量更低；1 个明显更差（+711 Wh），1 个持平。
- 局部增强不是免费：它用 47 个局部点替换了 42 个全局点；在 12 个种子里有 1 个反向。
- 轨迹均值：`cei_latent` 最优中位 37,749 Wh、在线可行 96；`cei_latent_local` 37,331 Wh、在线可行 131；首次可行调用均值都为 6.08。
- 能量比较是诊断性目标，不是 PR100 的主统计检验；12 个种子不足以支持默认替换。

## 4. 判读

- 局部 CEI 增强在同一潜空间、同一预算下给出了**方向一致但幅度有限**的改善：Brier 和 AUC 更好，能量配对 10/12，在线可行评价更多。
- 代价是 feasible recall 下降约 1.6 个百分点，且能量配对存在一个 +711 Wh 的反例；不能写成“局部增强全面占优”。
- 没有显式 vs 潜空间的独立对照；本实验只回答“局部与全局候选增强”的差异。
- 没有 RCAIDE、没有闭环优化收益、没有安全保证；12 个 seed 属于探索性。

## 5. 文件

- `smoke/`、`smoke-analysis/`：26 次评价与验证/分析。
- `pilot/`、`pilot-analysis/`：1,104 次评价、24 条轨迹、`metrics.csv`、`joint_pairs.csv`、`energy_pairs.csv`、`analysis.json`。
- `logs/`：测试、plan、smoke/pilot run/verify/analyze 全日志与退出码。
- `fixed_sha.txt`、`environment.txt`、`environment_packages.txt`、`SHA256SUMS.txt`。
- `.gitattributes`：`* -text`。
