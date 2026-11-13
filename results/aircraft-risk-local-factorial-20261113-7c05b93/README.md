# PR101：局部/全局候选增强的飞机风险-能量因子实验

固定代码 SHA：`7c05b93eca1b5930a1ede49dbd22a394afab0382`  
分支：`research/aircraft-risk-local-factorial-20261113`

本目录是简化飞机任务的原始运行、独立验证、分析与日志。没有调用 RCAIDE/MDA。

## 1. 执行与费用

| 配置 | 种子 | 评价数 | 结构 | 状态 |
|---|---|---:|---|---|
| smoke | 28424100 | 30 | 8 初始 +10 在线 +12 留出 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| nominal | 28425000–28426100（12） | 1,392 | 144 +864 +384 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| headwind | 28427000–28428100（12） | 1,392 | 144 +864 +384 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| **合计** | | **2,814** | | |

- RCAIDE/MDA：**0**
- 10 项测试通过；三套 verify 均为 `VERIFIED_AIRCRAFT_RISK_COLD_START`
- 三套 analyze 均为 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`
- 两个正式配置的 `energy_pairs.csv` 各有 **48** 条配对
- 四臂：`cei_latent`、`cei_latent_local`、`ei_latent_global`、`ei_latent_local`
- 12 初始 +36 在线/臂；局部切换在第 36 次；每 8 次更新目标

## 2. 能量配对（负值表示候选更好）

### 无风 nominal

| 比较 | W/L/T | 平均 ΔWh | 中位 ΔWh | 在线可行 Δ/seed | 首次可用调用（候选/对照） |
|---|---:|---:|---:|---:|---:|
| `cei_latent_local` vs `cei_latent` | **11/0/1** | **−634.66** | −684.20 | +2.42 | 11.92 / 11.92 |
| `ei_latent_local` vs `ei_latent_global` | **10/0/2** | **−532.70** | −514.45 | +1.67 | 11.92 / 11.92 |
| `cei_latent_local` vs `ei_latent_local` | 1/1/10 | −64.70 | 0 | +2.00 | 11.92 / 11.92 |
| `cei_latent` vs `ei_latent_global` | 2/1/9 | +37.26 | 0 | +1.25 | 11.92 / 11.92 |

### 逆风 4 m/s headwind

| 比较 | W/L/T | 平均 ΔWh | 中位 ΔWh | 在线可行 Δ/seed | 首次可用调用（候选/对照） |
|---|---:|---:|---:|---:|---:|
| `cei_latent_local` vs `cei_latent` | 7/2/3 | −52.57 | −452.59 | +1.42 | 13.30 / 13.30 |
| `ei_latent_local` vs `ei_latent_global` | **9/0/3** | **−636.65** | −452.59 | +0.92 | 13.30 / 13.30 |
| `cei_latent_local` vs `ei_latent_local` | 2/0/10 | −376.30 | 0 | +1.75 | 13.30 / 13.30 |
| `cei_latent` vs `ei_latent_global` | 4/0/8 | **−960.38** | 0 | +1.25 | 13.30 / 13.30 |

- 无风时局部增强对 CEI、EI 都有明显能量优势（11/0/1、10/0/2），不是单一臂的偶然。
- 逆风时局部 CEI 优势缩小到 7/2/3、均值 −52.6 Wh；局部 EI 仍为 9/0/3、−636.7 Wh，但 CEI 全局对 EI 全局的均值更好（−960.4 Wh）。
- 首次可用调用四臂相同（11.92 / 13.30）；局部增强没有改变冷启动发现时间。

## 3. 风险预测（n=48，joint 头，12 个种子均值）

### nominal

| arm | Brier | AUROC | feasible recall | false-feasible | ECE |
|---|---:|---:|---:|---:|---:|
| `cei_latent` | **.0577** | .9540 | .6848 | .0497 | .0938 |
| `cei_latent_local` | .0584 | **.9559** | **.6864** | **.0464** | .0904 |
| `ei_latent_global` | .0552 | .9616 | **.7485** | .0493 | .0940 |
| `ei_latent_local` | .0580 | .9510 | .5909 | .0435 | .0935 |

### headwind

| arm | Brier | AUROC | feasible recall | false-feasible | ECE |
|---|---:|---:|---:|---:|---:|
| `cei_latent` | **.0300** | **.9547** | **.6400** | .0116 | .0746 |
| `cei_latent_local` | .0331 | .9351 | .5900 | .0116 | .0735 |
| `ei_latent_global` | .0364 | .9268 | .5233 | .0171 | .0741 |
| `ei_latent_local` | .0335 | .9414 | .5900 | **.0199** | **.0705** |

- 无风时局部 CEI 的 Brier 略差于全局（.0584 vs .0577），AUC/召回/假可行略好，差异很小。
- 逆风时局部 CEI 的 Brier/AUC/召回都差于全局 CEI；局部 EI 的 Brier 略好于全局 EI，但召回相同。
- 局部增强没有带来一致的风险预测改善；不能把其能量收益归因于风险表示。

## 4. 判读

- **无风**：局部候选增强在 CEI 与 EI 上均显示能量收益（CEI 11/0/1，均值 −635 Wh；EI 10/0/2，−533 Wh），方向一致。
- **逆风**：CEI 的局部优势大幅缩水（7/2/3，−53 Wh）；EI 仍为 9/0/3（−637 Wh）；但 CEI 全局已优于 EI 全局（−960 Wh），说明逆风条件下简单 CEI 更强，局部增强没有翻盘。
- 四臂首次可用时间相同；没有冷启动发现差异。在线可行数在逆风时也更低。
- 风险 Brier/recall 没有随能量收益同步改善，说明能量差异不是由风险头变好解释的。
- 12 个新种子、简化飞机模型、有限 48 次评价；结果是探索性的，不涉及 RCAIDE，也不支持默认方法或安全声明。

## 5. 文件

- `smoke/`、`smoke-analysis/`：30 次评价与 12 候选验证。
- `nominal/`、`nominal-analysis/`：1,392 次评价，48 条 energy_pairs，48 候选/检查点。
- `headwind/`、`headwind-analysis/`：同上，逆风 4 m/s。
- `logs/`：测试、plan、run、verify、analyze 全日志与退出码。
- `fixed_sha.txt`、`environment.txt`、`environment_packages.txt`、`SHA256SUMS.txt`。
- `.gitattributes`：`* -text`。
