# PR104：sign-aware 潜空间风险 + LSBO 能耗采集

固定代码 SHA：`be614874c174415ff3ee2602727d5716e70e3196`  
分支：`research/aircraft-risk-sign-lsbo-20261114`

本目录包含 smoke、4 m/s 逆风与 payload 三套简化飞机实验的原始账本、独立验证、分析和日志。没有 RCAIDE/MDA 调用。

## 1. 执行与验收

| 配置 | 调用数 | 结构 | 状态 |
|---|---:|---|---|
| smoke | 36 | 8 初始 + 8 在线 + 20 留出 | `VERIFIED_AIRCRAFT_RISK_COLD_START` / `COMPLETE_AIRCRAFT_RISK_ANALYSIS` |
| headwind 4 m/s | 3,136 | 144 初始 + 2,304 在线 + 688 留出 | 同上 |
| payload | 3,136 | 同上 | 同上 |
| **合计** | **6,308** | | |

- RCAIDE/MDA：**0**
- 20 项测试通过；每套正式分析有 **80** 条 `discovery_pairs.csv`、**480** 条 `budget_pairs.csv`。
- 四臂：`lsbo_sign_latent`、`lsbo_sign_explicit`、`lsbo_coverage_latent`、`lsbo_coverage_explicit`。
- 初始 12 个共享评价；首解后切回 LSBO 能耗采集；每 8 次更新目标。
- 16 seeds/配置；初始无解：headwind 12/16，payload 11/16。

## 2. Terminal（第 36 次调用）主比较

负 ΔWh 表示候选更好。

### 逆风 4 m/s

| 候选 vs 对照 | 全种子 W/L/T | 平均 ΔWh | 中位 ΔWh | 初值无解 subgroup W/L/T | 平均 ΔWh |
|---|---:|---:|---:|---:|---:|
| `sign_latent` vs `coverage_latent` | 4/3/9 | −484.4 | 0 | 4/3/5 | −678.2 |
| `sign_latent` vs `exact_latent` | 10/1/5 | −911.3 | 0 | 10/1/1 | −1640.4 |
| `sign_latent` vs `sign_explicit` | 1/0/15 | −297.4 | 0 | 0/0/12 | 0 |
| `sign_explicit` vs `coverage_explicit` | 3/4/9 | −119.5 | 0 | 3/4/5 | −167.3 |
| `coverage_latent` vs `coverage_explicit` | 1/1/14 | +25.0 | 0 | 0/1/11 | +510.9 |

### Payload

| 候选 vs 对照 | 全种子 W/L/T | 平均 ΔWh | 中位 ΔWh | 初值无解 subgroup W/L/T | 平均 ΔWh |
|---|---:|---:|---:|---:|---:|
| `sign_latent` vs `coverage_latent` | 7/2/7 | −918.6 | 0 | 7/2/2 | −1684.1 |
| `sign_latent` vs `exact_latent` | 7/1/8 | +61.4 | 0 | 7/1/3 | +138.1 |
| `sign_latent` vs `sign_explicit` | 2/2/12 | +133.3 | 0 | 2/2/7 | +193.8 |
| `sign_explicit` vs `coverage_explicit` | 8/1/7 | −1134.1 | 0 | 8/1/2 | −2079.2 |
| `coverage_latent` vs `coverage_explicit` | 1/0/15 | −107.8 | 0 | 1/0/10 | −197.6 |

## 3. 初值无解子组：首次发现

| 工况 | 候选 vs 对照 | W/L/T | 平均 capped-call Δ | 中位 Δ | 候选/对照发现数 |
|---|---|---|---:|---:|---:|
| headwind | `sign_latent` vs `coverage_latent` | 10/2/0 | −6.67 | −2.5 | 12/10 |
| headwind | `sign_latent` vs `exact_latent` | 11/1/0 | **−17.08** | −20 | 12/5 |
| headwind | `sign_latent` vs `sign_explicit` | 0/0/12 | 0 | 0 | 12/12 |
| payload | `sign_latent` vs `coverage_latent` | 8/3/0 | **−14.18** | −17 | 11/6 |
| payload | `sign_latent` vs `exact_latent` | 10/1/0 | **−16.09** | −22 | 11/4 |
| payload | `sign_latent` vs `sign_explicit` | 0/0/11 | 0 | 0 | 11/11 |

- sign-aware 潜空间在两种工况下都显著加快初值无解种子的首次发现，并提高发现率（12/12 vs 10/12 或 5/12；11/11 vs 6/11 或 4/11）。
- 相对 sign-explicit，发现过程完全一致（0/0/12 或 0/0/11）；潜空间没有额外发现速度增益。

## 4. 判读

- **对 coverage_latent**：两种工况的 terminal 能量都占优（逆风 −484 Wh，payload −919 Wh），且初值无解子组更明显。
- **对 exact_latent**：逆风明显占优（−911 Wh，10/1/5）；payload 的 W/L 为 7/1/8，但均值 +61 Wh，说明少数损失种子抵消了多数小胜，不能写成 payload 全面更好。
- **对 sign_explicit**：逆风是 1/0/15，均值 −297 由单个种子贡献，基本可视为平局；payload 为 2/2/12、均值 +133 Wh，sign-aware 潜空间没有超过显式表示。
- 因此，sign-aware 的主要价值是**相对 coverage/exact 的发现与 terminal 改善**，而不是相对 sign-explicit 的表示优势。
- 36 次调用、16 seeds、简化飞机模型；没有 RCAIDE，不能外推真实飞机或安全结论。

## 5. 文件

- `sign-lsbo-smoke/`、`-analysis/`：36 次评价。
- `sign-lsbo-headwind/`、`-analysis/`：3,136 次评价，80 discovery /480 budget pairs。
- `sign-lsbo-payload/`、`-analysis/`：3,136 次评价，同样结构。
- `tests.log`、`*-plan.json`、`*-run.log`、`*-verify.log`、`*-analyze.log`。
- `fixed_sha.txt`、`environment.txt`、`environment_packages.txt`、`SHA256SUMS.txt`。
- `.gitattributes`：`* -text`。
