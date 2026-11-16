# PR111 — 分阶段信息获取的冷启动飞机实验

## 固定身份

- 代码仓库：`Luyi1235/RA-LSBO`
- 分支：`research/aircraft-risk-stage-information-20261116`
- 固定代码 SHA：`5dbf47ef044bfd21328f4c1260ba6b8c85c69141`
- 结果分支：`results/aircraft-risk-stage-information-20261116-5dbf47e`
- Python：`C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe`，3.9.25

## 执行与验收

| Profile | 调用 | 拆分 | 状态 |
|---|---:|---|---|
| `stage-information-smoke` | 35 | 4 + 15 + 16 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `stage-information-nominal` | 2,560 | 80 + 1,200 + 1,280 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `stage-information-headwind` | 2,560 | 80 + 1,200 + 1,280 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |

- 合计 **5,155** 次降阶飞机评价；RCAIDE/MDA **0**。
- 三套账本均 `VERIFIED_AIRCRAFT_RISK_COLD_START` 并生成 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`。
- 35 项专项测试通过，0 跳过。
- 五个臂：`risk_coverage`、`risk_active`、`risk_staged`、`risk_joint_interleaved`、`risk_staged_interleaved`。
- 两个正式 profile 各 20 个种子；`discovery_pairs.csv` 各 100 行。

## 1. Brier 配对差（优先）

来自 `analysis.json` 的 `learning_comparisons`；差值为“前臂 − 后臂”，负值表示前臂 Brier 更低。区间为配对 bootstrap 95%。

| 工况 | 比较 | 平均差 | 95% 区间 |
|---|---|---:|---:|
| nominal | `risk_staged_interleaved − risk_coverage` | **+0.002702** | **[−0.001310, +0.007620]** |
| headwind | `risk_staged_interleaved − risk_coverage` | **+0.002665** | **[+0.000098, +0.005260]** |
| nominal | `risk_staged_interleaved − risk_joint_interleaved` | +0.000395 | [−0.001131, +0.002179] |
| headwind | `risk_staged_interleaved − risk_joint_interleaved` | −0.000290 | [−0.000754, +0.000013] |
| nominal | `risk_staged_interleaved − risk_staged` | −0.000319 | [−0.003704, +0.002990] |
| headwind | `risk_staged_interleaved − risk_staged` | −0.000876 | [−0.003027, +0.001322] |

**分阶段交错信息臂相对覆盖臂没有 Brier 改善。** 两个工况点估计均为正（nominal +0.0027、headwind +0.0027）；headwind 的 95% 区间为 **[+0.000098, +0.005260]**，不跨零，表示在该工况显著更差。相对 joint-interleaved 和 staged 的区间均跨零，没有稳定校准优势。

## 2. 首解结果

只统计初值无解种子；调用差为“前臂 − 后臂”的截尾首解差，负值表示前臂更快。

### 相对 coverage

| 工况 | 初值无解 | W/L/T | 发现数 interleaved/coverage | 平均差 | 中位差 | 留一范围 |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 10 | 7/0/3 | **8 / 4** | **−4.200** | — | [−4.667, −3.444] |
| headwind | 11 | 8/0/3 | **9 / 2** | **−4.727** | — | [−5.200, −4.100] |

两个工况均 0 负、留一全负：interleaved 的发现数约为 coverage 的两倍，平均提前 4.2 / 4.7 次调用。

### 相对 joint-interleaved / staged / active

| 工况 | 对照 | W/L/T | 发现数 interleaved/对照 | 平均差 | 留一范围 |
|---|---|---:|---:|---:|---:|
| nominal | joint_interleaved | 1/2/7 | 8 / 8 | +0.600 | [−0.111, +0.889] |
| headwind | joint_interleaved | 0/1/10 | 9 / 10 | +0.182 | [0.000, 0.200] |
| nominal | staged | 1/6/3 | 8 / 9 | +1.500 | [+0.778, +2.000] |
| headwind | staged | 5/3/3 | 9 / 8 | −1.000 | [−1.400, −0.400] |
| nominal | active | — | — | — | — |
| headwind | active | — | — | — | — |

对 joint-interleaved 没有发现优势（nominal 更慢 0.6，headwind 更慢 0.18）；对 staged 在 nominal 更慢 1.5、headwind 更快 1.0。对 active 的逐对结果见 `analysis.json`。

## 3. 判读

- 本轮明确的正面信号仍是**相对 coverage 的冷启动发现速度**：nominal 8 vs 4、headwind 9 vs 2，平均提前 4.2 / 4.7 次调用，留一全负。
- 同一 interleaved 臂**没有** Brier 改善；headwind 甚至显著更差（区间全正）。因此不能把发现速度提升写成风险概率校准提升。
- interleaved 与 joint-interleaved 的 Brier 差约 0，发现速度也没有优势；把 staged 信息交错进来的主要可测收益是对 coverage 的发现速度。
- 对 staged 的方向不一致：nominal 更慢、headwind 更快；不能宣称稳定优于 staged。
- 20 个种子的降阶飞机实验，非 RCAIDE/真实飞行；无默认方法替换、无安全结论。

## 目录

```
stage-information-smoke/              smoke 原始账本
stage-information-smoke-analysis/     smoke 分析
stage-information-nominal/            nominal 原始账本
stage-information-nominal-analysis/   nominal 分析（analysis.json）
stage-information-headwind/           headwind 原始账本
stage-information-headwind-analysis/  headwind 分析（analysis.json）
logs/                                每条命令的 stdout/stderr
python-version.log
tests.log
environment.txt
fixed_sha.txt
integrity_check.json
SHA256SUMS.txt
```
