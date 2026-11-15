# PR108 — 保留结构裕量的终止反馈

## 固定身份

- 代码仓库：`Luyi1235/RA-LSBO`
- 分支：`research/aircraft-risk-structural-margin-feedback-20261115`
- 固定代码 SHA：`fe9cdde03ace78b1ae47386ac564b6a332f3008b`
- 结果分支：`results/aircraft-risk-structural-margin-feedback-20261115-fe9cdde`
- 参考证据：`results/aircraft-risk-termination-feedback-20261115-8980cb5` @ `fa1b4537b38265b64eb8a7fb6976b10bbd1b0773`
- 参考代码 SHA：`8980cb5850234e1feb6290b05d985d588dd6bf92`

## 执行与验收

| Profile | 调用 | 拆分 | 状态 |
|---|---:|---|---|
| `termination-margins-smoke` | 24 | 4 + 8 + 12 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `termination-margins` | 1,824 | 96 + 960 + 768 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |

- 合计新增 **1,848** 次降阶飞机评价；RCAIDE/MDA **0**。
- 两个 profile 均通过 `VERIFIED_AIRCRAFT_RISK_COLD_START`，并生成 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`。
- 跨参考核验：`VERIFIED_STRUCTURAL_MARGIN_FEEDBACK`。
- 参考证据先以 `git archive` 导出，`SHA256SUMS.txt` 全部通过后才运行。
- 正式分析有 20 个初值无解种子、24 个匹配的分阶段冷启动前缀；匹配评价设计 1,160 个、固定设计 864 个。

## 留出终止频数

保留裕量后的 enriched 运行与参考 baseline 的终止频数相同：

| 终止类别 | 数量 |
|---|---:|
| `complete` | 31 |
| `physical_soc_reserve_abort` | 508 |
| `root_bending` | 128 |
| `stall` | 101 |

合计 768 条留出评价。结构裕量增强改变的是风险/裕量表示，而不是本轮留出终止分布的构成。

## 发现数与首解调用差

只统计 20 个初值无解种子；调用差为 `lsbo_sign_latent − discovery_staged`，负值表示 sign_latent 更快。

### 保留裕量前（PR107 structural baseline）

| 项目 | 值 |
|---|---:|
| 发现数（sign_latent / staged） | **12 / 8** |
| W/L/T | 8 / 7 / 5 |
| 平均调用差 | **−0.45** |
| 中位调用差 | **0.0** |
| 留一范围 | **[−1.474, +0.526]** |

### 保留裕量后（PR108 enriched）

| 项目 | 值 |
|---|---:|
| 发现数（sign_latent / staged） | **15 / 8** |
| W/L/T | 12 / 4 / 4 |
| 平均调用差 | **−3.80** |
| 中位调用差 | **−1.00** |
| 留一范围 | **[−4.421, −3.000]** |

保留裕量后 sign_latent 的发现数由 12 增至 15，平均首解差由 −0.45 变为 −3.80；留一范围从跨零变为全负。

## 裕量效应（enriched − baseline）

| 项目 | 值 |
|---|---:|
| 初值无解种子 | 20 |
| W/L/T | 11 / 4 / 5 |
| 平均效应 | **−3.35 次调用** |
| 中位效应 | **−1.50 次调用** |
| 留一范围 | **[−4.158, −2.684]** |

效应定义为“保留裕量后的首解差 − 保留裕量前的首解差”；负值表示保留裕量后 sign_latent 相对 staged 更快。均值 −3.35、留一范围全负，说明该增强在本实验的 20 个初值无解种子上稳定改善首解速度。

## 判读与边界

- 这是 PR107 structural 参考上的增强实验；未重跑 baseline，baseline 指标直接引用自固定 PR107 证据。
- 保留裕量后留出终止频数不变（31/508/128/101），因此本轮不能把差异归因于终止分布改变；它对应的是同一终止结构下的候选/裕量表示变化。
- 发现数 15/20、平均 −3.8 次、留一全负，是首解速度信号；不是最终目标、能耗或安全优势。
- 参考 baseline 与 enriched 使用相同 24 个匹配 staged 前缀和相同 20 个初值无解种子；没有改算法、种子或预算。
- 没有默认方法替换；没有 RCAIDE/真实飞行结论。

## 目录

```
termination-margins-effect.json      正式跨参考效应
termination-margins-smoke-effect.json
margins/                             enriched 原始账本
margins-analysis/                    enriched 分析
margins-smoke/                       smoke 原始账本
margins-smoke-analysis/              smoke 分析
logs/                                plan/run/verify/analyze/effect、测试、解释器
fixed_sha.txt
SHA256SUMS.txt
```
