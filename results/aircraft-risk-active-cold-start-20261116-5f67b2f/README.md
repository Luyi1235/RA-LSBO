# PR110 — 主动冷启动风险信息获取

## 固定身份

- 代码仓库：`Luyi1235/RA-LSBO`
- 分支：`research/aircraft-risk-active-cold-start-20261116`
- 固定代码 SHA：`5f67b2f1f03f8a108012c95438c25e1204af3aa0`
- 结果分支：`results/aircraft-risk-active-cold-start-20261116-5f67b2f`
- Python：`C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe`，3.9.25

## 执行与验收

| Profile | 调用 | 拆分 | 状态 |
|---|---:|---|---|
| `active-risk-smoke` | 35 | 4 + 15 + 16 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `active-risk-nominal` | 2,048 | 64 + 960 + 1,024 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `active-risk-headwind` | 2,048 | 64 + 960 + 1,024 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |

- 合计 **4,131** 次降阶飞机评价；RCAIDE/MDA **0**。
- 三个 profile 均通过 `VERIFIED_AIRCRAFT_RISK_COLD_START`，并生成 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`。
- 33 项专项测试通过，0 跳过。
- 五个臂：`risk_coverage`、`risk_greedy`、`risk_pointwise`、`risk_active`、`risk_active_latent`。
- 正式 profile 各 16 个种子、64 次初始评价、960 次在线评价、1,024 次留出评价；初值无解种子：nominal **13/16**，headwind **15/16**。

## 1. Brier 配对差（首先报告）

来自 `analysis.json` 的 `learning_comparisons`；差值为“主动臂 − 覆盖臂”，负值表示主动臂 Brier 更低。区间是配对 bootstrap 95% 区间。

### Nominal

| 比较 | 平均 Brier 差 | 95% 区间 |
|---|---:|---:|
| `risk_active` − `risk_coverage` | **+0.000732** | **[−0.007098, +0.007925]** |
| `risk_active` − `risk_greedy` | −0.003199 | [−0.010109, +0.003932] |
| `risk_active` − `risk_pointwise` | −0.002669 | [−0.009302, +0.004230] |

### Headwind

| 比较 | 平均 Brier 差 | 95% 区间 |
|---|---:|---:|
| `risk_active` − `risk_coverage` | **+0.001986** | **[−0.001673, +0.005339]** |
| `risk_active` − `risk_greedy` | −0.002251 | [−0.006638, +0.002742] |
| `risk_active` − `risk_pointwise` | −0.002251 | [−0.006638, +0.002742] |

**结论：主动臂相对覆盖臂没有 Brier 改善。** 两个工况的点估计均为正（nominal +0.0007、headwind +0.0020），且 95% 区间都跨零；headwind 更接近“略差”而非“略好”。相对 greedy/pointwise 的点估计为负，但区间同样跨零。

n=16 风险指标（各臂 16 个种子均值，供定位，不替代上面的配对差）：

| 工况 | 臂 | Brier | AUROC | 可行召回 | 假可行 |
|---|---|---:|---:|---:|---:|
| nominal | coverage | .1073 | .749 | .538 | .212 |
| nominal | active | .1030 | .754 | .602 | .249 |
| nominal | active_latent | .1115 | .730 | .605 | .279 |
| headwind | coverage | .0922 | .728 | .508 | .194 |
| headwind | active | .0961 | .737 | .534 | .222 |
| headwind | active_latent | .1073 | .706 | .505 | .229 |

## 2. 首解结果

只统计初值无解种子；调用差为“主动臂 − 对照臂”的截尾首解调用差，负值表示主动臂更快。

### Active vs coverage

| 工况 | 初值无解 | W/L/T | 发现数 active/coverage | 平均差 | 中位差 | 留一范围 |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 13 | 12/1/0 | **13 / 7** | **−5.154** | −5 | [−5.917, −4.583] |
| headwind | 15 | 9/6/0 | **13 / 7** | **−2.200** | −4 | [−3.000, −1.500] |

这是本轮最强的信号：主动臂在两个工况都把覆盖臂的发现数从 7 提到 13，平均首解提前 5.15 / 2.20 次调用，留一范围全负。

### Active vs greedy

| 工况 | W/L/T | 发现数 active/greedy | 平均差 | 中位差 | 留一范围 |
|---|---:|---:|---:|---:|---:|
| nominal | 5/6/2 | 13 / 12 | −0.231 | 0 | [−0.667, +0.167] |
| headwind | 7/6/2 | 13 / 10 | −0.400 | 0 | [−1.071, +0.357] |

对 greedy 的优势很小且留一跨零；主动臂的收益主要来自相对 coverage 的覆盖策略，而不是稳定超过 greedy。

## 3. 潜空间消融

`risk_active_latent` 相对 `risk_active`：

### Brier

| 工况 | 平均差（latent − active） | 95% 区间 |
|---|---:|---:|
| nominal | −0.000519 | [−0.003940, +0.003643] |
| headwind | −0.001067 | [−0.004285, +0.001419] |

点估计略负（latent 略低），但区间均跨零，不能称为独立 Brier 增益。

### 首解

| 工况 | W/L/T | 发现数 latent/active | 平均差 | 中位差 | 留一范围 |
|---|---:|---:|---:|---:|---:|
| nominal | 5/4/4 | 13 / 13 | −0.308 | 0 | [−0.583, +0.083] |
| headwind | 2/9/4 | 10 / 13 | **+1.133** | +1 | [+0.857, +1.500] |

nominal 无差异；headwind 下 latent 明显更慢（10 vs 13，平均多 1.13 次调用，留一全正）。潜空间没有独立首解增益，且在 headwind 有代价。

## 4. 判读与边界

- 本轮主动 arm 的明确收益是**相对 coverage 的冷启动发现速度**：13/13 vs 7/7，平均 −5.15 / −2.20 次调用，留一全负。
- 但同一主动 arm **没有**带来 Brier 改善：相对 coverage 的配对差为 +0.0007 / +0.0020，区间跨零；不能把首解改善写成校准改善。
- 潜空间版本没有独立 Brier 增益，headwind 首解反而更慢；不支持“潜空间是额外增益”的说法。
- 这是 16 个种子的降阶飞机实验，不是 RCAIDE 或真实飞行；没有默认方法替换，也没有安全结论。
- 结果只对应固定 SHA、固定种子和固定预算；没有改算法、种子或评价上限。

## 目录

```
active-risk-smoke/                  smoke 原始账本
active-risk-smoke-analysis/         smoke 分析
active-risk-nominal/                nominal 原始账本
active-risk-nominal-analysis/       nominal 分析（analysis.json）
active-risk-headwind/               headwind 原始账本
active-risk-headwind-analysis/      headwind 分析（analysis.json）
logs/                               每条命令的 stdout/stderr
python-version.log
tests.log
fixed_sha.txt
environment.txt
integrity_check.json
SHA256SUMS.txt
```
