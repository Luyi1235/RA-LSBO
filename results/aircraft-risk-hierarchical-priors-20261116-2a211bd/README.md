# PR112 — 布局级分层先验的冷启动风险

## 固定身份

- 固定代码 SHA：`2a211bd1178d904bed8384a84adf4fce1c3caa96`
- 分支：`research/aircraft-risk-hierarchical-priors-20261116`
- 结果分支：`results/aircraft-risk-hierarchical-priors-20261116-2a211bd`
- Python：`C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe`，3.9.25

## 执行与验收

| Profile | 调用 | 拆分 | 状态 |
|---|---:|---|---|
| `hierarchical-prior-smoke` | 29 | 4 + 9 + 16 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `hierarchical-prior-nominal` | 2,496 | 96 + 864 + 1,536 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `hierarchical-prior-headwind` | 2,496 | 96 + 864 + 1,536 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |

- 合计 **5,021** 次降阶飞机评价；RCAIDE/MDA **0**。
- 三套账本均 `VERIFIED_AIRCRAFT_RISK_COLD_START`，并生成 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`。
- 38 项专项测试通过；两个正式配置各 24 个种子、`discovery_pairs.csv` 各 **144** 行。
- 五个臂：`risk_coverage`、`risk_global_coverage`、`risk_layout_coverage`、`risk_active`、`risk_layout_active`。

## 1. 主比较：layout_coverage − coverage 的 Brier 配对差

来自 `analysis.json` 的 `learning_comparisons`；区间为配对 bootstrap 95%。

| 工况 | 平均 Brier 差 | 95% 区间 |
|---|---:|---:|
| nominal | **−0.001735** | **[−0.003375, −0.000309]** |
| headwind | **+0.000139** | **[−0.000321, +0.000704]** |

- **nominal：布局级分层先验小幅改善 Brier**，区间全负。
- **headwind：没有改善**，点估计略正，区间跨零。
- 同一比较在两个工况方向不一致，不能写成稳健的校准收益。

## 2. layout_coverage − global_coverage

| 工况 | 平均 Brier 差 | 95% 区间 |
|---|---:|---:|
| nominal | +0.000417 | [−0.000264, +0.001072] |
| headwind | **+0.000641** | **[+0.000314, +0.000974]** |

- nominal 对 global 无显著差异。
- **headwind 下 layout 显著差于 global**（区间全正）。布局级先验并不优于全局先验；在 headwind 反而更差。

## 3. 其他 Brier 比较

| 工况 | 比较 | 平均差 | 95% 区间 |
|---|---|---:|---:|
| nominal | global − coverage | −0.002152 | [−0.004139, −0.000430] |
| headwind | global − coverage | −0.000502 | [−0.001029, +0.000118] |
| nominal | active − coverage | +0.001956 | [−0.002652, +0.006960] |
| headwind | active − coverage | **+0.006763** | **[+0.003213, +0.011019]** |
| nominal | layout_active − layout_coverage | +0.004092 | [−0.001424, +0.010454] |
| headwind | layout_active − layout_coverage | **+0.005694** | **[+0.002517, +0.009315]** |

- global 在 nominal 明显好于 coverage（区间全负），headwind 只有边缘改善。
- 主动选择在 headwind 显著更差；把 active 叠在 layout prior 上也显著更差。
- 本轮没有任何“先验 + active 同时改善”的证据。

## 4. 首解结果

只统计初值无解种子；调用差为“前臂 − 后臂”，负值表示前臂更快。

### 三个 coverage 臂

| 工况 | 初值无解 | 比较 | W/L/T | 发现数 | 平均差 |
|---|---:|---|---:|---:|---:|
| nominal | 12 | layout − coverage | 0/0/12 | 5 / 5 | 0 |
| nominal | 12 | global − coverage | 0/0/12 | 5 / 5 | 0 |
| nominal | 12 | layout − global | 0/0/12 | 5 / 5 | 0 |
| headwind | 19 | layout − coverage | 0/0/19 | 4 / 4 | 0 |
| headwind | 19 | global − coverage | 0/0/19 | 4 / 4 | 0 |
| headwind | 19 | layout − global | 0/0/19 | 4 / 4 | 0 |

三个 coverage 臂的候选路径完全相同：分层先验改变概率估计，但没有改变覆盖选点，也没有改变首解。Brier 差异不能归因于发现路径变化。

### 主動臂

| 工况 | 比较 | W/L/T | 发现数 | 平均差 | 留一范围 |
|---|---|---:|---:|---:|---:|
| nominal | layout_active − active | 4/2/6 | 12 / 12 | −0.667 | [−1.000, −0.273] |
| headwind | layout_active − active | 6/4/9 | 14 / 15 | −0.263 | [−0.833, +0.278] |
| nominal | layout_active − layout_coverage | 10/2/0 | **12 / 5** | **−5.667** | [−6.818, −5.091] |
| headwind | layout_active − layout_coverage | 13/3/3 | **14 / 4** | **−5.000** | [−5.444, −4.611] |
| nominal | active − coverage | 10/2/0 | 12 / 5 | −5.000 | [−5.818, −4.364] |
| headwind | active − coverage | 13/3/3 | 15 / 4 | −4.737 | [−5.167, −4.333] |

- layout prior 叠在 active 上，对首解只有很小的 nominal 改善（−0.667，留一全负）和 headwind 无差异（−0.263，留一跨零）。
- 主动臂相对 coverage 的强首解收益（12/5、14/4；−5.7/−5.0）来自 active 机制本身，而不是 layout prior。

## 5. 判读

- layout prior 只在 nominal 相对 coverage 有小幅 Brier 改善；headwind 无改善。
- layout 相对 global 在 nominal 无差异、headwind 显著更差；没有证据支持“布局级分层”优于全局先验。
- 三个 coverage 臂选点完全相同，因此 nominal 的 Brier 改善没有伴随发现路径变化；是纯概率估计差异。
- 把 layout prior 叠到 active 上，Brier 在 headwind 显著变差；首解只有很小甚至无差异。
- 24 个种子的降阶飞机实验，非 RCAIDE/真实飞行；无默认方法替换、无安全结论。

## 目录

```
hierarchical-prior-smoke/                  smoke 原始账本
hierarchical-prior-smoke-analysis/         smoke 分析
hierarchical-prior-nominal/                nominal 原始账本
hierarchical-prior-nominal-analysis/       nominal 分析（analysis.json）
hierarchical-prior-headwind/               headwind 原始账本
hierarchical-prior-headwind-analysis/      headwind 分析（analysis.json）
logs/                                      每条命令 stdout/stderr
python-version.log
environment.txt
tests.log
fixed_sha.txt
integrity_check.json
SHA256SUMS.txt
```
