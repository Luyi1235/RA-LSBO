# PR113 — 构型均衡的冷启动风险信息获取

## 固定身份

- 固定代码 SHA：`87ec0dc4cdf662a7f41604d24a7d0dff22415e0e`
- 分支：`research/aircraft-risk-stratified-acquisition-20261116`
- 结果分支：`results/aircraft-risk-config-balance-20261116-87ec0dc`
- Python：`C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe`，3.9.25

## 执行与验收

| Profile | 调用 | 拆分 | 状态 |
|---|---:|---|---|
| `config-balance-smoke` | 35 | 4 + 15 + 16 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `config-balance-nominal` | 3,072 | 96 + 1,440 + 1,536 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `config-balance-headwind` | 3,072 | 96 + 1,440 + 1,536 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |

- 合计 **6,179** 次降阶飞机评价；RCAIDE/MDA **0**。
- 三套账本均 `VERIFIED_AIRCRAFT_RISK_COLD_START` 并生成 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`。
- 41 项专项测试通过；两个正式配置各 24 个种子、`discovery_pairs.csv` 各 **120** 行。
- 五个臂：`risk_coverage`、`risk_active`、`risk_joint_interleaved`、`risk_balanced_coverage`、`risk_balanced_active`。

## 1. 主比较：`risk_balanced_active − risk_coverage` 的 Brier 配对差

来自 `analysis.json` 的 `learning_comparisons`；区间为配对 bootstrap 95%。

| 工况 | 平均 Brier 差 | 95% 区间 |
|---|---:|---:|
| nominal | **−0.001034** | **[−0.004875, +0.002803]** |
| headwind | **+0.001466** | **[−0.001513, +0.004599]** |

- 两个工况区间均跨零：**构型均衡主动臂相对纯覆盖没有显著 Brier 改善**。
- nominal 点估计略负、headwind 略正；方向不一致。

## 2. `risk_balanced_active − risk_balanced_coverage`

与上一节数值完全一致：

| 工况 | 平均 Brier 差 | 95% 区间 |
|---|---:|---:|
| nominal | −0.001034 | [−0.004875, +0.002803] |
| headwind | +0.001466 | [−0.001513, +0.004599] |

原因是 `risk_balanced_coverage` 与 `risk_coverage` 在本轮完全相同：

| 工况 | Brier 差 balanced_coverage − coverage | 95% 区间 | 发现数 |
|---|---:|---:|---:|
| nominal | **0.000000** | **[0.0, 0.0]** | 9 / 9 |
| headwind | **0.000000** | **[0.0, 0.0]** | 7 / 7 |

`risk_balanced_coverage` 与 `risk_coverage` 的选点路径、Brier 和首解完全一致；构型均衡单独作用于覆盖臂时**没有产生可观察差异**。因此 balanced_active 的差异不能归因于“balanced coverage 本身更好”，而只能来自 active + balancing 的组合。

## 3. `risk_balanced_active` 相对其他主动臂

| 工况 | 比较 | 平均 Brier 差 | 95% 区间 |
|---|---|---:|---:|
| nominal | balanced_active − active | **−0.003201** | **[−0.005998, −0.000559]** |
| headwind | balanced_active − active | **−0.002341** | **[−0.004994, +0.000046]** |
| nominal | balanced_active − joint_interleaved | −0.001147 | [−0.003868, +0.001395] |
| headwind | balanced_active − joint_interleaved | −0.000444 | [−0.003064, +0.002051] |

- 相对普通 active，balanced_active 在 nominal 显著更好（区间全负），headwind 边缘更好（上界几乎为 0）。
- 相对 joint_interleaved 无显著差异。

## 4. 首解

只统计初值无解种子；调用差为“前臂 − 后臂”，负值表示前臂更快。

| 工况 | 初值无解 | 比较 | W/L/T | 发现数 | 平均差 | 留一范围 |
|---|---:|---|---:|---:|---:|---:|
| nominal | 15 | balanced_active − coverage | 12/3/0 | **15 / 9** | **−4.267** | [−4.786, −3.714] |
| headwind | 19 | balanced_active − coverage | 11/5/3 | **13 / 7** | **−3.263** | [−3.944, −2.778] |
| nominal | 15 | balanced_active − active | 1/3/11 | 15 / 15 | +0.133 | [−0.143, +0.429] |
| headwind | 19 | balanced_active − active | 4/1/14 | 13 / 12 | **−0.632** | [−0.889, −0.278] |
| nominal | 15 | balanced_active − joint_interleaved | 5/7/3 | 15 / 15 | +0.133 | [−0.071, +0.429] |
| headwind | 19 | balanced_active − joint_interleaved | 4/6/9 | 13 / 12 | +0.474 | [+0.056, +0.889] |

- 相对纯覆盖，balanced_active 的发现数为 15/9、13/7，平均提前 4.27 / 3.26 次，留一全负：这是稳定的首解收益。
- 相对普通 active，nominal 略慢（+0.133，方向混合），headwind 略快（−0.632，留一全负）；两个工况不一致。
- 相对 joint_interleaved，两个工况均略慢或打平，没有发现优势。

## 5. 判读

- 构型均衡主动臂相对纯覆盖有强首解收益，但没有显著 Brier 改善。
- 单独把构型均衡加到覆盖臂没有改变任何选点或 Brier；它只在 active 组合中产生差异。
- 相对普通 active，balanced_active 的 Brier 有改善（nominal 显著、headwind 边缘），首解方向随工况不一致。
- 没有证据支持“构型均衡本身”是独立机制；也没有证据支持它优于 joint_interleaved。
- 24 个种子的降阶飞机实验，非 RCAIDE/真实飞行；无默认方法替换、无安全结论。

## 目录

```
config-balance-smoke/                  smoke 原始账本
config-balance-smoke-analysis/         smoke 分析
config-balance-nominal/                nominal 原始账本
config-balance-nominal-analysis/       nominal 分析（analysis.json）
config-balance-headwind/               headwind 原始账本
config-balance-headwind-analysis/      headwind 分析（analysis.json）
logs/                                  每条命令 stdout/stderr
python-version.log
environment.txt
tests.log
fixed_sha.txt
integrity_check.json
SHA256SUMS.txt
```
