# PR99：PR98 飞机候选池真值审计

固定代码 SHA：`48a26e41a0559dcbc4353b79086cfc629c6c19c0`  
输入 PR98 结果 HEAD：`5e521d234d8ce264156f72b67155949e9423de23`
（对应实验代码 `971bd052afb546dba501a4864ae3954dc3babeaa`）

本目录是只读审计结果；没有重新运行 PR98 优化，也没有调用飞机/RCAIDE。唯一新增计算是池候选真值评价。

## 1. 规模与完整性

- Smoke：20 次新增评价，`VERIFIED_AIRCRAFT_RISK_POOL_AUDIT`；24 个池候选，1 个已知可用；4 个端点、12 个逐步诊断。
- Pilot：836 次新增评价，`VERIFIED_AIRCRAFT_RISK_POOL_AUDIT`；1,536 个池候选、48 条轨迹终点、1,728 条原始选点诊断；`COMPLETE_AIRCRAFT_RISK_POOL_ANALYSIS`。
- 1538 个候选真值：**155 个 usable**、1,381 个不可用、0 unknown。
- 0 RCAIDE/MDA 调用；源 zip 解压 131 项，`testzip=None`。

## 2. 终点池差距

`known_pool_gap_Wh` 是轨迹最终已观测最优与已知池最优之差；`unseen_better_count_at_end` 是结束时仍在池中但未被选中的更优点数。

| arm | 最终最优 Wh 均值 | 池差距均值 Wh | 差距为 0 的 seed | 结束未选更优点 | opportunity steps | conversion steps |
|---|---:|---:|---:|---:|---:|---:|
| `ei_explicit` | 39,638.39 | 476.65 | 9/12 | 0.25 | 17.67/36 | 1.67 |
| `cei_explicit` | 39,473.15 | 311.41 | 11/12 | 0.08 | 13.75/36 | 2.08 |
| `ei_latent` | 39,638.39 | 476.65 | 9/12 | 0.25 | 17.08/36 | 1.58 |
| `cei_latent` | 39,473.15 | 311.41 | 11/12 | 0.08 | 13.42/36 | 2.17 |

- CEI 的终点池差距明显小于 EI（311 vs 477 Wh），但 12 个 seed 中仍有 1 个存在未选中的更优设计。
- 潜空间臂与对应显式臂的终点最优完全相同；没有独立增益。

## 3. 机会与转化

“opportunity step”指当时池中仍存在已知更优可用设计；“selected_improves”指实际选点是否改善 incumbent。

| arm | opportunity steps /432 | 实际改善 incumbent | 机会转化率 | best improver 中位排名 | 最优点被选中 |
|---|---:|---:|---:|---:|---:|
| `ei_explicit` | 212（49.1%） | 20 | 9.4% | 51.5 | 8 |
| `cei_explicit` | 165（38.2%） | 25 | 15.2% | 26 | 10 |
| `ei_latent` | 205（47.5%） | 19 | 9.3% | 46 | 8 |
| `cei_latent` | 161（37.3%） | 26 | 16.1% | 26 | 10 |

- 有大量池内机会没有被选中：EI 约一半在线步骤存在更优池内点，CEI 约 38%。
- 真正改善 incumbent 的在线选点只有 20–26 次；CEI 的转化率高于 EI，但仍然很低。
- 选中的最优点排名中位数 26–52；CEI 更接近池最优，但远非精确。

## 4. 同状态 oracle 对比

在同一状态下比较实际选点与三个“事后 oracle”选择：同状态 EI、同状态 CEI、同状态已知可用 EI。

| arm | 等于同状态 EI 选择 | 等于同状态 CEI 选择 | 等于 known-usable EI 选择 | EI oracle 会改善 | CEI oracle 会改善 | known-usable oracle 会改善 |
|---|---:|---:|---:|---:|---:|---:|
| `ei_explicit` | 399/432 | 59/432 | 49/432 | 16 | 50 | 91 |
| `cei_explicit` | 80/432 | 399/432 | 49/432 | 31 | 21 | 93 |
| `ei_latent` | 399/432 | 62/432 | 55/432 | 15 | 48 | 55 |
| `cei_latent` | 68/432 | 399/432 | 49/432 | 35 | 22 | 83 |

- 每个臂基本沿用自己的同状态 oracle（399/432），但实际选点只在少数步骤改善。
- known-usable EI oracle 在 83–93 个步骤会改善，而实际选点只匹配该 oracle 49–55 次；说明“知道候选可用”与“当前采集规则选中它”之间存在明显缺口。
- 这不是反事实收益：oracle 使用事后真值，只能说明池内机会存在，不能说明在线可获得。

## 5. 判读

- 池审计确认：在这 12 个 seed 的 140 km sparse 飞机任务中，候选池包含可观的可用设计（155/1536），但两种采集规则都只利用了其中很小一部分。
- CEI 比 EI 更接近池最优（311 vs 477 Wh），但仍有 38% 的机会步骤没有转化；没有方法在 36 次在线评价内稳定找到池最优。
- 潜空间与显式表示终点完全一致，没有独立收益；这与 PR98 的能量结论一致。
- 本审计只回答“错过的池内机会有多大”，不支持任何新采集算法或安全结论，也不能外推到 RCAIDE。

## 6. 文件

- `smoke/`、`smoke-analysis/`：20 次评价的审计输入与 24 候选结果。
- `pilot/`、`pilot-analysis/`：836 次评价、1,536 候选、48 端点、1,728 步骤诊断。
- `source/`、`pr98-source.zip`：被审计的 PR98 结果快照（131 项，zip SHA256 `9c99df9a...740d`）。
- `logs/`：plan/run/verify/analyze 全日志与退出码。
- `fixed_sha.txt`、`environment.txt`、`environment_packages.txt`、`SHA256SUMS.txt`。
- `.gitattributes`：`* -text`。
