# PR102：首可用解的分阶段发现策略

固定代码 SHA：`9234809ffd35bfd49fc6f5fa06ce60ae3ef7334b`  
分支：`research/aircraft-risk-staged-discovery-20261114`

本目录是简化飞机原始运行、独立验证、分析与日志。没有调用 RCAIDE/MDA。

## 1. 执行与费用

| 配置 | 种子 | 评价数 | 结构 | 状态 |
|---|---|---:|---|---|
| discovery-smoke | 28429000 | 38 | 8 初始 +10 在线 +20 留出 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| discovery-nominal | 28430000–28431500（16） | 2,112 | 144 +768 +1,200 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| discovery-headwind（4 m/s） | 28430000–28431500（16） | 2,112 | 144 +768 +1,200 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| **合计** | | **4,262** | | |

- 14 项专项测试通过；三套 verify 均为 `VERIFIED_AIRCRAFT_RISK_COLD_START`
- 三套 analyze 均为 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`
- 两套正式 `discovery_pairs.csv` 各 **80** 行（16 seed ×5 对照）
- RCAIDE/MDA：**0**

## 2. 初值无解与主比较

“初值无解”指 12 个共享初始评价没有任何 usable 设计。此时比较首次发现 usable 设计的调用位置。

| 工况 | 初值无解 seed | 主比较（lookahead vs collapsed）W/L/T | 受限平均首次发现差（候选−对照，call） | 中位差 |
|---|---:|---:|---:|---:|
| nominal | **9/16** | **0/5/11** | **+3.11** | +2 |
| headwind | **12/16** | **4/2/10** | **−0.58** | 0 |

- 无风时 lookahead 从未更好、5 次更差；平均晚约 3.1 次调用才首次发现 usable 设计。
- 逆风时 lookahead 4 胜 2 负，平均早约 0.6 次；方向与无风相反，幅度也小。
- 两个工况都有大量平局：无风 11/16，逆风 10/16。

## 3. 第 20 次调用时的发现率

在初值无解 seed 中，统计首次 usable 调用 ≤20 的比例：

| 工况 | lookahead | collapsed | 差值 |
|---|---:|---:|---:|
| nominal | 7/9 = **77.8%** | 8/9 = **88.9%** | −11.1 pp |
| headwind | 8/12 = **66.7%** | 7/12 = **58.3%** | +8.3 pp |

无风时 collapsed 的 20 次内发现率更高；逆风时 lookahead 略高。两个方向都不稳定。

## 4. 其他配对（W/L/T，平均调用差）

### nominal

| 比较 | W/L/T | 平均 Δcall |
|---|---:|---:|
| lookahead vs staged | 4/2/10 | +0.25 |
| lookahead vs coverage | 4/2/10 | −1.44 |
| staged vs direct | 4/2/10 | −0.69 |
| staged vs constraints | 2/5/9 | +1.75 |

### headwind

| 比较 | W/L/T | 平均 Δcall |
|---|---:|---:|
| lookahead vs staged | 4/3/9 | +0.19 |
| lookahead vs coverage | 6/0/10 | −2.88 |
| staged vs direct | 5/2/9 | −2.50 |
| staged vs constraints | 7/3/6 | −1.06 |

- 无风时 staged 相对 direct 略好（−0.69），但相对 constraints 更差（+1.75）。
- 逆风时 staged 相对 direct 明显更早（−2.50），相对 constraints 也更早（−1.06）。
- 无风与逆风的主比较方向相反，不能把 lookahead 写成稳定改进。

## 5. 判读

- 本实验只检验“初始无解时首可用解发现速度”；不评价最终能量、风险 Brier 或优化收益。
- 无风时 collapsed 更好（0/5/11；20 次发现率 88.9% vs 77.8%）。
- 逆风时 lookahead 略好（4/2/10；20 次发现率 66.7% vs 58.3%），但平均只早 0.58 次，且 10/16 平局。
- 两种工况方向相反，说明该机制对风况敏感；没有跨工况一致的发现优势。
- 16 个种子/工况、简化飞机、最多 48 次评价；结果探索性，不支持默认方法或安全声明，也不能外推到 RCAIDE。

## 6. 文件

- `discovery-smoke/`、`discovery-smoke-analysis/`：38 次评价、20 条 discovery pair。
- `discovery-nominal/`、`discovery-nominal-analysis/`：2,112 次评价、80 条 pair。
- `discovery-headwind/`、`discovery-headwind-analysis/`：同上，4 m/s 逆风。
- `logs/`：tests、三套 plan/run/verify/analyze 全日志和退出码。
- `fixed_sha.txt`、`environment.txt`、`environment_packages.txt`、`SHA256SUMS.txt`、`README.md`。
- `.gitattributes`：`* -text`。
