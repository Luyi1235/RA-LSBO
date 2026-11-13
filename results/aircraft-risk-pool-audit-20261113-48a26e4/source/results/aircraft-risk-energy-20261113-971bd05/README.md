# Aircraft risk + energy search pilot（PR #98）

固定代码 SHA：`971bd052afb546dba501a4864ae3954dc3babeaa`  
分支：`research/aircraft-risk-energy-20261113`  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本目录保留 smoke 与 energy-pilot 的全部原始输出、独立 verify、独立 analyze、日志和 SHA256 清单。

## 1. 执行与验收

| 项目 | 结果 |
|---|---:|
| 单元测试 | 8 passed |
| Smoke | 32 calls；`VERIFIED_AIRCRAFT_RISK_COLD_START`；`COMPLETE_AIRCRAFT_RISK_ANALYSIS` |
| Pilot | **2,640 calls** = 144 initial + 1,728 online + 768 held-out |
| Pilot 状态 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| Verify | `VERIFIED_AIRCRAFT_RISK_COLD_START`，2,640 calls |
| Analysis | `COMPLETE_AIRCRAFT_RISK_ANALYSIS` |
| 种子 | 28417000–28418100，共 12 个 |
| 实验臂 | `ei_explicit`、`cei_explicit`、`ei_latent`、`cei_latent` |
| 每臂 | 12 initial + 36 online；n=12/24/36/48 |
| 飞机 / RCAIDE | 0 / 0 |
| 未结算 / 未知 | 0 / 0 |

## 2. n=48 联合风险

| arm | Brier | Skill | AUROC | feasible recall | false feasible | ECE |
|---|---:|---:|---:|---:|---:|---:|
| ei_explicit | .05152 | .3758 | .9470 | .6539 | .0402 | .0893 |
| cei_explicit | **.05124** | **.4005** | .9565 | .6651 | .0447 | .0923 |
| ei_latent | .05467 | .3354 | .9398 | .6053 | .0456 | **.0823** |
| cei_latent | .05184 | .3850 | **.9612** | **.6866** | **.0358** | .0988 |

- `cei_explicit` 的 Brier 最低，但与 `ei_explicit` 只差 **.00028**；`cei_latent` 的 AUROC 与召回最高，Brier 略高于 `cei_explicit`。
- `ei_latent` 在 EI 臂上比显式差（Brier +.00315、AUC −.0072、召回 −4.9 pp）。
- `cei_latent` 在 CEI 臂上比显式略好（AUC +.0047、召回 +2.2 pp、FF −0.9 pp），但 Brier 略差（+.0006）。
- 潜空间没有一致独立增益：一个臂改善、一个臂变差。

## 3. 能量主比较

`pilot-analysis/energy_pairs.csv` 记录 4 个消融对 ×12 种子：

| 比较 | 胜/负/平 | 胜出种子 | ΔWh |
|---|---:|---|---:|
| `cei_explicit` vs `ei_explicit` | 2/0/10 | 28417400、28417700 | −1,543.77、−439.09 |
| `cei_latent` vs `ei_latent` | 2/0/10 | 28417400、28417700 | −1,543.77、−439.09 |
| `ei_latent` vs `ei_explicit` | 0/0/12 | — | 0 |
| `cei_latent` vs `cei_explicit` | 0/0/12 | — | 0 |

- 全部 48 个配对中只有 **4 次 CEI 胜出**，其余 44 次为平局，**0 次失败**。
- 能量优势只出现在 2 个种子；其余种子的最终最优 Wh 与 EI 完全相同。
- **latent 与 explicit 的最优能量在所有 12 个种子上完全相同**；本轮没有证据表明潜空间改变了能量最优值。
- 这不能写成 CEI 的能量收益：样本只有 2 个非平局种子，且最终最优值多由共享初始点决定。

描述性轨迹：

| arm | 首次可行均值 | online 可行数 | 最优 Wh 中位数 |
|---|---:|---:|---:|
| ei_explicit | 8.5 | 53 | 39,349.9 |
| cei_explicit | 8.5 | 99 | **39,310.4** |
| ei_latent | 8.5 | 59 | 39,349.9 |
| cei_latent | 8.5 | 103 | **39,310.4** |

CEI 明显增加了 online 可行评价数（约 +46–48），但只把最优能量中位数降低约 40 Wh；这是“更多可行探索”，不是已证明的优化收益。

## 4. 判读

- **风险表示**：显式与潜空间的结果互有胜负；`ei_latent` 更差，`cei_latent` 在排序/召回上略好。没有一致的潜空间优势。
- **采集准则**：CEI 相对 EI 提高了 AUC、召回和 online 可行数，但 Brier 改善极小（.0003–.0028），最优能量只在 2/12 种子上改善。
- **Brier 与召回没有明显取舍**：CEI 的召回提高、FF 未恶化；但这是同一批 12 个种子的描述性结果，不是确认性显著性结论。
- 单任务、12 种子、简化飞机模型、无 RCAIDE；不能外推为飞机优化闭环或安全保证。

## 5. 文件

- `smoke/`、`smoke-analysis/`：32-call 技术检查。
- `pilot/`：2,640-call 原始证据（evaluations、progress、plan、source receipt、manifest、报告）。
- `pilot-analysis/`：`metrics.csv`、`joint_pairs.csv`、`energy_pairs.csv`、`analysis.json`、`manifest.json`。
- `logs/`：测试、plan、smoke run/verify/analyze、pilot run/verify/analyze 的完整 stdout/stderr 和退出码。
- `fixed_sha.txt`、`environment.txt`、`environment_packages.txt`、`SHA256SUMS.txt`。
- `.gitattributes`：`* -text`，防止换行转换改变证据字节。
