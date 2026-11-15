# PR105 — 约束反馈基线：简化飞机首解诊断

## 固定身份

- 代码仓库：`Luyi1235/RA-LSBO`
- 分支：`research/aircraft-risk-feedback-baselines-20261115`
- 固定代码 SHA：`8dd16819e82ef25126f6b3f8fd75b64558bcea34`
- 结果分支：`results/aircraft-risk-feedback-baselines-20261115-8dd1681`
- 本目录是结果分支中的证据根目录。

## 执行与验收

| Profile | 数学/简化飞机调用 | 角色拆分 | 状态 |
|---|---:|---|---|
| `feedback-baseline-smoke` | 28 | 4 initial + 12 online + 12 held-out | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `feedback-baseline-headwind` | 2,304 | 96 initial + 1,440 online + 768 held-out | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| `feedback-baseline-payload` | 2,304 | 96 initial + 1,440 online + 768 held-out | `COMPLETE_AIRCRAFT_RISK_COLD_START` |

- 合计 **4,636** 次简化飞机调用；RCAIDE/MDA 调用 **0**。
- 三个 profile 均通过独立 `verify`：`VERIFIED_AIRCRAFT_RISK_COLD_START`。
- 三个 profile 均生成 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`。
- 正式 profile 各有 **72** 条 `discovery_pairs.csv` 数据行。
- 运行耗时：headwind 2,749.2 s；payload 2,604.0 s；smoke 17.4 s。

## 初值无解种子

每个正式工况共 24 个种子。

| 工况 | 初值无解种子 |
|---|---:|
| headwind | 22 / 24 |
| payload | 24 / 24 |

下面的胜负平和受限平均调用差只针对初值无解种子；调用差定义为“该臂首次可用解调用数 − 基线首次可用解调用数”，负值表示更快，未发现按上限截断。

## 主比较：`lsbo_sign_latent` vs `discovery_staged`

| 工况 | W/L/T | 平均调用差 | 中位调用差 | 留一范围 | 发现数（臂/基线） |
|---|---:|---:|---:|---|---:|
| headwind | 11 / 11 / 0 | −0.091 | −0.5 | [−0.810, +0.857] | 21 / 20 |
| payload | 12 / 9 / 3 | −0.083 | −0.5 | [−0.696, +0.609] | 24 / 22 |

主比较在两种工况下都接近平局：平均差不到 0.1 次调用，留一范围跨零。不能据此声称 sign-aware latent 比 staged discovery 更快。

## 次比较：相对 `discovery_direct`

| 工况 | 比较 | W/L/T | 平均调用差 | 中位调用差 | 留一范围 | 发现数（臂/基线） |
|---|---|---|---:|---:|---|---:|
| headwind | sign_latent vs direct | 17 / 4 / 1 | −6.364 | −8.0 | [−7.381, −5.714] | 21 / 13 |
| headwind | staged vs direct | 12 / 6 / 4 | −6.273 | −5.5 | [−6.952, −5.714] | 20 / 13 |
| payload | sign_latent vs direct | 19 / 5 / 0 | −8.917 | −12.0 | [−9.565, −8.435] | 24 / 10 |
| payload | staged vs direct | 18 / 4 / 2 | −8.833 | −11.5 | [−9.913, −8.348] | 22 / 10 |

相对 `discovery_direct`，两个带首解阶段的臂都稳定更快；但这属于“是否使用 staged/discovery 机制”的差异，不是 sign-aware latent 相对 staged 的增量。

## 第 24 次调用发现率（初值无解子组）

分母为该工况初值无解种子数；分子为该臂在 ≤24 次在线调用内首次找到可用解的种子数。

| 工况 | 方法 | ≤24 次发现 / 初值无解 | 比例 |
|---|---|---:|---:|
| headwind | `lsbo_sign_latent` | 21 / 22 | 95.5% |
| headwind | `discovery_staged` | 20 / 22 | 90.9% |
| headwind | `discovery_direct` | 13 / 22 | 59.1% |
| payload | `lsbo_sign_latent` | 24 / 24 | 100.0% |
| payload | `discovery_staged` | 22 / 24 | 91.7% |
| payload | `discovery_direct` | 10 / 24 | 41.7% |

第 24 次调用前的发现率与最终发现数一致：本实验中的首解事件都发生在 24 次在线调用以内或未发生；因此该列不是“第 24 次之后”的尾部发现率。

## 判读与边界

- 本轮是简化飞机首解诊断；不是 RCAIDE 高保真验证，也不是新的在线算法胜出证明。
- 主比较 `sign_latent` 相对 `discovery_staged` 基本平局；相对 `discovery_direct` 的大幅领先主要由 staged 首解阶段贡献。
- 没有默认方法替换；没有安全或真实飞行结论。
- 联合/能耗配对与终端指标保留在 `analysis.json`、`joint_pairs.csv` 和 `metrics.csv`，但本轮文档只按要求展开首解子组；不把探索性能写成最终优化收益。

## 目录

```
smoke/                   smoke 原始账本、契约、报告、manifest、calls
smoke-analysis/          smoke 分析
headwind/                headwind 原始账本
headwind-analysis/       headwind 分析（analysis.json、discovery_pairs.csv、budget_pairs.csv）
payload/                 payload 原始账本
payload-analysis/        payload 分析
logs/                    plan/run/verify/analyze 日志、测试日志、Python 与依赖版本
fixed_sha.txt
SHA256SUMS.txt
```

## 失败与恢复记录

第一次 headwind 批量运行使用了 PATH 中解析到的 Python 3.14（无 numpy），在入口导入阶段失败；失败日志和部分目录保存在代码 worktree 之外的本地证据根目录（`.../feedback-baseline-headwind-failed-*`），未纳入本上传。随后使用 `Rcaide_vsp` Python 3.9.25 完整重跑；本目录中的 headwind/payload 均为重跑并通过 verify/analyze 的结果。

## 完整性

- 代码 worktree 在运行前后均为固定 SHA 且干净。
- 所有正式运行 `source_sha` 均等于 `8dd16819e82ef25126f6b3f8fd75b64558bcea34`。
- `SHA256SUMS.txt` 覆盖本目录除自身外的全部文件；上传后逐文件核验远端字节。
- 目录级 `.gitattributes` 为 `* -text`，避免 Git 换行转换破坏原始证据。
