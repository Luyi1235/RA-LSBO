# PR107 — 终止反馈策略与冷启动策略的飞机因子实验

## 固定身份

- 代码仓库：`Luyi1235/RA-LSBO`
- 分支：`research/aircraft-risk-termination-feedback-20261115`
- 固定代码 SHA：`8980cb5850234e1feb6290b05d985d588dd6bf92`
- 结果分支：`results/aircraft-risk-termination-feedback-20261115-8980cb5`

## 执行与验收

| 组 | Profile | 调用 | 角色拆分 | 状态 |
|---|---|---|---:|---|
| smoke | `termination-native-smoke` | 24 | 4 + 8 + 12 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| smoke | `termination-structural-smoke` | 24 | 4 + 8 + 12 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| formal | `termination-native` | 1,824 | 96 + 960 + 768 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |
| formal | `termination-structural` | 1,824 | 96 + 960 + 768 | `COMPLETE_AIRCRAFT_RISK_COLD_START` |

- 合计 **3,696** 次降阶飞机调用；RCAIDE/MDA **0**。
- 四个 profile 均通过 `verify`：`VERIFIED_AIRCRAFT_RISK_COLD_START`。
- 四个 profile 均生成 `COMPLETE_AIRCRAFT_RISK_ANALYSIS`。
- 两套正式分析各 24 条 `discovery_pairs.csv` 数据行。
- 跨策略核验：`factorial.json` 状态 `VERIFIED_TERMINATION_FEEDBACK_FACTORIAL`；`factorial-smoke.json` 状态 `VERIFIED_TERMINATION_FEEDBACK_FACTORIAL`。
- 匹配评价设计 1,126 个；匹配固定设计 864 个。

## 留出终止频数

`factorial.json` 的 `runs`：

| 策略 | complete | physical SOC reserve abort | root bending | stall |
|---|---:|---:|---:|---:|
| native | 114 | 654 | — | — |
| structural | 31 | 508 | 128 | 101 |

- native 视图的留出终止几乎只有 `physical_soc_reserve_abort`，没有结构/失速终止项。
- structural 视图把 508 条归为 SOC 预留中止，另有 128 条 `root_bending` 与 101 条 `stall`；`complete` 降至 31。
- 两套都保留 `physical_soc_reserve_abort` 为主终止类别；改变的是终止原因的可分辨结构，而不是消除 SOC 中止。

## 首解差（`lsbo_sign_latent` vs `discovery_staged`）

只统计初值无解种子；调用差为“sign_latent 首次可用解调用数 − discovery_staged 首次可用解调用数”，负值表示更快。

| 视图 | 初值无解 | W/L/T | 平均调用差 | 中位差 | 留一范围 | 发现数（臂/基线） |
|---|---:|---:|---:|---:|---|---:|
| native | 20 | 12 / 6 / 2 | −1.05 | −1.5 | [−1.684, −0.579] | 20 / 18 |
| structural | 20 | 8 / 7 / 5 | −0.45 | 0.0 | [−1.474, +0.526] | 12 / 8 |

- native 视图下 sign_latent 相对 staged 有约 1 次调用的小幅首解优势，留一范围完全为负。
- structural 视图下平均差 −0.45，但中位差为 0、留一范围跨零，不能称为稳定优势。
- structural 视图的绝对发现数明显更低（12/20 vs native 20/20），说明终止反馈视图改变了可发现性，而不只是提前/延后。

## 主交互

`factorial.json.interaction`：

| 项目 | 值 |
|---|---:|
| 初值无解种子 | 20 |
| 平均交互差 | **+0.600** 次调用 |
| 中位交互差 | **+2.000** 次调用 |
| 留一范围 | **[−0.368, +1.632]** |

交互差按同种子配对计算；正号表示 native 视图的首解优势相对 structural 视图更大（或劣势更小），详见 `factorial.json` 的 `seed_pairs`。均值 +0.6 但留一范围跨零，且种子间从 −19 到 +19，不能把该交互写成稳定方向。

## 判读与边界

- 这是降阶飞机、已知任务族的探索性因子实验；不是 X-57/RCAIDE 高保真验证。
- native 与 structural 的终止频数差异说明终止反馈视图实质改变了终态标签结构；但它没有带来稳定的跨视图首解交互。
- native 的小幅首解优势留一范围全负，但幅度约 1 次调用；structural 的均值优势不稳健。
- 主交互均值 +0.6、留一跨零，不能支持“终止反馈视图与冷启动策略存在稳定协同”的结论。
- 没有默认方法替换；没有安全或真实飞行结论。

## 目录

```
factorial.json           正式跨策略因子核验
factorial-smoke.json     smoke 因子核验
native/                  native 原始账本
native-analysis/          native 分析
structural/              structural 原始账本
structural-analysis/      structural 分析
native-smoke/            native smoke 原始账本
structural-smoke/        structural smoke 原始账本
logs/                    plan/run/verify/analyze/factorial 日志、测试、解释器版本
fixed_sha.txt
SHA256SUMS.txt
```
