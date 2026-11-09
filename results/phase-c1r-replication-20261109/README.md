# PR81 / C1R：冻结 C1 方法的新种子复现

固定代码 SHA：`0a532a16e59c0f94e5263ad77a8a7e6f145196d1`  
实验身份：`c1r_fresh_seed_replication_v1`  
父实现：`62b46f7a30ab5c3475cd59b86c7f3ad51974a1da`  
PR80 结果 HEAD（仅引用）：`b3ebfec2c7771eef57919a0a96eb7a0223a266ed`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/81>  
环境：`Rcaide_vsp` / Python 3.9.25。

本目录是 PR81 的完整原始结果：smoke、full、独立 full-analysis、日志、环境、指令与 SHA256 清单。没有重跑 PR80，也没有读取旧标签训练。

## 1. 执行与费用

- 预检：38 项标准库测试 + 15 项原生测试通过，0 skip；compileall / help / plan 均 exit 0。
- 计划：`c1r_fresh_seed_replication_v1`，full 384 trials，12,288 次数学评价；种子块 930000，16 次/任务。
- smoke：264 次评价（192 initial + 72 online），24 trials；状态 `COMPLETE_C1_ONLINE_PILOT`，`model_failure_calls=0`；verify 11 files；十类篡改全部拒绝。
- full：12,288 次评价（3,072 initial + 9,216 online），384 trials，8,280.2 s；状态 `COMPLETE_C1_ONLINE_PILOT`；verify 71 files；`VERIFIED_C1_ONLINE`；0 new math/GP。
- 独立分析：`COMPLETE_C1R_ANALYSIS`；2 文件 manifest；只读，0 新增评价/拟合。
- 工作机正式评价 smoke+full = **12,552** 次；CI/测试计算另计，未重复执行。

## 2. 六臂总览（k24，全部 64 组）

| arm | 最终可行/64 | 改善初始目标 | 在线不可行/9216 | 新增精确目标标签 |
|---|---:|---:|---:|---:|
| `ei_frozen` | 64 | 32 | 1,226 | 394 |
| `risk_ei_frozen` | 64 | 47 | 951 | 586 |
| `ei_updated` | 64 | 37 | 1,433 | 178 |
| `risk_ei_updated` | 64 | 54 | 1,113 | 423 |
| `risk_only_updated` | 63 | 36 | 81 | 1,455 |
| `uniform_hash` | 64 | 16 | 1,409 | 155 |

- 所有 64 组中有 62 组初始已有可行解；2 组 `initial_cold`（s2 r15、s3 r4）。
- `risk_only_updated` 有 1 条无解轨迹；其余五臂最终 64/64 可行。
- 预测费用：11,945 GP heads，95,560 kernel started / 95,526 completed / **34 kernel rejects**，5,952 predictions，0 prediction errors；`model_failure_calls=0`。VAE / fantasy / aircraft / RCAIDE 均为 0。

## 3. 主比较：`risk_ei_updated` vs `risk_ei_frozen`

协议终点：initial_warm、24 次在线、每任务独立；双侧精确符号检验 + 四任务 Holm。

| task | n | W/L/T | 均值 Δ | 中位数 Δ | 原始 p | Holm p | Holm 拒绝 |
|---|---:|---:|---:|---:|---:|---:|---|
| tradeoff_s0 | 16 | 8/1/7 | −0.241422 | −0.051249 | 0.0391 | 0.0771 | 否 |
| tradeoff_s1 | 16 | 9/0/7 | −0.073967 | −0.005712 | 0.0039 | 0.0156 | **是** |
| tradeoff_s2 | 15 | 8/0/7 | −0.145763 | −0.015873 | 0.0078 | 0.0234 | **是** |
| tradeoff_s3 | 15 | 10/2/3 | −0.330876 | −0.078452 | 0.0386 | 0.0771 | 否 |

- 方向在所有任务一致为负（risk updated 目标更低），但只有 s1、s2 通过四任务 Holm 校正。
- 全部 64 组（含冷启动）：36/4/24；warm 汇总 35/3/24。
- **代价**：主比较的在线不可行评价增加 **+169**（warm），新增精确目标标签减少 **−170**。目标变好与失败增加同时存在，不能称为安全提升。
- 若把 cold 组也纳入，失败差 +162、标签差 −163；协议主分析仍是 warm 分层。

## 4. 次比较（k24，warm；逐任务均值 Δ）

| comparison | s0 | s1 | s2 | s3 | 总 W/L/T |
|---|---:|---:|---:|---:|---:|
| `ei_updated` vs `ei_frozen` | −0.3907 | +0.0910 | −0.4343 | −0.0798 | 17/17/28 |
| `risk_ei_updated` vs `ei_updated` | −0.0525 | −0.2577 | −0.1377 | +0.1358 | 43/7/14 |
| `risk_ei_frozen` vs `ei_frozen` | −0.2017 | −0.0927 | −0.4263 | +0.3868 | 26/12/26 |
| `risk_ei_updated` vs `risk_only_updated` | −0.2151 | −0.1906 | −0.5827 | −0.9946 | 52/6/6 |
| `risk_ei_updated` vs `uniform_hash` | −0.2524 | −0.1432 | −0.5756 | −0.0816 | 48/8/8 |

- 单纯“在线更新”（`ei_updated` vs `ei_frozen`）没有稳定收益；风险更新臂相对普通更新臂更好（43/7/14），但相对冻结风险臂的优势主要来自更新。
- `risk_only_updated` 目标差、失败和标签量与 `risk_ei_updated` 方向相反：它只有 81 次在线失败、1,455 个新标签，但最终只有 63/64 可行、36 次改善，目标更差；`risk_ei_updated` 用更多失败/更少标签换取目标改善。
- `risk_ei_frozen` 相对 `ei_frozen` 的收益在 s3 反向（+0.3868），说明冻结风险不是一致优势。
- 6/12 步为次要描述结果，不构成新的独立检验。

## 5. 机制与分层

- 32 组 warm + 2 组 cold；s0/s1 各 16 warm，s2/s3 各 15 warm + 1 cold。
- **冻结/更新首动作一致**：`ei_frozen` vs `ei_updated` 64/64；`risk_ei_frozen` vs `risk_ei_updated` 64/64。与 PR80 一致，差异从第一步之后才出现。
- 主比较路径改变 64/64；普通更新 vs 冻结 48/64；风险冻结 vs 普通冻结 63/64；其他匹配比较均 64/64。
- 首次可行位置：
  - 62/64 组 `first_feasible_call=0`（初始已有可行）。
  - s2 r15 cold：除 `risk_only_updated` 无解外，五臂均在第 12 次在线评价首次发现。
  - s3 r4 cold：除 `risk_only_updated` 在第 2 次发现外，五臂均在第 14 次发现。
- 无模型回退、无无可行初值重采样、无目标模型偷加；配置覆盖按臂记录在 `full-analysis/report.json` 的 `online_only_branch_counts`。

## 6. 判读

**C1R 复现了“风险 EI + 在线更新”相对冻结风险臂的方向性目标优势，但只在 4 个任务中的 2 个通过 Holm 校正；同时失败数增加、精确目标标签减少。**

- 主比较 35/3/24（warm），四个任务均值全为负；s1/s2 显著，s0/s3 未通过 Holm。
- 方向性证据比 PR80 的 8 种子 pilot 更强，但仍是 16 种子、4 个已知任务，不是 held-out s4–s7 确认。
- `risk_only_updated` 展示了低失败/高标签但目标更差的另一极；不能只看失败数或只看目标。
- 在线更新本身没有稳定收益；风险信号与更新的交互才是本轮观察对象。
- 无安全保证；34 个 kernel reject 与 0 model fallback 分开报告，未删除任何轨迹。

## 7. 限制

- 4 个已知 tradeoff 任务 × 16 种子；不是 held-out 任务确认，也没有飞机/RCAIDE。
- 主检验只检验胜负方向，不检验均值效应大小；未拒绝不是等价证明。
- 失败增加与目标改善同时存在；本轮不把失败折算成未注册的惩罚，也不声称安全。
- 相同初始前缀/首动作不意味着后续路径相同；路径改变 64/64。
- 一次轨迹的 k6/k12/k24 不是独立重复；不同任务原始目标不混合平均。

## 8. 文件

- `smoke/`、`full/`：完整原始运行，含 plan/source_receipt/environment/report/events/evaluations/progress/units/*/records.jsonl/manifest。
- `full-analysis/`：独立只读分析，含 report.json、trials.json、manifest.json。
- `logs/`：预检、smoke/full 控制台、verify、tamper、analysis 日志与退出码。
- `environment.txt`、`environment_packages.txt`、`fixed_sha.txt`、`instructions.txt`、`integrity_check.json`、`sha256_manifest.txt`。
- `.gitattributes`：`* -text`，防止换行转换改变证据字节。
