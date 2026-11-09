# PR83 C2：风险头 × 目标头 2×2 因子消融

固定代码 SHA：`975d96caac53d675ebe1b790a0647479a75a2ed5`  
CI run：`42611798409`（固定 SHA，success）  
环境：`Rcaide_vsp` / Python 3.9.25。

本目录是 PR83 C2 的完整原始证据：smoke、full、full-analysis、日志、测试、计划、账本与逐 trial 记录。所有数值来自真实在线数学优化；本轮没有 VAE、幻想、飞机或 RCAIDE 调用。

## 1. 结论先行

**2×2 因子消融没有产生任何在 Holm 校正后显著的风险头或目标头主效应。**

- 风险更新的点估计在四个任务上全部为负（更好的目标），但只有 s0/s1 的名义符号检验 p=0.00635，Holm 后均为 0.0762，未拒绝。
- 目标更新的点估计也全为负，但 s1 的 p=0.0574（Holm 0.574）也不显著。
- 交互项在四个任务上方向不一致，s0 +0.0397、s1 −0.0218、s2 −0.0135、s3 +0.2209，均不显著。
- 风险更新确实降低了在线失败次数（927→497，冻结目标头时），但目标头更新反而增加失败（927→1306）；两者同时更新为 1115。这个失败/标签代价与目标收益方向相反，不能只报目标差。
- 所有 64 个 case/seed 在四臂中最终都有可行解；没有无解运行、没有冷启动、首次可行调用均为 0。因子效应只在“已有可行解的 warm 状态”的目标值上，不在可行性上。

## 2. 执行与完整性

| 项目 | 结果 |
|---|---|
| 34 core + 15 native 测试 | 通过，0 skip |
| compileall / --help / plan | exit 0 |
| smoke | 264 calls（192 initial+72 online），24 trials，`COMPLETE_C2_FACTORIAL` |
| smoke verify | 11 files，`VERIFIED_C2_FACTORIAL` |
| full | 12,288 calls（3,072 initial+9,216 online），384 trials，7,732.4 s，`COMPLETE_C2_FACTORIAL` |
| full verify | 71 files，`VERIFIED_C2_FACTORIAL`，0 new math/GP |
| full analysis | `COMPLETE_C2_FACTORIAL_ANALYSIS` |
| tamper | smoke/full 均 `REJECTED_ALL_C2_TAMPERING` |
| 未知/未结算 | 0；12,288 started = settled |
| 飞机 / RCAIDE / VAE / 幻想 | 0 / 0 / 0 / 0 |

## 3. 冻结设计

四个 task：`tradeoff_s0..s3`；16 repeats/task；四格因子：

- 风险头：`frozen` vs `updated`
- 目标头：`frozen` vs `updated`

四格为 `risk_frozen_objective_frozen`、`risk_updated_objective_frozen`、`risk_frozen_objective_updated`、`risk_updated_objective_updated`；另有两个参照臂 `risk_only_updated`、`uniform_hash`。

每 task/repeat 预注册种子 `960000+block+repeat`；**四格完整 16/16，missing 0/16，四任务合计 64/64**。没有零填充，也没有把 missing 当 0。所有 64 个初始状态均为 warm；0 个 cold，因此本实验不估计冷启动效应。

## 4. 主因子效应

公式（同一 task/repeat 的四格）：

```
ΔR = [(y10 - y00) + (y11 - y01)] / 2
ΔO = [(y01 - y00) + (y11 - y10)] / 2
ΔRO = (y11 - y10) - (y01 - y00)
```

负的目标差表示该头更新带来更低（更好）的目标；W/L/T 按该差值的符号计。

| task | 效应 | 均值 | 中位数 | W/L/T | 双侧符号 p | Holm p | Holm 拒绝 |
|---|---|---:|---:|---:|---:|---:|---|
| s0 | 风险 | −0.078541 | −0.041876 | 11/1/4 | 0.006348 | 0.076172 | 否 |
| s0 | 目标 | −0.153598 | −0.005267 | 9/5/2 | 0.423950 | 1.0 | 否 |
| s0 | 交互 | +0.039734 | 0.0 | 7/5/4 | 0.774414 | 1.0 | 否 |
| s1 | 风险 | −0.121169 | −0.069036 | 11/1/4 | 0.006348 | 0.076172 | 否 |
| s1 | 目标 | −0.153677 | −0.068661 | 11/3/2 | 0.057373 | 0.573730 | 否 |
| s1 | 交互 | −0.021834 | 0.0 | 7/5/4 | 0.774414 | 1.0 | 否 |
| s2 | 风险 | −0.073112 | −0.007393 | 9/3/4 | 0.145996 | 1.0 | 否 |
| s2 | 目标 | −0.089932 | 0.0 | 7/6/3 | 1.0 | 1.0 | 否 |
| s2 | 交互 | −0.013515 | −0.002940 | 8/4/4 | 0.387695 | 1.0 | 否 |
| s3 | 风险 | −0.053188 | −0.005330 | 8/3/5 | 0.226562 | 1.0 | 否 |
| s3 | 目标 | −0.147535 | 0.0 | 4/6/6 | 0.753906 | 1.0 | 否 |
| s3 | 交互 | +0.220899 | 0.0 | 6/3/7 | 0.507812 | 1.0 | 否 |

**没有任何 Holm 拒绝。** 风险头的 s0/s1 名义 p 值只是提示，不足以支持“风险头有效”的确认性结论。

## 5. 六个 arm 的实际代价与结果

| arm | 最终可行/64 | 无解 | 冷启动 | 在线失败/9,216 | 在线精确目标标签 | GP heads | GP kernels |
|---|---:|---:|---:|---:|---:|---:|---:|
| risk_frozen_objective_frozen | 64 | 0 | 0 | 927 | 609 | 0 | 0 |
| risk_updated_objective_frozen | 64 | 0 | 0 | 497 | 1,039 | 2,944 | 23,552 |
| risk_frozen_objective_updated | 64 | 0 | 0 | 1,306 | 230 | 1,472 | 11,776 |
| risk_updated_objective_updated | 64 | 0 | 0 | 1,115 | 421 | 4,416 | 35,328 |
| risk_only_updated | 64 | 0 | 0 | 36 | 1,500 | 2,944 | 23,552 |
| uniform_hash | 64 | 0 | 0 | 1,413 | 166 | 0 | 0 |

- 六个臂都是 64/64 最终可行；失败数不改变最终可行，只反映在线选点的失败代价。
- 风险头更新把失败从 927 降到 497，并把在线精确标签从 609 增到 1,039；目标头更新把失败升到 1,306、精确标签降到 230；两者同时更新为 1,115/421。
- `risk_only_updated` 的 36 次失败/1,500 标签说明“只看风险”能压低失败，但它不是 2×2 因子的主比较，不能据此提升为默认方法。
- `uniform_hash` 的 1,413 次失败/166 标签是随机参照。

全局账单：11,968 GP heads、95,744 kernels、7,680 predictions，0 errors；其中四格在线模型拟合合计 11,776 heads/94,208 kernels，余下 192 heads/1,536 kernels 属于共享初始状态，未按 arm 重复计费。单 arm 的 prediction 数没有在 `arm_composite_sealed` 中独立拆分，不能从该字段补造；全局 7,680 已记录。

## 6. 逐任务解释

### s0

- 风险效应 −0.0785（11/1/4，名义 p=.0063，Holm .076）；目标效应 −0.1536（9/5/2，p=.424）；交互 +0.0397。
- 四格 16/16 完整；所有 64 个 arm trial 最终可行。
- 目标头点估计更大，但符号检验不显著；风险头名义显著但 Holm 不拒绝。

### s1

- 风险效应 −0.1212（11/1/4，p=.0063，Holm .076）；目标效应 −0.1537（11/3/2，p=.0574，Holm .574）；交互 −0.0218。
- 目标头最接近名义显著，但 Holm 后仍不拒绝；不能写成目标头有效。

### s2

- 风险效应 −0.0731（9/3/4，p=.146）；目标效应 −0.0899（7/6/3，p=1.0）；交互 −0.0135。
- 风险/目标方向为负但噪声大；四格 16/16 完整。

### s3

- 风险效应 −0.0532（8/3/5，p=.2266）；目标效应 −0.1475（4/6/6，p=.7539）；交互 +0.2209（6/3/7，p=.5078）。
- 交互方向与其他任务相反且幅度最大，但没有显著；不能解释为稳定机制。

## 7. 代价与限制

- 所有 64 个初始状态都是 warm；本实验**没有冷启动样本**，不能回答冷启动发现、冷启动失败或首次可行位置差异。
- 所有 arm 最终 64/64 可行，首次可行调用均为 0；主结果只比较目标值，不比较最终可行性。
- 风险头更新降低失败数但目标收益只有名义趋势；目标头更新目标略好但失败数大幅增加、精确标签大幅减少。两者不能只看目标均值。
- 没有 Holm 拒绝；12 个检验的 family-wise 结论是“没有确认性主效应”。
- `risk_only_updated` 和 `uniform_hash` 是参照，不是四格因子；前者低失败不等于主方法有效。
- 本实验没有 VAE/幻想/飞机/RCAIDE；不能外推到 100/140 km 或安全保证。

## 8. 文件

- `smoke/`、`full/`：完整原始输出（plan、source_receipt、report、progress、events、units、manifest）。
- `smoke-analysis/`、`full-analysis/`：因子效应、Holm、四格完整性与逐 repeat 成本。
- `logs/`：测试、compile、help、plan、smoke/full、verify、analysis、tamper 全部日志与退出码。
- `analysis_summary.json`、`task_effects.csv`、`arm_costs.csv`、`integrity_check.json`、`sha256_manifest.txt`。
- `protocol.txt`、`instructions.txt`、`fixed_sha.txt`、`environment.txt`、`environment_packages.txt`。
- `.gitattributes`：`* -text`。
