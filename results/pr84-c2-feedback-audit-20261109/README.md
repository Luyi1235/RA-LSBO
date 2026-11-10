# PR84 / PR #85：C2 归档的只读反馈审计

本目录只审计已经完成的 PR83/C2 归档，不运行新的优化实验，不拟合 GP/VAE，不调用真值评价器或飞机/RCAIDE。

## 1. 固定身份

| 项目 | 值 |
|---|---|
| 审计代码 SHA | `54a3fd4d578852e846d6769a3f0ff9d72e3a5e85` |
| 输入结果 HEAD | `ade354de8b35d20a2bb32b469b04bc7da92aa004` |
| 输入实验代码 SHA | `975d96caac53d675ebe1b790a0647479a75a2ed5` |
| CI run | `42743617268`，success |
| 审计入口 | `run_pr84_c2_audit.py` |
| 代码 worktree | `D:\LSBO\RA-LSBO-pr84-code-20261110-112950` |
| 输入 worktree（复用） | `D:\LSBO\RA-LSBO-c2-results-worktree-20261109` |

源 worktree 与输入 worktree 在审计后均保持干净；输入结果 HEAD 未变。

## 2. 执行与状态

| 项目 | 结果 |
|---|---|
| 31 项纯标准库审计测试 | 31 passed，0 skip |
| compileall / `--help` / `plan --profile full` | 全部 exit 0 |
| smoke 审计 | `COMPLETE_PR84_AUDIT`，24 trials / 4 units / 72 在线观测 |
| smoke verify | `VERIFIED_PR84_AUDIT` |
| full 审计 | `COMPLETE_WITH_REPORTING_DISCREPANCIES`，384 trials / 64 units / 9,216 在线观测 |
| full verify | `VERIFIED_PR84_AUDIT` |

full 的 `COMPLETE_WITH_REPORTING_DISCREPANCIES` 是**审计完成并发现一个汇总差异**，不是数值失败，也不是重跑 C2 的理由。

## 3. 已确认的差异（1 条）

`discrepancies.json` 只有一条：

| 字段 | 归档值 | 重算值 |
|---|---|---|
| `integrity/arms_all_final_feasible` | `false` | `true` |

重算确认 **六臂各自 64/64 个 trial 都有最终可行解**（`no_solve=0`），因此归档汇总中的布尔值 `false` 与逐 trial 事实不一致。

**差异来源未确定（UNRESOLVED）**：没有原始 summary-generator 源码，无法判定该布尔值为何被写成 false；这不是“某条轨迹失败”的证据。其余汇总、12 项因子统计、轨迹重建和账本均一致。

## 4. 规模与原始账单

| 项目 | smoke | full |
|---|---:|---:|
| tasks / repeats | 4 | 64 |
| trials | 24 | 384 |
| 历史在线观测 | 72 | 9,216 |
| 历史真实评价 | 264（192 初始 + 72 在线） | 12,288（3,072 初始 + 9,216 在线） |
| GP heads | — | 11,968 started = completed，0 errors |
| GP kernels | — | 95,744 started = completed，0 errors |
| GP predictions | — | 7,680 completed，0 errors |
| VAE / fantasy | 0 / 0 | 0 / 0 |
| aircraft / RCAIDE | 0 / 0 | 0 / 0 |
| 本轮新增真值/GP/预测/VAE/幻想/飞机/RCAIDE | 全 0 | 全 0 |

full 的历史账单是读取和复核的对象，不是本轮新增计算。smoke 与 full 的 `new_bill` 全部为 0。

## 5. 六臂原始结果一致性

每臂 full 的分母是 **1,536** 次在线评价（64 trials × 24）；不是六臂合计 9,216。

| 臂 | 最终可行 | 无解 | warm/cold | 首次可行位置 | 在线不可行 | 精确目标标签 | GP heads | kernels | archived predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `risk_frozen_objective_frozen` | 64 | 0 | 64/0 | 全 0 | 927 | 609 | 0 | 0 | 1,536 |
| `risk_updated_objective_frozen` | 64 | 0 | 64/0 | 全 0 | 497 | 1,039 | 2,944 | 23,552 | 1,536 |
| `risk_frozen_objective_updated` | 64 | 0 | 64/0 | 全 0 | 1,306 | 230 | 1,472 | 11,776 | 1,536 |
| `risk_updated_objective_updated` | 64 | 0 | 64/0 | 全 0 | 1,115 | 421 | 4,416 | 35,328 | 1,536 |
| `risk_only_updated` | 64 | 0 | 64/0 | 全 0 | 36 | 1,500 | 2,944 | 23,552 | 1,536 |
| `uniform_hash` | 64 | 0 | 64/0 | 全 0 | 1,413 | 166 | 0 | 0 | 0 |

- 32 个 task/repeat 组全部 `initial_warm`，**cold=0、warm=64**；因此所有臂的首次可行位置都是 0，没有冷启动发现。
- 六臂的最终可行数在重算后都是 64/64；差异只在**失败次数、精确目标标签数和计算账单**。
- 冻结臂（`risk_frozen_objective_frozen`）的 GP 重建账单为 0，是因为它复用归档前缀/预测记录；其 1,536 条 prediction 记录来自归档，不是本轮新增。

## 6. 原 12 项因子统计复算

每任务 16 个 repeat；`complete_four_cell=16`、`missing=0`。效应定义保持原样：

- risk = `0.5*((y10-y00)+(y11-y01))`
- objective = `0.5*((y01-y00)+(y11-y10))`
- interaction = `(y11-y10)-(y01-y00)`

| 任务 | 效应 | 均值 | 中位数 | W/L/T | 原始 sign p | Holm p | 拒绝 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s0 | risk | −0.078541 | −0.041876 | 11/1/4 | 0.00635 | 0.07617 | 否 |
| s0 | objective | −0.153598 | −0.005267 | 9/5/2 | 0.42395 | 1.0 | 否 |
| s0 | interaction | +0.039734 | 0 | 7/5/4 | 0.77441 | 1.0 | 否 |
| s1 | risk | −0.121169 | −0.069036 | 11/1/4 | 0.00635 | 0.07617 | 否 |
| s1 | objective | −0.153677 | −0.068661 | 11/3/2 | 0.05737 | 0.57373 | 否 |
| s1 | interaction | −0.021834 | 0 | 7/5/4 | 0.77441 | 1.0 | 否 |
| s2 | risk | −0.073112 | −0.035230 | 9/3/4 | 0.145996 | 1.0 | 否 |
| s2 | objective | −0.089932 | 0 | 7/6/3 | 1.0 | 1.0 | 否 |
| s2 | interaction | −0.013515 | −0.002940 | 8/4/4 | 0.38770 | 1.0 | 否 |
| s3 | risk | −0.053188 | −0.029325 | 8/3/5 | 0.226563 | 1.0 | 否 |
| s3 | objective | −0.147535 | 0.009324 | 4/6/6 | 0.753906 | 1.0 | 否 |
| s3 | interaction | +0.220899 | 0 | 6/3/7 | 0.507813 | 1.0 | 否 |

- 12 项全部 **Holm reject=false**；原统计家族没有显著项。
- risk 在 s0/s1 的 nominally 小 p 值经 Holm 后为 0.076，不能称为显著。
- 这份复算与归档的 `factor_effects`/`holm_12` 一致；差异只在上述布尔汇总字段。

## 7. 预测诊断摘要

反馈构成（9,216 条被选中评价）：

| 反馈 | 数量 |
|---|---:|
| 精确 objective / 精确 region / 精确 resource | 3,965 |
| objective missing / region interval>0 / resource exact | 5,251 |
| objective censored 区间 | 0 |
| joint feasible=true | 3,922 |
| joint feasible=false | 5,294 |

预期诊断必须按“被选中评价”解释，不是全候选池校准。整体 joint Brier：

| 臂 | exact objective 率 | joint Brier | joint log loss | joint AUROC | region Brier | resource Brier | 条件 objective RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen/frozen | 39.6% | 0.3184 | 1.0718 | 0.771 | 0.3209 | 0.00124 | 1.095 |
| updated/frozen | 67.6% | 0.1334 | 0.4223 | 0.946 | 0.1336 | 5.5e-5 | 1.341 |
| frozen/updated | 15.0% | 0.3392 | 1.0828 | 0.742 | 0.3414 | 0.00201 | 0.801 |
| updated/updated | 27.4% | 0.1984 | 0.6034 | 0.890 | 0.1990 | 0.00089 | 0.811 |
| risk_only_updated | 97.7% | 0.0172 | 0.0671 | 0.978 | 0.0173 | 4.2e-7 | 缺失（无 objective 预测） |
| uniform_hash | 10.8% | — | — | — | — | — | 缺失 |

- 这些 Brier 是**选中样本、不同覆盖/不同 exact 率**下的描述性诊断，不能把六臂差异直接读成全候选池校准差异。
- objective RMSE/MAE 只在实际返回精确目标的子集上计算；缺失目标不插补，未选候选没有真值。
- 完整 per task/arm/step/branch/six-step 分层在 `diagnostics.json`（796 个分层键）；这里只给整体摘要。

## 8. 解释边界

- 这是只读审计，不是新的在线算法结果；不能据此声称任何风险头、目标头或组合策略更优。
- 发现一个布尔汇总差异，但**没有发现轨迹失败、账本不闭合或候选重建失败**；差异的生成逻辑仍 UNRESOLVED。
- 12 项因子检验没有 Holm 显著项；“不显著”不等于“无作用”。
- 选中样本的 Brier/AUROC/可靠性分箱不是独立全候选池校准，也不能直接证明“选择性反馈导致目标模型有偏”。
- 不同任务的原始目标误差未混合成总体结论；一次运行中的多个决策不是独立种子样本。
- 没有飞机/RCAIDE 数据；不能外推到 100/140 km 或安全保证。

## 9. 文件

- `smoke-audit/`、`full-audit/`：各 7 个封存文件（`audit_report.json`、`diagnostics.json`、`discrepancies.json`、`factorial_recomputed.json`、`per_trial.csv`、`selected_feedback.csv`、`manifest.json`）。
- `logs/`：python-version、31 项测试、compile、help、plan、smoke/full run 与 verify 的 stdout/stderr/exit。
- `identity.txt`、`fixed_sha.txt`、`environment.txt`、`protocol.txt`、`instructions.txt`、`summary_tables.json`。
- `sha256_manifest.txt`：本目录全部文件，排除自身。
- `.gitattributes`：`* -text`。

本分支只保存审计输出，不复制 646 MB 的原始 C2 归档；原始归档通过输入 HEAD `ade354de...` 与其 manifest 引用。
