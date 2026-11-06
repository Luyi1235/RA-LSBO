# PR63：候选级局部风险可信度与软收益/失败成本采集的六臂等成本验证

固定代码：`18b48e25dda6ff7d17296087981d0c4dc3ce99b2`  
环境：`Rcaide_vsp` / Python 3.9.25。

本分支只上传 PR63 新生成的解析实验结果；没有读取或修改 PR45–PR62 的飞机/SQLite 历史数据。

## 实验范围与完整性

- 4 个自定义层级数学任务：Branin ordinary/sparse、Rosenbrock ordinary/sparse。
- 12 个全新种子：`997201–997212`；与 PR54/55/58/60/61 的种子块不重叠。
- 6 个算法：`legacy`、`semantic_only`、`budgeted`、`soft_gp`、`local_blend`、`local_value`。
- 每个 case/seed 共享 12 个真实初始评价；随后每臂严格执行 **6 次**在线评价。
- 共 **2,304 次**真实解析评价（576 初始 + 1,728 在线），**288 条** on-policy trial。
- 新增飞机/RCAIDE 评价：**0**；新增 GP/VAE 拟合：真实执行。
- 原始 PR54 三个对照臂字节一致；所有臂每次在线成本一致；`stop_reason` 全部为 `planned_equal_six_real_calls`。
- 测试：`20 passed in 0.08s`（`tests/test_risk_budget63_local_value_policy.py`、`tests/test_risk_budget63_protocol.py`、`tests/test_risk_budget54_policy.py`）。
- 完整状态：`COMPLETE_FRESH_FIXED_SIX_ON_POLICY`，exit 0。

## 三个新增选择器

- `soft_gp`：`U = G(1−r_GP) − λ_t r_GP`。只把原 GP 的边际失败代理放进软效用，不使用硬门。
- `local_blend`：`r_blend = (1−ω)r_GP + ω r_local`，其中 `r_local` 和 `ω` 只来自同一离散分支、同一活动连续坐标、时间前缀已结算的真实邻居；不是 PR60 的 branch Beta 下限。
- `local_value`：在 `local_blend` 上再加有界信息项 `0.08 u(x) G(1−r)^2`；`u` 由局部支持、局部/GP 分歧和歧义构成。

三者都没有“无候选过门 → 选最高失败率回退”的机制。`local_value` 若最大 `U≤0`，只记录 `would_defer_in_unconstrained_utility_model=true`，本次固定六次协议**仍然真实评价**该候选；没有伪造拒评或节省成本。

## 总体结果

| arm | 最终可行 | 冷启动新可行 | 实际在线不可行 | 有风险预测 | 无模型预测 | 净效用≤0 且仍评价 | 其中实际失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 40/48 | 3 | 119/288 | 0 | 288 | 0 | 0 |
| semantic_only | 41/48 | 4 | 134/288 | 255 | 33 | 0 | 0 |
| budgeted | 41/48 | 4 | 124/288 | 255 | 33 | 0 | 0 |
| soft_gp | 41/48 | 4 | 99/288 | 255 | 33 | 22 | **22/22** |
| local_blend | 43/48 | 6 | 128/288 | 260 | 28 | 27 | **27/27** |
| local_value | 43/48 | 6 | 129/288 | 260 | 28 | 27 | **27/27** |

`local_blend`/`local_value` 的最终可行数从 41 升到 43，但全部来自 Rosenbrock sparse 的 2 个种子；见下文代价。

## 逐项消融与主对照

表中为 `候选 vs 对照` 的 feasible-first 胜/负/平，跨 48 对种子；目标差仅在双方最终都有精确可行目标时统计。正目标差表示候选更差。

| 对比 | W/L/T | 双方可行 | 目标差均值 | 目标差中位数 | 实际不可行差/48 |
|---|---:|---:|---:|---:|---:|
| **local_value vs budgeted（主对照）** | **10/13/25** | 41 | **+49.85** | 0 | **+5.0** |
| local_value vs legacy | 13/10/25 | 40 | +35.19 | 0 | +10.0 |
| soft_gp vs semantic_only | 10/7/31 | 41 | −15.07 | 0 | −35.0 |
| local_blend vs soft_gp | 9/14/25 | 41 | +52.09 | 0 | +29.0 |
| local_value vs local_blend | 2/1/45 | 43 | −0.15 | 0 | +1.0 |
| budgeted vs semantic_only | 3/0/45 | 41 | −12.98 | 0 | −10.0 |
| local_value vs soft_gp | 10/14/24 | 41 | +51.94 | 0 | +30.0 |

主对照没有通过：`local_value` 相对 `budgeted` 10 胜、13 负、25 平，实际失败多 5 次/48 对，目标差均值受长尾影响为正 49.85（中位数 0）。

## 分任务结果

每格为 `最终可行数 / 平均在线不可行数`（每任务 12 个种子、每臂 6 次在线）。

| case | legacy | semantic_only | budgeted | soft_gp | local_blend | local_value |
|---|---:|---:|---:|---:|---:|---:|
| Branin ordinary | 12 / 0.92 | 12 / 1.08 | 12 / 1.08 | 12 / 0.42 | 12 / 0.75 | 12 / 0.75 |
| Branin sparse | 8 / 4.25 | 9 / 4.42 | 9 / 4.00 | 9 / 4.08 | 9 / 4.67 | 9 / 4.75 |
| Rosenbrock ordinary | 12 / 1.08 | 12 / 1.58 | 12 / 1.58 | 12 / 0.42 | 12 / 1.00 | 12 / 1.00 |
| Rosenbrock sparse | 8 / 3.67 | 8 / 4.08 | 8 / 3.67 | 8 / 3.33 | 10 / 4.25 | 10 / 4.25 |

- **Branin sparse**：所有新臂最终都 9/12；`budgeted` 4.00 次失败，`soft_gp` 4.08、`local_blend` 4.67、`local_value` 4.75。局部/软价值没有改善发现率，反而增加失败。条件中位目标：`budgeted`/`soft_gp` 5.98，`local_blend`/`local_value` 9.18。
- **Rosenbrock sparse**：`local_blend`/`local_value` 最终 10/12（`budgeted` 8/12），冷启动发现 3（`budgeted` 1）；但平均失败从 3.67 升到 4.25，条件中位目标从 31.13 升到 **130.05**。多出的 2 个可行种子以更高失败和更差目标为代价，不能解释为更优优化。
- 两个 ordinary 任务所有臂最终都 12/12；`local_value` 失败数低于 `budgeted`，但 Branin ordinary 中位目标 1.77 vs 0.88，Rosenbrock ordinary 2.56 vs 2.54，目标质量没有稳定改善。

## 局部风险支持与延迟诊断

| arm | 模型收据 | 局部支持均值 | 局部信任 ω 均值 | 局部有效样本均值 | 同分支观测中位数 | 零支持调用 | 歧义均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft_gp | 255 | 2.16 | 0.466 | 3.44 | 4 | 2 | 0.546 |
| local_blend | 260 | 2.00 | 0.446 | 3.27 | 4 | 3 | 0.515 |
| local_value | 260 | 1.94 | 0.438 | 3.19 | 4 | 3 | 0.520 |

局部支持并不低：中位同分支观测 4 个，零支持仅 2–3 次；因此“没有局部邻居可用”不是主要失败解释。

`local_value`/`local_blend` 的 27 次 `U≤0` 调用全部实际不可行（27/27），`soft_gp` 的 22 次也全部失败（22/22）；这些调用集中在稀疏任务：

- Branin sparse：`soft_gp` 12 次外部选项、`local_blend` 14 次、`local_value` 14 次；
- Rosenbrock sparse：分别 10、13、13 次；
- ordinary 任务没有任何 `would_defer`。

这说明负净效用在该批稀疏样本上与真实失败高度一致，可能是有用的**观察性延迟信号**；但本次协议固定评价这些点，因此没有节省任何调用，也不能把“如果拒评会避免失败”当作已验证的成本收益。

## 判读

1. **局部风险融合没有带来可行首次收益。** `local_blend` 相对 `soft_gp` 9 胜、14 负、25 平，失败多 29 次/48 对；目标差均值 +52.09。把局部经验风险混入 GP 代理后，选择变差而非变好。
2. **信息价值项几乎无效。** `local_value` 相对 `local_blend` 仅 2 胜、1 负、45 平，只有 3 条实际路径改变；`0.08uG(1−r)^2` 没有形成可辨识的收益。
3. **相对旧硬门主对照失败。** `local_value` vs `budgeted` 为 10/13/25，失败 +5/48；目标差中位数为 0，均值 +49.85 主要由 Rosenbrock sparse 的 640/660 级长尾造成，不能作为稳定改善。
4. **唯一的失败数下降来自 `soft_gp`，不是局部机制。** `soft_gp` 相对 `semantic_only` 失败少 35 次/48 对、10 胜 7 负；但它没有提高最终可行数（41），且在 Rosenbrock ordinary 上中位目标 3.86 vs 2.54 更差。该信号值得后续独立验证，不能在本轮称为成功算法。
5. **稀疏任务出现“更多可行但更差”的代价。** Rosenbrock sparse 的 2 个额外可行种子伴随失败 +0.58/seed、中位目标 130 vs 31；Branin sparse 则没有增加可行数，失败和目标都更差。
6. **没有安全或校准声明。** 三个新臂都无硬风险门，`r_GP` 是边际乘积代理，`r_local` 是自适应选择的经验比率；两者都不是校准联合概率或置信上界。负净效用与失败的相关也不构成因果或安全保证。

## 限制

- 4 个自定义数学任务、每任务 12 个种子；不是官方 CBOE/约束 BO 基准复刻，也不是 100/140 km 飞机或 RCAIDE 验证。
- 目标差均值对长尾极敏感；中位数为 0 表示多数配对无差异，不能只报告均值。
- `local_value` 与 `local_blend` 的差异很小，信息项在本设计中没有被有效识别。
- 所有选择都在固定六次真实评价下执行，`would_defer` 只是诊断；没有测试可变成本拒评、未评价候选或反事实选择。
- 结果支持“局部风险/软价值没有改善本批可行首次表现”，不支持“软价值或局部可信度在所有 BO 中无效”的泛化结论。

## 文件

- `fixed_six_onpolicy_report.json`：完整 288 条 trial、配对统计、逐提议收据与真实标签。
- `plan.json`、`source_receipt.json`、`progress.jsonl`：固定协议、代码/引擎身份和逐 trial 进度。
- `pr63-*-trials.zip`：四个任务各自的 `shared_initial/` 与 `trials/` 完整证据（各 444 个条目，ZIP 完整性检查通过）。
- `smoke-console.log`、`full-console.log`、`tests.log`、`aggregate.log`：smoke/full 控制台、pytest 和独立聚合日志。
- `pr63_aggregate.json`：独立聚合结果（臂总计、配对、逐 case、局部支持）。
- `fixed_sha.txt`、`artifact_sha256.txt`：固定 Git SHA、协议/runner/引擎 SHA256，以及所有上传文件的 SHA256。
