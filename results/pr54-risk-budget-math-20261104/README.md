# PR54：风险预算在线选点闭环（解析 Branin 双视图）

固定代码 SHA：`fd5492d33b21421ff8f54ad8bba453543caf39fe`  
PR53 基线合并提交：`a7151a7bb44e2570eaec64f1ce3125024deadffc`  
环境：`Rcaide_vsp` / Python 3.9.25 / 真实 torch、scipy、numpy。

本分支是**前瞻 on-policy 解析实验**的新证据。每个方法真实选点、真实调用解析评价器、真实结算后更新 GP/VAE；不是 PR45 历史轨迹复播。新增飞机/RCAIDE 调用为 0，未读取或修改 PR45–PR53 原始实验。

## 冻结设计

- 两任务：`branin_ordinary`、`branin_sparse`（自定义层级混合变量扩展，不是上游标准基准复制，也不是飞机）。
- 三臂：`legacy`（原 completion-status 风险因子）、`semantic_only`（仅约束概率乘积）、`budgeted`（在 semantic 排序上增加动态预测失败额度）。
- 每任务 8 个固定种子：995401–995408。
- 每组 12 个共享初始评价；最多 12 次在线评价；所有臂共享 **6 次已观察在线不可行即停止** 的硬上限。初始不可行不计入在线失败。
- 候选池每步固定 128 个，三臂共享同一无标签候选流；`local candidate augmentation=False`；4×2 stable GP/EP 与 2000 结构 VAE 保持冻结。
- 目标视图为 2000 结构 VAE 潜空间；风险视图为显式 One-hot；物理反馈保留 exact/interval/missing。

## 运行与完整性

- smoke：`--profile smoke`，2 任务 × 1 种子 × 3 臂 × 2 步，36 次解析评价（24 初始 + 12 在线），exit 0。
- full：默认每任务 8 种子，**696 次新解析评价 = 192 次共享初始 + 504 次在线**，48 个 trial，全部 exit 0；低于冻结上限 768，因为部分路径在 6 次在线失败后提前停止。
- 48/48 trial 的 `proposals`、`observations`、`result.json` 与 `result.online_calls` 数量一致；所有 `online_nonfeasible ≤ 6`。
- `new_aircraft_evaluations=0`、`new_rcaide_calls=0`、`reused_original_PR45_physical_truth=false`。
- 单元测试 `tests/test_risk_budget54_policy.py`：7 passed（0.04 s）。
- `source_receipt.json` 记录 policy SHA256 `d2c32b77...`、engine SHA256 `3f6105f0...`；PR54 未改任何 PR45–PR53 证据。

## 终点结果

| 任务 | 臂 | 初始可行 | 最终可行 | 在线调用均值 | 在线失败均值 | 触 6 次上限 | 最终最优可行均值 / 中位数 |
|---|---|---:|---:|---:|---:|---:|---|
| ordinary | legacy | 8/8 | 8/8 | 12.00 | 1.12 | 0/8 | 2.0890 / 1.7587 |
| ordinary | semantic_only | 8/8 | 8/8 | 12.00 | 1.12 | 0/8 | 2.0756 / 2.0181 |
| ordinary | budgeted | 8/8 | 8/8 | 12.00 | 1.12 | 0/8 | 2.0756 / 2.0181 |
| sparse | legacy | 3/8 | 5/8 | 8.63 | 5.75 | 7/8 | 11.2599 / 14.5232（n=5） |
| sparse | semantic_only | 3/8 | 6/8 | 9.38 | 5.75 | 6/8 | 11.7100 / 14.8949（n=6） |
| sparse | budgeted | 3/8 | 6/8 | 9.00 | 5.62 | 6/8 | 9.4504 / 9.4201（n=6） |

注意：sparse 的最终可行数不同（5 vs 6），均值/中位数是在各自可行子集上计算，不能直接当作同点配对比较；配对统计见下。

### 首次发现可行解

- ordinary：三臂的 8 个种子在初始 12 点中均已有可行解，因此“首次发现”不是有效终点。
- sparse：初始可行 3/8（995401、995402、995405）；无初始可行的 5 个种子为 995403、995404、995406、995407、995408。
  - `legacy`：2/5 找到。995406 在第 6 次在线评价（iteration 5）找到；995408 在第 2 次（iteration 1）找到。
  - `semantic_only`：3/5 找到。额外在 995407 的第 2 次在线评价找到；995406/995408 与 legacy 相同。
  - `budgeted`：3/5 找到，首次发现步数与 `semantic_only` 完全相同。
  - 995403、995404 三臂均未找到；995407 仅 legacy 未找到（最终无可行值）。

**结论：风险预算没有改变首次发现步数；semantic 修正本身使 995407 从“未找到”变为第 2 步找到。**

## 是否改变选点

### semantic_only vs legacy

- ordinary：8/8 种子的候选序列均改变；最终均值几乎相同（2.0756 vs 2.0890），中位数 legacy 更低（1.7587 vs 2.0181）。
- sparse：8/8 种子改变；最终可行数 6/8 vs 5/8，配对均值 −0.3143（仅两臂都可行的 5 对），中位数差 0；实际调用多 0.75 次/种子，失败数相同。
- 这证明“去掉 completion-status 因子”确实改变选点；但 8 个种子、单任务，不能作确认性表示优势结论。

### budgeted vs semantic_only

- ordinary：**0/8 种子改变**。96/96 次预算门均通过，预测风险都低于额度，因此预算过滤器不改变排序。
- sparse：**5/8 种子改变**（995401、995402、995405、995407、995408），共 19 个共同步骤的候选不同；995403、995404、995406 与 semantic 完全相同。
- 变化发生在第 3–10 步，主要因为 sparse 的预测风险接近或超过动态额度，预算门开始筛选或触发回退。

逐种子：

| seed | budgeted 调用/失败 | semantic 调用/失败 | budgeted 最终最优 | semantic 最终最优 | 变化 |
|---|---:|---:|---:|---:|---|
| 995401 | 12/4 | 10/6 | 4.3170 | 4.3170 | 预算避免了第 5、6 次失败，可继续到 12 步；最终值相同 |
| 995402 | 8/6 | 12/6 | 16.8382 | 15.2666 | 更早触发 6 次失败；最终值更差 |
| 995405 | 10/6 | 10/6 | 15.9356 | 18.3535 | 预算选点改进最终值 |
| 995407 | 11/6 | 12/5 | 2.8206 | 15.5318 | 预算选点显著改进最终值，但多 1 次在线失败 |
| 995408 | 12/5 | 12/5 | 2.2678 | 2.2678 | 选点不同，终点相同 |

因此预算的收益/代价高度依赖种子：2 个改进、1 个变差、2 个终点相同；不能只报均值。

## 配对描述统计

| 对比 | 两臂均可行 | 候选唯一可行 | 对照唯一可行 | 能量差均值* | 能量差中位数* | 调用差均值 | 失败差均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary budgeted − semantic | 8/8 | 0 | 0 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| ordinary semantic − legacy | 8/8 | 0 | 0 | −0.0134 | +0.1766 | 0.00 | 0.00 |
| sparse budgeted − semantic | 6/8 | 0 | 0 | −2.2596 | 0.0000 | −0.375 | −0.125 |
| sparse budgeted − legacy | 5/8 | 1 (995407) | 0 | −0.4836 | 0.0000 | +0.375 | −0.125 |
| sparse semantic − legacy | 5/8 | 1 (995407) | 0 | −0.3143 | 0.0000 | +0.750 | 0.00 |

\* 负值表示候选臂目标更小（更好）；均值只在两臂都可行的种子上计算。`n=8`，描述性，不作显著性声称。

## 风险预算筛选与回退

### ordinary

- `semantic_only`：96/96 次 receipt 记录 `constraint_only_ranking`；预算门全部通过（`True=96`），但 semantic 不应用预算过滤。
- `budgeted`：96/96 次 `predicted_allowance_filter`；无回退。因此两臂候选序列完全相同。

### sparse

`budgeted` 共 72 个在线 receipt：

- 49 次 `predicted_allowance_filter`（预算门通过）；
- 10 次 `no_candidate_below_allowance_min_predicted_failure`（回退）；
- 13 次没有预算决策（scheduled coverage / degenerate gate coverage 等既有回退）。

10 次预算回退：

- 995402 的 iteration 6、7（已有可行解，额度收紧）；
- 995403 的 iteration 1、2、3；995404 的 1、2；995406 的 1、2（feasible discovery 阶段，候选预测风险 0.93–0.96，均无候选满足额度）；
- 995407 的 iteration 1（预测 0.9578，实际该点为可行，是唯一“回退后成功”的案例）。

`semantic_only` 的 75 个 sparse receipt 中，39 次门通过、23 次门不通过但因策略不应用预算而仍按约束排序选点；62 次记录 `constraint_only_ranking`，13 次无预算决策。也就是说，预算门确实在 sparse 中“有东西可拦”，但拦截并不等于安全。

## 预测风险 vs 实际失败

| 任务/臂 | 平均预测失败概率代理 | 实际在线失败率 |
|---|---:|---:|
| ordinary budgeted | 0.1219 | 0.09375 |
| sparse budgeted | 0.3491 | 0.5593 |
| sparse semantic | 0.3751 | 0.5484 |

预测概率仍然偏乐观，尤其在 sparse；预算回退也没有把实际失败率压到预测额度以下。所有 receipt 均带 `probability_guarantee=False`，本实验不构成机会约束或安全保证。

## 关键结论

1. **风险预算不是普适有效：在 ordinary 完全无效，在 sparse 才改变选点。**  
   ordinary 的 8/8 种子初始即可行、预测风险低，96/96 次预算门通过，budgeted 与 semantic 选点逐点相同。sparse 初始可行仅 3/8，预算门在 5/8 种子改变选点。

2. **预算在 sparse 的终点收益是“少数种子驱动”的，不是稳定提升。**  
   budgeted 最终中位数 9.4201，semantic 14.8949；均值 9.4504 vs 11.7100；但配对中位数差为 0，2 个种子改进、1 个变差、2 个相同。995407（15.53→2.82）和 995405（18.35→15.94）贡献了主要均值差，995402 则从 15.27 变差到 16.84。

3. **首次发现步数没有被预算改变。**  
   sparse 无初始可行的 5 个种子里，legacy 2/5 找到，semantic/budgeted 3/5；budgeted 与 semantic 的首次发现步数逐种子完全相同。预算影响的是发现可行解之后的开发选点。

4. **semantic 修正本身已经改变全部种子选点，且使 995407 从不找到变为第 2 步找到。**  
   这与 PR50–PR53 的“completion-status 头在任务完成上语义不匹配、约束概率才是风险信号”一致；但本实验只有 8 个种子，不能替代飞机实验。

5. **实际调用数和失败数必须与终点一起报告。**  
   sparse：legacy 8.63 次/5.75 失败，semantic 9.38/5.75，budgeted 9.00/5.62。三者共享 6 次失败硬停止，调用数不同；budgeted 的少量失败减少不是“同预算优化收益”，也不能把硬停止造成的提前终止归功于预算采集。995401 因预算避免失败而多跑 2 步，995402 则更早停止。

6. **预算回退是显式、可审计的，但仍不是安全机制。**  
   10 次 sparse 回退中 9 次实际失败，1 次成功；预测概率整体仍偏乐观（0.349 vs 实际 0.559）。预算只能作为“剩余额度提示”，不能证明满足 6 次失败上限，也不能保证找到可行最优。

## 限制

- 8 个种子/任务，描述性配对，不构成确认性统计结论。
- 仅两个自定义解析 Branin 任务；不是上游标准基准复制，也不是飞机/RCAIDE。
- 不同臂实际在线调用数不同；报告保留差异，不作“同预算优胜”声明。
- 预算使用边际约束概率乘积作为联合可行代理，不是机会约束；回退记录的是“没有候选满足预测额度”，不是“安全”。
- 所有方法共享 6 次失败硬停止，预算臂的失败差异可能来自停止时点和选点共同作用，不能单独归因于预算采集。
- 目标/风险 GP 与 VAE 均为真实拟合，但只有单次运行、固定种子；没有多种子重复的训练随机性。
- 本实验不能外推为 140 km 飞机稀疏资格或 RCAIDE 高保真结论。

## 文件

- `risk_budget_math_report.json`：48 trial 明细、配对比较、预算 receipt、限制。
- `plan.json`、`source_receipt.json`：冻结设计、代码/引擎/策略哈希。
- `progress.jsonl`：48 行逐 trial 进度。
- `trial_evidence.zip`：`trials/` 下 48 组 `proposals.jsonl`、`observations.jsonl`、`result.json`，以及 `shared_initial/` 的 16 个共享初始记录；共 415 个文件。
- `smoke/`：smoke 的 report、plan、source receipt、progress。
- `smoke-console.log`、`full-console.log`：两轮完整控制台。
- `tests.log`：7 项单元测试。
- `fixed_sha.txt`：固定 SHA、源哈希、计数。
