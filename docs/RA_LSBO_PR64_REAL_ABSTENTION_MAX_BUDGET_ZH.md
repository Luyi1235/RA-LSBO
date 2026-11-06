# PR64 — 将软收益/风险 outside option 变成真实拒评动作

## 为什么 PR63 后不再继续局部风险融合

PR63 完整结果在 \`results/pr63-local-value-20261106\`（结果 HEAD \`b1d66eed5c61c3b61253b456c961aea78ed90b8b\`）。关键证据：
- \`soft_gp\`：99/288 在线不可行，最终可行 41/48，冷启动可行发现 4。
- \`semantic_only\`：134/288 在线不可行，最终可行 41/48。
- \`budgeted\`：124/288 在线不可行，最终可行 41/48。
- \`local_blend\`：128/288 在线不可行，最终可行 43/48。
- \`local_value\`：129/288 在线不可行，最终可行 43/48。
- \`local_blend\` 对 \`soft_gp\` 是 9/14/25，失败多 29；局部支持中位同分支观测为 4，故不应只归因于无邻居。
- \`local_value\` 对 \`local_blend\` 仅 2/1/45，信息项改变实际路径很少。
- \`soft_gp\` 22 个最大净效用不大于 0 的评价全部实际不可行；局部融合臂 27 个全部不可行。

PR63 仍然强制执行 6 次评价，所以这些 outside-option 信号没有被用作真正的拒评。它们不能被称为已经避免的失败，也没有测到拒评后错失可行解的机会成本。

**PR64 因此只保留 soft-GP 净效用，暂停 local_blend 和 information bonus。** 新问题是：若所有模型支持候选的最大净效用不大于零，真实停止是否能在相同最大预算上限下减少昂贵/不可行调用，同时保留足够的已证实可行优化质量？

## 动作定义

原始 PR63 \`soft_gp\` 净效用与候选池、GP/VAE、约束概率乘积、原始阶段日历全部不变。

- \`soft_gp_continue\`：最多 6 次在线真实评价，PR63 原样继续。
- \`soft_gp_abstain\`：同一选点效用；只有当本轮存在真实、绑定候选的 GP 风险/效用收据，且候选池最大 \(U(x)\le0\) 时，返回 \`None\` 并记录 \`abstain_stop_before_evaluator\`。runner 必须在调用 \`problem.evaluate\` **之前**停止本 run。停止后不重启、不把被拒候选写成观测、不生成物理标签或目标值。
- 若 GP/表示模型失败、候选仅来自 coverage、没有有效风险/效用收据，则**不得**根据缺失预测而拒评；按既有 coverage 规则评价并记为无风险预测。

冻结对照为原 \`semantic_only\` 与 \`budgeted\`。不再加入 PR63 失败的局部融合或不可辨识的信息奖励。

## 最大预算比较（不是等实际评价数）

每任务/种子共享 12 个真实初始点。每个算法的在线调用上限 6 次。Continue / 两个旧对照最多执行 6 次；abstain 臂可提前停止。所有真实 evaluator 调用都计费，**未调用的评价不计为成功、失败或零成本性能收益**。

- 新种子：998101–998112，和 PR54/55/58/60/63 种子块不重叠。
- 4 个自定义数学任务，4 个算法，Full 最大解析真值调用：\(4\times12\times(12+4\times6)=1,728\)。实际调用可更少，报告真实数字和未用预算。
- Smoke：每任务 1 个新种子，每臂最多 2 次在线调用；最大 80 次解析评价。
- PR64 主要对照：\`soft_gp_abstain\` vs \`soft_gp_continue\`。
- 次要对照：abstain vs budgeted、continue vs budgeted、budgeted vs semantic_only。
- 每任务报告所有配对种子的 feasible-first W/L/T、真实在线评价数、节省的最大预算、实际不可行数、最终已证实可行目标、冷启动可行发现、停止发生在第几次真实评价后，以及 continue 对照之后是否发现 abstain run 错过的可行解。
- **成本—质量 Pareto** 才是主结论。少评价必然可能少失败；只展示不可行数下降而不展示实际调用与最终目标属于无效结论。

## E0/E2 硬契约

- abstain decision 的 \`evaluation_performed=false\`、\`realized_nonfeasible=null\`、\`no_evaluator_call=true\)。
- 同一 decision receipt 必须含负/零净效用和被拒候选哈希，且其历史前缀 hash 必须在决策之前冻结。
- \`online_calls\` 等于真实 observation 数；best-objective 曲线长度严格等于 online_calls+1。
- abstain 后绝不出现同一 run 的后续 proposal 或 observation。
- 非 abstain arm 不允许 abstain；coverage/no-model 不能被解释为负净效用。
- 共享初始数据身份一致，所有旧实验和 SQLite 只读，飞机/RCAIDE 调用为 0。

PR64 是看过 PR63 后提出的探索性后续研究，不是独立预注册确认。GP 风险和净效用仍未校准，不是安全上界，也不能外推真实飞机/飞行安全。若减少成本但丢失可行解或最终目标明显恶化，则策略失败。

## 工作机入口

\`run_ra_lsbo_risk_budget64_abstention_pycharm.py\`

先 smoke、再在全新目录 full。只上传结果与日志，不在工作机改代码或重新调系数。
