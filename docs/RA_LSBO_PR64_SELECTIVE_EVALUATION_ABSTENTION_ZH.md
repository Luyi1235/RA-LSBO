# PR64 — 风险感知选择性真实评价：外部选项、候选池宽度与真实拒评

## 1. 研究动机

PR63 的结果支持保留 GP-only soft utility，但没有支持当前 candidate-local blend / information bonus。尤其值得检验的是：soft_gp 在固定六次真实调用中，22 次 \`U<=0\` 的选中候选全部真实不可行；这只是已选点的小样本现象，不是概率保证。历史路径回放显示若在第一次负效用时停止，会节约 45 次评价，但会漏掉 2 个后来通过 coverage 发现的可行设计。因此 PR64 不是把旧的固定六次轨迹重新包装成“安全成功”，而是**第一次在真实运行循环中实现拒绝真实评价**，并把“廉价扩大候选筛选”与“是否真的评价”分开消融。

PR64 只研究解析数学任务；不调用飞机/RCAIDE，不读取 PR45–63 的任何历史真值。

## 2. 固定的候选净效用

完全复用 PR63 的 GP-only 软效用，不改权重或风险代理：

\[
U_t(x)=G_t(x)(1-r_{\rm GP,t}(x))-\lambda_t r_{\rm GP,t}(x)
\]

其中 G 是当前候选池内部的归一化 EI/原阶段收益代理，\(r_{GP}=1-\prod_j p_j\) 仍是未校准的约束边际概率乘积，\(\lambda_t=0.15+0.25F_t/6\)。外部选项固定为 \(U_{\rm outside}=0\)。候选池的所有原有乐观物理支持条件不变。

本算法**不**宣称 \(U\) 是精确经济期望、经过校准的概率或安全证书。阈值 0 是一项待检验的启发式外部选项。只有在模型真实预测可用时才可做此比较；coverage/no-risk-probability 继续作为独立未评分动作，绝不填入零风险。

## 3. 六个算法臂

| 臂 | 候选筛选宽度 | 负净效用时的真实动作 |
|---|---:|---|
| legacy | 128 | 原 PR54 评价 |
| semantic_only | 128 | 原 PR54 评价 |
| budgeted | 128 | 原 PR54 硬预算门 |
| soft_gp_fixed128 | 128 | 始终评价，PR63 软效用 |
| soft_gp_fixed512 | 512 | 始终评价，同一软效用 |
| soft_gp_abstain512 | 512 | 若最大 eligible U≤0，则在真值评价前 abstain |

**主要对照：** \`soft_gp_abstain512 vs soft_gp_fixed512\`。这两臂具有相同 512 候选池、同一 PR63 GP/VAE、同一风险/效用、相同初始前缀和相同**最大** 6 次在线评价；唯一处理差异是真正是否调用昂贵评价器。

**候选池宽度消融：** \`soft_gp_fixed512 vs soft_gp_fixed128\`。用于区分更宽廉价筛选的影响与 abstention 的影响。不得把宽池性能变化归因于拒评本身。

### 实际 abstain 语义

- 对 soft_gp_abstain512，如果模型选中的候选净效用 \(U\le0\)，它会写入 proposal / risk receipt，然后立即停止该 trial；**不调用 \`problem.evaluate(candidate)\`，不产生 observation，不计真实评价次数，不填充虚假结果**。
- 如果 \(U>0\)，则真实评价并按实际标签更新 GP/VAE。
- 如果 model-free scheduled/degenerate coverage 绕过风险预测，照常执行真实 coverage 评价，但标记为 no probability forecast、单独统计失败；不把 coverage 当成通过风险门。
- 对固定评价臂，即使 U≤0，也照常评价。这是用于衡量被拒候选实际代价的对照。

每个 abstain 候选的 shadow label 只能在全试验结束后，从同 case/seed/step 的 fixed512 臂读取，且必须验证候选 SHA 完全相同。它仅用于事后审计“被拒候选在控制臂里实际如何结算”，不能回流至 abstain 臂或训练模型。

## 4. 冻结前瞻实验

- 四个任务：自定义层级 Branin ordinary/sparse、Rosenbrock ordinary/sparse。
- 新种子：997301–997324（每任务 24 个）；与 PR54/55/58/60/63 种子不重叠。
- 六个算法臂；每 case/seed 共享 12 个真实初始评价一次。
- 每臂最多 6 次真实在线解析评价；固定臂必须做满 6 次，abstain512 可以少做。每臂内部仍使用原 12 步模型 planning horizon。
- Full 最大预算：\(4\times24\times(12+6\times6)=4608\) 次解析评价；实际计数等于共享初始 + 各臂真实调用数，**必须小于或等于**上限。Smoke：4 任务各 1 种子，每臂最多 2 次在线，最多 96 次解析评价。
- 六臂使用同 case/seed 的同一确定性候选随机流；512 pool 是 128 pool 的嵌套前缀。宽池只增加廉价候选筛选/GP 预测，不增加真实评价预算。
- 新 PR64 使用 PR63 完全冻结的 \`risk_budget63_local_value_engine.py\` / \`risk_budget63_local_value_policy.py\`，只调用 variant=\`soft_gp\`。原 PR54 控制 engine/policy 字节身份也必须严格核对。

## 5. 必须报告的结果

不可只给一个胜率或平均目标值。每任务分别报告：
1. feasible-first 配对 W/L/T、双方可行条件下的目标差、单方可行、双方均不可行；
2. 各臂真实调用数、调用节约数、实际不可行总数和每真实调用失败比例；
3. 最终可行 incumbent、cold-start 可行发现；
4. abstain 次数、负效用但仍评价次数、coverage/no-risk-forecast 调用和其真实失败；
5. 对每个 abstained candidate，fixed512 是否选出**相同 SHA 的候选**，若相同，其 shadow 真值如何结算（事后对照、非算法可见）；
6. 128→512 候选池对选择与计算耗时的影响；
7. 所有未成功 seed 和 objective long-tail。

**主要评价是最大资源预算下的成本—质量 Pareto，而非同实际调用次数性能优势。** 如果 abstain 省了调用但损失可行发现，应如实报告 tradeoff。跨任务 objective 量纲不同，不合并一个不透明的宏平均来夸大优势。

## 6. 通过/失败门槛

- E0：所有 abstain 决策都发生在 \`problem.evaluate\` 之前；无真实 observation、无虚构 label、无未来标签；fixed512 与 abstain512 在停止前的同前缀候选 SHA 必须一致。
- E1：Smoke 的 4 任务、所有 arm、源哈希、实际调用预算和输出账本全部通过。
- E2：Full 新种子前瞻实验完成，真实调用数逐项 reconcile，所有 shadow label SHA 绑定。
- E3：在新种子上，abstain512 相对 fixed512 是否在实际失败/调用节约与最终可行设计之间获得稳定更好的 Pareto 权衡。没有改善即记录负结论，不调零阈值。
- 任何 PR64 结果都只适用于当前数学函数族，不构成飞推控物理安全证明。

## 7. 执行与结果上传

PyCharm 入口：\`run_ra_lsbo_risk_budget64_selective_evaluation_pycharm.py\`。

先 Smoke，再用全新输出目录 Full。工作机不改代码，不覆盖旧输出。独立结果分支应包含：
- \`selective_evaluation_report.json\`、\`plan.json\`、\`source_receipt.json\`、\`progress.jsonl\`;
- 完整 \`trials/\`、\`shared_initial/\`，每任务 zip；
- 每次候选 proposal 和实际 observation，abstain 记录必须没有 observation；
- Smoke/Full console、pytest、固定 SHA、SHA256 清单。

PR64 的科学问题是“何时不值得为某候选支付真实评价成本”，而不是证明 GP 已经能给出安全概率。
