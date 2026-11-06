# PR63 — 候选级局部风险可信度与优化收益联合采集（新算法原型）

## 研究定位

PR50–61 暴露两个相互关联的问题：稀疏任务的约束 GP 边际概率乘积存在严重过度乐观；未校准的硬风险门遇到无候选满足时，回退到高失败候选，分支 Beta 风险下限反而增加失败。本 PR 不再为旧门增加更保守的阈值，不把分支内不同连续设计视为同一风险概率，而是检验“候选级局部数据支持 + 软收益/失败成本 + 有界信息价值”的**新候选选择机制**。

**注意：这只是待真实 on-policy 验证的新优化算法原型，不是已经成功的创新结果。** 候选级局部风险、支持质量和歧义都是明确命名的经验指标，不能宣称校准、置信上界或 Safe BO 概率保证。

## 数学定义与可复核动作

保持 PR54 原始双视图 GP/VAE 完全不变：潜空间目标 GP、显式 One-hot 约束 GP，2000 结构语料、4×2 stable GP/EP 核格、原 feedback、模型失败 coverage、12 步内部 horizon、6 次已观察失败停止。原三个对照字节身份仍由运行器验证。

对当前真实 GP 预测候选 x，原始不可行代理

\(r_{\mathrm{GP}}(x)=1-\prod_j p_j(g_j(x)\le0\mid D_t)\).

这只是边际概率乘积代理，并非被证明校准的联合可行概率。**不**将记录 status completion 当作独立任务失败因子。

仅使用在 t 之前真实结算的同一运行前缀 \(D_t\)。与 x 具有完全相同离散模式、相同活动掩码的历史设计 x_i，按活动连续变量（各自物理范围归一化）的距离：

\(d^2(x,x_i)=\frac1{|A(x)|}\sum_{k\in A(x)}((x_k-x_{i,k})/(u_k-l_k))^2\).

固定长度尺度 h=0.25，权重 \(w_i=\exp[-d^2/(2h^2)]\)，其中全部 w_i 只来自**已知的历史设计与真值**，不是未评价候选的预测标签。定义候选局部经验失败率和支持度：

\(\hat r_{\mathrm{local}}(x)=(0.5+\sum_iw_i 1[\mathrm{unfeasible}(x_i)])/(1+\sum_i w_i)\)

\(\omega(x)=(\sum_i w_i)/(2+\sum_i w_i)\).

局部连续可信度修正为平滑融合（不是风险下限）：

\(r_{\mathrm{blend}}(x)=(1-\omega)r_{\mathrm{GP}}(x)+\omega\hat r_{\mathrm{local}}(x)\).

这与 PR60 “整离散分支独立 Bernoulli 后验、取 max 风险”本质不同：同一结构下不同连续位置有不同的权重，远距离记录贡献很小；支持度不足时自然回退到原始 GP 风险而**不抬高为全局 Beta 下限**。但是该局部权重仍是经验机制，**不能保证**其风险预测校准。

定义用于决策的启发式歧义量：

\(u(x)=\min(1,\sqrt{\hat r_{\mathrm{local}}(1-\hat r_{\mathrm{local}})/(1+\sum w_i)}+0.25(1-\omega)+\omega|r_{\mathrm{GP}}-\hat r_{\mathrm{local}}|)\).

模型只在当次 GP 预测有效时才产生风险收据。预期 EI 使用**原 GP 给出的 EI**，在同一真正 eligible 的候选池内部归一化为 \(G(x)=EI(x)/(EI(x)+\mathrm{median}(EI_{eligible>0}))\)，保持范围 [0,1]；在无可行目标的 discovery 阶段取 G=1；原 boundary 阶段取 \(G=4r_{\mathrm{GP}}(1-r_{\mathrm{GP}})\)。从已观察在线失败数 F_t 得到**软**失败代价系数

\(\lambda_t=0.15+0.25\,F_t/6\).

三种新增选择器逐级消融（均在原候选乐观物理支持内，永远无新的硬风险门）：

- soft_gp：\(U=G(1-r_{\mathrm{GP}})-\lambda_t r_{\mathrm{GP}}\).
- local_blend：\(U=G(1-r_{\mathrm{blend}})-\lambda_t r_{\mathrm{blend}}\).
- local_value：\(U=G(1-r_{\mathrm{blend}})-\lambda_t r_{\mathrm{blend}}+0.08\,u(x)G(1-r_{\mathrm{blend}})^2\).

使用最大 U 的候选，按候选池固定次序打破平局。额外信息价值由**目标潜在收益 G**和预计可行性 \((1-r)^2\) 同时门控，因此避免 PR60 的“为了最大不确定性无条件进入最高风险区域”。

**空门/拒评语义：** 三个新选择器都没有“所有候选不能过风险门 -> 再选失败率最高的回退”。在满足原优化器乐观结构支持的候选集合不为空时总选择最大 U。如果最大 U≤0，保留 \`would_defer_in_unconstrained_utility_model=true\` 和外部选项 0 的诊断收据；但本次固定六次等成本实验**仍真实评价**这一候选，不假装已经拒评或节约成本。是否允许实际 abstain 必须另建可变评价成本的独立协议，不能把未执行调用充作真实同工作量。

## 冻结消融与真实评价

三种旧对照完全不改：legacy、semantic_only、budgeted。三种新臂：soft_gp、local_blend、local_value。它们共享 PR54 的真实 GP/VAE、候选池与数值失败回退机制；只有新臂的模态选择评分不同。使用工作机 Rcaide_vsp 实际运行新解析真值和 GP/VAE。

- 任务：自定义层级 Branin ordinary/sparse 与 Rosenbrock ordinary/sparse。不是官方 benchmark 复刻，也不是 100/140km 飞机/RCAIDE。
- 新种子：997201–997212，各任务 12 个；与 PR54/55/58/60 所有既有实验种子不重合。
- 每任务/种子共享 12 个真值初始评价（只付费一次），六臂各执行精确 6 次自己真实提议、实际评价和重新拟合的在线调用。新臂仍用旧内部 horizon=12 做阶段调度，不能暗中更改原规划时钟。
- Smoke：4×(12+6×2)=**96** 个解析真值调用，4 任务各 1 种子、六臂各 2 在线；结果不是正式科学比较。
- Full：4×12×(12+6×6)=**2,304** 个解析真值调用、288 条真实 on-policy trial；没有提前停止造成的工作量差异，飞机/RCAIDE 调用为 0。
- 预先指定的主对照为 **local_value vs budgeted**，按任务分别报告每任务 12 对种子的 feasible-first W/L/T，冷启动可行发现，条件最优可行目标中位数/均值和实际不可行次数。其余三项关键消融 soft_gp vs semantic_only、local_blend vs soft_gp、local_value vs local_blend，另辅助展示 local_value vs legacy/soft_gp。
- 记录 GP 风险、同候选局部风险、支持权重/有效样本量/最近距离、风险歧义、归一化 EI、信息增益分数、失败价格、净效用与 \`would_defer\`；另记录最终真实评价标签。绝不对无预测的 coverage 选择伪造 GP 风险。
- 不能因为“local_value 风险概率较高但 objective 更好”就声称安全；须同时展示所有种子失败总数、无预测覆盖和长尾目标值。也不得仅挑选 Branin sparse 有利种子。

## 执行与审计

PR63 通过纯 Python E0/E2 测试后，固定 Git HEAD；先工作机 smoke，确认真实 GP/VAE 的候选绑定、status/SOC/物理标签、完整输出。再用新输出目录执行 full。无需上传/更改 PR45–62 历史 SQLite/报告。

PyCharm 入口：\`run_ra_lsbo_risk_budget63_local_value_pycharm.py\`。

上传全新独立结果分支：
- \`fixed_six_onpolicy_report.json\`、\`plan.json\`、\`source_receipt.json\`、\`progress.jsonl\`;
- \`trials/\` 中每条提议、真实评价和 result，\`shared_initial/\`（建议按任务分 ZIP）;
- smoke/full console、pytest 日志、固定 Git HEAD 与所有上传字节 SHA256。

一旦执行异常，上传完整 traceback 和已结算收据；**工作机不编辑研究代码、不覆盖结果、不自行调参**，修复由 ChatGPT 推送新 SHA 后再严格全新执行。

## 文献方法边界

基础的 constrained expected feasible improvement (EI×PoF) 属于既有经典 BO 文献，不是本研究发明；在线校准、分布移位与安全 GP 不确定性也已有相关研究。本 PR63 的候选局部信任与收益—失败代价组合仍属于**尚未被试验验证的特定算法构造**，是否有独立创新贡献必须根据消融和更广泛文献对照再评估，不应提前写成理论安全性保证。

延伸阅读：
- Gelbart et al., Bayesian Optimization with Unknown Constraints, 2014.
- Deshpande et al., Online Calibrated and Conformal Prediction Improves Bayesian Optimization, AISTATS 2024: https://proceedings.mlr.press/v238/deshpande24a.html
- Fiedler et al., Safety in Safe Bayesian Optimization and its Ramifications for Control, 2025: https://arxiv.org/abs/2501.13697

PR63 是算法开发和预先固定的真实解析研究协议，**不是 PR50–61 的结论修订，也不是飞推控安全认证**。
