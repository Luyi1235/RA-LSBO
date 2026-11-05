# PR60：分支历史可行性不确定性与风险回退信息探索（新 GP/VAE on-policy 实验）

## 来源与研究假设

PR59 从 PR58 真实在线选点评估发现：稀疏 Branin/Rosenbrock 的 GP 约束概率乘积平均低估不可行概率 0.171/0.224；预测失败概率不高于 0.2 的选点仍有 35.9%/27.5% 失败率；47 次无候选满足预算门槛回退有 42 次实际失败。边界与可行改进阶段都有风险低估，未拟合 GP 的 coverage 仍消耗实际失败费用。

因此 PR60 不再调旧阈值，而是引入**只依赖同一运行已完成评价的分支级经验风险/离散度**，观察它能否改善新选点的实际收益与失败数。PR59 的风险率不进入新策略训练、常数或调参；原来的三个对照均保持原 PR54 代码。

## 算法实现与原理

对每个层级离散结构分支 b，仅使用本算法当前步之前**真实已完成**的初始和在线评价。如果此前该分支出现 f 次不可行、s 次可行，以 Beta(1,1) 作为固定先验，计算：

- 分支不可行率的经验后验均值：m_b = (1+f)/(2+f+s)
- 分支不可行率经验离散度：d_b = sqrt[(1+f)(1+s) / ((2+f+s)^2(3+f+s))]
- GP 原始风险代理：r_GP(x)=1−∏各物理约束 GP 的边际满足概率

其中 Beta 是**工程启发式不确定度**：在线自适应选点不是相互独立的同分布 Bernoulli，且同分支内不同连续设计可能差别巨大。m_b+d_b **不构成统计置信上界、安全证明或真实候选概率校准**。

6 个方法（逐项消融，不能合并解释）：

1. legacy：PR54 冻结原选择器。
2. semantic_only：PR54 仅约束概率选点。
3. budgeted：PR54 原 GP 概率乘积 + 剩余失败额度门控。
4. branch_mean：以 max(r_GP(x),m_b) 代替风险排序/门槛代理，其余预算机制不变。
5. branch_dispersion：以 max(r_GP(x),min(1,m_b+d_b)) 替代风险代理，其余机制与 branch_mean 相同。
6. branch_explore：同 branch_dispersion，但无候选满足预测额度时，不再按“最小修正风险”回退，而选择已真实观察信息最不足的分支（最大 d_b），明确记录其属于**高风险信息性回退**，不将风险门槛失败装作通过。

仍沿用原 PR54 的目标潜空间 GP、One-hot 风险 GP、VAE 结构训练、4×2 stable GP/EP 核格、exact/interval 反馈、原阶段调度及内部 12 步 horizon、6 次已观察不可行硬停止规则。没有给旧三组改 GP 或采集；新分支代码是 opt-in 独立类，已记录 PR54 原始 policy/engine SHA256 的运行前身份校验。

新策略在模型失败或 scheduled/degenerate coverage 时不假造概率；这些真实调用及失败保留在所有评价费用中。风险收据应绑定最终实际选中的候选；预测筛选若被 coverage 覆盖则移除无关风险收据。

## 固定真实解析实验

- 任务：自定义层级 Branin ordinary/sparse 与 Rosenbrock ordinary/sparse；不是飞机/RCAIDE 或官方标准 benchmark 的完整复现。
- 种子：997101–997112，每任务 12 个，完全不与 PR54/PR55/PR58 已用的三块种子重复。
- 每个 case/seed 的 12 个真实初始评价只进行一次，由所有 6 臂共享；之后每臂恰好 6 次自己选、自己真实评价、自己拟合 GP/VAE 的在线调用。没有反事实“知道未评价候选”的真值。
- 所有方法使用相同原始随机候选池流、同样的 GP/VAE 拟合频率、相同硬停止协议；故每种子在线评价成本相等，且第六次前不可能触及六次已观察失败的停止条件。
- Smoke：四任务各第一个种子，6 臂最多 2 次真实在线评价，共 4×(12+6×2)=96 次新解析评价。
- Full：4×12×(12+6×6)=2,304 次新解析评价，288 条 on-policy trial。新增飞机/RCAIDE 均为 0。

严格看三层证据：

1. **同工作量的 feasible-first 胜/负/平**：双方只有一方有精确可行目标，该方胜；双方都无可行则平；双方都有则小目标胜。报告所有种子，不填缺失目标。
2. 双方都可行时的配对目标差、是否改变候选轨迹、实际在线不可行数、原始 GP 和修正代理风险的预测偏差、已评估但无预测的 coverage 费用。
3. 对 branch_mean − budgeted、branch_dispersion − branch_mean、branch_explore − branch_dispersion 的逐项对照，区分“使用分支经验均值”“加入离散度”“信息性回退”。仍提供 branch_explore − budgeted/legacy 与旧 budgeted − semantic_only 的关键比较。

固定常数为 Beta(1,1) 与 1×后验标准差，仅为事先约定的探索性启发式；不能依据本批试验结果再隐式调参并声称是同一策略。12 种子/任务只适合作为机制探索、不能用小样本后验区间推断安全保证。

## Windows 工作机操作

在新 worktree 检出 PR60 固定代码 SHA；Rcaide_vsp (Python 3.9.25)，PyCharm 运行：

run_ra_lsbo_risk_budget60_branch_uq_pycharm.py

先 Script parameters：--profile smoke

成功后在**全新输出路径**去掉参数执行 Full。新增 GP/VAE 拟合和解析数学评价为真实调用，绝不调用或修改 PR45–PR59 原飞机数据、SQLite 和历史结果。

上传至独立 GitHub results 分支：
- fixed_six_onpolicy_report.json、plan.json、source_receipt.json、progress.jsonl
- shared_initial/、各 trials/ 的 proposals.jsonl、observations.jsonl、result.json，建议按任务分 ZIP
- Smoke/Full 完整控制台及测试日志、固定 HEAD SHA、主要源/结果 SHA256、读取后核对的总评价数量
- 若测试失败、风险收据身份不一致、模型异常或运行途中失败，上传完整 traceback 与已经写出的只读证据；工作机不修改代码、不覆盖结果，由 ChatGPT 负责修复并新建固定 SHA。

**科学边界：** 本算法是否优于旧算法未知。经验分支离散度有结构依赖和自适应采样偏差，绝非保真置信域；即使实验减少了失败也不能称为可靠的飞行安全。这里的目标是验证候选级风险不确定性和回退信息策略有没有额外算法价值，然后才讨论更高保真度飞机任务。
