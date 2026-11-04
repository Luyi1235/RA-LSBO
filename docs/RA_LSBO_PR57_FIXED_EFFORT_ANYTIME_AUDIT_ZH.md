# PR57：前瞻优化真实轨迹的等评价次数审计

## 动机与来源

PR54/55 预算算法在稀疏 Branin 中有明显终点改善，但实际执行的在线调用次数也更多。论文需要区分算法早期选点价值与“因少触发六次失败停止而获得的额外评价机会”。

本轮仅仅读取两个已经完成的 GitHub 结果文件，不运行新优化器，不拟合 GP/VAE，也不调用任何解析评价器、飞机或 RCAIDE。

固定来源：
- PR54 完整报告：结果 commit 41db5094f29645cf5afd17b08e65ff62544d3f41，Git blob SHA1 7f94ed040ee88c2ae552f704516fcea003193ded，48 条真实闭环轨迹。
- PR56 修正的 PR55 完整报告：结果 commit cdd02b551d3233d8262d5adbf46db207d6703f93，Git blob SHA1 be839a224ef56868f048cfb161fc54065c2eea8b，192 条真实闭环轨迹。

对所有报告在计算前后严格验证 Git blob SHA1，并核查注册 case/seed/arm、相同共享初值、真实评估调用账本、六次已观察失败停止、逐次最优值单调性。PR56 只修复原 PR55 报告风险收据汇总，不修改原始试验。

## 固定分析

**A：共同固定六次在线评价**（主要可比口径）。

三臂均在第六次已观察不可行之后才能触发硬停止。因此每条实际轨迹都至少执行六次在线解析评价。使用相同 12 个共享初值，加上各算法自己的前六次真实选点和已结算评价，在每个配对 seed 上比较第六步时已知最优可行目标。报告双方都可行时的目标差、单方可行、双方不可行、前六次真实失败数。绝不补全尚未发生的后续评价。

**B：每个种子的双方共同实际长度**。

每对算法取 k=min(实际在线调用 A、实际在线调用 B)，分别使用各自在 k 时已经评价的结果。虽然单个种子内工作量相同，但 k 是运行结束后才能知道的，受早停影响，是事后受策略选择的结果，**不能当作随机分配的同成本因果实验**。

**C：原始完整终点和 k=8/10/12 的真正已观察子集**。

完整终点如实保留不同调用次数；对 k 大于 6 的观察点，只统计实际两臂都达到该步的种子，报告该子集规模及早停排除的组数，避免将失败停止后的未知结果当真值。对双方都取得可行目标的种子报告 common-prefix 差与后续真实改进的代数分解，但这只是数值恒等式，不能当作因果归因。

全部对照为 Budgeted vs Semantic-only、Semantic-only vs Legacy、Budgeted vs Legacy。按 PR54/PR55、case、seed 分层，报告可行覆盖、配对目标均值和中位数、改善/退化/持平数，以及实际在线失败数与调用数。分析构想是在观察 PR54/55 结果之后提出，属于探索性、非独立确认性分析，不作统计显著性或安全保证声称。

## 工作机操作

在新 Git 工作树检出 PR57 的固定 SHA。Rcaide_vsp / Python 3.9.25，PyCharm 运行根目录：

run_ra_lsbo_risk_budget57_anytime_pycharm.py

默认读取：
PR54：D:\LSBO\RA-LSBO-risk-budget-math54-20261104-full\risk_budget_math_report.json
PR56：D:\LSBO\RA-LSBO-pr56-corrected-20261104\risk_budget55_report_corrected_pr56.json

如路径不同，传入 Script parameters：
--pr54 "实际 PR54 完整报告路径" --pr56 "实际 PR56 修正报告路径"

必须选择全新的输出目录；只需要两个 JSON，无需 trial ZIP、原飞机数据库或 RCAIDE 环境中的物理模型。

上传至独立 GitHub 结果分支：fixed_effort_anytime_report.json、fixed_effort_paired_seeds.csv、完整控制台日志、固定 Git SHA、两输入文件前后 SHA256。如出现字节/身份/轨迹错误，直接上传 traceback，不在工作机修改代码。

本轮新增解析、飞机、RCAIDE 调用及 GP/VAE 拟合一律为 0。保留 PR45–PR56 全部历史证据不变。
