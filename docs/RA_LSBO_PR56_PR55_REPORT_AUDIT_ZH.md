# PR56：PR55 风险收据聚合修复（只读，不重跑实验）

PR55 已完成 192 个 on-policy 数学 trial。原始 README/analysis.txt 从 trial receipts 得到：Branin sparse budgeted 17 次预算回退，Rosenbrock sparse 22 次。但 PR55 JSON 汇总字段 coverage/fallback 显示 0。

根因是字段名不一致：
- runner 写入 prediction_available_for_selected
- protocol 汇总误读为 forecast_available

PR56 将汇总逻辑改为读取真实 runner 字段，并增加以下字段：有效预测收据数、回退次数、回退实际失败数、预测收据上的实际失败率、平均预测不可行代理、预算门通过/未通过数。未更改 acquisition、GP/VAE、候选选择或 PR55 原始证据。

## 只读重建

在包含 PR55 结果目录的工作机，检出 PR56 固定 SHA，使用同一 Python 环境执行：

    python -m ra_lsbo.experiments.risk_budget56_report_audit --source "D:\LSBO\RA-LSBO-pr55-independent-budget-math-full-20261104\risk_budget55_report.json" --output "D:\LSBO\RA-LSBO-pr55-independent-budget-math-full-20261104\risk_budget55_report_corrected_pr56.json"

脚本只读取原报告的 trial_details 和逐轮风险收据，不导入或调用模型，不做任何新的解析评价；源报告字节保持不变，修订版写到新的文件路径。会校验 192 trial roster、逐轮 receipt、目标曲线、实际停止账本。原始 trial ZIP 和 PR55 结果 commit 不修改。

## 需要核验

- Branin sparse budgeted 预测收据应为 148，回退 17。
- Rosenbrock sparse budgeted 预测收据应为 149，回退 22。
- 两项回退的真实失败数分别为 12、16。
- Ordinary 两项 budgeted 不应有回退。
- 原始 README 的摘要表与新 JSON 聚合字段应一致。
- 报告中所有概率保证仍为 false；不把预算回退解释为安全过滤。

## 边界

这是 PR55 结果文件的纯聚合修复，不是新算法，不是新实验，不是 PR55 结果重跑。新解析、飞机、RCAIDE 评价均为 0。PR55 原始 report、README、ZIP 及结果分支保留原样。
