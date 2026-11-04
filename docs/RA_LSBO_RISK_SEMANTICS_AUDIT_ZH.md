# PR50：PR45 真实评价的风险事件语义审计（零新增飞机调用）

## 科学问题

必须区分 RunObservation.completion_label() 的**评价记录状态**、diagnostics.mission_completed 的**任务完整执行**、精确目标值、各项工程约束的真/假/未知，以及 DesignRecord.usable_feasible 的**优化器可用可行结果**。

特别注意：status=completed 但任务因 SOC/电池功率提前终止、目标仅有区间下界，是合法的部分观测记录，不可将其误称为完整飞行任务成功，也不可抹除有效约束区间。

## 运行

本 PR 基于 PR49 源码分支，追加只读新模块，**不修改 PR45/46/48/49 的算法、物理模型、数据库或结果**。

在新 Git 工作树使用 Rcaide_vsp，PyCharm 运行根目录：

run_ra_lsbo_risk_semantics_pycharm.py

首次 Script parameters：--max-records 1000，产出 PARTIAL_DIAGNOSTIC，仅作接口试跑。移除参数后，使用**全新输出目录**运行默认完整 151006 条真实评价审计。

默认源目录：D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z

输出：risk_semantics_report.json、risk_semantics_by_group.csv、risk_event_records.csv，以及用户保存的完整控制台日志和 Git HEAD SHA。输出默认位于 D:\LSBO\RA-LSBO-risk-semantics-no-aircraft-<UTC>。遇到异常保存 failure.json 和控制台 traceback，**工作机不修改代码**。

## 分析口径和限制

- 按 case × role × arm 计数：initial、online、verification 以及 7 个算法；初始评价只计算一次，不复制到多个算法统计。
- status_completion：复现既有 RunObservation 的标签定义，不重新训练或改变旧模型。
- mission_completed：使用布尔诊断或有明确含义的最终终止码；未知保持 null。
- physical_feasibility：有任何已知工程约束违反则为假；全部 8 个必要约束均已知满足才为真；其余为未知。
- usable_feasible：按现有代码 status=completed ∧ 全部约束已知满足 ∧ 目标为精确值 判定；额外提供 mission_and_usable_feasible 供交叉核查。
- 每项约束严格保留 exact/interval/missing，并报告可判定约束符号的覆盖；0 点在开闭区间端点按原语义处理。
- status=completed 同时 mission_completed=False 是**应统计的标签差异**，不是默认的程序错误。
- 本实验只做语义及样本覆盖诊断，没有概率模型拟合/校准，更不证明隐藏失败识别、优化收益、形式化安全保证或 140 km 稀疏资格。
- 原始计划、SQLite SHA256 与记录 SHA 身份需严格匹配 PR45，源 DB 以 SQLite mode=ro 打开并在运行前后检查 SHA 和 mtime，不回写原始证据。

分析后再决定是否需要单独建模任务完成、审视旧 completion GP 的物理解释，并开展独立留出校准。不得用全体标签直接调整 PR45 历史采集行为或冒充前瞻结果。
