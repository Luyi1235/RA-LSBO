# X-57 47.5 nmi 五臂对比 · 检查点 16

这是 2026 年 11 月 27 日完成的 RCAIDE 仿真批次。每个种子共享 8 个初始设计，再由五个算法各评价 8 次，到达检查点 16。计划为每个算法配置的完整预算是 320 次，因此这份归档只覆盖前 16 次。能耗质量阈值为 47,500 Wh。

| 算法 | 达到阈值的种子（共 6 个） |
| --- | ---: |
| LSBO full budget | 3 |
| RA-LSBO energy sign | 3 |
| RA-LSBO energy interval | 3 |
| screened LogCEI | 3 |
| screened SCBO | 1 |

表中计入了种子 953731，它在共享初始化的第 2 次评价就达到了阈值。共享初始化后，5 个种子有 1–2 个可用设计，1 个种子有至少 3 个，零可用设计组为空。这一批次可用于比较少量可用初始设计下的早期表现，零可用设计的表现需要另外的实验。

## 从哪里开始

- [report.json](report.json) 保存每个种子和算法的结果。看 `per_seed` 中的 `first_quality_call`、`best_usable_Wh` 和 `trajectory`，也可以用 `group_metrics` 比较初始可用设计数量不同的组。
- [calls.jsonl](calls.jsonl) 保存逐次事件。用 `event == "settled"` 筛选完成记录，并以 `task_id` 去重。账本有 288 条独立结算，其中 48 条是共享初始化，240 条是在线评价，每个算法 48 条。
- [run_plan.json](run_plan.json) 和 [distance_47.5.json](distance_47.5.json) 给出冻结的随机种子、输入校验值、预算、并发和任务配置。[recovery_manifest.json](recovery_manifest.json) 记录原始尝试及恢复情况。[SHA256SUMS](SHA256SUMS) 可用于核对归档字节。
- 执行器固定为 `ae5dd445c9656ffa31c9e340b1364bc2c746eef7` 的 [run_x57_five_arm.py](../../run_x57_five_arm.py)。历史资格输入来自提交 `9c5ac77c5ecc3987787292981dc679045441972c` 的 `results/`，启动器按冻结哈希校验。对比分析可先从本归档读取数据，开发新实验时另建目录保存计划和结果。

## Windows 续跑修复

[launcher/run.py](launcher/run.py) 在外部优化器使用隔离 Python 时，为其虚拟环境写入作用域受限的 `.pth` 钩子，以便优化器服务能导入冻结源码中的 `ra_lsbo`。同目录的 [resume_recovery.py](launcher/resume_recovery.py) 核对已承接的任务账本，并分别计算原批次和恢复批次的峰值并发。

原批次有 8 次初始化因内存分配失败，恢复阶段用 4 个 worker 补齐这 8 项，随后在线评价恢复到 12 个 worker。最终报告标记为 `complete`，错误为空，峰值并发为 12。报告记录 40 次承接的有效 worker 启动和 248 次恢复批次启动，另有 8 次原批次基础设施失败。

复跑脚本依赖 Windows、RCAIDE/VSP Python、独立的优化器 Python、冻结资格计划及其 `physical-base`。运行 `launcher/run.py --help` 可以查看路径参数。原始数据中的绝对路径用于记录来源。此目录是结果和源码归档，直接续跑还需要原机器上的逐任务文件和原始账本。报告中的 `next_proposal_replay_verified` 也未标记为通过。后续要扩大预算，应在新输出目录验证恢复和下一候选点重放，再追加评价。
