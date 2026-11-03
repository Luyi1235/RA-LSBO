# PyCharm 直接运行真实 151,040 上限的七算法实验

在 Windows 的新 Git 工作树中检出本 PR，使用 **Rcaide_vsp** 解释器。

1. 打开仓库根目录的 **`run_ra_lsbo_real_12workers_pycharm.py`**。
2. 右键 **Run**。这是**真实简化飞机模型评价**入口，不是哈希演练，也不是构造反馈。
3. 它自动生成独立的 `results/true_aircraft_12workers_<UTC时间>` 目录，使用固定 **12 个 Windows spawn 进程**。
4. 预算：100/140 km 各 1,024 个主种子，主两算法；其中各 256 个嵌套种子另运行五辅助算法。每个任务/种子 12 个共享初值，在线每方法 18 次，可能有 1 次 5s 数值复核，**真值调用上限 151,040**，RCAIDE 调用 0。
5. 观测与优化轨迹写在 `trials/<任务>/<种子>/<算法>`，全局真实评价事务在 `real_calls.sqlite`；进度 `progress.jsonl`，完整结束生成 `audit.json` 和 `completed.json`。
6. 如果运行出错，保留整个结果目录和控制台 traceback，向我发送 ZIP。失败后不会默默重试已开始的真值调用；程序会停止新的派单。**运行时间不设 24 小时截止。**

也可在现有 `run_ra_lsbo_confirm_12workers.py` 的 PyCharm Script parameters 使用 `--mode execute --workers 12 --output <全新目录>`。

**解释边界**：这是用户主动启动、实际能算的七方法研究数据；100/140 km 的原 q_A 600 次资格评价和历史种子完全排除尚未完成，新主法的归因和 CBOB 作者等价性也尚未确证。因此结果保留 `PROVISIONAL_UNQUALIFIED_SEVEN_METHOD_REDUCED_AIRCRAFT` 标记，不能只因计算完整就写成经过确认的论文优势结论。损坏的旧 ZIP 不阻止当前代码执行。
