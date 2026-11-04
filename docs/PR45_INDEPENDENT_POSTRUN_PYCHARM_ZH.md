# PR #45 完成后独立核查：直接 PyCharm 运行

代码：[PR #46](https://github.com/Luyi1235/RA-LSBO/pull/46)，基于**原 PR45 源码 SHA `386082beeacac8ae43d0dfb1ccda5c5ad3b356cf`**新建。原实测结果 GitHub commit `d42ac115231372a39e0557aa8f61948b7783f61b`，在 `results/pr45-real-aircraft-20261104/`；勿改原账本。

## 直接操作

- 拉取 PR #46 至新的 Git 工作树，用 **Rcaide_vsp** 打开 PyCharm。
- 右键 **Run** `run_pr45_postrun_audit_pycharm.py`。脚本自动尝试工作机原路径：

  `D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z`

- 如果原路径另存了，配置环境变量 `PR45_REAL_RESULT_DIR`，或者 Script parameters 设为 `--root <原真实实验结果目录>`。结果默认写入 `D:\LSBO\RA-LSBO-pr45-independent-postrun-<UTC>` **全新目录**。
- 仅生成 `postrun_report.json` 和 `paired_main_by_seed.csv`，不会运行任何飞机/RCAIDE 求解，不会改写原 `real_calls.sqlite`、`RUNNING_REAL` 元数据、`audit.json` 或约 20 GB 的历史数据。

## 检查内容

1. 原 SQLite 只读校验、冻结 1024/256 种子与完整七算法登记、2,048 组进度唯一性、151,040 预算、15 万条结算计数、不可变事件链 SHA；完成标记和原始 `RUNNING_REAL` 状态分别显示。若所有证据一致，单独报告 `COMPLETE_AUDITED_POSTRUN`，**不伪造原账本终态**。
2. 每方法 `result.json`：6,656 份、完整 18 在线、共享初始摘要一致、对应 6,622 份可选 5s 复核；剩余 34 项按实验合同不计费。
3. 从 SQLite **24,576 条真实共享初始记录**计算 100/140km 单设计 `q_A` 可行率、Clopper-Pearson 95% CI 和原本期望的普通/稀疏域阈值。与优化最终发现率独立看待。这是事后、并非独立 600 样本资格审批。
4. 1024 个主方法配对种子的 100 km 能耗差及 140 km 可行发现差；5 辅助方法严格只用预先选出的各 256 个嵌套种子。所有 p 值和 CI 标注为*事后探索性*，不能倒填原始预注册 Holm 等科研结论。

## 回传给我

把新生成的 `postrun_report.json` 和 `paired_main_by_seed.csv`、完整控制台输出、源码 SHA 上传独立 GitHub 结果目录。**不必上传 20 GB 原始提案/VAE 权重，也不许删除原件**；原始数据继续本机只读封存。

当前可独立给出：PR #45 已运行完毕、2,048/2,048；151,006=24,576 初始+119,808 在线+6,622 复核；34 空余复核。是否为符合原研究设想的 140km 稀疏域，需要本工具原始 `q_A` 统计结果。是否有 RA-LSBO 的可信统计优越性，需要配对分析结果及独立科学复核。
