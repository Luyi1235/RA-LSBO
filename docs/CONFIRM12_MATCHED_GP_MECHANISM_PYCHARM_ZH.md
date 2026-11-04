# RA-LSBO 双视图 GP 机制检验：Windows 直接执行

代码位于 PR #48，原 PR #45 151006 条飞机真值数据库只读；零新增飞机/RCAIDE 调用。

科学问题：潜空间对能耗预测是否有价值？显式 One-hot 对 SOC、其他约束和完成状态的预测是否更有价值？不同优化器的已选设计具有选择偏差，因此必须在相同训练数据及相同测试设计上比较，而不能直接比较各方法各自采集点的 Brier。

方案：对普通及 140km 两任务，按预先固定种子列表选第 1–8 个作为训练种子、第 129–136 个作为独立测试种子；各自使用 12 条真实共享初始记录。每个训练种子训练一次 VAE 潜空间与一次显式 One-hot GP 后验，使用 2000 条无标签结构语料和 PR45 固定的核/EP 超参数。测试标签没有参加拟合。目标 GP 和风险 GP 两套输出通过 2×2 交叉分配对应 RA-LSBO、Full Latent、Full One-hot、Reverse Representation。不是 4 次不同随机 GP 训练。

报告内容：精确目标 RMSE、约束余量 RMSE、SOC/约束 Brier、边界 Brier、假可行数量、完成概率 Brier；同一测试目标的潜空间与 One-hot 配对误差，置信区间按独立训练种子聚类。逐点预测保留，以便论文分析。

请将 PR #48 拉到全新 Windows 工作树，选 Rcaide_vsp / Python 3.9.25，打开仓库根目录的 run_ra_lsbo_matched_gp_pycharm.py。

默认读取 D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z。

第一轮在 PyCharm Script parameters 中填写 --pairs-per-task 1 ，运行两任务各一个独立训练/测试种子对。成功后移除参数，另起一个输出目录运行默认每任务 8 对。输出位于 D:\LSBO\RA-LSBO-matched-GP-no-aircraft-<UTC>，包含 matched_gp_report.json 和 matched_gp_predictions.csv。如失败保存 failure.json、traceback、源 SHA。

工作机不改算法和分析代码，不重新执行飞机模型；只需将两份结果文件、完整控制台日志、PR HEAD SHA 上传独立 GitHub 结果分支。原始数据库/20GB 资料原封不动。

当前研究状态：100km 主比较与 140km 主比较的配对能耗均值略优于 Full Latent，但 256 嵌套种子中 Full One-hot 优于 RA-LSBO；这说明应优先验证潜空间预测收益，而不是预设双视图胜出。140km qA CI 上界超过 10%，不宜声称正式稀疏可行域资格通过。机制检验属事后预测诊断，不是确认性的优化收益证明。
