# PR53：真实在线选点的阶段风险校准与前缀校准

## 科学定位
PR52 已表明原算法在真实在线选点上的约束概率存在明显乐观偏差（原始 Brier 约 0.188–0.200）。PR52 的整体跨种子单截距校准改善总体预测，但边界探索阶段可能反而变差。因此本 PR 对比阶段校准和仅使用同种子过去评价的前缀校准，均只作**原始在线已选点的回顾性风险诊断**，不更改候选策略。

## 来源与固定分割
- 只读 PR52 结果 commit 694b098cacba75bf5168eec3a1c55dbf84289051 的 prequential_risk_predictions.csv 和 prequential_risk_report.json，验证两文件 Git blob SHA1、PR45 源 SHA、SQLite 身份、9,216 次在线选点、原始概率和 seed/iteration 序列。
- 原始 100/140km 两任务、ra_lsbo/full_latent 两臂；每任务每臂 64 个校准种子（索引 640–703）和 64 个测试种子（704–767），每种子 18 次。预先固定，不选择有利种子。默认输出 4,608 个原始测试选点 × 2 个事件（可用可行及任务完成）。
- 原状态 completion 恒为 1，不能解释为飞机任务完成；本试验以原 SOC GP 满足概率预测该数据中的任务完成，以全部非解析约束概率乘积近似可用可行概率，禁止重复乘入错误状态头。

## 对照定义（每个任务、每个主臂、每个事件）
1. raw：原始 PR45 选点前预测。
2. pooled_other_seeds：用 PR52 原来其他校准种子的 GP 预测与已观察真值拟合的单截距；完整输入时要求与 PR52 保存的单截距数值一致。
3. phase_other_seeds：以 mode == budgeted_boundary_information 划分 boundary 和 other；每组截距向总体单截距进行 ridge=0.10 收缩。每组必须同时包含正负类、≥32 条可评分校准数据及 ≥8 个独立校准种子，否则回退总体截距。此方案使用其他种子的已观察结果，属于回顾性评估。
4. prefix_only：严格只使用**本种子本次提案之前**已选点、已结算且有预测的记录，≥6 条且有正负两类时用 ridge=0.25 的单截距修正当前预测，否则使用 raw。原轨迹中按时间可获得，但未被 PR45 的原候选策略实际采用。

分阶段构想来自 PR52 结果，属于**事后探索性构想**，测试种子仍是 PR52 同一批，不是全新独立确认性数据。不得把新校准收益写成确认性独立泛化或在线优化收益。

报告 Brier、Log Loss、5-bin ECE、p≥0.5/0.8/0.9 下假可行数量与被接受率、按种子聚类的描述性配对差区间。风险预算 4/6/8 单位仅核对**原始 18 次已选点路径**上的预测累计失败费用与实际失败次数，缺乏预测的覆盖点按最坏 1 计入；**不是**新风险预算控制器的反事实效果，也不是实际失败上界保证。

## 工作机 PyCharm 操作
在新工作树检出 PR53 固定 SHA，Rcaide_vsp 环境运行根目录：

run_ra_lsbo_phase_prefix_risk_pycharm.py

默认读取 PR52 完整结果目录：
D:\LSBO\RA-LSBO-native-risk-prequential-no-aircraft-20261104-full

若路径不同，使用 --source "完整 PR52 输出目录" 指定；只需要该目录的原始 CSV/JSON，不需要 20GB 飞机数据库。

第一轮 Script parameters：--per-block 1。
成功后去掉参数，另建全新输出目录完成全部 64+64 个种子测试。

上传独立 GitHub 结果分支：phase_prefix_report.json、phase_prefix_predictions.csv、smoke/full 完整 console.log、固定 Git SHA 和 PR52 输入文件运行前后 SHA256。若有字节、身份、概率或数值异常，上传 traceback/failure.json，不修改工作机代码。

全程新增飞机、RCAIDE、GP/VAE 拟合及优化器选点均为 0，保留原 PR45–PR52 实验数据。
