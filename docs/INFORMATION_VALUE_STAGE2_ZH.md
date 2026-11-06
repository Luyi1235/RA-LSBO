# RA-LSBO 信息价值路线第二阶段：全候选短视对照

日期：2026-11-06。代码基线：PR66 固定代码 `1917f305b39f93d79f6b16dd83e56af365098500`。
本阶段先检验 PR66 发现的支持域退化是否会影响实际在线优化；不声称已经实现 GP 两步前瞻。

## 1. 唯一研究问题

PR66 的一个 Branin sparse 检查点出现 128 个合法 GP 候选中仅 1 个通过 optimistic-violation eligible 支持条件，且该点不可行；原候选池另有 11 个可行候选。本阶段检验：移除这一 eligible mask、但保持同一风险预测与软效用后，是否改善同成本在线选点。

注意：PR63 的 `soft_gp` 没有概率硬门，但仍有 optimistic-violation eligible 支持筛选。本阶段不是“是否使用风险概率硬门”的实验，而是“是否保留 eligible 支持筛选”的精确消融。

## 2. 两个臂

- `soft_gp_eligible`：原 PR63 `LocalCredibilityValueBO(variant="soft_gp")`，源码和选择行为不改。
- `soft_gp_all_support`：相同 PR63 GP/VAE、相同风险代理、EI、soft utility、阶段计划和候选池；唯一变化是将原始 `eligible` mask 改为全部合法 GP 候选均可参与软效用排序。仍保留候选 canonical、重复点、precheck、模型覆盖及真实 evaluator 保护。

不增加风险惩罚、不调阈值、不加入局部风险融合、不加入信息奖励。全支持臂可能评价高预测风险点，必须如实计入实际失败。

## 3. 冻结实验

四个自定义数学任务：Branin ordinary/sparse、Rosenbrock ordinary/sparse。
新种子 `998701–998712`，与 PR54–66 已注册种子块不重叠；smoke 使用 `998701`，full 使用全部 12 个种子。
每个 case/seed 共享 12 个真实初始评价；两个臂各自在线实际评价 6 次。候选池 128，使用相同初始与候选随机流；两臂从各自真实历史独立拟合 GP/VAE。

| 项目 | smoke | full |
|---|---:|---:|
| case-seed 轨迹 | 4 | 48 |
| 初始评价（共享） | 48 | 576 |
| 在线评价 | 16 | 576 |
| 数学评价合计 | 64 | 1,152 |
| 飞机/RCAIDE | 0/0 | 0/0 |

full 每臂 288 次真实在线评价。主比较按任务分别报告 feasible-first 胜/负/平；目标差只在双方都获得可行解时报告。禁止将目标尺度不同的任务混合成一个总体平均。

## 4. 费用与证据

每次真实 `problem.evaluate` 都记录候选、proposal、观测和调用次数。两臂初始前缀必须一致；每臂恰好完成 6 次真实在线评价。模型预测缺失/覆盖动作没有风险概率收据。不得使用未来真值、PR66 后置审计标签或 PR45–66 历史 SQLite。

报告每个任务/种子的：实际失败数、最终可行、冷启动发现、原 eligible 候选数、all-support 候选数、选择模式、预测可用性和最终目标。保留全部逐轨迹数据。

## 5. 进入下一阶段的门槛

- 若 all-support 改善主要来自 Branin sparse 的支持坍塌样本，说明 eligible 支持筛选值得进一步研究，但不能直接称为普遍优势。
- 若 all-support 失败显著增加，说明扩大决策支持域需要更好的约束证据或其他机制，不应继续简单放开。
- 若两臂路径几乎相同，说明 eligible mask 不是主要瓶颈，应优先处理目标收益排序/预测。

无论结果如何，本阶段仍是 myopic selector ablation。只有在同成本在线结果显示模型决策具备可靠基础后，才接入真实 GP fantasy update 的两步 lookahead。有限两步机制的构造性 PR66 结果不能替代该验证。

## 6. 工作机执行

使用现有 `Rcaide_vsp` 环境，在全新 worktree 检出 PR67 固定 SHA。不得修改代码、物理模型或依赖。

```text
python -m unittest discover -s tests -p test_information_value_stage2.py -v
python -m compileall -q ra_lsbo/experiments/information_value_stage2 run_ra_lsbo_information_stage2_pycharm.py
python run_ra_lsbo_information_stage2_pycharm.py --profile smoke --output <新smoke目录>
```

smoke 成功且完整账本核对通过后，在另一全新目录运行 full。异常时保留失败目录与 traceback，停止，不原地修复或重跑。上传独立 `results/information-value-stage2-20261106` 分支，保留 report、plan、source_receipt、progress、每条 proposal/observation、日志和 SHA256。禁止覆盖 PR66 结果、合并代码 PR 或修改 main。

本阶段不运行飞机/RCAIDE，不声称两步 GP 算法胜出或安全保证。
