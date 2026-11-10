# RA-LSBO：先审计证据，再研究风险与目标代理的决策价值

> **当前阶段：PR84 研究任务，实际实现为 PR #85；Issue #84 为设计记录。**
> PR83/C2 已完成并封存，禁止重跑。当前只做原始证据一致性复算和已选候选的预测诊断。

## 当前判断与边界

C2 的原定 12 项因子效应检验没有 Holm 拒绝，不能包装为已确认的主效应，也不能据此断言代理更新无效。原汇总存在 `arms_all_final_feasible` 布尔值与逐臂 final-feasible 数量不一致，需要由初始与在线可观测反馈重算。

本阶段不追加种子、不修改采集函数、不拟合模型。研究重点是核对风险与目标模型在实际选点上的预测、反馈可得性及失败代价。选择样本上的指标不是独立测试集校准，更不证明选择性采样本身导致模型有偏；这仍是待验证假设。

## 固定输入

结果 HEAD：`ade354de8b35d20a2bb32b469b04bc7da92aa004`

原 C2 代码：`975d96caac53d675ebe1b790a0647479a75a2ed5`

审计入口：`run_pr84_c2_audit.py`。`smoke/full` 仅指读取既有档案，不产生新的优化运行。

输出单独封存，原 646 MB 证据不修改；新增数学评价、GP 拟合/预测、VAE、幻想、飞机/RCAIDE调用均为 0。

## 导航

- [PR84 审计协议](docs/PR84_C2_READONLY_AUDIT.txt)
- [实现 PR #85](https://github.com/Luyi1235/RA-LSBO/pull/85)
- [设计 Issue #84](https://github.com/Luyi1235/RA-LSBO/issues/84)
- [研究总览 Issue #76](https://github.com/Luyi1235/RA-LSBO/issues/76)
- [转向前 C2 README 原文](docs/README_C2_BEFORE_PR84.md)

工作机只使用已完成真实归档集成 CI 的固定 SHA，不安装 ZIP，不升级环境，不推送代码分支或合并 PR。
