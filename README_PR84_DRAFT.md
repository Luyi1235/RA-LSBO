# PR84 研究任务 / GitHub PR #85：C2 只读审计

PR83 已完成，不重跑。固定输入结果为 `ade354de8b35d20a2bb32b469b04bc7da92aa004`。

新版直接从可观测反馈重建最终可行目标、首次可行位置及轨迹；在线不可行同时检查 region 与 resource，不能用区域中止数替代。对 summaries 和 integrity flag 的矛盾单列报告，不改原始证据，不猜测生成程序的错误原因。

所有候选预测诊断限定在实际选中且已评价的样本。目标误差只在实际返回目标的子集计算，不等价于全候选池校准，也不证明选择偏差是性能下降的原因。

入口 `run_pr84_c2_audit.py` 仅有 `plan/run/verify`；`run` 只运行审计，不运行优化。纯标准库，`python -S`，零新增真值评价及 GP/VAE/飞机调用。

旧草稿 `11fba9f...` 只经过小型指标测试，存在主路径字段缺失和失败口径错误；不得按旧指令执行。以 PR #85 明确公布的完整集成验收 SHA 为准，禁止跟随浮动 HEAD。

[协议](docs/PR84_C2_READONLY_AUDIT.txt) · [设计 Issue #84](https://github.com/Luyi1235/RA-LSBO/issues/84) · [实现 PR #85](https://github.com/Luyi1235/RA-LSBO/pull/85)
