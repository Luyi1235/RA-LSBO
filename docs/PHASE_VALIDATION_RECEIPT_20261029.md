# Phase修订验证回执（2026-10-29）

固定执行提交：`155f0b0237183d3feefa124e430c01bcd5db1450`。
代码树：`4befc75c6ff2dce8566a01091ace115107c2113f`。
分支：`research/discovery-review-20261029`，PR #4，未合并。
CI： https://github.com/Luyi1235/RA-LSBO/actions/runs/40858151663
Job123394279484，结果success。

实际下载JUnit：**262 tests，0 failures，0 errors，0 skipped**。范围包括三份分阶段测试及七份既有相关测试，不是全部历史仓库测试。Linux/Python3.11，CPU依赖记录保存在artifact的environment.txt；Windows工作机仍必须核验。

Phase smoke：120/120次数学调用，28trial、56在线提案；4数学ordinary/sparse算例、1seed、每组2在线。运行/审计/报告均通过，120项证据哈希及166项源码绑定下载后再次核对相符。新ra_guard_only、ra_phase_coverage、ra_phase_discovery均无暖启动覆盖调用；全部保护组阈值内近重复为0。仅代表有限流程验证，不证明性能恢复。

完整已上传Windows历史pilot的806项封存证据和162项来源由只读导入器验证成功，未改路径/换行/注册哈希，未执行快照代码；在新Linux环境成功prepare新的8916上限phase计划并确认没有calls.jsonl。这是仅登记测试，不是启动完整pilot，也不是重新执行原历史环境审计。

本轮新增研究smoke合计**120次**；数据传输和离线评审新增真实评估0次；回归测试中的人工/廉价求值另计，未对全部内部求值统一计数。新增RCAIDE MDA 0，完整pilot0。没有用历史测试记录冒充当前验证。

Artifact：phase-validation-155f0b0237183d3feefa124e430c01bcd5db1450，ID12890097465。
ZIP SHA256：`e3f32e51a63fc329ecd5973aef33795f323a7a8633824486430d0c301ea09e9f`。
保留14天。包括JUnit、环境、提交、先决证据导入回执、smoke完整计划/快照/提案/观测/账本/审计/报告。

后续提交仅增加本回执、数据评审、工作机指南与独立离线分析工具；不更改已验证优化器。工作机固定到上面的执行SHA。离线工具在本地针对完整旧pilot运行成功；并未把它称作同一CI的回归测试内容。
