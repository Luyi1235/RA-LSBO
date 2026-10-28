# RA-LSBO comparison v2 验证回执

代码提交：`aea7607f0edc25f4a424d1d735d85bbb27ebfd68`。
代码树：`027785d530702ada8c0803716bcadca2ee0860d9`。
执行： https://github.com/Luyi1235/RA-LSBO/actions/runs/40690256091
Job `122822942141`，结果success，2026-10-28完成。

环境：Ubuntu24.04.5、CPython3.11.17、CPUtorch2.8.0+cpu、numpy2.0.2、scipy1.13.1、gpytorch1.11、botorch0.10.0、scikit-learn1.6.1、matplotlib3.9.4、pytest8.3.5。

| 验证 | 实际结果 |
|---|---|
| 影响范围回归 | 185 passed，0 failures，0 errors，0 skipped |
| 回归告警 | 14条Matplotlib/Pyparsing弃用告警，未全局屏蔽 |
| 数学smoke | 112/112调用，48个在线提案核对，149项来源绑定，审计和报告通过 |
| 简化飞机smoke | 108/108调用，24个在线提案核对，149项来源绑定，审计和报告通过 |
| RCAIDE MDA | 新增0 |
| 完整20种子pilot | 本轮未启动 |

两套smoke账本共220次。回归测试另含人工标签/廉价评价检查，未对所有测试内部evaluator调用做全局计数；220不是所有软件测试调用总数。用户工作机v1中止费用尚未读取，必须本地另行结算。

首轮修复CI run40689502835为181通过、1失败，未进入两套smoke。失败是旧绘图测试夹具未模拟新增报告复审接口；后续更新夹具并断言复审确被调用，独立真实审计和破坏性测试保留。最终上述185项通过，不把首轮失败隐藏。

这些结果不代表全部历史测试通过、Windows本机已验证、独立优化器回放、算法统计优势或真实飞机物理验证。

## 证据与保存范围

Artifact名称：`comparison-validation-aea7607f0edc25f4a424d1d735d85bbb27ebfd68`。
ID：12816662335；zip SHA256：`2bff41d26ee32d9a7485b7eb6beca4af84a477921a459fab4885c81cfd1757c7`。
入口： https://github.com/Luyi1235/RA-LSBO/actions/runs/40690256091/artifacts/12816662335
保存7天，含JUnit、环境、两套smoke的results/audit/report。没有上传完整逐调用目录和全部source_snapshot；故不能仅凭该精简artifact在其他机器独立重放完整审计。本地新研究必须保留完整目录。控制台保留执行核对日志。

从CI审计输出核对的文件SHA256：

- 数学plan：`b72a1289699a9e1527aca580cb968e8c5f31b9b37a061e7c300da83d262da1d2`
- 数学results：`09ba12890261561185e3fdc6b66bbade784b0de8d936ea597bedf93271831d53`
- 数学calls：`caf08fdc24ac9b4d19e4ad70667cdceecb09dcbc94926429e0cc1274ef911bc4`
- 飞机plan：`c1400f3de3de657cec530b659e5d2a5351e55d4bc6262a296f58de40e10294cb`
- 飞机results：`0c8653b65825221ea8a5a68ba251e33b1ecf08a53b4912cbe03eb5154adc61ae`
- 飞机calls：`53ff97985b8e93878d9d5ba9184dd410a870677ea12c4ed06a1652247e601431`

来源为上述Actions job执行日志与步骤状态，读取日期2026-10-28。此文为摘要回执，不冒充原始日志。
