# 简化飞机转场实现：验证回执（2026-10-30）

固定执行提交：`b8a94c324cd23a0ef9f15c668fe9a247e4ad9a00`。
分支：`research/aircraft-transition-20261030`；PR #7，未合并。
基础提交：`90dcfbb6e643e69c5836613fde77cbf103fafc93`。

CI：https://github.com/Luyi1235/RA-LSBO/actions/runs/41045597157
Job124028783160。下面的数字来自实际下载的JUnit、results/audits/ledgers/source_snapshot，
不是仅由配置推算。首次CI41045298412因引用不存在的旧测试总文件名而未收集测试；
没有启动smoke或真实模型调用。修正工作流为现存的acquisition/posterior测试文件后复跑成功。

## 实际完成

264 tests，0 failures，0 errors，0 skipped。36项新增转场测试、228项既有影响范围回归；
不是整个历史仓库全部测试，也不是复用旧280项记录。八个测试文件见工作流/本机指南。

| 项目 | 实际调用 | 提案数 | 证据JSON/JSONL | 源码绑定 | 执行/审计/报告 |
|---|---:|---:|---:|---:|---|
| 六方法、两任务smoke | 36/36 | 24 | 56 | 156 | 通过 |
| 缩小CI资格（每题仅2个随机面板点） | 64/64 | 0 | 4 | 156 | 通过 |
| 本机完整资格配置 | 仅prepare，0调用 | 0 | — | — | 计划上限316，无calls.jsonl |

云端研究调用累计100次降阶飞机模型，新增RCAIDE MDA0；未执行完整316资格、数学性能pilot或
飞机性能pilot。单元测试内部人工/廉价求值单列，不冒充研究模型调用或0总计算。
新增测试使用人工标签并禁止真实飞机evaluate；既有回归仍按各自测试边界运行。

## 科学边界与待检查事项

软件审计通过不等于物理资格通过。CI的14个固定设计检查/任务中，100km有10个完成输出的
积分筛查通过、4个因部分输出为inconclusive；140km有2个通过、12个inconclusive。
这些提前结束设计及其记录均保留，没有改阈值、种子或把不完整输出当收敛。
随机面板每题仅2点，不能用它判断真实稀疏率。本机316资格仍须完成并人工评审。

smoke只有每方法两步，两任务初值均冷启动；不能依据各方法终点比较优劣。
新阶段代码没有启用自适应权重，不修改冻结优化器、原评估方程或历史证据。
资格面板、锚点、分支探针不供优化器训练，不要求旧transfer结果或33项可选诊断。

## 独立下载核对

Artifact ID12971842999，保留30天：
`aircraft-transition-b8a94c324cd23a0ef9f15c668fe9a247e4ad9a00`。
ZIP SHA256：`6f49b07197bd6b99f1cb12806813c570fca9c9fea66a6491b2be7e9092630980`。

下载后复核两个运行的全部封存JSON/JSONL证据、所有156项source_snapshot绑定及账本配对相符。
artifact内三个受测试源码文件与本地开发副本逐字节一致。没有再拟合GP或重算真实模型。
Linux/Python3.11、CPU torch2.8.0、numpy2.0.2、scipy1.13.1、gpytorch1.11、botorch0.10.0、
scikit-learn1.6.1、pytest8.3.5；完整环境保留在artifact/environment.txt。Windows仍需本机验收。

后续提交只增加本回执和工作机指南，不更改受测试运行代码。不要把CI计划搬到Windows运行；
使用固定checkout在本机新建计划，保留源码/环境绑定。完成36+316后停止，不自动进入正式pilot。
