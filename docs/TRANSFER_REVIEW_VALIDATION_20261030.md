# Transfer只读诊断验证回执（2026-10-30）

固定诊断代码提交：`8c1ee02131124eb82be62acf261a7f97830e8b37`。
分支：`research/transfer-review-20261030`；基于`80eb368cd49ab512700fcaac441f35b18abe7dcb`。
CI：https://github.com/Luyi1235/RA-LSBO/actions/runs/41027924975

## 实际验证

实际下载JUnit：33 tests，0 failures，0 errors，0 skipped。测试只用数值预测和人工观测包，不导入/调用评估器，不拟合代理或编码器。不是重跑此前280项优化器测试，也不是新的性能smoke。

CI还下载原transfer归档并校验SHA256，然后运行完整只读重建工具：390项证据哈希、170项源码绑定、3,108个账本结算、96trial/2,880提案、206个冷阶段专家评分结算。等权/自适应480个实际候选和480条观测相同；103个冷阶段配对中有6次门控变化、3次模型首选变化、0次实际候选变化，3次变化均被覆盖回退替换。

诊断工具从保存的预测重算评分；不拟合GP、不重放优化器、不调用原数学函数，不给未送评点补算真值。数据获取工作流也没有启动实验。本轮新增数学模型评估0、新增RCAIDE MDA0；测试只是数学算术，不额外调用问题评估器。

## 产物身份

Artifact ID：12964561937。
名称：`transfer-review-validation-8c1ee02131124eb82be62acf261a7f97830e8b37`。
ZIP SHA256：`bc7d617371baea1e0e9108bf4e6e37136935a7c62fd00976b1c2d8afd8d89bc8`。
Artifact保留30天，含诊断代码副本、JUnit、环境、全量派生facts/trials/weights/decision_effects和运行日志。未把原86MB归档重复提交到仓库。

CI依赖：Python3.11，numpy2.0.2、scipy1.13.1、pytest8.3.5；完整环境见artifact/environment.txt。没有安装GPyTorch、BoTorch或Torch来运行诊断。

下载后确认三个已测试源码文件与本地分析副本逐字节相同。本地再次运行完整工具：facts、trials、decision_effects与CI解析后完全相同；weights有7个浮点字段末位不同，最大绝对差1.1102230246251565e-16，不改变记录内比较结论。不将此描述成所有派生文件逐字节一致。

后续提交仅增加评审、后续研究门槛及工作机说明，不改诊断代码或优化器。原结果、源快照、任务、种子、全部旧优化器与main均未改。工作机只需运行零调用核验，不启动新pilot。该验证不证明模型校准、算法等效、统计优势或物理安全。
