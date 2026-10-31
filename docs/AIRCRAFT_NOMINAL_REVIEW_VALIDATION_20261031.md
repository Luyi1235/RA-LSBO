# 飞机pilot评审与表示诊断：实际验证回执

日期2026-10-31；分支analysis/aircraft-nominal-review-20261031，基础70e34951b41aab662efafdd4fe1feaf6e70bc341。
所有旧优化器、飞机方程、实验配置、历史计划和账本未改。没有运行新的闭环优化或MDA。

## 原始记录核验

原ZIP35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd、manifest
6abee3f22e39b31f75bdd1ed40b61dcbd40b37e5fa230301bf37600e5c2633b1均已下载核验。
1164个归档文件的大小及哈希、pilot709封存证据、smoke92封存证据、两运行各160源码绑定全部匹配。
重建pilot3664调用/112trial/3360提案/112细步长复核；smoke54/14/28/14。
每条调用配对、候选契约、共同初值、提案前缀、首次可行和最优值对应。
没有执行快照、拟合GP、重新计算飞机真值或访问本机原件。

## 只读重建代码与测试

固定提交5f33e4b9174b23e232ebe5d5ce53f7859a58e1ed。
CI：https://github.com/Luyi1235/RA-LSBO/actions/runs/41170562126
20 tests，0 failures/errors/skipped，实际JUnit已下载读取；仅Python标准库数据处理与pytest。
完整原始记录上的脚本执行成功。派生facts/trials/steps在本地与CI解析后完全一致。

Artifact13030767431，nominal-evidence-review-5f33e4b9174b23e232ebe5d5ce53f7859a58e1ed。
ZIP SHA256 fda53738e6dbf922f858e2cc13bdd89477e6802b9a322a91710617400d90cd23，保留30天。
包含实际受测试脚本/测试、JUnit、环境、完整派生数据及日志。
这些20项不是旧327项回归，也不构成新方法性能验证。

## 表示重拟合代码执行

固定提交a72c520043bd31782df802e8cb8b87119e703499。
CI：https://github.com/Luyi1235/RA-LSBO/actions/runs/41170082480
全部16个latent运行的2个前缀，各做30/120epoch，实际64次编码器重拟合及32次one-hot转换。
GP拟合0、优化器重放0、新增数学/简化飞机评价0、新增RCAIDE MDA0。
156项历史Python源码文本绑定检查通过；保留原始快照，仅兼容CRLF/LF读取。
评估入口被替换成报错函数；原始观测仍原样使用。记录当前前缀后不使用未来标签。

Artifact13031130751，representation-diagnostic-a72c520043bd31782df802e8cb8b87119e703499。
ZIP SHA256 b2e6f056c850d533acf6513e94e137ac23a9faf55a01741d77bbc924de33df3b，保留30天。
已下载结果、日志、代码、环境和提交。该脚本在5f33e4b与受测a72c520之间未改动。

29个可比30epoch特征身份与原Windows记录匹配为0，另3个初始覆盖步骤未留原身份。
这是重要限制，不能把Linux诊断冒称原Windows编码器逐字节复现，也不直接认定原因是平台。
30epoch训练语法重构低且120epoch仍不完整，仅构成表示训练/几何问题假说。
解码器训练指标不是留出预测质量；当前选点不使用decoder生成设计，不将低重构单独当作失败因果证明。

## 成本和后续

本轮所有云端CI和本地复核新增物理/数学任务评价0；编码器64次不是零计算。
20个新单元测试使用人工数据和纯算术，不调用研究评估器。没有新的smoke或pilot。
CI成功不代替Windows原环境的表示身份检查，亦不证明校准、算法优势、泛化或安全。
后续文档提交仅增加评审/指南/回执，不修改已测试工具或实验算法。
工作机只运行固定64次表示重拟合及20项诊断测试，输出后停止，不继续训练网格或重跑3718调用。
